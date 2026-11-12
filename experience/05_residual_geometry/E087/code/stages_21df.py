"""Visualize an EBM checkpoint using a reproducible balanced 21DF sample.

Run inside the mentee container. No training, network access, or source edits.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

sys.dont_write_bytecode = True
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['MPLCONFIGDIR'] = '/home/mentee/.cache/matplotlib'

ROOT = Path('/home/mentee/results/CM/LA/EBM')
CHECKPOINT = ROOT / 'model/BestCLS_80_21DF.pt'
SOURCE = ROOT / 'backup/scripts'
DATA = Path('/data/ASVspoof2021_DF')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def sample_trials(n, seed):
    groups = {0: [], 1: []}
    protocol = DATA / 'keys/DF/CM/trial_metadata.txt'
    for line in protocol.read_text().splitlines():
        fields = line.split()
        if len(fields) < 8 or fields[7] != 'eval':
            continue
        label = int(fields[5] == 'bonafide')
        groups[label].append({'id': fields[1], 'label': label, 'attack': fields[4]})
    rng = random.Random(seed)
    selected = []
    for label in (1, 0):
        if len(groups[label]) < n:
            raise ValueError('Requested sample exceeds population')
        selected.extend(rng.sample(groups[label], n))
    rng.shuffle(selected)
    for row in selected:
        path = DATA / 'ASVspoof2021_DF_eval/flac' / (row['id'] + '.wav')
        if not path.is_file():
            raise FileNotFoundError(path)
    return selected, {str(k): len(v) for k, v in groups.items()}, digest(protocol)


def worker(rank, cfg, trials):
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import Wav2Vec2Model
    sys.path.insert(0, str(SOURCE))
    from model import B_Linear, BPL, OCSoftmax

    torch.set_num_threads(2)
    torch.manual_seed(cfg['seed'] + rank)
    np.random.seed(cfg['seed'] + rank)
    device = torch.device('cpu' if cfg['cpu'] else f'cuda:{rank}')
    if not cfg['cpu']:
        torch.cuda.set_device(rank)
    state = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    state = {k.removeprefix('module.'): v for k, v in state.items()}
    net = B_Linear(num_layer=25, hidden_size=1024, output_size=160, agg_size=128,
                   loss=OCSoftmax(160, r_real=.9, r_fake=.2, alpha=20),
                   loss_bpl=BPL(alpha=30))
    net.load_state_dict(state, strict=True)
    net.to(device).eval()
    plm = Wav2Vec2Model.from_pretrained(cfg['plm_path'], local_files_only=True).to(device).eval()
    center = torch.nn.functional.normalize(net.loss.weight.detach(), dim=1)
    assert center.shape == (1, 160)
    indices = list(range(rank, len(trials), cfg['workers']))
    collected, used = [], []
    stages = {k: [] for k in ('ssl', 'bridge', 'pool', 'linear', 'short')}
    started = time.time()
    with torch.inference_mode():
        for offset in range(0, len(indices), cfg['batch_size']):
            batch_ids = indices[offset:offset + cfg['batch_size']]
            audio_batch = []
            short_batch = []
            for idx in batch_ids:
                path = DATA / 'ASVspoof2021_DF_eval/flac' / (trials[idx]['id'] + '.wav')
                audio, sr = sf.read(path, dtype='float32')
                if sr != 16000 or audio.ndim != 1 or len(audio) == 0:
                    raise ValueError(f'Unexpected audio format: {path}')
                short_crop = 303 * 160 - 40
                start = random.Random(cfg['seed'] * 100000 + idx).randint(0, max(0, len(audio)-short_crop))
                short_audio = np.pad(audio, (0, max(0, short_crop-len(audio))), mode='wrap')[start:start+short_crop]
                short_batch.append(short_audio)
                crop = 404 * 160 - 40
                audio = np.pad(audio, (0, max(0, crop-len(audio))), mode='wrap')[:crop]
                audio_batch.append(audio)
            x = torch.from_numpy(np.stack(audio_batch)).to(device)
            hidden = torch.stack(plm(x, output_hidden_states=True).hidden_states, dim=1)
            bridge = net.bridge_module(hidden)
            pooled = net.AHP_ASP(bridge)
            linear = net.linear(pooled)
            z = net.layer_norm(linear)
            for key, value in [('ssl', hidden[:, -1].mean(1)), ('bridge', bridge.mean(1)), ('pool', pooled), ('linear', linear)]:
                stages[key].append(torch.nn.functional.normalize(value, dim=1).cpu().numpy())
            del hidden, bridge, pooled, linear
            hs = torch.stack(plm(torch.from_numpy(np.stack(short_batch)).to(device), output_hidden_states=True).hidden_states, dim=1)
            zs = net.layer_norm(net.linear(net.AHP_ASP(net.bridge_module(hs))))
            stages['short'].append(torch.nn.functional.normalize(zs, dim=1).cpu().numpy())
            del hs, zs
            z = torch.nn.functional.normalize(z, dim=1)
            if not torch.isfinite(z).all():
                raise ValueError('Non-finite embeddings')
            collected.append(z.cpu().numpy())
            used.extend(batch_ids)
            if offset == 0 or (offset // cfg['batch_size']) % 10 == 0:
                print(f'worker {rank}: {len(used)}/{len(indices)} samples, {time.time()-started:.0f}s', flush=True)
    np.savez_compressed(Path(cfg['out']) / f'shard_{rank}.npz',
                        indices=np.array(used), embeddings=np.concatenate(collected),
                        center=center.cpu().numpy()[0], **{k: np.concatenate(v) for k,v in stages.items()})


def render(out, trials, provenance):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    shards = [np.load(out / f'shard_{r}.npz') for r in range(provenance['workers'])]
    idx = np.concatenate([x['indices'] for x in shards])
    assert sorted(idx.tolist()) == list(range(len(trials)))
    order = np.argsort(idx)
    values = {k: np.concatenate([x[k] for x in shards])[order] for k in ('ssl','bridge','pool','linear','embeddings','short')}
    y = np.array([t['label'] for t in trials])
    center = shards[0]['center']
    for shard in shards:
        np.testing.assert_allclose(center, shard['center'], atol=1e-6)
    for z in values.values():
        assert np.isfinite(z).all()
        np.testing.assert_allclose(np.linalg.norm(z, axis=1), 1, atol=1e-5)
    old = json.loads(Path('/results/analysis/EBM80_21DF_centroid/sample_manifest.json').read_text())
    assert old == trials, 'Sample differs from prior analysis'
    colors = {1:'#2563A6', 0:'#D87527'}
    names = {1:'Bonafide', 0:'Spoof'}
    titles = [('ssl','XLS-R final layer: temporal mean (1024D)'), ('bridge','Bridge: temporal mean (128D)'), ('pool','Attentive statistics pooling (256D)'), ('linear','Linear projection (160D)'), ('embeddings','LayerNorm: input to both losses (160D)')]
    fig, axes = plt.subplots(2,3,figsize=(17,10),layout='constrained')
    summary = {'stages':{}, 'provenance':provenance}
    for ax,(key,title) in zip(axes.flat,titles):
        z = values[key]
        pca = PCA(n_components=2, svd_solver='full').fit(z)
        xy = pca.transform(z)
        stats = {}
        for label in (1,0):
            sub = z[y==label]; n = len(sub); m = sub.mean(0)
            stats[names[label]+'_within_cosine_mean'] = float((np.sum(sub.sum(0,dtype=np.float64)**2)-n)/(n*(n-1)))
            ax.scatter(*xy[y==label].T, c=colors[label], s=8, alpha=.32, marker='o' if label else '^', label=names[label])
            ax.scatter(*pca.transform(m[None])[0], c=colors[label], marker='D', s=75, edgecolors='white')
        stats['between_cosine_mean'] = float(z[y==1].mean(0) @ z[y==0].mean(0))
        stats['pca2_variance'] = float(pca.explained_variance_ratio_.sum())
        stats['dimension'] = z.shape[1]
        if key == 'embeddings':
            ax.scatter(*pca.transform(center[None])[0], c='black', marker='*', s=180, label='Learned OC center')
        summary['stages'][key] = stats
        ax.set_title(title+'\nPCA retained variance: '+f"{stats['pca2_variance']*100:.1f}%")
        ax.set(xlabel='PC1',ylabel='PC2'); ax.grid(alpha=.15); ax.legend(fontsize=8)
    ax = axes.flat[5]
    keys = [t[0] for t in titles]; xx=np.arange(len(keys))
    for field,color,label in [('Bonafide_within_cosine_mean',colors[1],'Within bonafide'),('Spoof_within_cosine_mean',colors[0],'Within spoof'),('between_cosine_mean','#333333','Between classes')]:
        ax.plot(xx,[summary['stages'][k][field] for k in keys], marker='o',color=color,label=label)
    ax.set_xticks(xx,['XLS-R','Bridge','Pooling','Linear','LayerNorm']); ax.set(ylabel='Mean pairwise cosine',ylim=(-1.05,1.05),title='Original-dimensional geometry'); ax.legend(); ax.grid(alpha=.15)
    fig.suptitle('Before BPL / EOC-S | trained epoch-80 model, same 2,000 trials\nEach stage L2-normalized; PCA fitted separately (axes are not shared)',fontsize=15)
    fig.savefig(out/'stage_geometry.png',dpi=170); fig.savefig(out/'stage_geometry.pdf'); plt.close(fig)
    long=values['embeddings']; short=values['short']; paired=np.clip(np.sum(long*short,axis=1),-1,1)
    score=long@center; shortscore=short@center
    def stat(a):
        return {'mean':float(a.mean()),'q05_q50_q95':np.quantile(a,[.05,.5,.95]).tolist()}
    summary['paired_long_short']={names[l]:stat(paired[y==l]) for l in (1,0)}
    b=long[y==1]@short[y==1].T
    off=b[~np.eye(len(b),dtype=bool)]
    summary['bpl_bonafide_cross_utterance']=stat(off)
    summary['eocs']={names[l]:{'long':stat(score[y==l]),'short':stat(shortscore[y==l]),'outside_training_margin_fraction':float(np.mean(score[y==l]<.9 if l else score[y==l]>.2))} for l in (1,0)}
    fig,axes=plt.subplots(1,3,figsize=(17,5),layout='constrained')
    for l in (1,0):
        axes[0].hist(paired[y==l],bins=np.linspace(-1,1,61),histtype='step',density=True,color=colors[l],label=names[l])
        axes[2].scatter(score[y==l],shortscore[y==l],s=8,alpha=.3,c=colors[l],label=names[l])
    axes[0].set(title='Same utterance: long vs short',xlabel='Cosine similarity',ylabel='Density'); axes[0].legend()
    axes[1].hist(np.diag(b),bins=np.linspace(-1,1,61),density=True,histtype='step',label='Same bonafide utterance')
    axes[1].hist(off,bins=np.linspace(-1,1,61),density=True,histtype='step',label='Different bonafide utterances')
    axes[1].set(title='BPL inputs: bonafide long vs short',xlabel='Cosine similarity',ylabel='Density'); axes[1].legend(fontsize=8)
    axes[2].plot([-1,1],[-1,1],c='gray',linestyle='--'); axes[2].set(title='EOC-S inputs: similarity to learned center',xlabel='Long-view cosine',ylabel='Short-view cosine',xlim=(-1.05,1.05),ylim=(-1.05,1.05)); axes[2].legend()
    fig.suptitle('Long 4.04s vs short 3.03s | seeded short crop, no augmentation\nDiagnostic on 21DF evaluation data; not a training-loss reproduction or ablation',fontsize=13)
    fig.savefig(out/'loss_inputs.png',dpi=170); fig.savefig(out/'loss_inputs.pdf'); plt.close(fig)
    summary['limitations']=['All stages belong to the jointly trained model; this is not a loss ablation.', 'Temporal means summarize SSL and bridge sequences, not their complete representations.', 'L2 normalization is applied for stage comparison; independent PCA axes cannot be compared directly.', 'Short crop is seeded per utterance; augmentation disabled. This is not training BPL loss.', 'BPL cross-utterance distribution uses all sampled bonafide pairs, not original training minibatches.', 'Bonafide/spoof sample balanced; no population EER inferred.']
    np.savez_compressed(out/'stage_embeddings.npz',**values,labels=y,center=center)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--per-class', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=4, help='Per GPU inference batch, independent of training')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--cpu', action='store_true', help='CPU-only inference inside container')
    parser.add_argument('--out', default='/results/analysis/EBM80_21DF_stages')
    args = parser.parse_args()
    if not Path('/.dockerenv').exists():
        raise SystemExit('Run only inside the mentee container.')
    if args.per_class < 3 or args.batch_size < 1:
        raise SystemExit('per-class >= 3 and batch-size >= 1 are required')
    import torch
    from huggingface_hub import snapshot_download
    workers = 1 if args.cpu else 2
    if not args.cpu and (os.environ.get('CUDA_VISIBLE_DEVICES') != '0,1' or torch.cuda.device_count() != 2):
        raise SystemExit('Use CUDA_VISIBLE_DEVICES=0,1; exactly two GPUs are required.')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    trials, population, protocol_hash = sample_trials(args.per_class, args.seed)
    plm_path = snapshot_download('facebook/wav2vec2-xls-r-300m', local_files_only=True)
    cfg = dict(vars(args), workers=workers, plm_path=plm_path)
    provenance = dict(cfg, checkpoint=str(CHECKPOINT), checkpoint_sha256=digest(CHECKPOINT),
                      protocol_sha256=protocol_hash, population=population,
                      torch_version=torch.__version__, sampling='uniform random within label, without replacement',
                      crop_samples=64600, sample_rate=16000,
                      embedding='stagewise L2 normalized; short crop seeded per trial, no augmentation', short_crop_samples=48440,
                      source_sha256={f: digest(SOURCE / f) for f in ('model/b_linear.py', 'model/loss.py', 'train.py')})
    (out / 'sample_manifest.json').write_text(json.dumps(trials, indent=2))
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    if args.cpu:
        worker(0, cfg, trials)
    else:
        torch.multiprocessing.spawn(worker, args=(cfg, trials), nprocs=2, join=True)
    render(out, trials, provenance)


if __name__ == '__main__':
    main()
