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
    used, scores_all, lengths_all, starts_all = [], [], [], []
    fractions = [0, .25, .5, .75, 1]
    started = time.time()
    with torch.inference_mode():
        for offset in range(0, len(indices), cfg['batch_size']):
            batch_ids = indices[offset:offset+cfg['batch_size']]
            audios=[]
            for idx in batch_ids:
                a,sr=sf.read(DATA/'ASVspoof2021_DF_eval/flac'/(trials[idx]['id']+'.wav'), dtype='float32')
                assert sr==16000 and a.ndim==1 and len(a)>0
                audios.append(a)
            lengths_all.extend([len(a) for a in audios])
            views=[]; starts=[]
            for size in (64600,48440):
                for fraction in fractions:
                    positions=[int(round(max(0,len(a)-size)*fraction)) for a in audios]
                    views.append(np.stack([np.pad(a,(0,max(0,size-len(a))),mode='wrap')[p:p+size] for a,p in zip(audios,positions)]))
                    starts.append(positions)
            # Repeated identical head inputs quantify stochastic pooling noise.
            views.extend([views[0],views[5]])
            batch_scores=[]
            for view in views:
                h=torch.stack(plm(torch.from_numpy(view).to(device),output_hidden_states=True).hidden_states,dim=1)
                z=net.layer_norm(net.linear(net.AHP_ASP(net.bridge_module(h))))
                z=torch.nn.functional.normalize(z,dim=1)
                score=(z@center.T).squeeze(1)
                assert torch.isfinite(score).all()
                batch_scores.append(score.cpu().numpy())
                del h,z
            scores_all.append(np.stack(batch_scores,axis=1))
            starts_all.append(np.stack(starts,axis=1))
            used.extend(batch_ids)
            if offset==0 or (offset//cfg['batch_size'])%25==0:
                print(f'worker {rank}: {len(used)}/{len(indices)} samples, {time.time()-started:.0f}s',flush=True)
    np.savez_compressed(Path(cfg['out'])/f'shard_{rank}.npz',indices=used,scores=np.concatenate(scores_all),lengths=lengths_all,starts=np.concatenate(starts_all))


def render(out,trials,provenance):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    old=Path('/results/analysis/EBM80_21DF_centroid')
    assert trials==json.loads((old/'sample_manifest.json').read_text())
    shards=[np.load(out/f'shard_{r}.npz') for r in range(provenance['workers'])]
    idx=np.concatenate([s['indices'] for s in shards]); order=np.argsort(idx)
    assert sorted(idx.tolist())==list(range(len(trials)))
    scores=np.concatenate([s['scores'] for s in shards])[order]
    lengths=np.concatenate([s['lengths'] for s in shards])[order]
    starts=np.concatenate([s['starts'] for s in shards])[order]
    labels=np.array([t['label'] for t in trials])
    assert np.isfinite(scores).all() and np.max(np.abs(scores))<1.00001
    previous=np.load(old/'embeddings.npz')['cosine']
    delta=np.abs(scores[:,0]-previous)
    assert delta.max()<1e-3, 'Head baseline differs from previous extraction'
    base_low=(labels==1)&(previous<.9)
    def stat(a):
        return {'n':int(len(a)),'mean':float(a.mean()),'median':float(np.median(a)),'q90':float(np.quantile(a,.9))} if len(a) else {'n':0}
    summary={'provenance':provenance,'head_reproduction_max_abs_diff':float(delta.max()),'repeat_noise':stat(np.abs(scores[:,:][:,[0,5]]-scores[:,10:12]).ravel()),'classes':{},'bonafide_groups':{}}
    for label,name in [(1,'Bonafide'),(0,'Spoof')]:
        result={}
        for sl,size,key in [(slice(0,5),64600,'long'),(slice(5,10),48440,'short')]:
            eligible=(labels==label)&(lengths>size)
            v=scores[eligible,sl]; ranges=np.ptp(v,axis=1)
            margin=.9 if label else .2
            result[key]={'eligible_n':int(eligible.sum()),'range':stat(ranges),'range_above_0p2_n':int((ranges>.2).sum()),'straddle_training_margin_n':int(((v.min(1)<margin)&(v.max(1)>=margin)).sum()),'padded_or_no_shift_n':int(((labels==label)&(lengths<=size)).sum())}
        summary['classes'][name]=result
    for islow,name in [(True,'head_below_0p9'),(False,'head_at_least_0p9')]:
        m=(labels==1)&(base_low==islow)
        result={'n':int(m.sum())}
        for sl,size,key in [(slice(0,5),64600,'long'),(slice(5,10),48440,'short')]:
            eligible=m&(lengths>size); v=scores[eligible,sl]
            result[key]={'eligible_n':int(eligible.sum()),'range':stat(np.ptp(v,axis=1)),'any_above_0p9_n':int((v.max(1)>=.9).sum()),'all_below_0p9_n':int((v.max(1)<.9).sum()),'any_below_0p9_n':int((v.min(1)<.9).sum())}
        summary['bonafide_groups'][name]=result
    summary['head_length_change']={name:stat((scores[:,5]-scores[:,0])[labels==l]) for l,name in [(1,'Bonafide'),(0,'Spoof')]}
    summary['limitations']=['Descriptive association, not causal attribution to BPL or EOC-S.', 'Five deterministic overlapping windows per length; range depends on audio duration and window overlap.', 'No-shift/padded clips excluded from range comparisons.', 'Bonafide groups defined by previous long-head score, introducing baseline selection effects.', 'Margins 0.9/0.2 are training margins, not calibrated thresholds or EER.', 'Balanced 2000-trial sample, no augmentation, epoch80 checkpoint only.']
    fig,axes=plt.subplots(2,2,figsize=(14,10),layout='constrained')
    colors={1:'#2563A6',0:'#D87527'}
    for label,name in [(1,'Bonafide'),(0,'Spoof')]:
        m=(labels==label)&(lengths>64600)
        axes[0,0].scatter(scores[m,0],np.ptp(scores[m,:5],axis=1),s=9,alpha=.35,c=colors[label],label=name)
        m=(labels==label)&(lengths>48440)
        axes[0,1].scatter(scores[m,0],np.ptp(scores[m,5:10],axis=1),s=9,alpha=.35,c=colors[label],label=name)
    for ax,title in zip(axes[0],['Same length: five 4.04s positions','Five 3.03s positions']):
        ax.set(title=title,xlabel='Original long-head cosine',ylabel='Maximum minus minimum cosine',xlim=(-1.05,1.05),ylim=(-.03,2.03)); ax.legend(); ax.grid(alpha=.15)
    data=[]; tick=[]
    for sl,size,typ in [(slice(0,5),64600,'4.04s'),(slice(5,10),48440,'3.03s')]:
        for islow,title in [(True,'Low head'),(False,'High head')]:
            m=(labels==1)&(base_low==islow)&(lengths>size)
            data.append(np.ptp(scores[m,sl],axis=1)); tick.append(typ+'\n'+title+f' (n={m.sum()})')
    axes[1,0].boxplot(data,showfliers=False)
    axes[1,0].set_xticks(range(1,5),tick,fontsize=9)
    axes[1,0].set(title='Bonafide: crop sensitivity by baseline group',ylabel='Score range (outliers hidden)'); axes[1,0].grid(axis='y',alpha=.15)
    m=np.flatnonzero(base_low&(lengths>64600))
    # Show 20 examples spanning baseline-score ranks, not cherry-picked crop changes.
    m=m[np.argsort(previous[m])]; chosen=m[np.unique(np.linspace(0,len(m)-1,min(20,len(m))).astype(int))]
    for i in chosen: axes[1,1].plot([0,.25,.5,.75,1],scores[i,:5],alpha=.6,linewidth=1)
    axes[1,1].axhline(.9,c='black',linestyle='--',label='Training margin 0.9')
    axes[1,1].set(title='20 low-head bonafide examples: fixed 4.04s length',xlabel='Relative start in available window range',ylabel='Cosine to learned center',ylim=(-1.05,1.05)); axes[1,1].legend(fontsize=9)
    fig.suptitle('21DF crop sensitivity | epoch80 | five positions per length\nSame checkpoint and 2,000 trials; padded/no-shift cases excluded from range plots',fontsize=14)
    fig.savefig(out/'multicrop.png',dpi=170); fig.savefig(out/'multicrop.pdf'); plt.close(fig)
    np.savez_compressed(out/'crop_scores.npz',scores=scores,labels=labels,lengths=lengths,starts=starts)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2),flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--per-class', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=4, help='Per GPU inference batch, independent of training')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--cpu', action='store_true', help='CPU-only inference inside container')
    parser.add_argument('--out', default='/results/analysis/EBM80_21DF_multicrop')
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
                      embedding='after layer_norm, L2 normalized', crop_lengths=[64600,48440], crop_start_fractions=[0,.25,.5,.75,1], repeats='long and short head',
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
