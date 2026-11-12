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
    indices=list(range(rank,len(trials),cfg['workers']))
    used,all_scores,all_lengths,all_kept=[],[],[],[]
    started=time.time()
    def infer(view):
        h=torch.stack(plm(torch.from_numpy(np.asarray(view,dtype=np.float32)).to(device),output_hidden_states=True).hidden_states,dim=1)
        z=net.layer_norm(net.linear(net.AHP_ASP(net.bridge_module(h))))
        z=torch.nn.functional.normalize(z,dim=1)
        result=(z@center.T).squeeze(1)
        assert torch.isfinite(result).all()
        return result.cpu().numpy()
    with torch.inference_mode():
        for offset in range(0,len(indices),cfg['batch_size']):
            ids=indices[offset:offset+cfg['batch_size']]
            audio=[]; raw=[]; trimmed=[]; kept=[]
            for idx in ids:
                a,sr=sf.read(DATA/'ASVspoof2021_DF_eval/flac'/(trials[idx]['id']+'.wav'),dtype='float32')
                assert sr==16000 and a.ndim==1 and len(a)>512
                all_lengths.append(len(a)); a=a[:64600]; raw.append(a)
                audio.append(np.pad(a,(0,max(0,64600-len(a))),mode='wrap'))
                frames=np.pad(a,(0,(-len(a))%320)).reshape(-1,320)
                rms=np.sqrt(np.mean(frames**2,axis=1)+1e-16)
                active=np.flatnonzero(rms>=rms.max()*.1)
                left=int(active[0])*320; right=min(len(a),(int(active[-1])+1)*320)
                t=a[left:right]
                if len(t)<512: t=a
                kept.append(len(t)/len(a))
                trimmed.append(np.pad(t,(0,max(0,64600-len(t))),mode='wrap'))
            baseline=np.stack(audio)
            fixed=[baseline,baseline*.25,np.stack([np.pad(a,(0,64600-len(a)),mode='constant') for a in raw]),np.stack([np.pad(a,(0,64600-len(a)),mode='reflect') for a in raw]),np.stack(trimmed)]
            result=[infer(v) for v in fixed]
            result.append(np.concatenate([infer(a[None]) for a in raw]))
            all_scores.append(np.stack(result,axis=1)); all_kept.extend(kept); used.extend(ids)
            if offset==0 or (offset//cfg['batch_size'])%25==0:
                print(f'worker {rank}: {len(used)}/{len(indices)} samples, {time.time()-started:.0f}s',flush=True)
    np.savez_compressed(Path(cfg['out'])/f'shard_{rank}.npz',indices=used,scores=np.concatenate(all_scores),lengths=all_lengths,edge_kept=all_kept)


def render(out,trials,provenance):
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    base=Path('/results/analysis/EBM80_21DF_centroid')
    assert trials==json.loads((base/'sample_manifest.json').read_text())
    old=np.load(base/'embeddings.npz')['cosine']
    shards=[np.load(out/f'shard_{i}.npz') for i in range(provenance['workers'])]
    indices=np.concatenate([s['indices'] for s in shards]); order=np.argsort(indices)
    assert sorted(indices.tolist())==list(range(len(trials)))
    scores=np.concatenate([s['scores'] for s in shards])[order]
    lengths=np.concatenate([s['lengths'] for s in shards])[order]
    kept=np.concatenate([s['edge_kept'] for s in shards])[order]
    labels=np.array([t['label'] for t in trials]); assert np.max(np.abs(scores[:,0]-old))<1e-3
    names=['wrap_baseline','gain_x0p25','zero_padding','reflect_padding','trim_edges_then_wrap','native_no_padding']
    lookup={}
    for line in (DATA/'keys/DF/CM/trial_metadata.txt').read_text().splitlines():
        f=line.split(); lookup[f[1]]=(f[0],f[2],f[3])
    sources=np.array([lookup[t['id']][2] for t in trials]); speakers=np.array([lookup[t['id']][0] for t in trials])
    tail=(labels==1)&(old<.9)
    def stat(a): return {'n':len(a),'mean':float(a.mean()),'median':float(np.median(a))} if len(a) else {'n':0}
    def eer(s,y):
        fpr,tpr,_=roc_curve(y,s,drop_intermediate=False); diff=fpr-(1-tpr); i=np.flatnonzero(diff>=0)[0]
        if i==0: return float(fpr[0]*100)
        w=-diff[i-1]/(diff[i]-diff[i-1]); return float(100*(fpr[i-1]+w*(fpr[i]-fpr[i-1])))
    summary={'provenance':provenance,'baseline_max_error':float(np.max(np.abs(scores[:,0]-old))),'overall':{},'by_source':{},'by_source_padding':{}}
    for j,name in enumerate(names):
        summary['overall'][name]={'sample_eer_pct':eer(scores[:,j],labels),'bona_below09_n':int(((labels==1)&(scores[:,j]<.9)).sum()),'spoof_above02_n':int(((labels==0)&(scores[:,j]>.2)).sum()),'tail_recovered09_n':int((tail&(scores[:,j]>=.9)).sum()),'core_new_below09_n':int(((labels==1)&~tail&(scores[:,j]<.9)).sum()),'tail_delta':stat((scores[:,j]-scores[:,0])[tail]),'spoof_delta':stat((scores[:,j]-scores[:,0])[labels==0])}
        summary['by_source'][name]={}
        for src in np.unique(sources):
            m=(sources==src)&(labels==1)
            summary['by_source'][name][src]={'n':int(m.sum()),'below09_n':int((scores[m,j]<.9).sum()),'mean_score':float(scores[m,j].mean()),'tail_delta':stat((scores[:,j]-scores[:,0])[m&tail])}
            for padded in (True,False):
                g=m&((lengths<64600)==padded)
                key=src+('_padded' if padded else '_not_padded')
                summary['by_source_padding'].setdefault(key,{})[name]={'n':int(g.sum()),'below09_n':int((scores[g,j]<.9).sum()),'tail_n':int((g&tail).sum()),'tail_recovered_n':int((g&tail&(scores[:,j]>=.9)).sum()),'delta':stat((scores[:,j]-scores[:,0])[g])}
    summary['limitations']=['Controlled changes measure this frozen model input sensitivity, not training causes.','Native input changes length and pooling statistics; padding replacements change temporal structure.','Edge trim also changes repetition pattern; it is not an isolated silence-only experiment.','Balanced evaluation sample, no official EER claim; 0.9 and 0.2 are training margins.','No modifications to original audio or training code.']
    np.savez_compressed(out/'intervention_scores.npz',scores=scores,labels=labels,lengths=lengths,edge_kept=kept,sources=sources,speakers=speakers,names=np.array(names))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    fig,axes=plt.subplots(1,3,figsize=(17,5),layout='constrained')
    xx=np.arange(len(names)); short=['Wrap','Gain x0.25','Zero pad','Reflect pad','Edge trim','No pad']
    for src,color in [('asvspoof','#2563A6'),('vcc2018','#D87527'),('vcc2020','#5D8C45')]:
        axes[0].plot(xx,[100*summary['by_source'][n][src]['below09_n']/summary['by_source'][n][src]['n'] for n in names],marker='o',label=src,color=color)
    axes[0].set(title='Bonafide tail rate by source',ylabel='Cosine below 0.9 (%)'); axes[0].legend(fontsize=8)
    axes[1].plot(xx,[summary['overall'][n]['sample_eer_pct'] for n in names],marker='o',color='#333333'); axes[1].set(title='All 2,000 trials: exploratory sample EER',ylabel='EER (%)')
    axes[2].plot(xx,[summary['overall'][n]['tail_delta']['mean'] for n in names],marker='o',label='Original bonafide tail',color='#2563A6')
    axes[2].plot(xx,[summary['overall'][n]['spoof_delta']['mean'] for n in names],marker='^',label='Spoof',color='#D87527'); axes[2].set(title='Mean score change vs original input',ylabel='Cosine change'); axes[2].legend(fontsize=8)
    for ax in axes: ax.set_xticks(xx,short,rotation=35,ha='right'); ax.grid(alpha=.15)
    fig.suptitle('Frozen epoch80 model: controlled input changes on the same 2,000 trials\nPadding interventions only alter clips shorter than 4.04s; tail margin is not a decision threshold',fontsize=13)
    fig.savefig(out/'interventions.png',dpi=170); fig.savefig(out/'interventions.pdf'); plt.close(fig)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['provenance','by_source_padding']},indent=2),flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--per-class', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=4, help='Per GPU inference batch, independent of training')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--cpu', action='store_true', help='CPU-only inference inside container')
    parser.add_argument('--out', default='/results/analysis/EBM80_21DF_tail_interventions')
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
                      embedding='after layer_norm, L2 normalized',
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
