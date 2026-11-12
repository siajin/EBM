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
    for m in net.modules():
        if m.__class__.__name__=='SelfWeightedPooling':m.noise_std=0.
    indices=list(range(rank,len(trials),cfg['workers']))
    used=[]; all_scores=[]; all_fraction=[]; energy_errors=[]; mean_errors=[]; started=time.time()
    def infer(a):
        h=torch.stack(plm(torch.from_numpy(np.asarray(a,dtype=np.float32)).to(device),output_hidden_states=True).hidden_states,dim=1)
        z=net.layer_norm(net.linear(net.AHP_ASP(net.bridge_module(h))))
        return (torch.nn.functional.normalize(z,dim=1)@center.T).squeeze(1).cpu().numpy()
    with torch.inference_mode():
        for offset in range(0,len(indices),cfg['batch_size']):
            ids=indices[offset:offset+cfg['batch_size']]; views=[[] for _ in range(5)]
            for idx in ids:
                raw,sr=sf.read(DATA/'ASVspoof2021_DF_eval/flac'/(trials[idx]['id']+'.wav'),dtype='float32')
                assert sr==16000 and raw.ndim==1
                raw=raw[:64600]; a=np.pad(raw,(0,64600-len(raw)),mode='wrap')
                chunks=[a[k:min(k+320,len(a))] for k in range(0,len(a),320)]
                rms=np.array([np.sqrt(np.mean(v.astype(np.float64)**2)+1e-20) for v in chunks])
                masks=[]
                for relative in [.01,.1]:
                    active=np.flatnonzero(rms>=rms.max()*relative)
                    mask=np.zeros(len(a),dtype=bool); mask[:active[0]*320]=True; mask[min((active[-1]+1)*320,len(a)):]=True
                    masks.append(mask)
                rng=np.random.default_rng(170000+idx)
                zero=a.copy(); perm40=a.copy(); noise40=a.copy(); perm20=a.copy()
                zero[masks[0]]=0
                err=0.; merr=0.
                for k,v in enumerate(chunks):
                    lo=k*320; hi=lo+len(v)
                    if masks[0][lo]:
                        shuffled=rng.permutation(v); perm40[lo:hi]=shuffled
                        noise=rng.normal(size=len(v)); orig_rms=np.sqrt(np.mean(v.astype(np.float64)**2))
                        noise*=orig_rms/np.sqrt(np.mean(noise**2)); noise40[lo:hi]=noise
                        err=max(err,abs(np.mean(shuffled.astype(np.float64)**2)-np.mean(v.astype(np.float64)**2)),abs(np.mean(noise40[lo:hi].astype(np.float64)**2)-np.mean(v.astype(np.float64)**2)))
                        merr=max(merr,abs(shuffled.astype(np.float64).mean()-v.astype(np.float64).mean()))
                    if masks[1][lo]:perm20[lo:hi]=rng.permutation(v)
                assert err<1e-10 and merr<1e-10
                for j,v in enumerate([a,zero,perm40,noise40,perm20]):views[j].append(v)
                all_fraction.append([float(m.mean()) for m in masks]); energy_errors.append(err); mean_errors.append(merr)
            all_scores.append(np.stack([infer(np.stack(v)) for v in views],axis=1)); used.extend(ids)
            if offset==0 or offset//cfg['batch_size']%50==0: print(f'worker {rank}: {len(used)}/{len(indices)}, {time.time()-started:.0f}s',flush=True)
    np.savez_compressed(Path(cfg['out'])/f'shard_{rank}.npz',indices=used,scores=np.concatenate(all_scores),fractions=all_fraction,energy_error=energy_errors,permutation_mean_error=mean_errors)

def render(out,trials,provenance):
    import numpy as np
    from sklearn.metrics import roc_curve
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path('/results/analysis')
    assert trials==json.loads((root/'EBM80_21DF_centroid/sample_manifest.json').read_text())
    previous=np.load(root/'EBM80_21DF_tail_interventions/intervention_scores.npz')
    shards=[np.load(out/f'shard_{i}.npz') for i in range(provenance['workers'])]
    ix=np.concatenate([v['indices'] for v in shards]); order=np.argsort(ix); assert sorted(ix.tolist())==list(range(2000))
    values={k:np.concatenate([v[k] for v in shards])[order] for k in ['scores','fractions','energy_error','permutation_mean_error']}
    scores=values['scores']; fractions=values['fractions']; y=previous['labels']; b=y==1; src=previous['sources']
    names=['baseline','zero_edges_40dB','permute_edges_40dB','rms_matched_noise_edges_40dB','permute_edges_20dB']
    assert np.max(np.abs(scores[:,0]-previous['scores'][:,0]))<1e-3
    def eer(a):
        fpr,tpr,_=roc_curve(y,a,drop_intermediate=False); diff=fpr-(1-tpr); i=np.flatnonzero(diff>=0)[0]
        if i==0:return float(100*fpr[0])
        w=-diff[i-1]/(diff[i]-diff[i-1]); return float(100*(fpr[i-1]+w*(fpr[i]-fpr[i-1])))
    threshold=-.5619228482246399
    summary={'provenance':provenance,'overall':{},'by_source':{},'selected_fraction':{},
       'energy_preservation_max_error':float(values['energy_error'].max()),'permutation_mean_max_error':float(values['permutation_mean_error'].max()),
       'baseline_reproduction_max_error':float(np.max(np.abs(scores[:,0]-previous['scores'][:,0])))}
    for k,condition in enumerate(['40dB','20dB']):
        summary['selected_fraction'][condition]={label:{'affected_n':int((fractions[m,k]>0).sum()),'median_fraction':float(np.median(fractions[m,k])),'mean_fraction':float(fractions[m,k].mean())} for label,m in [('bonafide',b),('spoof',~b)]}
    for j,n in enumerate(names):
        delta=scores[:,j]-scores[:,0]
        summary['overall'][n]={'sample_eer_pct':eer(scores[:,j]),'bona_tail_n':int((scores[b,j]<.9).sum()),
         'bona_mean_score':float(scores[b,j].mean()),'spoof_mean_score':float(scores[~b,j].mean()),
         'bona_mean_delta':float(delta[b].mean()),'spoof_mean_delta':float(delta[~b].mean()),
         'bona_mean_absolute_delta':float(np.abs(delta[b]).mean()),'spoof_mean_absolute_delta':float(np.abs(delta[~b]).mean()),
         'bona_reject_at_baseline_threshold_n':int((scores[b,j]<threshold).sum()),'spoof_accept_at_baseline_threshold_n':int((scores[~b,j]>=threshold).sum())}
        summary['by_source'][n]={str(k):{'bona_n':int((b&(src==k)).sum()),'bona_tail_n':int((b&(src==k)&(scores[:,j]<.9)).sum()),'bona_mean_delta':float(delta[b&(src==k)].mean())} for k in np.unique(src)}
        if j:
            nochange=fractions[:,1 if j==4 else 0]==0
            if nochange.any():assert np.max(np.abs(delta[nochange]))<1e-3
    summary['limitations']=['Only leading/trailing regions below the relative frame-RMS threshold are modified; no duration or padding changes.',
       '40dB below peak-frame RMS is a stricter low-energy proxy, not certified speech/non-speech annotation.',
       'Within-frame permutation preserves sample histogram, frame RMS, mean, duration and placement, but changes temporal/spectral texture.',
       'RMS-matched white noise preserves per-frame energy but changes waveform statistics and can be out of distribution.',
       'Frozen model sensitivity does not prove that the cue is spurious; genuine synthesis artifacts can occur in low-energy regions.',
       'Same balanced evaluation sample, not official full-set EER; no training or source modifications.']
    np.savez_compressed(out/'quiet_texture_scores.npz',**values,labels=y,sources=src,names=np.array(names))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    fig,ax=plt.subplots(1,2,figsize=(13,5),layout='constrained'); xx=np.arange(len(names)); short=['Original','Zero edges (-40dB)','Shuffle edges (-40dB)','Noise edges (-40dB)','Shuffle edges (-20dB)']
    ax[0].bar(xx,[summary['overall'][n]['bona_tail_n']/10 for n in names],color='#397C98'); ax[0].set(ylabel='Bonafide cosine below 0.9 (%)',title='Input length and edge positions stay fixed')
    ax[1].bar(xx,[summary['overall'][n]['sample_eer_pct'] for n in names],color='#C48350'); ax[1].set(ylabel='Sample EER (%)',title='Only low-energy edge texture is changed')
    for a in ax:a.set_xticks(xx,short,rotation=28,ha='right'); a.grid(axis='y',alpha=.15)
    fig.suptitle('Quiet-edge interventions | frozen epoch80 | 1,000 bonafide + 1,000 spoof')
    fig.savefig(out/'quiet_texture.png',dpi=170); fig.savefig(out/'quiet_texture.pdf'); plt.close(fig)
    print(json.dumps({k:v for k,v in summary.items() if k!='provenance'},indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--per-class', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=4, help='Per GPU inference batch, independent of training')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--cpu', action='store_true', help='CPU-only inference inside container')
    parser.add_argument('--out', default='/results/analysis/EBM80_21DF_quiet_texture')
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
