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
            audio=[]; raw=[]; trimmed=[]; kept=[]; trim_raw=[]; edge_zero=[]
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
                trim_raw.append(t)
                q=a.copy(); q[:left]=0; q[right:]=0
                edge_zero.append(np.pad(q,(0,64600-len(q)),mode='wrap'))
                trimmed.append(np.pad(t,(0,max(0,64600-len(t))),mode='wrap'))
            baseline=np.stack(audio)
            fixed=[baseline,np.stack(trimmed),
                   np.stack([np.pad(t,(0,64600-len(t))) for t in trim_raw]),
                   np.stack([np.pad(t,((64600-len(t))//2,(64600-len(t)+1)//2)) for t in trim_raw]),
                   np.stack(edge_zero)]
            result=[infer(v) for v in fixed]
            all_scores.append(np.stack(result,axis=1)); all_kept.extend(kept); used.extend(ids)
            if offset==0 or (offset//cfg['batch_size'])%25==0:
                print(f'worker {rank}: {len(used)}/{len(indices)} samples, {time.time()-started:.0f}s',flush=True)
    np.savez_compressed(Path(cfg['out'])/f'shard_{rank}.npz',indices=used,scores=np.concatenate(all_scores),lengths=all_lengths,edge_kept=all_kept)


def render(out,trials,provenance):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    root=Path('/results/analysis')
    assert trials==json.loads((root/'EBM80_21DF_centroid/sample_manifest.json').read_text())
    before=np.load(root/'EBM80_21DF_tail_interventions/intervention_scores.npz')
    shards=[np.load(out/f'shard_{i}.npz') for i in range(provenance['workers'])]
    ix=np.concatenate([v['indices'] for v in shards]); assert sorted(ix.tolist())==list(range(len(trials)))
    order=np.argsort(ix)
    scores=np.concatenate([v['scores'] for v in shards])[order]
    kept=np.concatenate([v['edge_kept'] for v in shards])[order]
    y=before['labels']; sources=before['sources']; b=y==1
    names=['wrap_baseline','trim_then_wrap','trim_then_zero_right','trim_then_zero_center','zero_original_edge_positions']
    baseline_error=float(np.max(np.abs(scores[:,0]-before['scores'][:,0])))
    trim_error=float(np.max(np.abs(scores[:,1]-before['scores'][:,4])))
    assert baseline_error<1e-3 and trim_error<1e-3
    def eer(s):
        fpr,tpr,thr=roc_curve(y,s,drop_intermediate=False); diff=fpr-(1-tpr); i=np.flatnonzero(diff>=0)[0]
        w=-diff[i-1]/(diff[i]-diff[i-1]) if i else 0
        return float(100*(fpr[i-1]+w*(fpr[i]-fpr[i-1]))) if i else float(fpr[0]*100)
    fpr,tpr,thr=roc_curve(y,scores[:,0],drop_intermediate=False)
    baseline_thr=float(thr[np.argmin(np.abs(fpr-(1-tpr)))])
    summary={'provenance':provenance,'baseline_max_error':baseline_error,'trim_max_error':trim_error,
             'baseline_sample_eer_threshold':baseline_thr,'overall':{},'by_source':{}}
    for j,n in enumerate(names):
        summary['overall'][n]={'sample_eer_pct':eer(scores[:,j]),'bona_below09_n':int((scores[b,j]<.9).sum()),
        'spoof_above02_n':int((scores[~b,j]>.2).sum()),'bona_mean_score':float(scores[b,j].mean()),
        'bona_median_score':float(np.median(scores[b,j])),'spoof_mean_score':float(scores[~b,j].mean()),
        'bona_reject_at_baseline_threshold_n':int((scores[b,j]<baseline_thr).sum()),
        'spoof_accept_at_baseline_threshold_n':int((scores[~b,j]>=baseline_thr).sum()),
        'recovery_of_trim_induced_tail_n':int((b&(scores[:,0]>=.9)&(scores[:,1]<.9)&(scores[:,j]>=.9)).sum())}
        summary['by_source'][n]={src:{'n':int((b&(sources==src)).sum()),
        'tail_n':int((b&(sources==src)&(scores[:,j]<.9)).sum()),
        'mean_score':float(scores[b&(sources==src),j].mean())} for src in np.unique(sources)}
    summary['bonafide_edge_removed_fraction']={'median':float(np.median(1-kept[b])),
       'min':float((1-kept[b]).min()),'max':float((1-kept[b]).max())}
    summary['limitations']=['Edge detector is relative frame RMS, not VAD; quiet speech may be removed.',
       'Trim then wrap and trim then zero share the same retained waveform and total model input length, but differ in filler and repetition.',
       'Zero-original-edge-positions preserves original cycle length and placement while replacing low-energy content with exact zeros.',
       'This establishes frozen-model input sensitivity, not the unique training cause or efficacy of retrained multi-centroid methods.',
       'EER is exploratory on 1000+1000 sampled trials; a change in training-margin count need not be an equivalent detection-error change.']
    np.savez_compressed(out/'silence_scores.npz',scores=scores,labels=y,sources=sources,names=np.array(names),edge_kept=kept)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    short=['Original wrap','Trim + repeat','Trim + end zeros','Trim + side zeros','Keep timing; zero edges']
    fig,ax=plt.subplots(1,2,figsize=(13,5.5),layout='constrained'); xx=np.arange(len(names))
    ax[0].bar(xx,[summary['overall'][n]['bona_below09_n']/10 for n in names],color=['#777777','#C96B3C','#3B7C9D','#6B97AF','#699760'])
    ax[0].set(ylabel='Bonafide cosine below 0.9 (%)',title='Sensitivity to low-energy context',ylim=(0,100))
    ax[1].bar(xx,[summary['overall'][n]['sample_eer_pct'] for n in names],color='#777777')
    ax[1].set(ylabel='Sample EER (%)',title='Detection trade-off on the same 2,000 trials')
    for a in ax: a.set_xticks(xx,short,rotation=25,ha='right'); a.grid(axis='y',alpha=.15)
    fig.suptitle('Same speech, different low-energy context | frozen epoch80 model\nAll model inputs are 64,600 samples; training-margin tail is not a classification error',fontsize=13)
    fig.savefig(out/'silence_context.png',dpi=170); fig.savefig(out/'silence_context.pdf'); plt.close(fig)
    print(json.dumps({k:v for k,v in summary.items() if k!='provenance'},indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--per-class', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=4, help='Per GPU inference batch, independent of training')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--cpu', action='store_true', help='CPU-only inference inside container')
    parser.add_argument('--out', default='/results/analysis/EBM80_21DF_tail_silence')
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
