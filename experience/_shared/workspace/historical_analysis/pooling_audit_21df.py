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
    from contextlib import contextmanager
    pools=[m for m in net.modules() if m.__class__.__name__=='SelfWeightedPooling']
    assert len(pools)==26
    for m in pools: m.noise_std=0.0
    @contextmanager
    def length_scale(reference, final_only=False):
        selected=[net.AHP_ASP] if final_only else pools
        old=[m.forward for m in selected]
        for module, original in zip(selected,old):
            def adjusted(inputs,*args,_original=original,**kwargs):
                v=_original(inputs,*args,**kwargs)
                assert isinstance(v,torch.Tensor)
                n=inputs.size(1)
                factor=n/reference*((n-1)/n*reference/(reference-1))**.5
                return torch.cat([v[:,:128],v[:,128:]*factor],dim=1)
            module.forward=adjusted
        try: yield
        finally:
            for module,original in zip(selected,old): module.forward=original
    def classify(pooled):
        z=net.layer_norm(net.linear(pooled))
        return (torch.nn.functional.normalize(z,dim=1)@center.T).squeeze(1)
    def hidden(a):
        return torch.stack(plm(torch.from_numpy(np.asarray(a,dtype=np.float32)).to(device),output_hidden_states=True).hidden_states,dim=1)
    def score_hidden(h): return classify(net.AHP_ASP(net.bridge_module(h)))
    indices=list(range(rank,len(trials),cfg['workers']))
    used=[]; all_scores=[]; ratios=[]; native_frames=[]; mean_diff=[]; predicted_diff=[]
    started=time.time()
    with torch.inference_mode():
        for offset in range(0,len(indices),cfg['batch_size']):
            ids=indices[offset:offset+cfg['batch_size']]
            raw=[]; fixed=[]
            for idx in ids:
                a,sr=sf.read(DATA/'ASVspoof2021_DF_eval/flac'/(trials[idx]['id']+'.wav'),dtype='float32')
                assert sr==16000 and a.ndim==1
                a=a[:64600]; raw.append(a); fixed.append(np.pad(a,(0,64600-len(a)),mode='wrap'))
            h=hidden(np.stack(fixed)); ref=h.size(2)
            bridge=net.bridge_module(h)
            pooled=net.AHP_ASP(bridge)
            double=net.AHP_ASP(bridge.repeat(1,2,1))
            # No waveform or SSL changes: exact duplication of frozen frame features.
            ratio=double[:,128:].norm(dim=1)/pooled[:,128:].norm(dim=1)
            expected=(2*(ref-1)/(2*ref-1))**.5/2
            assert torch.max(torch.abs(ratio-expected)).item()<1e-5
            diff=(pooled[:,:128]-double[:,:128]).abs().amax(dim=1)
            assert diff.max().item()<1e-4
            scaled=pooled.clone(); scaled[:,128:]*=expected
            pred_diff=(classify(double)-classify(scaled)).abs()
            assert pred_diff.max().item()<1e-4
            result=[classify(pooled),classify(double),score_hidden(h.repeat(1,1,2,1))]
            native=[]; corrected_final=[]; corrected_all=[]
            for a in raw:
                hn=hidden(a[None]); native_frames.append(hn.size(2))
                native.append(score_hidden(hn))
                with length_scale(ref,final_only=True): corrected_final.append(score_hidden(hn))
                with length_scale(ref): corrected_all.append(score_hidden(hn))
            result += [torch.cat(native),torch.cat(corrected_final),torch.cat(corrected_all)]
            all_scores.append(torch.stack(result,dim=1).cpu().numpy())
            ratios.extend(ratio.cpu().tolist()); mean_diff.extend(diff.cpu().tolist()); predicted_diff.extend(pred_diff.cpu().tolist()); used.extend(ids)
            if offset==0 or offset//cfg['batch_size']%50==0:
                print(f'worker {rank}: {len(used)}/{len(indices)}, {time.time()-started:.0f}s',flush=True)
    np.savez_compressed(Path(cfg['out'])/f'shard_{rank}.npz',indices=used,scores=np.concatenate(all_scores),std_ratio=ratios,mean_maxdiff=mean_diff,predicted_score_maxdiff=predicted_diff,native_frames=native_frames,reference_frames=ref)


def render(out,trials,provenance):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    root=Path('/results/analysis')
    assert trials==json.loads((root/'EBM80_21DF_centroid/sample_manifest.json').read_text())
    previous=np.load(root/'EBM80_21DF_tail_interventions/intervention_scores.npz')
    shards=[np.load(out/f'shard_{i}.npz') for i in range(provenance['workers'])]
    ix=np.concatenate([a['indices'] for a in shards]); order=np.argsort(ix)
    assert sorted(ix.tolist())==list(range(2000))
    arrays={k:np.concatenate([a[k] for a in shards])[order] for k in ['scores','std_ratio','mean_maxdiff','predicted_score_maxdiff','native_frames']}
    y=previous['labels']; source=previous['sources']; scores=arrays['scores']; b=y==1; tail=b&(previous['scores'][:,0]<.9)
    names=['baseline_fixed','duplicate_final_features','duplicate_all_SSL_features','native_length','native_scale_final_std','native_scale_all_std']
    assert np.max(np.abs(scores[:,0]-previous['scores'][:,0]))<1e-3
    assert np.max(np.abs(scores[:,3]-previous['scores'][:,5]))<1e-3
    def eer(a,yy):
        fpr,tpr,_=roc_curve(yy,a,drop_intermediate=False); diff=fpr-(1-tpr); i=np.flatnonzero(diff>=0)[0]
        if i==0:return float(100*fpr[0])
        w=-diff[i-1]/(diff[i]-diff[i-1]); return float(100*(fpr[i-1]+w*(fpr[i]-fpr[i-1])))
    summary={'provenance':provenance,'reference_frames':int(shards[0]['reference_frames']),'overall':{},'source':{},
       'std_ratio_mean':float(arrays['std_ratio'].mean()),'mean_branch_maxdiff':float(arrays['mean_maxdiff'].max()),
       'duplication_vs_explicit_std_scaling_max_score_diff':float(arrays['predicted_score_maxdiff'].max()),
       'baseline_reproduction_max_error':float(np.max(np.abs(scores[:,0]-previous['scores'][:,0]))),
       'native_reproduction_max_error':float(np.max(np.abs(scores[:,3]-previous['scores'][:,5])))}
    for j,n in enumerate(names):
        delta=scores[:,j]-scores[:,0]
        summary['overall'][n]={'sample_eer_pct':eer(scores[:,j],y),'bonafide_tail_n':int((scores[b,j]<.9).sum()),
         'bona_mean_delta':float(delta[b].mean()),'bona_median_delta':float(np.median(delta[b])),
         'spoof_mean_delta':float(delta[~b].mean()),'tail_mean_delta':float(delta[tail].mean()),
         'original_tail_recovered_n':int((tail&(scores[:,j]>=.9)).sum())}
        summary['source'][n]={}
        for src in np.unique(source):
            m=source==src
            summary['source'][n][src]={'bona_n':int((b&m).sum()),'spoof_n':int((~b&m).sum()),
             'bona_tail_n':int((b&m&(scores[:,j]<.9)).sum()),'sample_eer_pct':eer(scores[m,j],y[m])}
    short=arrays['native_frames']<summary['reference_frames']
    summary['native_scale_change_short']={n:{'bona_mean_delta_vs_native':float((scores[:,j]-scores[:,3])[b&short].mean()),
       'spoof_mean_delta_vs_native':float((scores[:,j]-scores[:,3])[~b&short].mean())} for j,n in enumerate(names) if j>=4}
    summary['limitations']=['Frozen old epoch80 model, same 1000 bonafide and 1000 spoof evaluation sample.',
      'Feature duplication bypasses waveform and SSL context changes; it isolates downstream pooling sensitivity.',
      'Std length scaling preserves the original reference-length formula and is not canonical weighted standard deviation.',
      'Pooling noise disabled to isolate the deterministic calculation; original outputs reproduced within reported error.',
      'No retraining; scores after intervention do not establish performance of a newly trained architecture.']
    np.savez_compressed(out/'pooling_scores.npz',**arrays,labels=y,sources=source,names=np.array(names))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    fig,ax=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    labels=['Fixed input','Repeat final features','Repeat SSL features','Native length','Native: final std scale','Native: all std scale']
    xx=np.arange(len(names))
    ax[0].bar(xx,[summary['overall'][n]['bonafide_tail_n']/10 for n in names],color='#387A9A'); ax[0].set(ylabel='Bonafide cosine below 0.9 (%)',title='Tail movement (not classification errors)')
    ax[1].bar(xx,[summary['overall'][n]['sample_eer_pct'] for n in names],color='#CE8653'); ax[1].set(ylabel='Sample EER (%)',title='Both classes on the same 2,000 trials')
    for a in ax:a.set_xticks(xx,labels,rotation=30,ha='right'); a.grid(axis='y',alpha=.15)
    fig.suptitle('Frozen-model pooling interventions | no training | duplication uses identical frame features')
    fig.savefig(out/'pooling_audit.png',dpi=170); fig.savefig(out/'pooling_audit.pdf'); plt.close(fig)
    print(json.dumps({k:v for k,v in summary.items() if k!='provenance'},indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--per-class', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=4, help='Per GPU inference batch, independent of training')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--cpu', action='store_true', help='CPU-only inference inside container')
    parser.add_argument('--out', default='/results/analysis/EBM80_21DF_pooling_audit')
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
