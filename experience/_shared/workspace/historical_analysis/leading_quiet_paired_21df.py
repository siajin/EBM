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

    names=['original','identity_repeat','zero20_006','zero20_048','zero20_102',
           'ownquiet20_006','ownquiet20_048','ownquiet20_102',
           'variable20_006','variable20_048','zero40_006','zero40_048','zero40_102']
    def fixed(a):
        a=a[:64600]
        return np.pad(a,(0,64600-len(a)),mode='wrap')
    def infer(a):
        h=torch.stack(plm(torch.from_numpy(np.asarray(a,dtype=np.float32)).to(device),output_hidden_states=True).hidden_states,dim=1)
        z=net.layer_norm(net.linear(net.AHP_ASP(net.bridge_module(h))))
        score=(torch.nn.functional.normalize(z,dim=1)@center.T).squeeze(1)
        assert torch.isfinite(score).all()
        return score.cpu().numpy()
    def onset(a,relative):
        f=np.pad(a,(0,(-len(a))%320)).astype(np.float64).reshape(-1,320)
        rms=np.sqrt(np.mean(f*f,axis=1)+1e-16)
        assert rms.max()>1e-7
        return int(np.flatnonzero(rms>=rms.max()*relative)[0])*320
    def make_views(a):
        targets=[960,7680,16320]; bodylen=64600-targets[-1]
        left20=onset(a,.1); left40=onset(a,.01)
        views=[a,a.copy()]; details=[left20,left40]; error=0.
        bodies={}
        for key,left in [('20',left20),('40',left40)]:
            available=a[left:]; body=np.zeros(bodylen,dtype=np.float32)
            kept=min(bodylen,len(available)); body[:kept]=available[:kept]
            bodies[key]=body
            details.extend([kept,max(0,len(available)-bodylen)])
        q=np.resize(a[:left20],16320) if left20 else np.zeros(16320,dtype=np.float32)
        for kind in ['zero20','ownquiet20','variable20','zero40']:
            body=bodies['40' if kind=='zero40' else '20']
            fill=q if kind=='ownquiet20' else np.zeros(16320,dtype=np.float32)
            this=[]
            for t in targets[:2] if kind=='variable20' else targets:
                v=np.concatenate([fill[:t],body] + ([] if kind=='variable20' else [fill[t:]]))
                assert np.array_equal(v[t:t+bodylen],body)
                if kind!='variable20':
                    assert len(v)==64600
                    # Identical sample multiset and body for every fixed-length dose.
                    assert np.array_equal(np.sort(v),np.sort(np.concatenate([fill,body])))
                this.append(v)
            if kind!='variable20':
                energies=[float(np.sum(v.astype(np.float64)**2)) for v in this]
                error=max(error,max(energies)-min(energies))
            views.extend(this)
        assert len(views)==len(names) and error<1e-8
        # All transformed views retain exactly the same body within their family.
        return views,details+[error], [onset(v,.1)/16000 for v in views]
    dev_all=[t for t in json.loads(Path('/results/analysis/EBM80_21DF_cue_audit/sample_manifest.json').read_text()) if t['split']=='dev']
    dev=[]
    for label in [1,0]:dev.extend([t for t in dev_all if t['label']==label][:cfg['per_class']])
    dev_ix=list(range(rank,len(dev),cfg['workers'])); dev_scores=[]
    started=time.time()
    with torch.inference_mode():
        for offset in range(0,len(dev_ix),cfg['batch_size']):
            ids=dev_ix[offset:offset+cfg['batch_size']]; batch=[]
            for idx in ids:
                a,sr=sf.read(dev[idx]['path'],dtype='float32'); assert sr==16000 and a.ndim==1
                batch.append(fixed(a))
            dev_scores.extend(infer(np.stack(batch)).tolist())
        print(f'worker {rank}: calibration {len(dev_ix)} done, {time.time()-started:.0f}s',flush=True)
        indices=list(range(rank,len(trials),cfg['workers'])); used=[]; all_scores=[]; checks=[]; measured=[]
        for offset in range(0,len(indices),cfg['batch_size']):
            ids=indices[offset:offset+cfg['batch_size']]; batch=[[] for _ in names]
            for idx in ids:
                a,sr=sf.read(DATA/'ASVspoof2021_DF_eval/flac'/(trials[idx]['id']+'.wav'),dtype='float32')
                assert sr==16000 and a.ndim==1
                views,detail,lead=make_views(fixed(a)); checks.append(detail); measured.append(lead)
                for j,v in enumerate(views):batch[j].append(v)
            scores=np.stack([infer(np.stack(v)) for v in batch],axis=1)
            all_scores.append(scores); used.extend(ids)
            if offset==0 or offset//cfg['batch_size']%25==0:print(f'worker {rank}: paired {len(used)}/{len(indices)}, {time.time()-started:.0f}s',flush=True)
    np.savez_compressed(Path(cfg['out'])/f'shard_{rank}.npz',indices=used,scores=np.concatenate(all_scores),checks=checks,measured_leading_s=measured,
        calibration_indices=dev_ix,calibration_scores=dev_scores,calibration_labels=[dev[i]['label'] for i in dev_ix])


def render(out,trials,provenance):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve,roc_auc_score
    names=['original','identity_repeat','zero20_006','zero20_048','zero20_102',
           'ownquiet20_006','ownquiet20_048','ownquiet20_102',
           'variable20_006','variable20_048','zero40_006','zero40_048','zero40_102']
    families={'zero20':[2,3,4],'ownquiet20':[5,6,7],'variable20':[8,9,4],'zero40':[10,11,12]}
    shards=[np.load(out/f'shard_{i}.npz') for i in range(provenance['workers'])]
    ix=np.concatenate([v['indices'] for v in shards]); order=np.argsort(ix); assert sorted(ix.tolist())==list(range(len(trials)))
    vals={k:np.concatenate([v[k] for v in shards])[order] for k in ['scores','checks','measured_leading_s']}
    scores=vals['scores']; labels=np.array([t['label'] for t in trials]); b=labels==1
    cal_ix=np.concatenate([v['calibration_indices'] for v in shards]); co=np.argsort(cal_ix)
    assert sorted(cal_ix.tolist())==list(range(len(trials)))
    cal=np.concatenate([v['calibration_scores'] for v in shards])[co]
    cy=np.concatenate([v['calibration_labels'] for v in shards])[co]
    assert np.sum(cy==1)==np.sum(cy==0)
    mid=len(cal)//2; sortedcal=np.sort(cal); assert sortedcal[mid-1]<sortedcal[mid]
    threshold=(float(sortedcal[mid-1])+float(sortedcal[mid]))/2
    def metric(a,y):
        fpr,tpr,_=roc_curve(y,a,drop_intermediate=False); diff=fpr+tpr-1; i=np.flatnonzero(diff>=0)[0]
        if i==0:e=float(100*fpr[0])
        else:
            w=-diff[i-1]/(diff[i]-diff[i-1]);e=float(100*(fpr[i-1]+w*(fpr[i]-fpr[i-1])))
        fr=float(100*np.mean(a[y==1]<threshold)); fa=float(100*np.mean(a[y==0]>=threshold))
        return {'eer_pct':e,'auc':float(roc_auc_score(y,a)),'bona_reject_pct':fr,'spoof_accept_pct':fa,'balanced_error_pct':(fr+fa)/2}
    meta={}
    for line in (DATA/'keys/DF/CM/trial_metadata.txt').read_text().splitlines():
        f=line.split();meta[f[1]]=(f[0],f[3])
    src=np.array([meta[t['id']][1] for t in trials]); spk=np.array([meta[t['id']][0] for t in trials])
    delta=scores[:,[0]]-scores
    identity_error=float(np.max(np.abs(delta[:,1]))); eps=max(1e-4,10*identity_error)
    summary={'provenance':provenance,'threshold':threshold,'threshold_source':'19LA dev unchanged balanced sample, pooled-median EER operating point',
       'calibration':metric(cal,cy),'baseline':metric(scores[:,0],labels),'identity_repeat_max_delta':identity_error,
       'meaningful_delta_epsilon':eps,'score_definition':'spoof score = -cosine; positive change means more spoof-like, not a probability',
       'primary_contrast':'1.02s prefix minus 0.06s prefix, same file and identical body within each family',
       'pairwise_vs_original':{},'four_conditions':{},'dose':{},'input_checks':{}}
    if len(trials)==2000:
        root=Path('/results/analysis/EBM80_21DF_centroid')
        assert trials==json.loads((root/'sample_manifest.json').read_text())
        original=np.load(root/'embeddings.npz')['cosine']
        summary['baseline_reproduction_max_error']=float(np.max(np.abs(scores[:,0]-original)))
        assert summary['baseline_reproduction_max_error']<1e-3
    rng=np.random.default_rng(813)
    def ci(values,mask):
        # Speaker-cluster bootstrap within source and label; source weights fixed.
        if values.ndim==1:values=values[:,None]
        boot=np.zeros((1000,values.shape[1]))
        for source in np.unique(src[mask]):
            m=mask&(src==source)
            groups=[np.flatnonzero(m&(spk==s)) for s in np.unique(spk[m])]
            sizes=np.array([len(g) for g in groups]); sums=np.stack([values[g].sum(0) for g in groups])
            draws=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=1000)
            boot+=(draws@sums)/(draws@sizes)[:,None]*(m.sum()/mask.sum())
        return np.quantile(boot,[.025,.975],axis=0)
    cis={y:ci(delta,labels==y) for y in [1,0]}
    for j,n in enumerate(names):
        summary['pairwise_vs_original'][n]={}
        for y,label in [(1,'bonafide'),(0,'spoof')]:
            m=labels==y; dd=delta[m,j]; before=(scores[m,0]>=threshold)==bool(y); after=(scores[m,j]>=threshold)==bool(y)
            summary['pairwise_vs_original'][n][label]={'n':int(m.sum()),'mean_delta_spoof':float(dd.mean()),'median_delta_spoof':float(np.median(dd)),
              'speaker_cluster_95ci':cis[y][:,j].tolist(),'more_spoof_pct':float(100*np.mean(dd>eps)),
              'correct_to_wrong_n':int((before&~after).sum()),'wrong_to_correct_n':int((~before&after).sum())}
        summary['four_conditions'][n]={k:metric(v,labels) for k,v in {
          'B_S':scores[:,0],'TB_S':np.where(b,scores[:,j],scores[:,0]),'B_TS':np.where(b,scores[:,0],scores[:,j]),'TB_TS':scores[:,j]}.items()}
    for family,idx in families.items():
        # Positive values mean adding leading quiet makes the model MORE spoof-like.
        d=scores[:,idx[0]]-scores[:,idx[2]]
        entry={}
        for source,m in [('all',np.ones(len(labels),dtype=bool))]+[(s,src==s) for s in np.unique(src)]:
            if len(np.unique(labels[m]))<2:continue
            e={'sample_counts':{'bonafide':int((m&b).sum()),'spoof':int((m&~b).sum())},
               'both_classes_by_dose':[metric(scores[m,j],labels[m]) for j in idx]}
            for y,label in [(1,'bonafide'),(0,'spoof')]:
                g=m&(labels==y); dd=d[g]
                intervals=ci(d,g)[:,0].tolist()
                before=(scores[g,idx[0]]>=threshold)==bool(y); after=(scores[g,idx[2]]>=threshold)==bool(y)
                e[label]={'mean_delta_long_minus_short':float(dd.mean()),'median_delta_long_minus_short':float(np.median(dd)),
                   'speaker_cluster_95ci':intervals,'more_spoof_pct':float(100*np.mean(dd>eps)),'less_spoof_pct':float(100*np.mean(dd<-eps)),
                   'mean_spoof_scores':[-float(scores[g,j].mean()) for j in idx],
                   'correct_to_wrong_n':int((before&~after).sum()),'wrong_to_correct_n':int((~before&after).sum())}
            entry[source]=e
        summary['dose'][family]=entry
    cc=vals['checks']; summary['input_checks']={'energy_permutation_max_error':float(cc[:,-1].max()),
       'original_leading_median_s':{},'no_own_quiet_n':int((cc[:,0]==0).sum()),
       'common_body_duration_s':48280/16000,'fixed_total_duration_s':64600/16000,
       'variable_total_durations_s':[(48280+k)/16000 for k in [960,7680,16320]],
       'cropped_right_samples20_median':float(np.median(cc[:,3])),'cropped_right20_n':int((cc[:,3]>0).sum()),
       'short_body_padded20_n':int((cc[:,2]<48280).sum()),'speech_body_exact_equality_assertions':True,'fixed_family_sample_multiset_assertions':True}
    for source,m in [('all',np.ones(len(labels),dtype=bool))]+[(s,src==s) for s in np.unique(src)]:
        summary['input_checks']['original_leading_median_s'][source]={label:float(np.median(cc[m&(labels==y),0])/16000) if (m&(labels==y)).any() else None for y,label in [(1,'bonafide'),(0,'spoof')]}
    summary['limitations']=['Low-energy onset uses 20ms frame RMS relative to maximum, not certified silence or speech annotation. Strict -40dB onset is a sensitivity check.',
       'Fixed-length variants hold body samples, total duration, sample multiset, total energy and total filler duration constant. Leading quiet grows by moving quiet from the end, so this tests placement and leading/trailing balance, not leading duration in isolation.',
       'A common body of at most 3.0175s is selected after the original low-energy prefix and padded if short. It is bit-identical across doses, but may omit later content relative to original. Direct original comparisons therefore mix editing effects.',
       'Own-quiet variants repeat each file original leading quiet to a common 1.02s filler; no leading quiet falls back to zeros. Repetition and joins may themselves be cues.',
       'Variable-duration variants keep the same 3.0175s body and zero prefix but remove trailing filler; total duration changes. Consistency with fixed duration is informative, not a perfect isolation of all factors.',
       'Frozen-model paired sensitivity plus label-cue association strengthens a shortcut hypothesis but cannot identify the historical training mechanism or prove non-speech artifacts are illegitimate.',
       'Independent 19LA dev threshold is used for errors and flips; EER separately changes threshold and is exploratory on a balanced sample. Threshold domain shift affects absolute errors.',
       'Bootstrap intervals condition on this checkpoint and resample speaker clusters within source and true label; numerous subgroup comparisons are descriptive, not multiplicity-adjusted tests.']
    np.savez_compressed(out/'paired_scores.npz',**vals,labels=labels,sources=src,speakers=spk,names=np.array(names),calibration_scores=cal,calibration_labels=cy)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained'); xx=[.06,.48,1.02]
    titles={'zero20':'Zeros; fixed total duration (-20dB onset)','ownquiet20':'Own quiet texture; fixed total duration','variable20':'Zeros; variable total duration','zero40':'Zeros; fixed duration (-40dB onset)'}
    for ax,(family,idx) in zip(axes.flat,families.items()):
        for y,label,color in [(1,'bonafide','#287DA8'),(0,'spoof','#D07A2B')]:
            g=labels==y; change=scores[:,[idx[0]]]-scores[:,idx]; intervals=ci(change,g); means=change[g].mean(0)
            ax.plot(xx,means,'o-',label=label,color=color);ax.fill_between(xx,intervals[0],intervals[1],alpha=.15,color=color)
        ax.axhline(0,lw=1,color='#888');ax.set(title=titles[family],xlabel='Inserted leading quiet (seconds)',ylabel='Spoof-score change vs 0.06s');ax.grid(alpha=.15);ax.legend()
    fig.suptitle('Same-file leading-quiet intervention | frozen epoch80 | positive = more spoof-like\nIdentical body within each family; shading = 95% speaker-cluster interval',fontsize=13)
    fig.savefig(out/'leading_quiet_effect.png',dpi=170);fig.savefig(out/'leading_quiet_effect.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for ax,family in zip(axes,['zero20','ownquiet20']):
        for source,color in [('asvspoof','#2867A0'),('vcc2018','#BB6F31'),('vcc2020','#4D8C55')]:
            if source not in summary['dose'][family]:continue
            for label,marker,ls in [('bonafide','o','-'),('spoof','^','--')]:
                mm=np.array(summary['dose'][family][source][label]['mean_spoof_scores']);ax.plot(xx,mm-mm[0],marker=marker,ls=ls,color=color,label=f'{source} {label}')
        ax.axhline(0,c='#999',lw=1);ax.set(title=titles[family],xlabel='Inserted leading quiet (seconds)',ylabel='Spoof-score change vs 0.06s');ax.legend(fontsize=8);ax.grid(alpha=.15)
    fig.suptitle('Source-specific paired response: pooled averages can hide opposing directions')
    fig.savefig(out/'leading_quiet_sources.png',dpi=170);fig.savefig(out/'leading_quiet_sources.pdf');plt.close(fig)
    print(json.dumps({k:summary[k] for k in ['threshold','calibration','baseline','identity_repeat_max_delta','input_checks','dose']},indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--per-class', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=4, help='Per GPU inference batch, independent of training')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--cpu', action='store_true', help='CPU-only inference inside container')
    parser.add_argument('--out', default='/results/analysis/EBM80_21DF_leading_quiet_paired')
    parser.add_argument('--render-only', action='store_true')
    args = parser.parse_args()
    if not Path('/.dockerenv').exists():
        raise SystemExit('Run only inside the mentee container.')
    if args.render_only:
        out = Path(args.out)
        render(out,json.loads((out/'sample_manifest.json').read_text()),json.loads((out/'provenance.json').read_text()))
        return
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
    provenance = dict(cfg, analysis_script_sha256=digest(Path(__file__)), checkpoint=str(CHECKPOINT), checkpoint_sha256=digest(CHECKPOINT),
                      protocol_sha256=protocol_hash, population=population,
                      torch_version=torch.__version__, sampling='uniform random within label, without replacement',
                      crop_samples=64600, sample_rate=16000,
                      embedding='after layer_norm, L2 normalized',
                      source_sha256={f: digest(SOURCE / f) for f in ('model/b_linear.py', 'model/loss.py', 'train.py', 'data_loaders.py', 'data_processing/augmentation/raw_boost.py', 'data_processing/augmentation/codec_aug.py')})
    (out / 'sample_manifest.json').write_text(json.dumps(trials, indent=2))
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    if args.cpu:
        worker(0, cfg, trials)
    else:
        torch.multiprocessing.spawn(worker, args=(cfg, trials), nprocs=2, join=True)
    render(out, trials, provenance)


if __name__ == '__main__':
    main()
