"""CPU-only audit; execute on server, keeping all file-level results there."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
import concurrent.futures as cf
import collections
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import soundfile as sf
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path('/home/mentee/results/analysis/EBM_padding_length_20261030_v1')
SOURCE = Path('/home/mentee/code/EBM')
PRIOR = Path('/home/mentee/results/analysis/ASV_quiet_BPL_distribution_20261030_v1')
TARGETS = {'long': 64600, 'short': 48440}
os.nice(10)

def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def hostpath(p):
    return p.replace('/data/', '/home/mentee/DB/', 1)

def records():
    rows = []
    protocols = []
    base = Path('/home/mentee/DB/ASVspoof2019/LA')
    for split, tail in [('train', 'train.trn'), ('dev', 'dev.trl'), ('eval', 'eval.trl')]:
        p = base / 'ASVspoof2019_LA_cm_protocols' / f'ASVspoof2019.LA.cm.{tail}.txt'
        protocols.append(p)
        for line in p.read_text().splitlines():
            f = line.split()
            rows.append(dict(dataset='19LA_' + split, id=f[1], label=int(f[4]=='bonafide'),
                             attack=f[3], source='asvspoof', speaker=f[0],
                             path=str(base / f'ASVspoof2019_LA_{split}/flac/{f[1]}.flac')))
    for kind in ['LA', 'DF']:
        base = Path('/home/mentee/DB/ASVspoof2021_LA_eval' if kind=='LA' else '/home/mentee/DB/ASVspoof2021_DF')
        p = base / f'keys/{kind}/CM/trial_metadata.txt'
        protocols.append(p)
        for line in p.read_text().splitlines():
            f = line.split()
            if f[7] != 'eval':
                continue
            path = base / (f'flac/{f[1]}.flac' if kind=='LA' else f'ASVspoof2021_DF_eval/flac/{f[1]}.wav')
            rows.append(dict(dataset=f'21{kind}_eval', id=f[1], label=int(f[4 if kind=='LA' else 5]=='bonafide'),
                             attack=f[3 if kind=='LA' else 4], source='asvspoof' if kind=='LA' else f[3],
                             speaker=f[0], codec=f[2], path=str(path)))
    return rows, protocols

def header(row):
    try:
        info = sf.info(row['path'])
        return info.frames, info.samplerate, info.channels, None
    except Exception as e:
        return -1, -1, -1, type(e).__name__ + ': ' + str(e)

def describe(x):
    x = np.asarray(x, dtype=float)
    if not len(x):
        return None
    return dict(n=len(x), mean=float(x.mean()), median=float(np.median(x)),
                p10=float(np.quantile(x, .1)), p25=float(np.quantile(x, .25)),
                p75=float(np.quantile(x, .75)), p90=float(np.quantile(x, .9)),
                min=float(x.min()), max=float(x.max()))

def group_summary(indices, rows, frames):
    n = frames[indices]
    ans = {'duration_s': describe(n / 16000), 'n': len(n), 'views': {}}
    for key, target in TARGETS.items():
        deficit = np.maximum(0, target-n)
        q = deficit/target
        padded = deficit > 0
        joins = np.maximum(0, (target-1)//n)
        bins = [0, 16000, 32000, 48440, 64600, 96000, 160000, 10**12]
        ans['views'][key] = dict(
            padded_n=int(padded.sum()), padded_pct=float(padded.mean()*100),
            added_fraction_all_pct=float(q.mean()*100),
            added_fraction_padded_pct=float(q[padded].mean()*100) if padded.any() else None,
            added_s_all=float(deficit.mean()/16000),
            majority_added_pct=float((q>.5).mean()*100),
            at_least_two_joins_pct=float((joins>=2).mean()*100),
            at_least_three_joins_pct=float((joins>=3).mean()*100),
            mean_joins=float(joins.mean()),
            added_fraction_padded_quantiles=describe(q[padded]*100),
        )
    ans['duration_histogram'] = dict(
        edges_s=[0, 1, 2, 3.0275, 4.0375, 6, 10, None],
        counts=np.histogram(n, bins=bins)[0].tolist(),
        percentages=(np.histogram(n, bins=bins)[0]/len(n)*100).tolist())
    return ans

def discrimination(indices, rows, frames):
    y = np.array([rows[i]['label'] for i in indices])
    n = frames[indices]
    if len(set(y))<2:
        return None
    ans = {}
    for name, value in [('shorter_file_is_bona', -n),
                        ('long_wrap_fraction_is_bona', np.maximum(0,64600-n)/64600),
                        ('short_wrap_fraction_is_bona', np.maximum(0,48440-n)/48440)]:
        fpr, tpr, thresholds = roc_curve(y, value)
        j = int(np.argmin(np.abs(fpr-(1-tpr))))
        ans[name] = dict(auc=float(roc_auc_score(y,value)),
                         approximate_eer_pct=float((fpr[j]+1-tpr[j])/2*100))
    for name, target in TARGETS.items():
        bona = n[y==1]<target
        spoof = n[y==0]<target
        ans[name+'_padded_is_bona_balanced_accuracy_pct'] = float((bona.mean()+1-spoof.mean())/2*100)
    return ans

def quiet_one(row):
    a, sr = sf.read(hostpath(row['path']), dtype='float64')
    assert sr==16000 and a.ndim==1 and len(a)>0
    n = len(a)
    k = (n+319)//320
    padded = np.pad(a,(0,k*320-n))
    widths = np.full(k,320)
    widths[-1] = n-320*(k-1)
    rms = np.sqrt(np.sum(padded.reshape(k,320)**2,axis=1)/widths)
    result = dict(dataset=row['dataset'], label=row['label'], n=n, thresholds={})
    for db in [20,40]:
        mask = rms<rms.max()*10**(-db/20) if rms.max()>0 else np.ones(k,bool)
        q = np.repeat(mask,320)[:n]
        cu = np.r_[0,np.cumsum(q)]
        stats = {'raw_quiet_pct':float(q.mean()*100), 'views':{}}
        for view,target in TARGETS.items():
            added = max(0,target-n)
            cycles,rem = divmod(added,n)
            added_quiet = int(cycles*cu[-1]+cu[rem])
            original_quiet = int(cu[min(n,target)])
            p = np.pad(a,(0,added),'wrap') if added else a[:target]
            joins = int((target-1)//n) if added else 0
            jump = abs(a[-1]-a[0]) if added else 0
            peak = math.sqrt(float(np.mean(a*a)))
            stats['views'][view] = dict(added=added, added_quiet=added_quiet,
                original_quiet=original_quiet, final_quiet=original_quiet+added_quiet,
                exact_periodic=bool(np.array_equal(p[n:],p[:-n])) if added else None,
                joins=joins, boundary_jump_over_rms=float(jump/max(peak,1e-15)))
        result['thresholds'][str(db)]=stats
    return result

def main():
    ROOT.mkdir(exist_ok=False)
    start=time.time()
    rows,protocols=records()
    keys=[(r['dataset'],r['id']) for r in rows]
    assert len(keys)==len(set(keys))
    source_hashes={str(p):digest(p) for p in [SOURCE/'data_loaders.py',SOURCE/'arguments.py',SOURCE/'data_processing/augmentation/codec_aug.py']}
    plan=dict(rows=len(rows), counts={str(k):v for k,v in collections.Counter((r['dataset'],r['label']) for r in rows).items()},
              targets_samples=TARGETS, protocol_hashes={str(p):digest(p) for p in protocols},source_hashes=source_hashes,
              scope='Full protocol census, exact current loader eval filters; 21DF all sources and ASV-only separately. No model execution.',
              worker_threads=4)
    (ROOT/'plan.json').write_text(json.dumps(plan,indent=2))
    print('PLAN',json.dumps(plan),flush=True)
    frames=np.empty(len(rows),np.int64)
    errors=[];sr_counts=collections.Counter();channel_counts=collections.Counter()
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        for i,(n,sr,ch,error) in enumerate(pool.map(header,rows)):
            frames[i]=n;sr_counts[sr]+=1;channel_counts[ch]+=1
            if error:errors.append({'index':i,'error':error})
            if (i+1)%50000==0: print('HEADERS',i+1,'seconds',round(time.time()-start),flush=True)
    assert not errors, errors[:3]
    assert sr_counts=={16000:len(rows)} and channel_counts=={1:len(rows)}
    assert np.all(frames>0)
    with gzip.open(ROOT/'file_lengths_private.csv.gz','wt') as f:
        w=csv.writer(f);w.writerow(['dataset','id','label','source','speaker','attack','frames'])
        for row,n in zip(rows,frames):w.writerow([row[k] for k in ['dataset','id','label','source','speaker','attack']]+[int(n)])
    summary={'plan':plan,'checks':dict(unique_keys=True,errors=errors,sample_rates=dict(sr_counts),channels=dict(channel_counts)),
             'groups':{},'discrimination':{},'attacks':{}}
    ds_names=sorted({r['dataset'] for r in rows})
    for ds in ds_names+['21DF_ASV_eval']:
        ix=np.array([i for i,r in enumerate(rows) if r['dataset']==ds or (ds=='21DF_ASV_eval' and r['dataset']=='21DF_eval' and r['source']=='asvspoof')])
        summary['groups'][ds]={str(lab):group_summary(np.array([i for i in ix if rows[i]['label']==lab]),rows,frames) for lab in [1,0]}
        summary['discrimination'][ds]=discrimination(ix,rows,frames)
    for ds in ds_names:
        keys=sorted({(r['attack'] if ds.startswith('19') else r['source']) for r in rows if r['dataset']==ds})
        summary['attacks'][ds]={}
        for key in keys:
            ix=np.array([i for i,r in enumerate(rows) if r['dataset']==ds and (r['attack'] if ds.startswith('19') else r['source'])==key])
            summary['attacks'][ds][key]=dict(n=len(ix), bonafide_n=sum(rows[i]['label'] for i in ix),
                mean_duration_s=float(frames[ix].mean()/16000),
                long_padded_pct=float((frames[ix]<64600).mean()*100),
                long_added_pct=float((np.maximum(0,64600-frames[ix])/64600).mean()*100))
    lookup={(r['dataset'],r['id']):int(n) for r,n in zip(rows,frames)}
    native=json.load(open(PRIOR/'native_manifest_private.json'))
    nf=np.load(PRIOR/'native_metrics_private.npz')['features']
    mismatches=[]
    for i,r in enumerate(native):
        ds='21DF_eval' if r['dataset']=='21DF_ASV_eval' else r['dataset']
        if abs(lookup[ds,r['id']]/16000-float(nf[i,0,0]))>2e-5:mismatches.append(i)
    oldplan=json.load(open(PRIOR/'plan.json'))
    assert digest(SOURCE/'data_loaders.py')==oldplan['source_sha256']
    assert all(digest(hostpath(p))==h for p,h in oldplan['protocol_sha256'].items())
    assert not mismatches
    summary['checks']['prior_source_and_protocol_hashes_match']=True
    summary['checks']['prior_duration_agreement_n']=len(native)
    pairs=[json.loads(line) for line in gzip.open(PRIOR/'pair_metadata_private.jsonl.gz','rt')]
    summary['actual_augmented_train_draws']={}
    for lab in [1,0]:
        rr=[r for r in pairs if r['label']==lab]
        stats={'draws':len(rr),'unique_files':len({r['id'] for r in rr}),'views':{}}
        for view,tag in [('long','L'),('short','S')]:
            q=np.array([r[tag+'_wrap_fraction'] for r in rr])
            raw=np.array([max(0,TARGETS[view]-round(r['raw_duration_s']*16000))/TARGETS[view] for r in rr])
            stats['views'][view]=dict(padded_pct=float((q>0).mean()*100),added_pct=float(q.mean()*100),
                padded_added_pct=float(q[q>0].mean()*100),raw_same_files_padded_pct=float((raw>0).mean()*100),
                raw_same_files_added_pct=float(raw.mean()*100),
                codec_changed_padding_status_pct=float(((q>0)!=(raw>0)).mean()*100),
                mean_codec_length_change_ms=float(np.mean([1000*(r[tag+'_before_crop_s']-min(r['raw_duration_s'],TARGETS[view]*1.2/16000) if r[tag+'_prefix_capped'] else r[tag+'_before_crop_s']-r['raw_duration_s']) for r in rr])))
        summary['actual_augmented_train_draws'][str(lab)]=stats
    rng=np.random.default_rng(20261030)
    sample=[]
    for ds in sorted({r['dataset'] for r in native}):
        for lab in [1,0]:
            rr=[r for r in native if r['dataset']==ds and r['label']==lab]
            selected=rng.choice(len(rr),size=min(1000,len(rr)),replace=False)
            sample.extend(rr[i] for i in selected)
    quiet=[]
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        for i,result in enumerate(pool.map(quiet_one,sample),1):
            quiet.append(result)
            if i%1000==0:print('QUIET',i,len(sample),'seconds',round(time.time()-start),flush=True)
    with gzip.open(ROOT/'quiet_repetition_private.jsonl.gz','wt') as f:
        for r in quiet:f.write(json.dumps(r)+'\n')
    summary['quiet_repeat_sample']={}
    for ds in sorted({r['dataset'] for r in quiet}):
        summary['quiet_repeat_sample'][ds]={}
        for lab in [1,0]:
            rr=[r for r in quiet if r['dataset']==ds and r['label']==lab]
            group={'n':len(rr),'thresholds':{}}
            for db in ['20','40']:
                result={'mean_raw_quiet_pct':float(np.mean([r['thresholds'][db]['raw_quiet_pct'] for r in rr])),'views':{}}
                for view,target in TARGETS.items():
                    vv=[r['thresholds'][db]['views'][view] for r in rr]
                    padded=[v for v in vv if v['added']>0]
                    assert all(v['exact_periodic'] for v in padded)
                    result['views'][view]=dict(padded_n=len(padded),
                        added_lowenergy_pct_of_input=float(np.mean([v['added_quiet']/target*100 for v in vv])),
                        final_lowenergy_pct_of_input=float(np.mean([v['final_quiet']/target*100 for v in vv])),
                        added_lowenergy_pct_of_added_samples=float(sum(v['added_quiet'] for v in vv)/max(1,sum(v['added'] for v in vv))*100),
                        join_jump_over_rms=describe([v['boundary_jump_over_rms'] for v in padded]))
                group['thresholds'][db]=result
            summary['quiet_repeat_sample'][ds][str(lab)]=group
    toy=np.array([1,2,3]);assert np.array_equal(np.pad(toy,(0,7),'wrap'),[1,2,3,1,2,3,1,2,3,1])
    assert all(digest(p)==h for p,h in source_hashes.items())
    summary['checks']['toy_wrap_and_exact_periodicity']=True
    summary['checks']['source_unchanged']=True
    summary['seconds']=time.time()-start
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2))
    print('COMPLETE',ROOT,'seconds',round(time.time()-start),flush=True)
    for ds,gg in summary['groups'].items():
        print('RESULT',ds,json.dumps({lab:dict(n=g['n'],mean_s=g['duration_s']['mean'],median_s=g['duration_s']['median'],long=g['views']['long'],short=g['views']['short']) for lab,g in gg.items()}),flush=True)

if __name__=='__main__':
    main()
