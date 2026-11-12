"""Train-only crop I/O, speech-activity and50-epoch sampler audit. No learning."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
import multiprocessing as mp
import os
import random
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import arguments
from positive_bpl import MatchedTrainSet,MatchedBatchSampler,rawboost

DS=None
ROOT=None


def save_json(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')


def activity(wave):
    import webrtcvad
    x=np.asarray(wave,dtype=np.float64)
    assert x.ndim==1 and len(x)%320==0 and np.isfinite(x).all()
    frames=x.reshape(-1,320)
    energy=np.sqrt((frames**2).mean(1))
    db=20*np.log10(np.maximum(energy,1e-12))
    pcm=np.clip(np.rint(x*32768),-32768,32767).astype('<i2').reshape(-1,320)
    clips=[row.tobytes() for row in pcm]
    result={'frames':len(frames),'seconds':len(x)/16000,
            'rms_dbfs':float(20*np.log10(max(float(np.sqrt((x*x).mean())),1e-12))),
            'frames_below_minus50_dbfs':int((db < -50).sum()),
            'frames_below_minus60_dbfs':int((db < -60).sum())}
    for mode in [0,2,3]:
        vad=webrtcvad.Vad(mode)
        flags=np.fromiter((vad.is_speech(row,16000) for row in clips),dtype=bool,count=len(clips))
        result[f'vad{mode}_speech_frames']=int(flags.sum())
        result[f'vad{mode}_speech_fraction']=float(flags.mean())
    result['vad2_speech_seconds']=.02*result['vad2_speech_frames']
    result['vad2_zero_speech']=result['vad2_speech_frames']==0
    result['vad2_speech_below100ms']=result['vad2_speech_seconds'] < .1
    result['vad2_speech_fraction_below20pct']=result['vad2_speech_fraction'] < .2
    return result


def audit_crop(task):
    serial,epoch,rank,step,role,spec,augmented=task
    item=DS.items[spec.index]
    raw=DS.read_crop(spec,augment=False).numpy()
    row={'serial':serial,'epoch':epoch,'rank':rank,'step':step,'role':role,
         'UID':Path(item.path).stem,'label':item.label,'speaker':item.speaker,'attack':item.type,
         'start_sample':spec.start,'stop_sample':spec.stop,'source_samples':int(DS.lengths[spec.index]),
         'crop_seconds':len(raw)/16000,'touches_start':spec.start==0,
         'touches_end':spec.stop==DS.lengths[spec.index]}
    row.update(activity(raw))
    if augmented:
        after=activity(DS.read_crop(spec,augment=True).numpy())
        row.update({'aug_'+k:v for k,v in after.items()})
    return row


def interval_size(intervals):
    if not intervals:return 0
    intervals.sort();a,b=intervals[0];total=0
    for start,stop in intervals[1:]:
        if start>b:total+=b-a;a,b=start,stop
        else:b=max(b,stop)
    return total+b-a


def full_crop(ds,spec):
    wave,sr=sf.read(ds.items[spec.index].path)
    assert sr==16000 and wave.ndim==1 and len(wave)==ds.lengths[spec.index]
    return torch.as_tensor(wave[spec.start:spec.stop].copy(),dtype=torch.float32)


def benchmark(tasks,out):
    groups=defaultdict(list)
    for task in tasks:
        spec=task[5];item=DS.items[spec.index]
        groups[(item.label,spec.stop-spec.start)].append(task)
    chosen=[]
    for key,rows in sorted(groups.items()):
        chosen.extend(sorted(rows,key=lambda x:hashlib.sha256(str(x[:6]).encode()).digest())[:32])
    # First verify all selected intervals before timing; both paths see warm files.
    for task in chosen:
        assert torch.equal(full_crop(DS,task[5]),DS.read_crop(task[5],augment=False))
    augmented_equal=0
    for task in chosen[::8]:
        spec=task[5];wave,_=sf.read(DS.items[spec.index].path)
        wave=wave[spec.start:spec.stop].copy()
        py_state,np_state=random.getstate(),np.random.get_state()
        try:
            random.seed(spec.augmentation_seed);np.random.seed(spec.augmentation_seed)
            expected=rawboost(wave,random.randint(0,8))
        finally:
            random.setstate(py_state);np.random.set_state(np_state)
        assert torch.equal(torch.as_tensor(expected,dtype=torch.float32),DS.read_crop(spec,augment=True))
        augmented_equal+=1
    samples=[]
    for repeat in range(3):
        for i,task in enumerate(chosen):
            order=['full','partial'] if (repeat+i)%2==0 else ['partial','full']
            for kind in order:
                start=time.perf_counter_ns()
                value=full_crop(DS,task[5]) if kind=='full' else DS.read_crop(task[5],augment=False)
                elapsed=(time.perf_counter_ns()-start)/1e6
                samples.append({'repeat':repeat,'serial':task[0],'label':DS.items[task[5].index].label,
                                'crop_seconds':len(value)/16000,'kind':kind,'milliseconds':elapsed})
    df=pd.DataFrame(samples);df.to_csv(out/'io_timings.csv',index=False)
    totals=df.groupby(['repeat','kind']).milliseconds.sum().unstack()
    by_length=df.groupby(['crop_seconds','kind']).milliseconds.mean().unstack()
    result={'pairs_verified_exact':len(chosen),'repeats':3,'all_samples_equal':True,
            'timing_scope':'warm-cache CPU file open,decode,slice/header checks,float32 tensor conversion; no augmentation or GPU',
            'median_total_full_ms':float(totals['full'].median()),
            'median_total_partial_ms':float(totals['partial'].median()),
            'median_paired_round_speedup':float((totals['full']/totals['partial']).median()),
            'by_length':by_length.reset_index().to_dict('records')}
    result['augmented_pairs_verified_exact']=augmented_equal
    means=df.groupby(['label','crop_seconds','kind']).milliseconds.mean()
    epoch_counts={key:len(rows) for key,rows in groups.items()}
    expected={kind:sum(count*means.loc[(label,length/16000,kind)] for (label,length),count in epoch_counts.items())/sum(epoch_counts.values()) for kind in ['full','partial']}
    result['epoch1_weighted_mean_ms']=expected
    result['epoch1_weighted_speedup']=expected['full']/expected['partial']
    save_json(out/'io_benchmark.json',result)
    return result


def coverage(epoch,counts,seen,intervals,lengths,labels):
    rows=[]
    for label,roles in [(1,[0]),(0,[1]),(1,[0,2])]:
        name='bona_OC' if roles==[0] else 'spoof_OC' if roles==[1] else 'bona_OC_plus_BPL'
        ids=np.flatnonzero(labels==label)
        actual=counts[roles].sum(0)[ids]
        previous=seen[roles].sum(0)[ids]
        selected=np.count_nonzero(actual)
        merged=defaultdict(list)
        for role in roles:
            for idx,parts in intervals[role].items():merged[idx].extend(parts)
        covered=sum(interval_size(parts.copy()) for parts in merged.values())
        rows.append({'epoch':epoch,'population':name,'originals':len(ids),'draws':int(actual.sum()),
                     'unique_originals':int(selected),'unique_fraction':selected/len(ids),
                     'duplicate_draw_fraction':1-selected/actual.sum(),
                     'cumulative_unique_originals':int(np.count_nonzero(previous)),
                     'cumulative_unique_fraction':np.count_nonzero(previous)/len(ids),
                     'covered_original_seconds':covered/16000,
                     'total_original_seconds':float(lengths[ids].sum()/16000),
                     'original_time_fraction':covered/lengths[ids].sum(),
                     'never_seen_this_epoch':int((actual==0).sum())})
    return rows


def summarize_activity(df):
    outputs=[]
    for (role,label,duration),g in df.groupby(['role','label','crop_seconds']):
        row={'role':role,'label':int(label),'crop_seconds':float(duration),'crops':len(g),
             'voiced_seconds_mean':float(g.vad2_speech_seconds.mean()),
             'voiced_seconds_median':float(g.vad2_speech_seconds.median()),
             'vad2_speech_fraction':float(g.vad2_speech_frames.sum()/g.frames.sum()),
             'vad0_speech_fraction':float(g.vad0_speech_frames.sum()/g.frames.sum()),
             'vad3_speech_fraction':float(g.vad3_speech_frames.sum()/g.frames.sum()),
             'zero_speech_crops':int(g.vad2_zero_speech.sum()),
             'zero_speech_fraction':float(g.vad2_zero_speech.mean()),
             'speech_below100ms_fraction':float(g.vad2_speech_below100ms.mean()),
             'speech_below20pct_fraction':float(g.vad2_speech_fraction_below20pct.mean()),
             'quiet_frames_below_minus50_fraction':float(g.frames_below_minus50_dbfs.sum()/g.frames.sum())}
        outputs.append(row)
    return pd.DataFrame(outputs)


def main(root):
    global DS,ROOT
    os.nice(10);torch.set_num_threads(1)
    ROOT=Path(root);out=ROOT/'results';out.mkdir(exist_ok=False)
    source=Path('/data/ASVspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt')
    audio_root=Path('/data/ASVspoof2019/LA/ASVspoof2019_LA_train/flac')
    items=[]
    for line in source.read_text().splitlines():
        fields=line.split();assert len(fields)==5
        speaker,uid,_,attack,kind=fields;y=int(kind=='bonafide')
        path=audio_root/(uid+'.flac')
        assert path.parent==audio_root
        items.append(SimpleNamespace(path=str(path),speaker=speaker,label=y,type='bonafide' if y else attack))
    args,_,_=arguments.get_args()
    args.update(bpl_pairing='matched_independent',duration_policy_path=str(ROOT/'policy.json'),
                rank=0,world_size=2,batch_size=120,num_workers=0)
    DS=MatchedTrainSet(args,items)
    assert len(DS)==25380 and len(DS.profiles)==140
    assert not torch.cuda.is_initialized()
    labels=np.array([v.label for v in DS.items]);lengths=DS.lengths
    fingerprint={'train_protocol_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                 'policy_sha256':DS.policy_sha256,'source_fingerprint':DS.source_fingerprint,
                 'Train_originals':len(DS),'profiles':len(DS.profiles),'seed':args['rand_seed'],
                 'epochs':[1,50],'draws_per_epoch':28512,'dev_used':False,'eval_used':False,
                 'model_training':False,'audio_whitelist':str(audio_root),
                 'speech_activity':'WebRTC VAD2 primary; VAD0/3 sensitivity; monoPCM16,16kHz,20ms,reset each crop',
                 'voice_metrics_caveat':'automatic speech-activity estimates, not human speech labels; not used to filter or tune training',
                 'rawboost_check':'64 selected crops per class/length fromepoch1, capped by available count; same stored augmentation seeds',
                 'coverage_definition':'original UID selected at least once; time fraction is union of chosen intervals per original/epoch, not repeated duration sum'}
    save_json(out/'provenance.json',fingerprint)
    seen=np.zeros((3,len(DS)),dtype=np.int32);coverage_rows=[];group_rows=[];length_rows=[]
    epoch1=[];role_names=['OC_bona','OC_spoof','BPL_right_bona'];start_time=time.monotonic()
    pool=None;activity_iterator=None
    for epoch in range(1,51):
        counts=np.zeros_like(seen);intervals=[defaultdict(list) for _ in range(3)]
        durations=np.zeros((3,8),dtype=np.int64)
        for rank in range(2):
            sampler=MatchedBatchSampler(DS,dict(args,rank=rank));sampler.set_epoch(epoch)
            for step,batch in enumerate(sampler):
                for role,section in enumerate([batch[:12],batch[12:120],batch[120:]]):
                    for spec in section:
                        counts[role,spec.index]+=1
                        intervals[role][spec.index].append((spec.start,spec.stop))
                        durations[role,(spec.stop-spec.start)//8000-1]+=1
                        if epoch==1:
                            epoch1.append((len(epoch1),epoch,rank,step,role_names[role],spec,False))
        seen+=counts;coverage_rows.extend(coverage(epoch,counts,seen,intervals,lengths,labels))
        for role in range(3):
            for k in range(8):length_rows.append({'epoch':epoch,'role':role_names[role],'seconds':.5*(k+1),'draws':int(durations[role,k])})
        for key,profile in DS.profiles.items():
            ids=np.array([i for i,k in enumerate(DS.keys) if k==key]);y,speaker,attack=key
            oc=counts[0 if y else 1,ids]
            group_rows.append({'epoch':epoch,'label':y,'speaker':speaker,'attack':attack,
                               'originals':len(ids),'OC_draws':int(oc.sum()),
                               'OC_unique':int(np.count_nonzero(oc)),
                               'OC_unique_fraction':float(np.count_nonzero(oc)/len(ids))})
        if epoch==1:
            io=benchmark(epoch1,out)
            strata=defaultdict(list)
            for task in epoch1:
                spec=task[5];strata[(DS.items[spec.index].label,spec.stop-spec.start)].append(task)
            aug_ids=set()
            for key,tasks in strata.items():
                aug_ids.update(t[0] for t in sorted(tasks,key=lambda t:hashlib.sha256(str(t[:6]).encode()).digest())[:64])
            epoch1=[(*task[:6],task[0] in aug_ids) for task in epoch1]
            save_json(out/'activity_plan.json',{'epoch1_crops':len(epoch1),'augmented_subset':len(aug_ids),'roles':role_names})
            pool=mp.get_context('fork').Pool(4)
            activity_iterator=pool.imap_unordered(audit_crop,epoch1,chunksize=32)
        pd.DataFrame(coverage_rows).to_csv(out/'epoch_coverage.csv',index=False)
        save_json(ROOT/'progress.json',{'stage':'Train50epoch sampler replay plus epoch1 activity','epochs_done':epoch,
                                      'elapsed_seconds':time.monotonic()-start_time})
    activity_rows=[]
    for row in activity_iterator:
        activity_rows.append(row)
        if len(activity_rows)%1024==0:
            save_json(ROOT/'progress.json',{'stage':'Train crop activity','epochs_done':50,
                       'activity_done':len(activity_rows),'activity_total':len(epoch1),'elapsed_seconds':time.monotonic()-start_time})
    pool.close();pool.join()
    df=pd.DataFrame(activity_rows).sort_values('serial');assert len(df)==28512 and df.serial.nunique()==28512
    df.to_csv(out/'epoch1_crop_activity.csv.gz',index=False,compression='gzip')
    summary=summarize_activity(df);summary.to_csv(out/'activity_by_length_role.csv',index=False)
    oc_summary=summarize_activity(df[df.role!='BPL_right_bona'].assign(role='OC'))
    oc_summary.to_csv(out/'OC_activity_by_length.csv',index=False)
    groups=df.groupby(['role','label']).agg(crops=('serial','size'),frames=('frames','sum'),speech_frames=('vad2_speech_frames','sum'),
                    zero_speech_crops=('vad2_zero_speech','sum'),short_speech_crops=('vad2_speech_below100ms','sum'),
                    quiet_frames=('frames_below_minus50_dbfs','sum')).reset_index()
    groups['speech_frame_fraction']=groups.speech_frames/groups.frames;groups.to_csv(out/'activity_overall.csv',index=False)
    aug=df[df.aug_vad2_speech_fraction.notna()].copy()
    aug.groupby(['label','crop_seconds']).agg(crops=('serial','size'),raw_speech_fraction=('vad2_speech_fraction','mean'),
        augmented_speech_fraction=('aug_vad2_speech_fraction','mean'),raw_zero=('vad2_zero_speech','mean'),
        augmented_zero=('aug_vad2_zero_speech','mean')).reset_index().to_csv(out/'rawboost_activity_check.csv',index=False)
    pd.DataFrame(group_rows).to_csv(out/'epoch_group_coverage.csv',index=False)
    pd.DataFrame(length_rows).to_csv(out/'epoch_length_draws.csv',index=False)
    usage=pd.DataFrame([{'UID':Path(v.path).stem,'label':v.label,'speaker':v.speaker,'attack':v.type,
        'source_seconds':DS.lengths[i]/16000,'OC_draws_50':int(seen[0 if v.label else 1,i]),
        'BPL_right_draws_50':int(seen[2,i])} for i,v in enumerate(DS.items)])
    usage.to_csv(out/'original_usage_50epochs.csv',index=False)
    cov=pd.DataFrame(coverage_rows)
    result={'status':'complete','dev_used':False,'eval_used':False,'model_training':False,
            'Train_originals':len(DS),'epoch1_crop_count':len(df),'epochs_replayed':50,
            'io_benchmark':io,'coverage_summary':cov.groupby('population').agg(
                epoch_unique_min=('unique_fraction','min'),epoch_unique_mean=('unique_fraction','mean'),
                epoch_unique_max=('unique_fraction','max'),original_time_mean=('original_time_fraction','mean')).reset_index().to_dict('records'),
            'coverage_epoch1':cov[cov.epoch==1].to_dict('records'),
            'coverage_epoch50':cov[cov.epoch==50].to_dict('records'),
            'activity_overall':groups.to_dict('records'),
            'OC_activity_by_length':oc_summary.to_dict('records'),
            'never_used_after50_OC':int((usage.OC_draws_50==0).sum()),
            'total_elapsed_seconds':time.monotonic()-start_time,'CUDA_initialized':torch.cuda.is_initialized()}
    assert not torch.cuda.is_initialized()
    save_json(out/'analysis.json',result)
    save_json(ROOT/'progress.json',{'stage':'complete','epochs_done':50,'activity_done':len(df)})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();main(a.root)
