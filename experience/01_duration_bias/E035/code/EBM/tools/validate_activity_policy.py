"""Train-only expected/realized joint-policy validation; no model training."""
from collections import defaultdict
import hashlib,itertools,json,multiprocessing as mp,os
from pathlib import Path
import sys,time
import numpy as np
import pandas as pd
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import arguments
from positive_bpl import MatchedTrainSet,MatchedBatchSampler
from dataset.ASVspoof2019_la_train import load_train_items_only

DS=None


def voice_fraction(wave):
    import webrtcvad
    x=np.asarray(wave,dtype=np.float64);assert len(x)%320==0
    frames=np.clip(np.rint(x*32768),-32768,32767).astype('<i2').reshape(-1,320)
    vad=webrtcvad.Vad(2)
    count=sum(vad.is_speech(row.tobytes(),16000) for row in frames)
    return count,len(frames)


def inspect_audio(task):
    serial,spec=task
    old=DS.read_crop(spec,augment=False).numpy();aug=DS.read_crop(spec,augment=True).numpy()
    raw,n=voice_fraction(old);after,_=voice_fraction(aug)
    return {'serial':serial,'raw_speech_frames':raw,'augmented_speech_frames':after,'total_frames':n}


def union_length(parts):
    parts=sorted(parts)
    if not parts:return 0
    a,b=parts[0];total=0
    for start,stop in parts[1:]:
        if start>b:total+=b-a;a,b=start,stop
        else:b=max(b,stop)
    return total+b-a


def expected(ds,data,kind):
    n,k=len(ds),len(ds.targets);offsets=data['context_offsets']
    context=np.repeat(np.arange(n*k),np.diff(offsets));orig=context//k;length=context%k
    if kind=='before':
        lp=np.stack([ds.tables[ds.keys[i]][ds.buckets[i]-1] for i in range(n)])
        mult=data['multiplicity'].astype(float)
        pos=mult/np.bincount(context,weights=mult,minlength=n*k)[context]
    else:lp=ds.activity_policy.lengths;pos=ds.activity_policy.positions
    label=np.array([v.label for v in ds.items])[orig]
    frames=ds.targets[length]//320;voice=data['speech_frames'];fraction=voice/frames
    bins=((voice.astype(np.int64)*5+frames-1)//frames).astype(int)
    mass=ds.source_probability[orig]*lp[orig,length]*pos
    hist=np.bincount(label*k*6+length*6+bins,weights=mass,minlength=2*k*6).reshape(2,k,6)
    rows=[]
    for y in [1,0]:
        for l in range(k):
            selected=(label==y)&(length==l);total=mass[selected].sum()
            vhist=np.bincount(voice[selected],weights=mass[selected],minlength=int(ds.targets[l]//320)+1)/total
            row={'policy':kind,'label':y,'seconds':float(ds.targets[l]/16000),
                 'length_probability':float(total),'mean_speech_fraction':float((mass[selected]*fraction[selected]).sum()/total),
                 'zero_speech_fraction':float(mass[selected&(voice==0)].sum()/total),
                 'speech_below100ms_fraction':float(mass[selected&(voice<5)].sum()/total),
                 'voice_frame_distribution':vhist.tolist()}
            row.update({f'bin_{b}':float(hist[y,l,b]/total) for b in range(6)})
            rows.append(row)
    return pd.DataFrame(rows),hist


if __name__=='__main__':
    root=Path(sys.argv[1]);os.nice(10);torch.set_num_threads(1);start=time.monotonic()
    output=root/'validation';output.mkdir(exist_ok=False)
    items=load_train_items_only('/data/ASVspoof2019');args,_,_=arguments.get_args()
    args.update(duration_policy_path=str(root/'duration_policy.json'),bpl_pairing='matched_independent',
                rank=0,world_size=2,batch_size=120,num_workers=0)
    before=MatchedTrainSet(args,items)
    DS=MatchedTrainSet(dict(args,activity_policy_path=str(root/'activity_policy.json')),items)
    np.testing.assert_array_equal(before.source_probability,DS.source_probability)
    with np.load(root/'raw_candidates.npz',allow_pickle=False) as f:data={key:f[key] for key in f.files}
    a,ha=expected(before,data,'before');b,hb=expected(DS,data,'after')
    expect=pd.concat([a,b],ignore_index=True)
    expect.drop(columns='voice_frame_distribution').to_csv(output/'expected_by_length.csv',index=False)
    differences=[]
    for name,df,hist in [('before',a,ha),('after',b,hb)]:
        for l,length in enumerate(DS.targets):
            p0=hist[0,l]/hist[0,l].sum();p1=hist[1,l]/hist[1,l].sum()
            c0=df[(df.label==0)&(df.seconds==length/16000)].iloc[0]
            c1=df[(df.label==1)&(df.seconds==length/16000)].iloc[0]
            v0=np.array(c0.voice_frame_distribution);v1=np.array(c1.voice_frame_distribution)
            differences.append({'policy':name,'seconds':length/16000,'binned_TV':float(np.abs(p0-p1).sum()/2),
                'speech_mean_gap':float(abs(c0.mean_speech_fraction-c1.mean_speech_fraction)),
                'fine_activity_KS':float(np.abs(v0.cumsum()-v1.cumsum()).max())})
    pd.DataFrame(differences).to_csv(output/'distribution_differences.csv',index=False)
    role_names=['OC_bona','OC_spoof','BPL_right_bona'];rows=[];old_intervals=[defaultdict(list) for _ in range(3)];new_intervals=[defaultdict(list) for _ in range(3)]
    tasks=[];uid_equal=True;augmentation_equal=True
    for rank in range(2):
        old=MatchedBatchSampler(before,dict(args,rank=rank));new=MatchedBatchSampler(DS,dict(args,rank=rank))
        old.set_epoch(1);new.set_epoch(1)
        for step,(old_batch,new_batch) in enumerate(zip(old,new)):
            assert [v.index for v in old_batch]==[v.index for v in new_batch]
            assert [v.augmentation_seed for v in old_batch]==[v.augmentation_seed for v in new_batch]
            for slot,(old_spec,spec) in enumerate(zip(old_batch,new_batch)):
                role=0 if slot<12 else 1 if slot<120 else 2
                i=spec.index;length=spec.stop-spec.start;k=length//8000-1
                lo,hi=DS.activity_policy.offsets[i*len(DS.targets)+k:i*len(DS.targets)+k+2]
                j=int(lo)+int(np.searchsorted(DS.activity_policy.starts[lo:hi],spec.start))
                assert j<hi and DS.activity_policy.starts[j]==spec.start
                voiced=int(DS.activity_policy.voice[j]);frames=length//320
                item=DS.items[i];serial=len(rows)
                rows.append({'serial':serial,'role':role_names[role],'label':item.label,'UID':Path(item.path).stem,
                    'speaker':item.speaker,'attack':item.type,'seconds':length/16000,'speech_frames':voiced,'frames':frames,
                    'speech_fraction':voiced/frames,'zero_speech':voiced==0,'speech_below100ms':voiced<5,
                    'bin':int((voiced*5+frames-1)//frames),'start_sample':spec.start,'stop_sample':spec.stop})
                tasks.append((serial,spec))
                old_intervals[role][i].append((old_spec.start,old_spec.stop));new_intervals[role][i].append((spec.start,spec.stop))
    realized=pd.DataFrame(rows);assert len(realized)==28512
    realized.to_csv(output/'epoch1_selected_crops.csv.gz',index=False,compression='gzip')
    realized.groupby(['role','label','seconds','bin']).size().reset_index(name='crops').to_csv(output/'epoch1_activity_bins.csv',index=False)
    realized.groupby(['role','label','seconds']).agg(crops=('serial','size'),speech_fraction=('speech_fraction','mean'),
        zero_speech=('zero_speech','mean'),below100ms=('speech_below100ms','mean')).reset_index().to_csv(output/'epoch1_activity_summary.csv',index=False)
    # Actual cached raw VAD is independently checked on stratified real waveforms.
    strata=defaultdict(list)
    for row,task in zip(rows,tasks):strata[(row['label'],row['seconds'])].append(task)
    selected=[]
    for key,group in strata.items():selected.extend(sorted(group,key=lambda t:hashlib.sha256(str(t).encode()).digest())[:64])
    with mp.get_context('fork').Pool(4) as pool:
        audio=pd.DataFrame(pool.imap(inspect_audio,selected,chunksize=16))
    audio=audio.merge(realized[['serial','label','seconds','speech_frames']],on='serial',validate='one_to_one')
    assert np.array_equal(audio.raw_speech_frames,audio.speech_frames)
    audio['raw_fraction']=audio.raw_speech_frames/audio.total_frames
    audio['augmented_fraction']=audio.augmented_speech_frames/audio.total_frames
    audio.to_csv(output/'augmentation_check.csv',index=False)
    time_coverage=[]
    for y,roles,name in [(1,[0],'bona_OC'),(0,[1],'spoof_OC'),(1,[0,2],'bona_OC_plus_BPL')]:
        denom=sum(DS.lengths[i] for i,item in enumerate(items) if item.label==y)
        for title,source in [('before',old_intervals),('after',new_intervals)]:
            merged=defaultdict(list)
            for role in roles:
                for i,parts in source[role].items():merged[i].extend(parts)
            covered=sum(union_length(v) for v in merged.values())
            time_coverage.append({'policy':title,'population':name,'unique_originals':len(merged),
                                  'original_time_fraction':float(covered/denom)})
    pd.DataFrame(time_coverage).to_csv(output/'epoch1_time_coverage.csv',index=False)
    offsets=data['context_offsets'];context=np.repeat(np.arange(len(DS)*len(DS.targets)),np.diff(offsets));orig=context//len(DS.targets);length=context%len(DS.targets)
    old_lp=np.stack([before.tables[before.keys[i]][before.buckets[i]-1] for i in range(len(before))])
    mult=data['multiplicity'].astype(float);base_pos=mult/np.bincount(context,weights=mult,minlength=len(DS)*len(DS.targets))[context]
    old_p=old_lp[orig,length]*base_pos;new_p=DS.activity_policy.lengths[orig,length]*DS.activity_policy.positions
    old_eff=1/np.bincount(orig,weights=old_p**2,minlength=len(DS));new_eff=1/np.bincount(orig,weights=new_p**2,minlength=len(DS))
    diversity={'median_effective_candidates_before':float(np.median(old_eff)),
               'median_effective_candidates_after':float(np.median(new_eff)),
               'fraction_effective_candidates_below2_before':float((old_eff<2).mean()),
               'fraction_effective_candidates_below2_after':float((new_eff<2).mean()),
               'per_original_length_TV_median':float(np.median(np.abs(old_lp-DS.activity_policy.lengths).sum(1)/2))}
    result={'status':'complete','dev_used':False,'eval_used':False,'model_training':False,
        'unit_scope':'Train only; expected full-candidate probabilities and actualepoch1 draws',
        'source_probabilities_exactly_unchanged':True,'epoch1_UID_sequences_exactly_equal':True,
        'augmentation_seed_sequences_exactly_equal':True,'actual_raw_VAD_checks':len(audio),
        'cached_VAD_exact_on_checked_waveforms':True,'expected_distribution_differences':differences,
        'validated_probability_statistics':DS.activity_policy.statistics,'epoch1_time_coverage':time_coverage,
        'candidate_diversity':diversity,'policy_sha256':DS.activity_policy.sha256,
        'augmentation_check':'Stratified sample; RawBoost is unchanged and post-augmentation VAD equality is not a constraint',
        'elapsed_seconds':time.monotonic()-start,'CUDA_initialized':torch.cuda.is_initialized()}
    assert not torch.cuda.is_initialized()
    (output/'analysis.json').write_text(json.dumps(result,indent=2))
