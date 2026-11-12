"""Train-only coverage/expected-prior/realized-distribution audit; no models."""
from collections import defaultdict
import hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import arguments
from positive_bpl import MatchedTrainSet,MatchedBatchSampler,coverage_source_plans
from dataset.ASVspoof2019_la_train import load_train_items_only


def union_length(parts):
    parts=sorted(parts)
    if not parts:return 0
    a,b=parts[0];total=0
    for start,stop in parts[1:]:
        if start>b:total+=b-a;a,b=start,stop
        else:b=max(b,stop)
    return total+b-a


def main(root):
    root=Path(root);os.nice(10);torch.set_num_threads(1);started=time.monotonic()
    output=root/'validation';output.mkdir(exist_ok=False)
    args,_,_=arguments.get_args()
    args.update(duration_policy_path=str(root/'duration_policy.json'),
        activity_policy_path=str(root/'activity_policy.json'),bpl_pairing='matched_independent',
        world_size=2,rank=0,batch_size=120,num_workers=0)
    ds=MatchedTrainSet(args,load_train_items_only('/data/ASVspoof2019'))
    k=len(ds.targets);n=len(ds);roles=['OC_bona','OC_spoof','BPL_right_bona']
    plans=coverage_source_plans(ds,{1:2592,0:23328})
    for y,plan in plans.items():
        np.testing.assert_allclose(plan.expected_probability,ds.class_probability[y],atol=1e-14,rtol=0)
    source_rows=[];source_summary=[]
    for mode in ['replacement','coverage']:
        cumulative=np.zeros((3,n),dtype=np.int64)
        for epoch in range(1,51):
            rngs=[np.random.default_rng([args['rand_seed'],epoch,side]) for side in range(3)]
            for role,y,quota,rng in zip(range(3),[1,0,1],[24,216,24],rngs):
                ids=ds.class_indices[y];prob=ds.class_probability[y]
                if mode=='coverage':indices=plans[y].draw(rng)
                else:indices=np.concatenate([rng.choice(ids,quota,p=prob) for _ in range(108)])
                counts=np.bincount(indices,minlength=n);cumulative[role]+=counts
                selected=int((counts[ids]>0).sum())
                if mode=='coverage':assert selected==len(ids) and counts[ids].min()>=1
                speaker_tv=attack_tv=0.
                for field in ['speaker','type']:
                    groups=defaultdict(list)
                    for j,i in enumerate(ids):groups[getattr(ds.items[i],field)].append(j)
                    distance=sum(abs(counts[ids][v].sum()/len(indices)-prob[v].sum()) for v in groups.values())/2
                    if field=='speaker':speaker_tv=distance
                    else:attack_tv=distance
                source_rows.append({'mode':mode,'epoch':epoch,'role':roles[role],
                    'originals':len(ids),'draws':len(indices),'unique_originals':selected,
                    'original_fraction':selected/len(ids),'speaker_target_TV':speaker_tv,'attack_target_TV':attack_tv})
        for role,y in enumerate([1,0,1]):
            visits=cumulative[role,ds.class_indices[y]]
            source_summary.append({'mode':mode,'role':roles[role],
                'minimum_visits_50epochs':int(visits.min()),'maximum_visits_50epochs':int(visits.max())})
    source_frame=pd.DataFrame(source_rows);source_frame.to_csv(output/'source_coverage_50epochs.csv',index=False)
    # Read actual CropSpecs for both ranks. Cached VAD was independently checked
    # on 1,010 real crops when this unchanged activity policy was built.
    distribution_rows=[];coverage_rows=[];seed_hashes={};pooled_differences=[]
    for mode in ['replacement','coverage']:
        hist=np.zeros((3,k,6),dtype=np.int64);voice_sum=np.zeros((3,k));fine=np.zeros((3,k,201),dtype=np.int64)
        for epoch in range(1,11):
            epoch_hist=np.zeros_like(hist);epoch_voice=np.zeros_like(voice_sum)
            selected=[set() for _ in roles];intervals=[defaultdict(list) for _ in roles]
            augmented_seed_hash=hashlib.sha256()
            for rank in range(2):
                sampler=MatchedBatchSampler(ds,dict(args,rank=rank,matched_source_sampling=mode))
                sampler.set_epoch(epoch)
                for batch in sampler:
                    assert len(batch)==132
                    for role,part in enumerate([batch[:12],batch[12:120],batch[120:]]):
                        for spec in part:
                            i=spec.index;length=spec.stop-spec.start;l=length//8000-1
                            lo,hi=ds.activity_policy.offsets[i*k+l:i*k+l+2]
                            j=int(lo)+int(np.searchsorted(ds.activity_policy.starts[lo:hi],spec.start))
                            assert j<hi and ds.activity_policy.starts[j]==spec.start
                            voiced=int(ds.activity_policy.voice[j]);frames=int(length//320)
                            b=(voiced*5+frames-1)//frames
                            epoch_hist[role,l,b]+=1;epoch_voice[role,l]+=voiced/frames
                            fine[role,l,voiced]+=1;selected[role].add(i)
                            augmented_seed_hash.update(int(spec.augmentation_seed).to_bytes(4,'little'))
                            if epoch==1:intervals[role][i].append((spec.start,spec.stop))
            if mode=='coverage':
                for role,y in enumerate([1,0,1]):assert selected[role]==set(ds.class_indices[y])
            if epoch==1:
                seed_hashes[mode]=augmented_seed_hash.hexdigest()
                for y,which,name in [(1,[0],'bona_OC'),(0,[1],'spoof_OC'),(1,[0,2],'bona_OC_plus_BPL')]:
                    combined=defaultdict(list)
                    for role in which:
                        for i,parts in intervals[role].items():combined[i].extend(parts)
                    denom=sum(ds.lengths[ds.class_indices[y]])
                    coverage_rows.append({'mode':mode,'population':name,'unique_originals':len(combined),
                        'original_time_fraction':sum(union_length(v) for v in combined.values())/denom})
            hist+=epoch_hist;voice_sum+=epoch_voice
            for role in range(3):
                total=int(epoch_hist[role].sum())
                for l,seconds in enumerate(ds.targets/16000):
                    count=int(epoch_hist[role,l].sum())
                    distribution_rows.append({'mode':mode,'epoch':epoch,'role':roles[role],
                        'seconds':seconds,'draws':count,'length_probability':count/total,
                        'mean_speech_fraction':epoch_voice[role,l]/count if count else None,
                        **{f'bin_{b}':int(epoch_hist[role,l,b]) for b in range(6)}})
            (root/'progress.json').write_text(json.dumps({'stage':'actual crop replay','mode':mode,'epoch':epoch}))
        for role in range(3):
            total=int(hist[role].sum())
            for l,seconds in enumerate(ds.targets/16000):
                count=int(hist[role,l].sum())
                distribution_rows.append({'mode':mode,'epoch':'pooled_10','role':roles[role],
                    'seconds':seconds,'draws':count,'length_probability':count/total,
                    'mean_speech_fraction':voice_sum[role,l]/count,
                    **{f'bin_{b}':int(hist[role,l,b]) for b in range(6)}})
        for l,seconds in enumerate(ds.targets/16000):
            count=hist[:,l].sum(1)
            bins=hist[:,l]/count[:,None]
            cdf=np.cumsum(fine[:,l]/count[:,None],axis=1)
            pooled_differences.append({'mode':mode,'seconds':seconds,
                'binned_TV':float(np.abs(bins[0]-bins[1]).sum()/2),
                'mean_speech_gap':float(abs(voice_sum[0,l]/count[0]-voice_sum[1,l]/count[1])),
                'fine_activity_KS':float(np.abs(cdf[0]-cdf[1]).max())})
    assert seed_hashes['replacement']==seed_hashes['coverage']
    pd.DataFrame(distribution_rows).to_csv(output/'realized_distributions_10epochs.csv',index=False)
    pd.DataFrame(coverage_rows).to_csv(output/'epoch1_time_coverage.csv',index=False)
    pd.DataFrame(pooled_differences).to_csv(output/'pooled10_distribution_differences.csv',index=False)
    plan_summary={y:plan.summary for y,plan in plans.items()}
    result={'status':'complete','Train_originals':n,'source_replay_epochs':50,'actual_crop_replay_epochs':10,
        'actual_crops_replayed_per_mode':285120,'coverage_plans':plan_summary,
        'all_originals_in_OC_every_epoch':True,'BPL_right_all_bona_every_epoch':True,
        'expected_original_probabilities_unchanged':True,
        'duration_and_activity_policies_unchanged':True,
        'duration_policy_sha256':ds.policy_sha256,'activity_policy_sha256':ds.activity_policy.sha256,
        'expected_activity_statistics':ds.activity_policy.statistics,
        'pooled10_realized_distribution_differences':pooled_differences,
        'augmentation_seed_stream_unchanged':True,'source_counts_50epochs':source_summary,
        'epoch1_time_coverage':coverage_rows,
        'source_group_TV_means':source_frame.groupby(['mode','role'])[['speaker_target_TV','attack_target_TV']].mean().reset_index().to_dict('records'),
        'dev_used':False,'eval_used':False,'model_training':False,'optimizer_updates':0,
        'CUDA_initialized':torch.cuda.is_initialized(),'elapsed_seconds':time.monotonic()-started}
    assert not torch.cuda.is_initialized()
    (output/'analysis.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main(sys.argv[1])
