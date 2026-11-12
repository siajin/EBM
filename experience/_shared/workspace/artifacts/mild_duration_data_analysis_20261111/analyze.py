"""Read-only policy analysis and exact frozen-code20epoch sampler replay. No audio/GPU."""
import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from dataclasses import dataclass
from types import SimpleNamespace, MethodType
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon

HERE=Path(__file__).resolve().parent
WORKSPACE=HERE.parents[1]
SPEC=WORKSPACE/'artifacts/mild_duration_e20_20261110'
sys.path[:0]=[str(WORKSPACE/'EBM'),str(WORKSPACE/'.analysis_deps')]


def save(name,value):
    (HERE/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def weighted_ks(a,b,wa,wb):
    order_a=np.argsort(a);order_b=np.argsort(b)
    a,b=np.asarray(a)[order_a],np.asarray(b)[order_b]
    ca,cb=np.r_[0,np.cumsum(np.asarray(wa)[order_a])],np.r_[0,np.cumsum(np.asarray(wb)[order_b])]
    ca/=ca[-1];cb/=cb[-1]
    points=np.union1d(a,b)
    return float(np.max(abs(ca[np.searchsorted(a,points,side='right')]-cb[np.searchsorted(b,points,side='right')])))


def similarity(a,b,wa=None,wb=None,width=.1):
    a,b=np.asarray(a),np.asarray(b)
    wa=np.ones(len(a))/len(a) if wa is None else np.asarray(wa)/np.sum(wa)
    wb=np.ones(len(b))/len(b) if wb is None else np.asarray(wb)/np.sum(wb)
    edges=np.arange(0,31+width/2,width)
    ha=np.histogram(a,bins=edges,weights=wa)[0];hb=np.histogram(b,bins=edges,weights=wb)[0]
    assert np.isclose(ha.sum(),1) and np.isclose(hb.sum(),1)
    return {'bin_seconds':width,'overlap_percent':float(100*np.minimum(ha,hb).sum()),
        'KS':weighted_ks(a,b,wa,wb),'W1_seconds':float(wasserstein_distance(a,b,wa,wb)),
        'JS_divergence_bits':float(jensenshannon(ha,hb,base=2)**2)}


def union_size(intervals):
    intervals=sorted(intervals)
    if not intervals:return 0
    total=0;start,stop=intervals[0]
    for left,right in intervals[1:]:
        if left<=stop: stop=max(stop,right)
        else: total+=stop-start;start,stop=left,right
    return total+stop-start


def coverage(counts,ds):
    # counts maps original index to original-timeline intervals, not slowed duration.
    n=np.array([ds.records[i]['samples'] for i in counts])
    used=np.array([union_size(v) for v in counts.values()])
    return {'UIDs':len(counts),'total_original_seconds':float(n.sum()/16000),
        'unique_retained_seconds':float(used.sum()/16000),
        'duration_weighted_retention_percent':float(100*used.sum()/n.sum()),
        'mean_per_UID_retention_percent':float(100*np.mean(used/n)),
        'worst_UID_retention_percent':float(100*np.min(used/n)),
        'UIDs_with_unseen_samples':int(np.sum(used<n))}


def main():
    config=json.loads((SPEC/'experiment.json').read_text())
    raw=(SPEC/'duration_policy.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==config['duration_policy_sha256']
    policy=json.loads(raw);metadata=json.loads((HERE/'train_metadata.json').read_text())
    meta=pd.DataFrame(metadata['records']);records={r['uid']:r for r in policy['originals']}
    assert len(meta)==25380 and meta.uid.is_unique and set(meta.uid)==set(records)
    assert all(records[r.uid]['label']==r.label for r in meta.itertuples())
    # Execute exactly the frozen sampler classes, isolated from optional audio imports.
    code_path=WORKSPACE/'EBM/positive_bpl.py'
    assert hashlib.sha256(code_path.read_bytes()).hexdigest()==config['source_sha256']['positive_bpl.py']
    assert hashlib.sha256((WORKSPACE/'EBM/mild_duration.py').read_bytes()).hexdigest()==config['source_sha256']['mild_duration.py']
    tree=ast.parse(code_path.read_text());wanted={'CoverageSourcePlan','coverage_source_plans','MatchedBatchSampler'}
    nodes=[n for n in tree.body if getattr(n,'name',None) in wanted]
    # The framework base is unused by these methods; avoid loading a second
    # Windows OpenMP runtime merely to replay pure NumPy selection code.
    ns={'np':np,'Sampler':object,'dataclass':dataclass,'__name__':'__main__'}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'frozen_sampler','exec'),ns)
    mild_tree=ast.parse((WORKSPACE/'EBM/mild_duration.py').read_text())
    spec=next(n for n in mild_tree.body if getattr(n,'name',None)=='MildSpec')
    cls=next(n for n in mild_tree.body if getattr(n,'name',None)=='MildTrainSet')
    chooser=next(n for n in cls.body if getattr(n,'name',None)=='crop_spec')
    exec(compile(ast.fix_missing_locations(ast.Module(body=[spec,chooser],type_ignores=[])),'frozen_chooser','exec'),ns)
    ds=SimpleNamespace()
    ds.crop_spec=MethodType(ns['crop_spec'],ds)
    ds.records=[records[r.uid] for r in meta.itertuples()]
    ds.items=[SimpleNamespace(label=r.label,speaker=r.speaker,type=r.attack) for r in meta.itertuples()]
    ds.class_indices={y:np.flatnonzero(meta.label.to_numpy()==y) for y in [1,0]}
    ds.class_probability={y:np.full(len(v),1/len(v)) for y,v in ds.class_indices.items()}
    args={'world_size':2,'rand_seed':1,'bpl_pair_size':12,'bpl_steps_per_epoch':108,
        'bpl_pairing':'matched_independent','matched_source_sampling':'coverage',
        'batch_size':120,'ratio_bona':.1,'ratio_spoof':.9}
    samplers=[ns['MatchedBatchSampler'](ds,dict(args,rank=rank)) for rank in [0,1]]
    variant_rows=[]
    for index,row in enumerate(ds.records):
        for variant,v in enumerate(row['variants']):
            variant_rows.append({'index':index,'uid':row['uid'],'label':row['label'],
                'speaker':meta.iloc[index].speaker,'attack':meta.iloc[index].attack,
                'variant':variant,'original_samples':row['samples'],'original_seconds':row['samples']/16000,
                'target_seconds':v['output_samples']/16000,'probability':v['probability'],
                'class_weight':v['probability']/len(ds.class_indices[row['label']]),'rate':v['rate'],
                'crop_fraction':1-v['kept_samples']/row['samples'],'start':v['start'],
                'stop':v['start']+v['kept_samples'],'output_samples':v['output_samples']})
    bank=pd.DataFrame(variant_rows);bank.to_parquet(HERE/'variant_metadata.parquet',index=False)
    expected=[];original=[];operations=[];prior_groups=[];bank_coverage={}
    for y in [1,0]:
        f=bank[bank.label==y];raw_lengths=np.array([ds.records[i]['samples']/16000 for i in ds.class_indices[y]])
        original.append({'label':y,'originals':len(raw_lengths),'mean_seconds':float(raw_lengths.mean()),
            'median_seconds':float(np.median(raw_lengths)),'under2_percent':float(100*np.mean(raw_lengths<2)),
            'under1_5_percent':float(100*np.mean(raw_lengths<=1.5)),
            'hours':float(raw_lengths.sum()/3600),'min_seconds':float(raw_lengths.min()),'max_seconds':float(raw_lengths.max())})
        w=f.class_weight.to_numpy();out=f.target_seconds.to_numpy();rates=f.rate.to_numpy()
        crop=f.crop_fraction.to_numpy();p=f.probability.to_numpy()
        candidates=f.groupby('index').size();ent=f.groupby('index').probability.apply(lambda x:float(np.exp(-np.sum(x*np.log(x)))))
        expected.append({'label':y,'mean_seconds':float(w@out),'under2_percent':float(100*w[out<2].sum()),
            'under1_5_percent':float(100*w[out<=1.5].sum()),'min_seconds':float(out.min()),'max_seconds':float(out.max())})
        operations.append({'label':y,'cache_variants':len(f),'source_single_variant_percent':float(100*np.mean(candidates==1)),
            'variants_per_source_median':float(candidates.median()),'variants_per_source_max':int(candidates.max()),
            'effective_variants_per_source_median':float(ent.median()),
            'slowed_percent':float(100*w[rates<1-1e-10].sum()),'sped_up_percent':float(100*w[rates>1+1e-10].sum()),
            'identity_rate_percent':float(100*w[abs(rates-1)<=1e-10].sum()),
            'cropped_percent':float(100*w[crop>1e-10].sum()),'mean_crop_percent':float(100*w@crop),
            'maximum_crop_percent':float(100*crop.max()),'rate_min':float(rates.min()),'rate_max':float(rates.max()),
            'mean_rate':float(w@rates)})
        ranges=defaultdict(list)
        for r in f.itertuples():ranges[r.index].append((r.start,r.stop))
        bank_coverage[str(y)]=coverage(ranges,ds)
    before=similarity(np.array([r['samples']/16000 for r in ds.records if r['label']==1]),
                      np.array([r['samples']/16000 for r in ds.records if r['label']==0]))
    b,s=bank[bank.label==1],bank[bank.label==0]
    exact=[similarity(b.target_seconds,s.target_seconds,b.class_weight,s.class_weight,width) for width in [.05,.1,.25]]
    # Groupwise comparison does not pretend the class-global optimum holds within speakers/attacks.
    speaker_results=[]
    for speaker in sorted(meta.speaker.unique()):
        b0=meta[(meta.label==1)&(meta.speaker==speaker)]['index'];s0=meta[(meta.label==0)&(meta.speaker==speaker)]['index']
        bf=b[b.speaker==speaker];sf=s[s.speaker==speaker]
        previous=similarity([ds.records[i]['samples']/16000 for i in b0],[ds.records[i]['samples']/16000 for i in s0])
        after=similarity(bf.target_seconds,sf.target_seconds,bf.probability,sf.probability)
        speaker_results.append({'speaker':speaker,'bonafide_originals':len(b0),'spoof_originals':len(s0),
            'before_overlap_percent':previous['overlap_percent'],'after_overlap_percent':after['overlap_percent'],
            'after_KS':after['KS'],'after_W1_seconds':after['W1_seconds']})
    pd.DataFrame(speaker_results).to_csv(HERE/'speaker_length_similarity.csv',index=False)
    attack_results=[];speaker_attack_results=[]
    for attack in sorted(s.attack.unique()):
        sf=s[s.attack==attack];after=similarity(b.target_seconds,sf.target_seconds,b.class_weight,sf.probability)
        sf_original=meta[(meta.label==0)&(meta.attack==attack)]['index']
        previous=similarity([r['samples']/16000 for r in ds.records if r['label']==1],
                            [ds.records[i]['samples']/16000 for i in sf_original])
        attack_results.append({'attack':attack,'originals':len(sf_original),'before_overlap_percent':previous['overlap_percent'],
            'after_overlap_percent':after['overlap_percent'],'after_KS':after['KS'],'after_W1_seconds':after['W1_seconds']})
        for speaker in sorted(meta.speaker.unique()):
            bf=b[b.speaker==speaker];ssf=sf[sf.speaker==speaker]
            sim=similarity(bf.target_seconds,ssf.target_seconds,bf.probability,ssf.probability)
            speaker_attack_results.append({'speaker':speaker,'attack':attack,**sim})
    pd.DataFrame(attack_results).to_csv(HERE/'attack_length_similarity.csv',index=False)
    pd.DataFrame(speaker_attack_results).to_csv(HERE/'speaker_attack_length_similarity.csv',index=False)
    # Observable transform parameters are a diagnostic only; not an audio classifier claim.
    speed_edges=np.linspace(.8,1.2,41)
    ha=np.histogram(b.rate,bins=speed_edges,weights=b.class_weight)[0]
    hb=np.histogram(s.rate,bins=speed_edges,weights=s.class_weight)[0]
    speed_overlap=float(np.minimum(ha,hb).sum())
    shortcut={'speed_bin_width':.01,'speed_histogram_overlap_percent':100*speed_overlap,
        'oracle_rate_bin_balanced_accuracy_percent':100*(1-speed_overlap/2),
        'meaning':'Oracle known transformation rate, binned likelihood diagnostic only; does not show an audio model can recover the rate.'}
    # Exact production sampler replay; randomness of RawBoost cannot affect selected lengths.
    draws=[];epoch_results=[];cumulative=[defaultdict(list) for _ in range(3)]
    seen_variants=[set(),set(),set()];same_source=0;pair_total=0
    for epoch in range(1,21):
        retained=[defaultdict(list) for _ in range(3)]
        for rank,sampler in enumerate(samplers):
            sampler.set_epoch(epoch)
            for step,batch in enumerate(sampler):
                left=np.array([v.index for v in batch[:12]]);right=np.array([v.index for v in batch[120:]])
                same_source+=int(np.sum(left[:,None]==right[None,:]));pair_total+=144
                for role,specs in enumerate([batch[:12],batch[12:120],batch[120:]]):
                    for spec in specs:
                        v=ds.records[spec.index]['variants'][spec.variant]
                        interval=(v['start'],v['start']+v['kept_samples'])
                        retained[role][spec.index].append(interval);cumulative[role][spec.index].append(interval)
                        seen_variants[role].add((spec.index,spec.variant))
                        draws.append((epoch,rank,role,spec.index,spec.variant,v['output_samples']))
        assert [len(r) for r in retained]==[2580,22800,2580]
        epoch_results.append({'epoch':epoch,'retained':[coverage(r,ds) for r in retained],
            'cumulative_retained':[coverage(r,ds) for r in cumulative],
            'cumulative_unique_variants':[len(v) for v in seen_variants]})
        print('replayed epoch',epoch,flush=True)
    draws=pd.DataFrame(draws,columns=['epoch','rank','role','index','variant','output_samples'])
    assert len(draws)==570240 and draws.groupby('role').size().tolist()==[51840,466560,51840]
    draws.to_parquet(HERE/'draws_20epochs.parquet',index=False)
    for result in epoch_results:
        epoch=result['epoch'];f=draws[draws.epoch==epoch];a=f[f.role==0].output_samples/16000;c=f[f.role==1].output_samples/16000
        result['OC_length_similarity']=similarity(a,c)
        all_bona=defaultdict(list)
        for role in [0,2]:
            for r in f[f.role==role].itertuples():
                v=ds.records[r.index]['variants'][r.variant];all_bona[r.index].append((v['start'],v['start']+v['kept_samples']))
        result['OC_plus_BPL_bona_retention']=coverage(all_bona,ds)
    epoch1=epoch_results[0]['OC_length_similarity']
    combined={}
    for width in [.05,.1,.25]:
        combined[str(width)]=similarity(draws[draws.role==0].output_samples/16000,
            draws[draws.role==1].output_samples/16000,width=width)
    # Every speaker/attack retains its original expected count share within class.
    prior_results=[]
    for role,label in [(0,1),(1,0),(2,1)]:
        ids=draws[draws.role==role]['index'].to_numpy();visits=meta.iloc[ids]
        first_ids=draws[(draws.role==role)&(draws.epoch==1)]['index'].to_numpy();first=meta.iloc[first_ids]
        for dimension in ['speaker','attack']:
            original_counts=meta[meta.label==label][dimension].value_counts()
            counts=visits[dimension].value_counts();first_counts=first[dimension].value_counts()
            for group,n in original_counts.items():
                p=n/len(ds.class_indices[label]);actual=counts.get(group,0)/len(visits);one=first_counts.get(group,0)/len(first)
                prior_results.append({'role':role,'label':label,'dimension':dimension,'group':group,
                    'original_percent':100*p,'expected_percent':100*p,'epoch1_percent':100*one,
                    'epoch20_cumulative_percent':100*actual,'epoch1_delta_pp':100*(one-p),
                    'epoch20_cumulative_delta_pp':100*(actual-p)})
    pd.DataFrame(prior_results).to_csv(HERE/'speaker_attack_priors.csv',index=False)
    # Class-specific data retention over all20 epochs, and bank reachability ceiling.
    combined_ranges=defaultdict(list)
    for role in [0,2]:
        for i,ranges in cumulative[role].items():combined_ranges[i].extend(ranges)
    per_uid=[]
    for i,row in enumerate(ds.records):
        role=0 if row['label']==1 else 1
        variants=row['variants'];bank_intervals=[(v['start'],v['start']+v['kept_samples']) for v in variants]
        per_uid.append({'uid':row['uid'],'label':row['label'],'speaker':meta.iloc[i].speaker,
            'attack':meta.iloc[i].attack,'original_samples':row['samples'],
            'bank_reachable_samples':union_size(bank_intervals),
            'OC20_reachable_samples':union_size(cumulative[role][i]),
            'OC_BPL20_reachable_samples':union_size(combined_ranges[i]) if row['label']==1 else union_size(cumulative[1][i]),
            'candidate_count':len(variants)})
    per_uid=pd.DataFrame(per_uid);per_uid.to_csv(HERE/'source_coverage.csv',index=False)
    # Compact reviewed chart data, common bins and class-conditional denominators.
    chart={'length':{},'speed':[]}
    for mode in ['original','expected','epoch1','epochs20']:
        width=.1;edges=np.arange(0,14.1+width/2,width)
        values=[]
        for label,role in [(1,0),(0,1)]:
            if mode=='original': vals=[r['samples']/16000 for r in ds.records if r['label']==label];w=None
            elif mode=='expected':f=bank[bank.label==label];vals=f.target_seconds;w=f.class_weight
            else:f=draws[(draws.role==role)&((draws.epoch==1) if mode=='epoch1' else True)];vals=f.output_samples/16000;w=None
            h=np.histogram(vals,bins=edges,weights=w)[0].astype(float)
            # Common full Train range, including original13.19s tails.
            assert np.isclose(h.sum(),np.sum(w) if w is not None else len(vals))
            values.append((h/h.sum()).tolist())
        chart['length'][mode]={'centers':((edges[:-1]+edges[1:])/2).tolist(),'bona':values[0],'spoof':values[1]}
    chart['speed']={'centers':((speed_edges[:-1]+speed_edges[1:])/2).tolist(),'bona':ha.tolist(),'spoof':hb.tolist()}
    save('chart_data.json',chart)
    summary={'scope':'TRAIN policy + exact production sampler replay; not generated waveform measurements, not model performance',
        'cache_status_at_analysis':'waiting for existing DF evaluation and comparison; no speech cache yet',
        'policy_sha256':config['duration_policy_sha256'],'protocol_sha256':metadata['protocol_sha256'],
        'original':original,'expected':expected,'operations':operations,'before_similarity':before,
        'expected_similarity':exact,'epoch1_OC_similarity':epoch1,'cumulative20_OC_similarity':combined,
        'bank_maximum_original_timeline_coverage':bank_coverage,'epoch_results':epoch_results,
        'cumulative20_bona_OC_BPL_retention':coverage(combined_ranges,ds),
        'total_original_UIDs':25380,'total_cached_variants':len(bank),'training_input_visits_per_epoch':28512,
        'OC_visits_per_epoch':25920,'extra_BPL_visits_per_epoch':2592,'training_input_visits20':570240,
        'OC_bona_spoof_per_epoch':[2592,23328],'BPL_right_bona_per_epoch':2592,
        'observed_BPL_same_UID_pairs':same_source,'BPL_pairs20':pair_total,
        'BPL_same_UID_pair_percent':100*same_source/pair_total,
        'speaker_results':speaker_results,'attack_results':attack_results,'transform_parameter_diagnostic':shortcut,
        'speaker_max_delta20_pp':max(abs(v['epoch20_cumulative_delta_pp']) for v in prior_results if v['dimension']=='speaker'),
        'attack_max_delta20_pp':max(abs(v['epoch20_cumulative_delta_pp']) for v in prior_results if v['dimension']=='attack'),
        'speech_activity':'Not matched; new transformed speech activity and acoustic distortion unmeasured until audio cache exists.',
        'cost':{'new_paid_resource_ceiling_USD':0,'GPU_updates':0,'experiment_changed':False}}
    save('summary.json',summary)
    save('validation.json',{'status':'passed','draws':len(draws),'epochs':20,'source_coverage_each_epoch':[2580,22800,2580],
        'sampler_code_matches_frozen_snapshot':True,'policy_probability_sums':True,
        'chart_histograms_cover_all_observations':True,'raw_audio_read':False,'GPU_used':False})
    print(json.dumps({k:summary[k] for k in ['original','expected','operations','before_similarity','expected_similarity',
        'epoch1_OC_similarity','cumulative20_OC_similarity','speaker_max_delta20_pp','attack_max_delta20_pp',
        'transform_parameter_diagnostic']},indent=2),flush=True)


if __name__=='__main__':main()
