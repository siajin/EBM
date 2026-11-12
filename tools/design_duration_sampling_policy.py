"""Fit and audit a Train-only sampling proposal using cached duration metadata."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from duration_sampling_policy import OriginalFirstDurationPolicy

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/duration_sampling_design_20261110'
OUT.mkdir(parents=True,exist_ok=True)
STEP=8000
TARGET_PERCENT=np.array([5,10,15,20,18,15,8,4,2,1.5,1,.5],dtype='float64')
q=TARGET_PERCENT/100
assert np.isclose(q.sum(),1)
targets=np.arange(1,len(q)+1)*STEP
previous=json.loads((ROOT/'artifacts/original_first_length_sampling_20261110/analysis.json').read_text(encoding='utf-8'))
source=previous['sources'][0]
path=ROOT/source['path']
assert hashlib.sha256(path.read_bytes()).hexdigest()==source['sha256']
f=pd.read_parquet(path,columns=['split','utterance_id','speaker_group','attack_id','y','duration'])
f=f.loc[f.split=='T'].reset_index(drop=True)
assert len(f)==25380 and f.utterance_id.is_unique and not f.isna().any().any()
n=np.rint(f.duration.to_numpy()*16000).astype('int64')
assert (n>=STEP).all() and np.max(np.abs(n/16000-f.duration.to_numpy()))<1e-9
support=np.arange(1,len(q)+1)[:,None]>=np.arange(1,len(q)+1)[None,:]


def fit_conditionals(freq):
    """IPF with fixed source-bucket marginal and requested duration marginal."""
    freq=freq/freq.sum()
    log_weights=np.log(q)
    for iteration in range(1,50001):
        weights=np.exp(log_weights-np.max(log_weights))
        table=support*weights[None,:]
        table/=table.sum(axis=1,keepdims=True)
        marginal=freq@table
        error=float(np.max(np.abs(marginal-q)))
        if error<1e-12:
            return table,iteration,error
        log_weights+=np.log(q/marginal)
        log_weights-=np.max(log_weights)
    raise RuntimeError('The conditional fit did not converge; do not emit a policy')


policy={'version':1,'status':'proposal_not_connected_to_training','sample_rate':16000,'step_samples':STEP,
        'target_samples':targets.tolist(),'target_seconds':(targets/16000).tolist(),'target_probabilities':q.tolist(),
        'original_selection':'uniform within class; class quotas remain external and unchanged',
        'length_selection':'class-specific conditional table fitted using Train metadata only',
        'crop_selection':'front/back 50/50, then uniform nonoverlapping tile within selected anchor',
        'maximum_selected_seconds':6,'duplicate_crop_policy':'retain identical front/back views',
        'under_half_second':'none in Train; policy rejects unsupported sources instead of silently changing the distribution',
        'source':source,'classes':{},
        'limits':['Targets are a proposed exposure budget, not accuracy-optimal values.',
                  'Targets match in expectation in Train, not necessarily in each batch/epoch or another dataset.',
                  'Original speaker/attack/corpus proportions are retained within class in expectation.',
                  'Matching the single duration marginal does not remove joint speaker-duration or acoustic shortcuts.',
                  'Sources longer than 6 s remain eligible; sample positions span both ends of the whole source.',
                  'A selected input has at most 6 s context; >6 s contiguous context is not trained under this policy.']}
rows=[]
class_marginals=[]
group_checks=[]
baseline=next(d for d in previous['datasets'] if d['key']=='19LA_train')
for y,name in [(1,'bonafide'),(0,'spoof')]:
    ids=np.flatnonzero(f.y.to_numpy()==y)
    lengths=n[ids]
    buckets=np.minimum(lengths//STEP,len(q))
    frequency=np.bincount(buckets,minlength=len(q)+1)[1:]
    eligible=np.array([np.count_nonzero(lengths>=target) for target in targets])
    requested_suffix=np.cumsum(q[::-1])[::-1]
    available=eligible/len(lengths)
    assert np.all(requested_suffix<=available+1e-12), 'Infeasible target under fixed original probabilities'
    table,iterations,error=fit_conditionals(frequency)
    per_original=table[buckets-1]
    # Independent verification over every source, rather than the grouped fit.
    actual=per_original.mean(axis=0)
    assert np.allclose(actual,q,atol=2e-12,rtol=0)
    assert np.allclose(per_original.sum(axis=1),1,atol=1e-12)
    assert np.all(per_original[np.broadcast_to(targets,(len(ids),len(q)))>lengths[:,None]]==0)
    for dim in ['speaker_group','attack_id']:
        groups=f.iloc[ids][dim].to_numpy()
        for group in np.unique(groups):
            mask=groups==group
            before=mask.sum()/len(ids)
            after=float(per_original[mask].sum()/len(ids))
            assert np.isclose(before,after,atol=1e-12)
        group_checks.append({'class':name,'dimension':dim,'exact_original_marginal_preserved':True})
    policy['classes'][str(y)]={'name':name,'original_count':len(ids),'bucket_original_counts':frequency.tolist(),
                              'conditional_probability':table.tolist(),'eligible_original_counts':eligible.tolist(),
                              'fit_iterations':iterations,'maximum_marginal_error':error}
    old=np.array(baseline['classes'][name]['original_first'])
    class_marginals.append(actual)
    for i,target in enumerate(targets):
        rows.append({'class':name,'target_seconds':target/16000,'target_percent':TARGET_PERCENT[i],
                     'expected_percent':100*actual[i],'eligible_originals':int(eligible[i]),
                     'eligible_original_percent':100*available[i],
                     'requested_at_or_above_percent':100*requested_suffix[i],
                     'available_at_or_above_percent':100*available[i]})
    print(name,'fit iterations',iterations,'max error',error,'under2',actual[:3].sum(),
          'baseline_under2',baseline['classes'][name]['under2_new_percent'])
pd.DataFrame(rows).to_csv(OUT/'target_and_feasibility.csv',index=False,encoding='utf-8-sig')
(OUT/'policy.json').write_text(json.dumps(policy,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
runtime=OriginalFirstDurationPolicy.load(OUT/'policy.json')
rng=np.random.default_rng(20261110)
anchors={'front':0,'back':0}
for y in [1,0]:
    ids=np.flatnonzero(f.y.to_numpy()==y)
    for chosen in rng.choice(ids,size=2000,replace=True):
        crop=runtime.sample_crop(int(n[chosen]),y,rng)
        assert crop.target_samples in targets and 0<=crop.start_sample<crop.stop_sample<=n[chosen]
        assert crop.stop_sample-crop.start_sample==crop.target_samples
        if crop.anchor=='front':
            assert crop.start_sample%crop.target_samples==0
        else:
            assert (n[chosen]-crop.stop_sample)%crop.target_samples==0
        anchors[crop.anchor]+=1
assert all(x>0 for x in anchors.values())
try:
    runtime.length_probabilities(7999,1)
except ValueError:
    pass
else:
    raise AssertionError('Unsupported short source was silently accepted')
overlap=100*np.minimum(*class_marginals).sum()
validation={'source_sha_match':True,'train_original_count':len(f),'target_is_feasible_under_fixed_original_marginals':True,
            'maximum_class_length_difference':float(np.max(np.abs(class_marginals[0]-class_marginals[1]))),
            'expected_train_length_overlap_percent':float(overlap),'group_marginals':group_checks,
            'runtime_crops_checked':4000,'anchors_seen':anchors,'short_source_rejected':True,
            'mean_selected_seconds':float(q@(targets/16000)),
            'training_loop_modified':False,'training_started':False,'heldout_data_used_to_choose_or_fit_policy':False}
(OUT/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
charts={}
for mode,title in [('baseline','현재 · 원음 균등 → 모든 조각 균등'),('proposed','설계안 · 원음 균등 → 길이 확률 보정')]:
    labels=baseline['length_labels']
    if mode=='baseline':
        values=[baseline['classes'][c]['original_first'] for c in ['bonafide','spoof']]
    else:
        values=[a.tolist()+[0.]*(len(labels)-len(q)) for a in class_marginals]
    charts[mode]={'id':'duration-proposal','title':title,'height':340,
                  'spec':{'type':'bar','x':'길이','y':['Bona fide','Spoof'],'xLabel':'선택 길이 (초)',
                          'yLabel':'클래스 내 선택 확률','sortOrder':'original','yDomain':[0,.5],
                          'valueFormat':{'style':'percent','maximumFractionDigits':2}},
                  'columns':['길이','Bona fide','Spoof'],
                  'rows':[[label,values[0][i],values[1][i]] for i,label in enumerate(labels)]}
(OUT/'chart-configs.json').write_text(json.dumps(charts,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(validation,ensure_ascii=False,indent=2))
