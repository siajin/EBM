"""Train-only feasibility and proposal for duration x speaker x attack matching.

This does not alter the training loop or audio. Matching an existing label
within-class group mixture differs from making speaker mixtures identical
between labels; both cases are checked explicitly.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.optimize import linprog

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/joint_duration_group_balance_20261110'
OUT.mkdir(parents=True,exist_ok=True)
old=json.loads((ROOT/'artifacts/duration_sampling_design_20261110/policy.json').read_text(encoding='utf-8'))
path=ROOT/old['source']['path']
assert hashlib.sha256(path.read_bytes()).hexdigest()==old['source']['sha256']
f=pd.read_parquet(path,columns=['split','utterance_id','speaker_group','attack_id','y','duration'])
f=f.loc[f.split=='T'].reset_index(drop=True)
assert len(f)==25380 and f.utterance_id.is_unique and not f.isna().any().any()
f['samples']=np.rint(f.duration*16000).astype('int64')
assert (f.samples>=8000).all() and np.max(np.abs(f.samples/16000-f.duration))<1e-9
groups=list(f.groupby(['y','speaker_group','attack_id'],sort=True))
all_targets=np.asarray(old['target_samples'])
old_q=np.asarray(old['target_probabilities'])
old_tail=np.cumsum(old_q[::-1])[::-1]
availability=[]
for j,t in enumerate(all_targets):
    cells=[]
    for key,g in groups:
        eligible=int((g.samples>=t).sum())
        cells.append({'label':int(key[0]),'speaker':key[1],'attack':key[2],
                      'eligible_originals':eligible,'originals':len(g),'available_fraction':eligible/len(g)})
    limiting=min(cells,key=lambda x:x['available_fraction'])
    availability.append({'target_seconds':t/16000,'old_requested_at_or_above_fraction':float(old_tail[j]),
                         'minimum_group_available_fraction':limiting['available_fraction'],
                         'limiting_group':limiting})
limits=np.array([a['minimum_group_available_fraction'] for a in availability])
feasible_tail=np.minimum(old_tail,limits)
closest_downward_q=feasible_tail-np.r_[feasible_tail[1:],0.]
assert np.all(closest_downward_q>=-1e-12) and np.isclose(closest_downward_q.sum(),1)

# Rounded interior proposal with some margin from each limiting capacity.
target_percent=np.array([10,20,30,20,8,5,6,1.],dtype='float64')
q=target_percent/100
targets=np.arange(1,len(q)+1)*8000
assert np.isclose(q.sum(),1)
tail=np.cumsum(q[::-1])[::-1]
assert np.all(tail<=limits[:len(q)]+1e-12)

speaker_before={y:f.loc[f.y==y,'speaker_group'].value_counts(normalize=True).sort_index() for y in [1,0]}
assert speaker_before[1].index.equals(speaker_before[0].index)
speaker_target=(speaker_before[1]+speaker_before[0])/2
assert np.isclose(speaker_target.sum(),1)
source_prob=np.zeros(len(f))
original_uniform=np.zeros(len(f))
relative_rows=[]
for y in [1,0]:
    class_count=int((f.y==y).sum())
    for speaker in speaker_target.index:
        mask=(f.y==y)&(f.speaker_group==speaker)
        source_prob[mask]=speaker_target.loc[speaker]/mask.sum()
        original_uniform[mask]=1/class_count
        ratio=float(speaker_target.loc[speaker]/speaker_before[y].loc[speaker])
        relative_rows.append({'label':y,'speaker':speaker,'original_share':float(speaker_before[y].loc[speaker]),
                              'common_target_share':float(speaker_target.loc[speaker]),'original_weight_multiplier':ratio})
    assert np.isclose(source_prob[f.y==y].sum(),1)

def fit(freq,support,target):
    freq=freq/freq.sum()
    log_weights=np.log(target)
    for iteration in range(1,100001):
        weights=np.exp(log_weights-np.max(log_weights))
        a=support*weights
        a/=a.sum(axis=1,keepdims=True)
        actual=freq@a
        if np.max(np.abs(actual-target))<1e-12:
            return a,iteration
        log_weights+=np.log(target/actual)
        log_weights-=log_weights.max()
    raise RuntimeError('Group kernel did not converge')

support=np.arange(1,len(q)+1)[:,None]>=np.arange(1,len(q)+1)[None,:]
conditional=np.zeros((len(f),len(q)))
kernels=[]
max_group_error=0.
for (y,speaker,attack),g in groups:
    bucket=np.minimum(g.samples.to_numpy()//8000,len(q))
    freq=np.bincount(bucket,minlength=len(q)+1)[1:]
    a,iterations=fit(freq,support,q)
    conditional[g.index]=a[bucket-1]
    actual=a[bucket-1].mean(axis=0)
    error=float(np.max(np.abs(actual-q)))
    max_group_error=max(max_group_error,error)
    assert error<2e-12
    kernels.append({'label':int(y),'speaker':speaker,'attack':attack,'original_count':len(g),
                    'bucket_counts':freq.tolist(),'conditional_probability':a.tolist(),'iterations':iterations})
assert np.allclose(conditional.sum(axis=1),1,atol=1e-12)
assert np.all(conditional[targets[None,:]>f.samples.to_numpy()[:,None]]==0)

# Independent small transportation LP for every group: fixed original-bucket
# mass + target columns with only physically realizable edges.
lp_checked=0
for kernel in kernels:
    freq=np.array(kernel['bucket_counts'],dtype='float64')
    rows,cols=np.where(support & (freq[:,None]>0))
    A=np.zeros((len(q)*2,len(rows)))
    A[rows,np.arange(len(rows))]=1
    A[len(q)+cols,np.arange(len(rows))]=1
    b=np.r_[freq,freq.sum()*q]
    solved=linprog(np.zeros(len(rows)),A_eq=A,b_eq=b,bounds=(0,None),method='highs')
    assert solved.success and np.max(np.abs(A@solved.x-b))<1e-7
    lp_checked+=1

group_checks=[]
cross_speaker=[]
for mode,weights in [('original_uniform',original_uniform),('speaker_matched',source_prob)]:
    joint=conditional*weights[:,None]
    length_marginals=[]
    speaker_marginals=[]
    for y in [1,0]:
        mask=f.y.to_numpy()==y
        actual=joint[mask].sum(axis=0)
        length_marginals.append(actual)
        assert np.allclose(actual,q,atol=2e-12)
        target_speaker=speaker_before[y] if mode=='original_uniform' else speaker_target
        by_s=[]
        for speaker in speaker_target.index:
            smask=mask & (f.speaker_group.to_numpy()==speaker)
            actual_joint=joint[smask].sum(axis=0)
            assert np.allclose(actual_joint,target_speaker.loc[speaker]*q,atol=2e-12)
            by_s.append(actual_joint)
        speaker_marginals.append(np.stack(by_s))
        if y==0:
            for attack in sorted(f.loc[f.y==0,'attack_id'].unique()):
                amask=mask & (f.attack_id.to_numpy()==attack)
                assert np.allclose(joint[amask].sum(axis=0),q/6,atol=2e-12)
        group_checks.append({'mode':mode,'label':y,'duration_matches_target':True,
                             'speaker_duration_factorizes':True,'attack_duration_factorizes':True})
    for j,t in enumerate(targets):
        b=speaker_marginals[0][:,j]/q[j]
        s=speaker_marginals[1][:,j]/q[j]
        cross_speaker.append({'mode':mode,'target_seconds':t/16000,
                              'bona_spoof_speaker_overlap_percent':float(100*np.minimum(b,s).sum())})
    assert np.allclose(length_marginals[0],length_marginals[1],atol=2e-12)
ratios=np.array([r['original_weight_multiplier'] for r in relative_rows])
result={'source':old['source'],'scope':'Train only; no audio access, model training, or loader edits',
        'dimensions':'protocol speaker and attack; Train corpus is ASVspoof only; no room/microphone identity inferred',
        'previous_6_second_target_fully_group_balanced_feasible':False,
        'group_count':len(groups),'speakers_per_class':20,
        'bonafide_speakers_without_6_second_original':int((f[f.y==1].groupby('speaker_group').duration.max()<6).sum()),
        'maximum_length_available_in_every_group_seconds':float(min(g.duration.max() for _,g in groups)),
        'availability':availability,'closest_downward_reallocation_probabilities':closest_downward_q.tolist(),
        'minimum_under2_fraction_if_fixed_original_within_every_group':float(1-limits[3]),
        'proposed_target_seconds':(targets/16000).tolist(),'proposed_target_probabilities':q.tolist(),
        'proposed_under2_fraction':float(q[:3].sum()),'proposed_mean_seconds':float(q@(targets/16000)),
        'matching_modes':{
            'original_uniform':'Each original retains 1/N in class; every speaker x attack cell has the shared duration target; original cross-label speaker differences remain.',
            'speaker_matched':'Common speaker target is the mean of original class speaker mixtures; originals remain uniform within speaker; each speaker x attack cell has the shared duration target.'},
        'speaker_matching_min_original_weight_multiplier':float(ratios.min()),
        'speaker_matching_max_original_weight_multiplier':float(ratios.max()),
        'speaker_matching_max_relative_original_weight_change_percent':float(100*np.abs(ratios-1).max()),
        'original_speaker_crossclass_overlap_percent':float(100*np.minimum(speaker_before[1],speaker_before[0]).sum()),
        'maximum_per_cell_length_error':max_group_error,'independent_transport_LPs_passed':lp_checked,
        'validation':group_checks,'crossclass_speaker_overlap_by_length':cross_speaker,
        'limitations':['Protocol group balance does not guarantee acoustic shortcut removal.',
                       'Uniform original probabilities and exactly equal cross-label speaker marginals cannot both hold because original marginals differ.',
                       'Target matches are expectations, not guarantees within a single mini-batch or epoch.',
                       'Longest conditional cells still rely on a small number of originals; group balance does not create new independent source diversity.',
                       'No heldout metadata or performance was used to fit these kernels.'],
        'solver_documentation':'https://docs.scipy.org/doc/scipy/reference/optimize.linprog-highs.html'}
policy={'status':'validated_proposal_not_connected_to_training','sample_rate':16000,'step_samples':8000,
        'target_samples':targets.tolist(),'target_probabilities':q.tolist(),'original_selection_mode':'speaker_matched',
        'original_relative_weights':relative_rows,'crop_selection':'front/back 50/50 then uniform tile',
        'conditional_profiles':kernels,'validation':{'maximum_per_cell_length_error':max_group_error,'independent_LPs_passed':lp_checked}}
(OUT/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(OUT/'policy.json').write_text(json.dumps(policy,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
pd.DataFrame(relative_rows).to_csv(OUT/'speaker_weight_adjustments.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(cross_speaker).to_csv(OUT/'speaker_overlap_by_length.csv',index=False,encoding='utf-8-sig')
pd.DataFrame([{'length_seconds':a['target_seconds'],'old_requested_at_or_above_percent':100*a['old_requested_at_or_above_fraction'],
               'minimum_group_available_percent':100*a['minimum_group_available_fraction'],
               'limiting_speaker':a['limiting_group']['speaker'],'limiting_attack':a['limiting_group']['attack']} for a in availability]).to_csv(OUT/'joint_feasibility.csv',index=False,encoding='utf-8-sig')
print(json.dumps({k:result[k] for k in ['group_count','bonafide_speakers_without_6_second_original',
      'maximum_length_available_in_every_group_seconds','minimum_under2_fraction_if_fixed_original_within_every_group',
      'proposed_target_seconds','proposed_target_probabilities','proposed_mean_seconds',
      'speaker_matching_min_original_weight_multiplier','speaker_matching_max_original_weight_multiplier',
      'speaker_matching_max_relative_original_weight_change_percent','original_speaker_crossclass_overlap_percent',
      'maximum_per_cell_length_error','independent_transport_LPs_passed']},ensure_ascii=False,indent=2))
