from pathlib import Path
import json
import numpy as np
from scipy.stats import t
assert Path('/.dockerenv').exists()
root=Path('/results/analysis/EBM80_native_quiet_strict');out=root/'analysis'
j=json.loads((out/'summary.json').read_text());d=np.load(out/'merged_private.npz')
check=json.loads((root/'inference/verification.json').read_text());assert check['all_checks_passed']
s=d['scores'].astype(float);y=d['labels'];cohort=d['cohort'];source=d['source'];speaker=d['speaker'];eligible=d['eligible'].astype(bool)
eps=j['provenance']['direction_epsilon'];outj={'independent_primary_comparisons':[],'dose_monotonicity':{},'low_splice_subsets':{}}
for row in j['primary']:
    family=row['family'];label=1 if row['label']=='bonafide' else 0
    f,b=(10,11) if family=='zero' else (12,13)
    before=0 if row['contrast']=='front_original' else b
    mask=(cohort=='new_files')&(y==label)
    if family=='own40':mask &= eligible
    delta=s[:,f]-s[:,before];v=delta[mask]
    reference=check['primary_point_estimates'][family][row['label']]['front_minus_original' if before==0 else 'front_minus_back']
    assert row['n']==len(v)==reference['n']
    for key,refkey in [('mean','mean_delta_cosine'),('median','median_delta_cosine'),('toward_n','toward_center_n'),('away_n','away_from_center_n'),('within_epsilon_n','within_epsilon_n'),('mean_actual_embedding_displacement','mean_embedding_displacement')]:
        assert abs(row[key]-reference[refkey])<1e-7
    # Independent stratified cluster-sandwich approximation, with finite-cluster correction.
    components=[];dfparts=[];leave_one=[];cluster_counts={};full=float(v.mean())
    for src in np.unique(source[mask]):
        m=mask&(source==src);N=int(m.sum());mu=float(delta[m].mean());w=N/len(v)
        groups=[m&(speaker==sp) for sp in np.unique(speaker[m])];G=len(groups);assert G>1
        residual=np.array([np.sum(delta[g]-mu) for g in groups]);var=w*w*G/(G-1)*np.sum(residual**2)/(N*N)
        components.append(var);dfparts.append(var*var/(G-1));cluster_counts[str(src)]=G
        for g in groups:
            keep=m&~g;assert keep.any()
            leave_one.append(full+w*(float(delta[keep].mean())-mu))
    variance=sum(components);df=variance*variance/sum(dfparts) if sum(dfparts)>0 else 1e9
    half=float(t.ppf(.996875,df)*np.sqrt(variance));ci=[full-half,full+half]
    direction='toward' if ci[0]>0 else ('away' if ci[1]<0 else 'unresolved')
    outj['independent_primary_comparisons'].append({'family':family,'label':row['label'],'contrast':row['contrast'],'n':len(v),
      'mean':full,'cluster_sandwich_se':float(np.sqrt(variance)),'approx_df':float(df),'multiplicity_adjusted_t_interval':ci,
      'bootstrap_interval':row['mean_simultaneous_ci'],'bootstrap_direction':row['primary_mean_direction'],
      'independent_direction':direction,'directions_agree':direction==row['primary_mean_direction'],
      'leave_one_speaker_cluster_out_mean_range':[float(min(leave_one)),float(max(leave_one))],'speaker_clusters':cluster_counts})
for fam,front,back in [('zero',[2,6,10],[3,7,11]),('own40',[4,8,12],[5,9,13])]:
    outj['dose_monotonicity'][fam]={};outj['low_splice_subsets'][fam]={}
    for label,word in [(1,'bonafide'),(0,'spoof')]:
        mask=(cohort=='new_files')&(y==label)
        if fam=='own40':mask &= eligible
        item={}
        for cols,key in [(front,'front'),(back,'back')]:
            allviews=s[mask][:,[0]+cols];steps=np.diff(allviews,axis=1)
            item[key]={'n':int(mask.sum()),'nondecreasing_with_epsilon_n':int(np.all(steps>=-eps,axis=1).sum()),
              'nonincreasing_with_epsilon_n':int(np.all(steps<=eps,axis=1).sum()),
              'strictly_increasing_each_step_n':int(np.all(steps>eps,axis=1).sum())}
        outj['dose_monotonicity'][fam][word]=item
        outj['low_splice_subsets'][fam][word]={}
        for limit in [.01,.1]:
            m=mask&(np.max(d['seams'][:,2,0 if fam=='zero' else 1],axis=1)<=limit)
            if not m.any():outj['low_splice_subsets'][fam][word][str(limit)]={'n':0};continue
            a=s[m,front[-1]]-s[m,0];b=s[m,front[-1]]-s[m,back[-1]]
            outj['low_splice_subsets'][fam][word][str(limit)]={'n':int(m.sum()),'front_original_mean':float(a.mean()),'front_original_median':float(np.median(a)),
              'front_original_toward_n':int((a>eps).sum()),'front_back_mean':float(b.mean()),'front_back_median':float(np.median(b)),
              'front_back_toward_n':int((b>eps).sum())}
outj['all_point_estimates_independently_match']=True
outj['all_primary_direction_decisions_agree']=all(x['directions_agree'] for x in outj['independent_primary_comparisons'])
outj['limits']=['Cluster-sandwich/t intervals are independent approximations, not an exact finite-sample proof. Primary inference remains the prespecified cluster bootstrap.',
 'Leave-one-cluster and splice/monotonicity checks are secondary descriptive sensitivity analyses. No extra confirmatory significance claims.']
(out/'robustness.json').write_text(json.dumps(outj,indent=2,allow_nan=False));print(json.dumps(outj,indent=2))
