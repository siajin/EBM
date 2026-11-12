"""Independent derived-score, Shapley, duplication algebra, and direction audit."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from run_padding_causal import OUT


def dump(path,x):
    Path(path).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    folders=[OUT,OUT/'pooling_mechanism',OUT/'frontend_context']
    for folder in folders:
        complete=json.loads((folder/'complete.json').read_text());assert complete['status']=='pass'
        for n,sha in complete['files'].items():assert hashlib.sha256((folder/n).read_bytes()).hexdigest()==sha
    native=pd.read_parquet(OUT/'scores.parquet');pool=pd.read_parquet(OUT/'pooling_mechanism/results.parquet');context=pd.read_parquet(OUT/'frontend_context/results.parquet')
    assert len(pool)==len(context)==471 and set(pool.utterance_id)==set(context.utterance_id)
    # Explicit two-order path decomposition, independently recomputed.
    ds=.5*((pool.std_only_score-pool.native_wrap_score)+(pool.native_original_score-pool.mean_only_score))
    dm=.5*((pool.mean_only_score-pool.native_wrap_score)+(pool.native_original_score-pool.std_only_score))
    assert np.max(np.abs(ds-pool.std_shapley_score_change))<1e-12
    assert np.max(np.abs(dm-pool.mean_shapley_score_change))<1e-12
    assert np.max(np.abs(ds+dm-pool.native_original_score+pool.native_wrap_score))<1e-12
    dup={}
    for k in [2,4]:
        delta=np.abs(pool[f'dup{k}_none_score']-pool.noise0_base_score);corrected=np.abs(pool[f'dup{k}_both_score']-pool.noise0_base_score)
        dup[str(k)]={'mean_abs_score_change':float(delta.mean()),'p95_abs_score_change':float(delta.quantile(.95)),'max_abs_score_change':float(delta.max()),'corrected_max_abs_score_change':float(corrected.max())}
        assert corrected.max()<1e-4
    # Independently demonstrate the std formula on explicit repeated arrays.
    rng=np.random.default_rng(8);a=rng.normal(size=(99,128));logit=rng.normal(size=99);w=np.exp(np.tanh(logit));w/=w.sum();sd=(a*w[:,None]).std(0,ddof=1)
    mathchecks=[]
    for k in [2,4]:
        duplicated=np.tile(a,(k,1))*np.tile(w/k,k)[:,None];new=duplicated.std(0,ddof=1);factor=np.sqrt(k*(k*len(a)-1)/(len(a)-1))
        error=float(np.max(np.abs(new*factor-sd)));assert error<1e-14
        mathchecks.append({'k':k,'observed_std_ratio':float(np.mean(new/sd)),'predicted_std_ratio':float(1/factor),'corrected_error':error})
    # Full factorial block swaps: sum the six marginal orders independently.
    import itertools
    contributions=np.zeros((len(context),3))
    for order in itertools.permutations(range(3)):
        mask=0
        for i in order:
            nxt=mask|(1<<i);contributions[:,i]+=(context[f'mask{nxt}_score']-context[f'mask{mask}_score']).to_numpy()/6;mask=nxt
    assert np.max(np.abs(contributions.sum(1)-(context.mask7_score-context.mask0_score)))<1e-12
    for i,label in enumerate(['early_0_8','middle_9_16','late_17_24']):context[label+'_shapley']=contributions[:,i]
    summaries=[]
    for (source,y,binname),f in context.groupby(['source_corpus','y','duration_bin'],sort=True):
        row={'source':source,'y':int(y),'duration_bin':binname,'N':len(f)}
        for name in [f'mask{i}_score' for i in range(8)]+['original_unpadded_score']:
            row[name+'_mean']=float(f[name].mean());row[name+'_mean_angle']=float(np.degrees(np.arccos(np.clip(f[name],-1,1))).mean())
        for i,label in enumerate(['early_0_8','middle_9_16','late_17_24']):row[label+'_mean_shapley']=float(f[label+'_shapley'].mean())
        summaries.append(row)
    pd.DataFrame(summaries).to_csv(OUT/'frontend_context/group_effects.csv',index=False)
    context.to_parquet(OUT/'frontend_context/decomposed_results.parquet',index=False)
    short_normal_summaries=[]
    for source in ['asvspoof','vcc2018','vcc2020']:
        f=context[(context.source_corpus==source)&(context.y==1)&(context.duration_bin!='long')]
        row={'source':source,'N':len(f)}
        for name in ['mask0_score','mask1_score','mask2_score','mask4_score','mask7_score','original_unpadded_score']:
            row[name+'_mean']=float(f[name].mean());row[name+'_mean_angle']=float(np.degrees(np.arccos(np.clip(f[name],-1,1))).mean())
        for label in ['early_0_8','middle_9_16','late_17_24']:row[label+'_mean_shapley']=float(f[label+'_shapley'].mean())
        short_normal_summaries.append(row)
    dump(OUT/'frontend_context/short_normal_summary.json',short_normal_summaries)
    stats=[]
    for source in ['asvspoof','vcc2018','vcc2020']:
        f=pool[(pool.source_corpus==source)&(pool.y==1)&(pool.duration_bin!='long')]
        stats.append({'source':source,'N':len(f),'wrap_score_mean':float(f.native_wrap_score.mean()),'original_score_mean':float(f.native_original_score.mean()),'mean_branch_score_contribution':float(f.mean_shapley_score_change.mean()),'std_branch_score_contribution':float(f.std_shapley_score_change.mean())})
    dump(OUT/'pooling_mechanism/summary.json',{'duplication':dup,'explicit_std_math_checks':mathchecks,'short_normal_pooled_stat_contributions':stats,'interpretation':'Nonstandard std scaling exists but does not explain the large waveform-padding shift in these tests; final mean feature change dominates.'})
    # Direction concentration: raw paired h changes vs post-LN z changes.
    ix=native.copy();ix['i']=range(len(ix));ids=ix[(ix.y==1)&(ix.duration_bin!='long')].utterance_id.unique();pivot=ix.pivot(index='utterance_id',columns='condition',values='i').loc[ids]
    directions={}
    for kind in ['h','z']:
        arr=np.concatenate([np.load(OUT/f'rank{r}_{kind}.npy') for r in [0,1]]).astype(float)
        delta=arr[pivot.native_original.to_numpy(int)]-arr[pivot.native_wrap.to_numpy(int)]
        _,sv,vt=np.linalg.svd(delta,full_matrices=False);energy=float(sv[0]**2/np.sum(sv**2))
        # Quadratic form/trace independently reproduces uncentered SVD energy.
        projected=np.sum((delta@vt[0])**2)/np.sum(delta**2);assert abs(projected-energy)<1e-12
        directions[kind]={'N_paired_normal_changes':len(delta),'first_axis_uncentered_energy':energy,'is_intrinsic_dimension_claim':False}
    dump(OUT/'paired_shift_direction.json',directions)
    # All long controls are no-ops and all native scores match the original arrays.
    long=context[context.duration_bin=='long']
    maxnoop=max(float(np.max(np.abs(long[f'mask{i}_score']-long.mask0_score))) for i in range(8));assert maxnoop<1e-5
    cross=context.merge(pool[['utterance_id','native_wrap_score']],on='utterance_id',validate='one_to_one')
    native_error=float(np.max(np.abs(cross.mask0_score-cross.native_wrap_score)));assert native_error<1e-5
    outcome={'status':'pass','original_context_native_max_error':native_error,'long_control_noop_max_error':maxnoop,
        'file_SHA256_verified':True,'pooled_stat_shapley_independently_recomputed':True,'three_block_shapley_all6_orders_verified':True,
        'explicit_std_duplication_algebra_verified':True,'paired_h_z_SVD_energy_verified_by_quadratic_form':True,
        'no_untested_claims':['No full-DF performance improvement established','No microphone/pitch/silence attribution','No assertion standard-deviation scaling is main cause','Hybrid feature swaps are internal functional sensitivity, not audio-realizable intervention'],
        'limitations':['Main12 contrast CIs prelocked; later mechanism followups exploratory','source-family dependence and selected-cohort uncertainty remain','Backend fixed201 control localizes input sensitivity, not the unique internal frontend operation or training cause']}
    dump(OUT/'independent_verification.json',outcome)
    print(json.dumps({'verification':outcome,'short_normals':short_normal_summaries,'paired_direction':directions},ensure_ascii=False,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
