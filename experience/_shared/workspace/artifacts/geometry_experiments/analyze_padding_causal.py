"""Paired, cluster-aware analysis of the prelocked padding intervention.

Does not fit a classifier or choose a condition using the intervention outcomes.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
import json
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
from geometry_cache import resolve_run,load_embeddings
from run_padding_causal import OUT


def save(name,x):
    def conv(v):
        if isinstance(v,np.generic):return v.item()
        if isinstance(v,np.ndarray):return v.tolist()
        raise TypeError(type(v).__name__)
    (OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=conv,allow_nan=False)+'\n',encoding='utf-8')


def cluster_ci(values,groups,seed,primary=False):
    g,codes=np.unique(np.asarray(groups).astype(str),return_inverse=True)
    values=np.asarray(values,float)
    assert np.isfinite(values).all()
    sums=np.bincount(codes,weights=values,minlength=len(g));counts=np.bincount(codes,minlength=len(g))
    draws=np.random.default_rng(seed).multinomial(len(g),np.full(len(g),1/len(g)),size=5000)
    b=(draws@sums)/(draws@counts)
    ans={'N_utterances':len(values),'speaker_clusters':len(g),'mean_effect':float(values.mean()),'median_effect':float(np.median(values)),
        'individual95_CI':np.quantile(b,[.025,.975]).tolist(),'bootstrap_draws':len(b),'source_family_dependency_unresolved':True}
    if primary:ans['family95_CI_12comparisons']=np.quantile(b,[.05/(2*12),1-.05/(2*12)]).tolist()
    return ans


def main():
    done=json.loads((OUT/'complete.json').read_text());assert done['status']=='pass'
    for n,h in done['files'].items():assert hashlib.sha256((OUT/n).read_bytes()).hexdigest()==h
    f=pd.read_parquet(OUT/'scores.parquet');selected=pd.read_parquet(OUT/'selected.parquet')
    protocol=json.loads((OUT/'protocol.json').read_text());threshold=protocol['diagnostic_error_threshold']
    assert len(selected)==471 and f.utterance_id.nunique()==471 and len(f)==3141
    assert not f[['utterance_id','condition']].duplicated().any()
    z=np.concatenate([np.load(OUT/f'rank{r}_z.npy') for r in [0,1]]).astype(float)
    h=np.concatenate([np.load(OUT/f'rank{r}_h.npy') for r in [0,1]]).astype(float)
    c=np.load(OUT/'rank0_centroid.npy');assert np.max(np.abs(c-np.load(OUT/'rank1_centroid.npy')))<1e-12
    assert len(z)==len(f) and np.isfinite(z).all() and np.isfinite(h).all()
    x=z/np.linalg.norm(z,axis=1,keepdims=True);s=x@c
    assert np.max(np.abs(s-f.cosine_s.to_numpy()))<1e-10
    theta=np.degrees(np.arccos(np.clip(s,-1,1)))
    assert np.max(np.abs(theta-f.theta_deg))<1e-9
    f['theta_deg']=theta
    baseline=f[f.condition=='native_wrap'].set_index('utterance_id')
    # Independent local correspondence to immutable original cache arrays.
    maxerrors={'h':0.,'z':0.,'score':0.}
    for alias in selected.cache_alias.unique():
        folder,_,split=resolve_run(alias);old=pd.read_parquet(folder/'extract/scores.parquet').set_index('utterance_id')
        arrays={kind:{r:np.load(folder/f'extract/rank{r}_{kind}.npy',mmap_mode='r') for r in [0,1]} for kind in ['h','z']}
        ids=set(selected.loc[selected.cache_alias==alias,'utterance_id'])
        ix=f.index[(f.condition=='native_wrap')&f.utterance_id.isin(ids)]
        for j in ix:
            row=old.loc[f.loc[j,'utterance_id']];r,i=int(row['rank']),int(row.embedding_row)
            maxerrors['z']=max(maxerrors['z'],float(np.max(np.abs(z[j]-arrays['z'][r][i]))))
            maxerrors['h']=max(maxerrors['h'],float(np.max(np.abs(h[j]-arrays['h'][r][i]))))
            maxerrors['score']=max(maxerrors['score'],abs(s[j]-row.cosine_s))
    assert max(maxerrors.values())<=1e-5
    # A global direction defined using prior pilot samples, excluding ALL new intervention IDs.
    d=load_embeddings('DF_pilot');gg=d.geometry();oldtheta=np.degrees(np.arccos(np.clip(gg['cosine'],-1,1)));exclude=set(selected.utterance_id)
    eligible=(d.rows.y.to_numpy()==1)&~d.rows.utterance_id.isin(exclude).to_numpy()
    near=eligible&(oldtheta<10);far=eligible&(oldtheta>90)
    bad=gg['unit_z'][far].mean(0)-gg['unit_z'][near].mean(0);bad/=np.linalg.norm(bad)
    reference={'near_N':int(near.sum()),'far_N':int(far.sum()),'source':'prior DF pilot excluding all471 newly selected IDs',
        'definition':'unit(mean(x of bona theta>90)-mean(x of bona theta<10)); descriptive, not a fitted detector or physical acoustic axis',
        'cosine_to_minus_c':float(bad@(-c))}
    native_index={u:i for i,u in zip(f.index[f.condition=='native_wrap'],f.loc[f.condition=='native_wrap','utterance_id'])}
    jbase=np.array([native_index[u] for u in f.utterance_id]);xb=x[jbase]
    dx=x-xb
    f['delta_score']=s-s[jbase];f['delta_theta']=theta-theta[jbase]
    f['embedding_movement_deg']=np.degrees(np.arccos(np.clip(np.sum(x*xb,axis=1),-1,1)))
    f['movement_along_bad_axis']=dx@bad
    tangent=x-s[:,None]*c;tn=np.linalg.norm(tangent,axis=1);u=tangent/tn[:,None]
    f['tangent_change_deg']=np.degrees(np.arccos(np.clip(np.sum(u*u[jbase],axis=1),-1,1)))
    f['diagnostic_error']=((f.y==1)&(s<threshold))|((f.y==0)&(s>=threshold))
    assert np.max(np.abs(f.loc[f.condition=='native_wrap','delta_score']))==0
    # Long original/reflect inputs are independently inferred exact no-ops.
    noop=f[(f.duration_bin=='long')&f.condition.isin(['native_original','native_reflect'])]
    assert np.max(np.abs(noop.delta_score))==0
    summaries=[]
    for (source,y,binname,condition),g in f.groupby(['source_corpus','y','duration_bin','condition'],sort=True):
        summaries.append({'source':source,'y':int(y),'duration_bin':binname,'condition':condition,'N':len(g),
            'median_theta':float(g.theta_deg.median()),'mean_theta':float(g.theta_deg.mean()),'mean_score':float(g.cosine_s.mean()),
            'mean_delta_score':float(g.delta_score.mean()),'mean_delta_theta':float(g.delta_theta.mean()),
            'error_N':int(g.diagnostic_error.sum()),'error_percent':float(100*g.diagnostic_error.mean()),
            'median_embedding_movement_deg':float(g.embedding_movement_deg.median()),'mean_bad_axis_move':float(g.movement_along_bad_axis.mean()),
            'median_frontend_mean_cosine':float(g.ssl_last_mean_cosine_to_native.median())})
    pd.DataFrame(summaries).to_csv(OUT/'condition_summaries.csv',index=False)
    pairs=[]
    for source in sorted(selected.source_corpus.unique()):
        normal=f[(f.source_corpus==source)&(f.y==1)]
        short=normal[normal.duration_bin!='long']
        wide=short.pivot(index='utterance_id',columns='condition',values='theta_deg')
        groups=baseline.loc[wide.index,'speaker_group'].to_numpy()
        for cond in ['native_original','native_reflect']:
            effect=wide[cond]-wide.native_wrap
            pairs.append({'source':source,'contrast':cond+' minus native_wrap on original short normals','primary':True,'units':'degrees; negative is toward c',**cluster_ci(effect,groups,2026110601,True)})
        long=normal[normal.duration_bin=='long'];ww=long.pivot(index='utterance_id',columns='condition',values='theta_deg');groups=baseline.loc[ww.index,'speaker_group'].to_numpy()
        for seconds in [1.5,2.0]:
            mean=lambda kind:.5*(ww[f'crop{seconds:g}_head_{kind}']+ww[f'crop{seconds:g}_tail_{kind}'])
            for tag,effect in [('crop_wrap minus native',mean('wrap')-ww.native_wrap),('crop_original minus crop_wrap',mean('original')-mean('wrap')),('crop_reflect minus crop_wrap',mean('reflect')-mean('wrap'))]:
                primary=seconds==2.0 and tag!='crop_reflect minus crop_wrap'
                pairs.append({'source':source,'contrast':f'{seconds:g}s {tag}; head/tail averaged per utterance','primary':primary,'units':'degrees; negative is toward c',**cluster_ci(effect,groups,2026110602,primary)})
    assert sum(p['primary'] for p in pairs)==12
    save('paired_effects.json',pairs)
    # Transparent aggregate error transitions in the deliberately stratified sample.
    transitions=[]
    for source in sorted(selected.source_corpus.unique()):
        for y in [0,1]:
            q=f[(f.source_corpus==source)&(f.y==y)&(f.duration_bin!='long')]
            ww=q.pivot(index='utterance_id',columns='condition',values='diagnostic_error')
            for cond in ['native_original','native_reflect']:
                a,b=ww.native_wrap.to_numpy(bool),ww[cond].to_numpy(bool)
                transitions.append({'source':source,'y':y,'condition':cond,'N':len(a),'native_error_N':int(a.sum()),'condition_error_N':int(b.sum()),'corrected_N':int((a&~b).sum()),'new_errors_N':int((~a&b).sum())})
    save('error_transitions.json',transitions)
    f.to_parquet(OUT/'paired_derived.parquet',index=False)
    save('direction_reference.json',reference)
    save('analysis_verification.json',{'status':'pass','N_utterances':len(selected),'N_forward_outputs':len(f),
        'input_SHA256_checked':True,'native_cache_max_errors':maxerrors,'native_score_max_error':float(np.max(np.abs(s-f.cosine_s))),
        'long_noop_max_delta_score':float(np.max(np.abs(noop.delta_score))),'primary_contrasts':12,
        'same_utterance_head_tail_not_counted_independently':True,'silence_distribution_excluded':True,'no_classifier_training':True,
        'bootstrap_groups':'speaker within each source; original-family dependence unresolved','figure_review':'pending'})
    print('Primary paired effects:',json.dumps([p for p in pairs if p['primary']],ensure_ascii=False,indent=2),flush=True)
    print('Error transitions:',json.dumps(transitions,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
