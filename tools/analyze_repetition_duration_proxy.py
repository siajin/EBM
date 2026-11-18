"""Train-only duration proxy and composition sensitivity; metadata/derived scores only."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from analyze_repetition_shortcut import eer

BASE=Path(__file__).resolve().parents[2]
OUT=BASE/'artifacts/repetition_shortcut_20261111'
BINS=[0,16000,24000,32000,40000,48000,56000,64600,np.inf]

def weights(g,policy):
    counts=pd.crosstab(g['bin'],g.label).reindex(columns=[0,1],fill_value=0)
    common=counts[(counts[0]>0)&(counts[1]>0)]
    a=g[g.bin.isin(common.index)].copy()
    if policy=='common_support_observed':return a,None
    target=common[1] if policy=='bona_marginal' else pd.Series(1.,index=common.index)
    target=target/target.sum();lookup=np.zeros((8,2))
    for b in common.index:lookup[b,:]=target.loc[b]/counts.loc[b].to_numpy()
    return a,lookup[a.bin.to_numpy(int),a.label.to_numpy(int)]

def main():
    source=BASE/'artifacts/mild_duration_data_analysis_20261111/variant_metadata.parquet'
    train=pd.read_parquet(source).drop_duplicates('uid');assert len(train)==25380 and train.label.sum()==2580
    train['bin']=pd.cut(train.original_samples,BINS,labels=False)
    counts=pd.crosstab(train['bin'],train.label).reindex(index=range(8),columns=[0,1],fill_value=0)
    # Jeffreys pseudocount and equal class priors. Fit ONLY on original training metadata.
    conditional=(counts+.5)/(counts.sum(axis=0)+.5*8)
    bona_probability=conditional[1]/conditional.sum(axis=1)
    table=counts.copy();table.columns=['spoof_n','bona_n'];table['duration_proxy_bona_score']=bona_probability
    table.to_csv(OUT/'train_duration_proxy_lookup.csv')
    proxy=[]
    for phase in ['dev','19LA','21LA','21DF']:
        f=pd.read_parquet(BASE/f'artifacts/geometry_cache/local_analysis/B_final_analysis_20261108_v1/paired_scores_{phase}.parquet')
        b=pd.cut(f.original_samples,BINS,labels=False);s=bona_probability.to_numpy()[b.to_numpy(int)]
        proxy.append(dict(phase=phase,n=len(f),eer=eer(f.label,s),auc=float(roc_auc_score(f.label,s)),feature='original duration bin / idealized known repeated-period length; metadata only, not a measured network feature'))
    pd.DataFrame(proxy).to_csv(OUT/'train_only_duration_proxy_evaluation.csv',index=False)
    file=BASE/'artifacts/coverage20_length_shortcut_20261111/joined_scores_21LA.parquet'
    f=pd.read_parquet(file);assert len(f)==148176 and f.uid.is_unique;f['bin']=pd.cut(f.original_samples,BINS,labels=False)
    result=[]
    for col in [c for c in f if c.startswith('score_')]:
        result.append(dict(model=col.removeprefix('score_'),policy='observed',n=len(f),eer=eer(f.label,f[col])))
        for policy in ['common_support_observed','bona_marginal','uniform_common']:
            a,w=weights(f,policy);result.append(dict(model=col.removeprefix('score_'),policy=policy,n=len(a),eer=eer(a.label,a[col],w)))
    pd.DataFrame(result).to_csv(OUT/'current_and_reference_length_reweighted_eer.csv',index=False)
    # Preserve spoof attack x codec marginal while matching bona duration marginal.
    # Iterative proportional fitting is applied to metadata cells, never to scores.
    meta=pd.read_csv(BASE/'artifacts/coverage20_length_shortcut_20261111/trial_metadata_21LA.txt',sep=r'\s+',header=None,names=['speaker_m','uid','codec','transmission','attack_m','class_m','trim','subset'])
    f=f.merge(meta[['uid','codec']],on='uid',validate='one_to_one')
    a,_=weights(f,'common_support_observed');spoof=a[a.label==0].copy();bona=a[a.label==1].copy()
    spoof['group']=spoof.attack+'|'+spoof.codec
    counts=pd.crosstab(spoof['bin'],spoof['group']).astype(float)
    rt=bona['bin'].value_counts(normalize=True).reindex(counts.index).to_numpy();ct=(counts.sum(axis=0)/counts.to_numpy().sum()).to_numpy()
    raw=counts.to_numpy();mass=raw/raw.sum()
    for iteration in range(10000):
        mass*=np.divide(rt,mass.sum(axis=1))[:,None]
        mass*=np.divide(ct,mass.sum(axis=0))[None,:]
        residual=max(abs(mass.sum(axis=1)-rt).max(),abs(mass.sum(axis=0)-ct).max())
        if residual<1e-11:break
    assert residual<1e-9,('No converged common-support IPF',residual)
    cellweight=np.divide(mass,raw,out=np.zeros_like(mass),where=raw>0)
    ilook={b:i for i,b in enumerate(counts.index)};jlook={c:j for j,c in enumerate(counts.columns)}
    sw=cellweight[spoof['bin'].map(ilook).to_numpy(int),spoof['group'].map(jlook).to_numpy(int)]
    w=np.r_[np.full(len(bona),1/len(bona)),sw];matched=pd.concat([bona,spoof],ignore_index=True)
    ipf=[]
    for col in [c for c in f if c.startswith('score_')]:ipf.append(dict(model=col.removeprefix('score_'),eer=eer(matched.label,matched[col],w)))
    pd.DataFrame(ipf).to_csv(OUT/'duration_matched_attack_codec_preserved_eer.csv',index=False)
    pd.DataFrame(mass,index=counts.index,columns=counts.columns).to_csv(OUT/'ipf_metadata_cell_mass.csv')
    complete={'status':'completed_metadata_only_experiment','train_n':len(train),'train_sources_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'test_join_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'proxy_results':proxy,'limits':['No audio/model learning: metadata proxy shows available class-duration correlation, not that EBM uses it','Target data already inspected; fixed bins were chosen before proxy fitting, exploratory diagnostic','Observed versus common-support isolates removal of3079spoofs<=1sec; bona-marginal additionally reweights coarse durations','Class-duration reweighting can change attack/codec mixture; not proof of independent causal duration effect','B45/C30 were previously best target checkpoints; not newly selected here; other rows retain original epochs'],'spending_usd':0}
    complete['attack_codec_preserving_match']={'iterations':iteration+1,'max_margin_residual':float(residual),'spoof_effective_n':float(sw.sum()**2/(sw@sw)),'actual_spoof_n':len(spoof),'bona_n':len(bona),'eer':ipf,'limitations':'Preserves marginal attack x codec proportions, not joint distributions with content/speaker; does not isolate acoustic duration causally.'}
    (OUT/'duration_proxy_complete.json').write_text(json.dumps(complete,indent=2)+'\n')
    print(pd.DataFrame(proxy).to_string(index=False));print(pd.DataFrame(result).query("policy in ['observed','bona_marginal']").to_string(index=False))

if __name__=='__main__':main()
