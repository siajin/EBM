"""Reuse audited derived scores; no audio, inference, or training."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

BASE = Path(__file__).resolve().parents[2]
OUT = BASE/'artifacts/repetition_shortcut_20261111'
CACHE = BASE/'artifacts/geometry_cache/local_analysis/B_final_analysis_20261108_v1'
BINS = [0, 16000, 24000, 32000, 40000, 48000, 56000, 64600, np.inf]

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p, x): Path(p).write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def eer(y, s, w=None):
    fpr,tpr,thr=roc_curve(y,s,sample_weight=w,drop_intermediate=False)
    fnr=1-tpr
    j=int(np.searchsorted(fpr-fnr,0)); j=max(1,min(j,len(fpr)-1))
    lo,hi=j-1,j
    d=(fpr[hi]-fnr[hi])-(fpr[lo]-fnr[lo])
    a=0 if d==0 else -(fpr[lo]-fnr[lo])/d
    return float(100*(fpr[lo]+a*(fpr[hi]-fpr[lo])))

def cluster_ci(g, field):
    # Equal utterance estimand; speaker cluster resampling preserves all codecs of a speaker.
    agg=g.groupby('speaker')[field].agg(['sum','count']).to_numpy()
    rng=np.random.default_rng(20261111); n=len(agg)
    ix=rng.integers(0,n,(3000,n)); total=agg[ix].sum(axis=1)
    return np.quantile(total[:,0]/total[:,1],[.025,.975]).tolist(),n

def main():
    OUT.mkdir(exist_ok=True)
    sources={}; distributions=[]; reweighted=[]
    for phase in ['dev','19LA','21LA','21DF']:
        path=CACHE/f'paired_scores_{phase}.parquet'; sources[str(path.relative_to(BASE))]=sha(path)
        f=pd.read_parquet(path);assert not f.uid.duplicated().any()
        f['bin']=pd.cut(f.original_samples,BINS,right=True,labels=False)
        for y,g in f.groupby('label'):
            distributions.append(dict(phase=phase,label=int(y),n=len(g),le2=int((g.original_samples<=32000).sum()),le2_percent=float(100*(g.original_samples<=32000).mean()),wrap_long_percent=float(100*(g.original_samples<64600).mean()),wrap_short_percent=float(100*(g.original_samples<48440).mean())))
        counts=pd.crosstab(f['bin'],f.label).reindex(columns=[0,1],fill_value=0)
        common=counts[(counts[0]>0)&(counts[1]>0)]
        targets={'overlap_min':common.min(axis=1),'bona_marginal':common[1], 'uniform_common':pd.Series(1.,index=common.index)}
        for model in ['legacy240','B']:
            reweighted.append(dict(phase=phase,model=model,policy='observed_native',eer=eer(f.label,f[f'score_{model}']),n=len(f),excluded=0))
            support=f[f.bin.isin(common.index)]
            reweighted.append(dict(phase=phase,model=model,policy='common_support_observed',eer=eer(support.label,support[f'score_{model}']),n=len(support),excluded=len(f)-len(support)))
            for name,target in targets.items():
                g=f[f.bin.isin(common.index)].copy();t=target/target.sum()
                lookup=np.zeros((len(BINS)-1,2))
                for b in common.index:
                    lookup[int(b),:]=t.loc[b]/counts.loc[b].to_numpy()
                w=lookup[g.bin.to_numpy(dtype=int),g.label.to_numpy(dtype=int)]
                assert abs(w[g.label==0].sum()-1)<1e-10 and abs(w[g.label==1].sum()-1)<1e-10
                reweighted.append(dict(phase=phase,model=model,policy=name,eer=eer(g.label,g[f'score_{model}'],w),n=len(g),excluded=len(f)-len(g)))
    pd.DataFrame(distributions).to_csv(OUT/'original_length_distribution.csv',index=False)
    pd.DataFrame(reweighted).to_csv(OUT/'native_length_reweighted_eer.csv',index=False)
    p=CACHE/'mechanism_2x2/intervention_scores.parquet';sources[str(p.relative_to(BASE))]=sha(p)
    f=pd.read_parquet(p);assert len(f)==1536 and not f.duplicated(['model','phase','uid']).any()
    assert ((f.score_native-f.score_wrap)-f.delta_score_native_minus_wrap).abs().max()<1e-7
    ops=json.loads((CACHE/'analysis/operating_points.json').read_text())
    rows=[]
    for (phase,model,y,binname),g in f.groupby(['phase','model','label','length_bin']):
        g=g.copy();g['wrap_minus_native']=g.score_wrap-g.score_native
        g['angle_wrap_minus_native']=g.angle_wrap-g.angle_native
        ci,clusters=cluster_ci(g,'wrap_minus_native')
        t=ops[model]['dev_threshold'];e0=(g.score_native<t) if y else (g.score_native>=t);e1=(g.score_wrap<t) if y else (g.score_wrap>=t)
        rows.append(dict(phase=phase,model=model,label=int(y),length_bin=binname,n=len(g),speakers=clusters,mean_native=float(g.score_native.mean()),mean_wrap=float(g.score_wrap.mean()),mean_delta=float(g.wrap_minus_native.mean()),ci_lo=ci[0],ci_hi=ci[1],mean_angle_delta=float(g.angle_wrap_minus_native.mean()),native_errors=int(e0.sum()),wrap_errors=int(e1.sum()),correct_to_error=int((~e0&e1).sum()),error_to_correct=int((e0&~e1).sum()),dev_native_threshold=t))
    pd.DataFrame(rows).to_csv(OUT/'cached_paired_interventions.csv',index=False)
    controls=f[f.length_bin=='over4.0375']
    # Label spelling is validated below if prior cache used another name.
    noop=f.groupby('length_bin').delta_score_native_minus_wrap.agg(lambda x:float(x.abs().max())).to_dict()
    save(OUT/'cpu_analysis_complete.json',dict(status='completed_reanalysis_existing_inference',sources=sources,paired_rows=len(f),length_bins=list(BINS[:-1]),long_noop_by_bin=noop,noise_note='cached paired probe noise=0; full saved native benchmark noise=1e-5; do not subtract them as controlled interventions',limits=['Reweighting changes class-conditional duration mixtures; not training and not a causal intervention on the model','Shared-support results exclude bins with no bona or no spoof; coarse-bin residual duration differences remain','Paired subset was metadata-selected, 32 per class x length x corpus; not population EER','Speaker-cluster intervals are exploratory individual95%, not multiplicity adjusted','Native-to-wrap changes input length and context as well as repetition; does not identify why the model learned its response']))
    print(pd.DataFrame(rows).query("phase=='21LA' and model=='legacy240'").to_string(index=False))
    print(pd.DataFrame(reweighted).query("phase=='21LA'").to_string(index=False))

if __name__=='__main__': main()
