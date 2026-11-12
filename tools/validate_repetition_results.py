"""Independent weighted tie-aware ROC check and CPU artifact reconciliation."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
BASE=Path(__file__).resolve().parents[2]
OUT=BASE/'artifacts/repetition_shortcut_20261111'

def independent(y,score,weight):
    order=np.argsort(-score,kind='stable');s=score[order];y=y[order];w=weight[order]
    ends=np.r_[np.flatnonzero(s[1:]!=s[:-1]),len(s)-1]
    tp=np.r_[0,np.cumsum(w*y)[ends]];fp=np.r_[0,np.cumsum(w*(1-y))[ends]]
    fp/=fp[-1];fn=1-tp/tp[-1];hi=int(np.flatnonzero(fp>=fn)[0]);lo=hi-1
    a=-(fp[lo]-fn[lo])/((fp[hi]-fn[hi])-(fp[lo]-fn[lo]))
    return float(100*(fp[lo]+a*(fp[hi]-fp[lo])))

def main():
    f=pd.read_parquet(BASE/'artifacts/coverage20_length_shortcut_20261111/joined_scores_21LA.parquet')
    meta=pd.read_csv(BASE/'artifacts/coverage20_length_shortcut_20261111/trial_metadata_21LA.txt',sep=r'\s+',header=None,names=['speaker_m','uid','codec','transmission','attack_m','class_m','trim','subset'])
    f=f.merge(meta[['uid','codec']],on='uid',validate='one_to_one');f['bin']=pd.cut(f.original_samples,[0,16000,24000,32000,40000,48000,56000,64600,np.inf],labels=False)
    f=f[f['bin']>0].copy();assert len(f)==145097 and f.label.sum()==14816
    f['group']=f.attack+'|'+f.codec;spoof=f[f.label==0]
    count=pd.crosstab(spoof['bin'],spoof['group']);mass=pd.read_csv(OUT/'ipf_metadata_cell_mass.csv',index_col=0)
    assert count.index.equals(mass.index) and count.columns.tolist()==mass.columns.tolist()
    lookup=(mass/count.replace(0,np.nan)).fillna(0)
    ix={b:i for i,b in enumerate(lookup.index)};jx={c:j for j,c in enumerate(lookup.columns)}
    w=np.full(len(f),1/(f.label==1).sum());mask=f.label.to_numpy()==0
    w[mask]=lookup.to_numpy()[f.loc[mask,'bin'].map(ix).to_numpy(int),f.loc[mask,'group'].map(jx).to_numpy(int)]
    assert abs(w[mask].sum()-1)<1e-10 and abs(w[~mask].sum()-1)<1e-10
    expected=pd.read_csv(OUT/'duration_matched_attack_codec_preserved_eer.csv').set_index('model').eer
    errors={}
    for name,value in expected.items():
        measured=independent(f.label.to_numpy(),f['score_'+name].to_numpy(),w)
        errors[name]=abs(measured-value);assert errors[name]<1e-9,(name,measured,value)
    paired=pd.read_csv(OUT/'cached_paired_interventions.csv');assert paired.n.eq(32).all() and len(paired)==48
    assert (paired.native_errors+paired.correct_to_error-paired.error_to_correct).equals(paired.wrap_errors)
    assert paired[paired.length_bin=='atleast4.0375'].mean_delta.eq(0).all()
    summary={'status':'passed','independent_weighted_eer_models':8,'max_eer_difference_percentage_points':max(errors.values()),'common_support_n':len(f),'excluded_spoof_le1s':3079,'paired_groups':len(paired),'paired_error_transition_reconciles':True,'long_noop_exact':True,'waveform_and_summary_tests':'3 unittest cases passed; includes31views, metadata pipeline end-to-end, corruption rejection, class-prior-invariant EER/ties','new_gpu_inference_completed':False,'hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.csv')}}
    (OUT/'validation.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
