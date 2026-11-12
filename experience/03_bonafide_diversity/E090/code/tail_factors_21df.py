"""Conditional diagnostics, using only server-resident derived data."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, SplineTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA

assert Path('/.dockerenv').exists()
root=Path('/results/analysis'); out=root/'EBM80_21DF_tail_factors_v2'
out.mkdir(exist_ok=False)
df=pd.read_csv(root/'EBM80_21DF_tail_diagnosis/bonafide_sample_descriptors.csv')
manifest=json.loads((root/'EBM80_21DF_centroid/sample_manifest.json').read_text())
assert [t['id'] for t in manifest if t['label']==1]==df.id.tolist()
required=['id','source','speaker','codec','score','tail','duration_s','rms_dbfs',
          'low_energy_fraction','below_minus50dbfs_fraction','zcr','hf_above4k_fraction','spectral_centroid_hz']
assert len(df)==1000 and df.id.nunique()==1000 and not df[required].isna().any().any()
d=np.load(root/'EBM80_21DF_tail_interventions/intervention_scores.npz')
b=d['labels']==1; score=d['scores'][b]; names=d['names'].tolist()
assert np.array_equal(d['sources'][b],df.source) and np.array_equal(d['speakers'][b],df.speaker)
assert np.max(np.abs(score[:,0]-df.score))<1e-3
df['speaker_key']=df.source+':'+df.speaker
df['log_duration']=np.log(df.duration_s)
df['duration_bin']=pd.cut(df.duration_s,[0,2,3.0275,4.0375,np.inf],labels=['<=2','2-3.03','3.03-4.04','>4.04'])
target=df['tail'].astype(int).to_numpy()
splits=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=17).split(df,target,df.speaker_key))
acoustic=['rms_dbfs','low_energy_fraction','below_minus50dbfs_fraction','zcr','hf_above4k_fraction','spectral_centroid_hz']
models={
 'duration_only':([],[],True),
 'source_only':(['source'],[],False),
 'source_duration':(['source'],[],True),
 'source_codec_duration':(['source','codec'],[],True),
 'duration_acoustic':([],acoustic,True),
 'source_codec_duration_acoustic':(['source','codec'],acoustic,True),
}
report={'tail_definition':'Bonafide cosine < 0.9; training margin, not decision error',
        'cv_speaker_folds':5,'cv':{},'source_duration':[], 'speaker_duration_adjusted':[], 'speaker_interventions':[]}
for name,(cat,num,length) in models.items():
    pre=[]
    if cat: pre.append(('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cat))
    if num: pre.append(('num',StandardScaler(),num))
    if length: pre.append(('len',make_pipeline(SplineTransformer(n_knots=4,degree=2),StandardScaler()),['log_duration']))
    pipe=make_pipeline(ColumnTransformer(pre),LogisticRegression(C=1,max_iter=1000,class_weight='balanced'))
    pred=np.zeros(len(df)); fold_auc=[]
    for tr,te in splits:
        assert set(df.speaker_key.iloc[tr]).isdisjoint(df.speaker_key.iloc[te])
        pipe.fit(df.iloc[tr],target[tr]); pred[te]=pipe.predict_proba(df.iloc[te])[:,1]
        fold_auc.append(float(roc_auc_score(target[te],pred[te])))
    report['cv'][name]={'pooled_oof_auc':float(roc_auc_score(target,pred)),'fold_auc':fold_auc}

st=np.load(root/'EBM80_21DF_stages/stage_embeddings.npz')
assert np.array_equal(st['labels'],d['labels'])
for stage in ['ssl','pool']:
    # Tail-prediction diagnostic only: representations belong to the same frozen model.
    x=st[stage][b]; pred=np.zeros(len(df)); fold_auc=[]
    for tr,te in splits:
        pipe=make_pipeline(PCA(n_components=16,svd_solver='full'),StandardScaler(),LogisticRegression(C=.1,max_iter=1000,class_weight='balanced'))
        pipe.fit(x[tr],target[tr]); pred[te]=pipe.predict_proba(x[te])[:,1]
        fold_auc.append(float(roc_auc_score(target[te],pred[te])))
    report['cv'][stage+'_pca16']={'pooled_oof_auc':float(roc_auc_score(target,pred)),'fold_auc':fold_auc}

for (src,dur),g in df.groupby(['source','duration_bin'],observed=True):
    report['source_duration'].append({'source':src,'duration':str(dur),'n':len(g),'tail_n':int(g['tail'].sum())})

for (src,spk),g in df.groupby(['source','speaker']):
    if len(g)<10: continue
    rest=df[(df.source==src)&(df.speaker!=spk)]
    if len(rest)==0: continue
    # Descriptive standardization. No causal effect or p-value claimed.
    expected=0.; bins=[]
    for dur,h in g.groupby('duration_bin',observed=True):
        ref=rest[rest.duration_bin==dur]
        if len(ref)==0: continue
        rate=float(ref['tail'].mean()); expected+=len(h)*rate
        bins.append({'duration':str(dur),'n':len(h),'tail_n':int(h['tail'].sum()),'reference_n':len(ref),'reference_tail_pct':100*rate})
    report['speaker_duration_adjusted'].append({'source':src,'speaker':spk,'n':len(g),'tail_n':int(g['tail'].sum()),'expected_tail_n_if_other_speakers_same_duration_bins':expected,'duration_bins':bins})
    ix=g.index.to_numpy()
    report['speaker_interventions'].append({'source':src,'speaker':spk,'n':len(g),'methods':{name:{'tail_n':int((score[ix,j]<.9).sum()),'median_score':float(np.median(score[ix,j]))} for j,name in enumerate(names)}})

# Confirm that padding-only modifications leave long clips unchanged.
not_padded=d['lengths']>=64600
report['validation']={'rows':len(df),'unique_ids':int(df.id.nunique()),'long_clip_n':int(not_padded.sum()),
 'zero_pad_long_max_difference':float(np.max(np.abs(d['scores'][not_padded,2]-d['scores'][not_padded,0]))),
 'reflect_pad_long_max_difference':float(np.max(np.abs(d['scores'][not_padded,3]-d['scores'][not_padded,0])))}
assert report['validation']['zero_pad_long_max_difference']<1e-3
assert report['validation']['reflect_pad_long_max_difference']<1e-3
report['limitations']=['Tail prediction AUC is not detection EER or evidence that a routing rule improves detection.',
 'Speaker-held-out folds help assess new-speaker transfer, but same-source and recording-family dependencies remain.',
 'Source and speaker associations may reflect voice, language, content, recording setup, or preprocessing; causal identity is not isolated.',
 'Duration-adjusted speaker rates use coarse bins and do not adjust every confounder.',
 'Pre-projection representations are from the frozen model being diagnosed; no independent training or validation performed.']
(out/'summary.json').write_text(json.dumps(report,indent=2))
print(json.dumps({'cv':report['cv'],'validation':report['validation'],
 'top_speakers':sorted(report['speaker_duration_adjusted'],key=lambda r:r['tail_n']/r['n'],reverse=True)[:5],
 'top_speaker_interventions':[x for x in report['speaker_interventions'] if x['speaker']=='VCC2SM3']},indent=2),flush=True)
