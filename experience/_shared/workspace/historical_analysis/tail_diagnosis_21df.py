import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MPLCONFIGDIR']='/home/mentee/.cache/matplotlib'
from pathlib import Path
import json,csv,hashlib
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import welch
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold,cross_val_predict
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

assert Path('/.dockerenv').exists()
root=Path('/results/analysis'); out=root/'EBM80_21DF_tail_diagnosis'; out.mkdir(exist_ok=False)
base=root/'EBM80_21DF_centroid'
trials=json.loads((base/'sample_manifest.json').read_text())
d=np.load(base/'embeddings.npz'); z=d['embeddings']; labels=d['labels']; scores=d['cosine']
mc=np.load(root/'EBM80_21DF_multicrop/crop_scores.npz'); assert np.array_equal(labels,mc['labels'])
protocol=Path('/data/ASVspoof2021_DF/keys/DF/CM/trial_metadata.txt')
selected={t['id'] for t in trials if t['label']==1}
meta={}
for line in protocol.read_text().splitlines():
    f=line.split()
    if f[1] in selected:
        assert f[5]=='bonafide' and f[7]=='eval'
        meta[f[1]]={'speaker':f[0],'codec':f[2],'source':f[3],'trim':f[6]}
assert len(meta)==1000
rows=[]
for i,t in enumerate(trials):
    if t['label']!=1: continue
    audio,sr=sf.read(Path('/data/ASVspoof2021_DF/ASVspoof2021_DF_eval/flac')/(t['id']+'.wav'),dtype='float32')
    assert sr==16000 and audio.ndim==1 and len(audio)==mc['lengths'][i]
    # Acoustic summaries from original head samples, before wrap padding.
    a=audio[:64600].astype(np.float64); n=len(a)
    frame=np.pad(a,(0,(-n)%320)).reshape(-1,320)
    rms=np.sqrt(np.mean(frame**2,axis=1)+1e-16)
    freq,power=welch(a,fs=sr,nperseg=min(512,n)); total=power.sum()+1e-20
    r={'id':t['id'],**meta[t['id']],'score':float(scores[i]),'tail':int(scores[i]<.9),'severe':int(scores[i]<.2),
        'duration_s':len(audio)/sr,'padding_fraction':max(0,64600-len(audio))/64600,
        'rms_dbfs':float(20*np.log10(np.sqrt(np.mean(a*a)+1e-16))),
        'low_energy_fraction':float(np.mean(rms<rms.max()*.1)),
        'below_minus50dbfs_fraction':float(np.mean(rms<10**(-50/20))),
        'clipping_fraction':float(np.mean(np.abs(a)>=.999)),
        'zcr':float(np.mean(a[1:]*a[:-1]<0)),
        'hf_above4k_fraction':float(power[freq>=4000].sum()/total),
        'spectral_centroid_hz':float((freq*power).sum()/total),
        'head_short_score':float(mc['scores'][i,5]),
        'long_range':float(np.ptp(mc['scores'][i,:5])) if len(audio)>64600 else np.nan,
        'short_range':float(np.ptp(mc['scores'][i,5:10])) if len(audio)>48440 else np.nan}
    rows.append(r)
    if len(rows)%200==0: print(f'Acoustic descriptors {len(rows)}/1000',flush=True)
df=pd.DataFrame(rows); df.to_csv(out/'bonafide_sample_descriptors.csv',index=False)
df.sort_values('score').head(50).to_csv(out/'lowest50_for_server_review.csv',index=False)
def rates(frame,col,min_n=1):
    rr=[]
    for key,g in frame.groupby(col):
        n=len(g); k=int(g['tail'].sum()); p=k/n; zz=1.96; den=1+zz**2/n
        mid=(p+zz**2/(2*n))/den; rad=zz*np.sqrt(p*(1-p)/n+zz**2/(4*n*n))/den
        if n>=min_n: rr.append({'group':str(key),'n':n,'tail':k,'tail_pct':100*p,'severe':int(g['severe'].sum()),'median_score':float(g.score.median()),'wilson95_pct':[100*(mid-rad),100*(mid+rad)]})
    return sorted(rr,key=lambda a:a['tail_pct'],reverse=True)
numeric=['duration_s','padding_fraction','rms_dbfs','low_energy_fraction','below_minus50dbfs_fraction','clipping_fraction','zcr','hf_above4k_fraction','spectral_centroid_hz']
stats={}
for col in numeric:
    b=df.loc[df['tail']==0,col]; t=df.loc[df['tail']==1,col]
    rho=spearmanr(df[col],df['score']).statistic if df[col].nunique()>1 else np.nan
    stats[col]={'tail_median':float(t.median()),'core_median':float(b.median()),'tail_q25_q75':t.quantile([.25,.75]).tolist(),'core_q25_q75':b.quantile([.25,.75]).tolist(),'spearman_score':float(rho) if np.isfinite(rho) else None}
df['speaker_key']=df['source']+':'+df['speaker']
df['duration_group']=pd.cut(df.duration_s,[0,2,3.0275,4.0375,np.inf],labels=['<=2s','2-3.03s','3.03-4.04s','>4.04s']).astype(str)
df['source_codec']=df['source']+'/'+df['codec']
cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=1)
splits=list(cv.split(df,df['tail'],df['speaker_key']))
cv_results={}
sets={'source_only':(['source'],[]),'codec_only':(['codec'],[]),'source_codec':(['source','codec'],[]),'acoustic_only':([],numeric),'source_codec_acoustic':(['source','codec'],numeric)}
for name,(cats,nums) in sets.items():
    trans=[]
    if cats: trans.append(('cat',OneHotEncoder(handle_unknown='ignore'),cats))
    if nums: trans.append(('num',make_pipeline(SimpleImputer(strategy='median'),StandardScaler()),nums))
    model=make_pipeline(ColumnTransformer(trans),LogisticRegression(C=1,max_iter=1500))
    pred=cross_val_predict(model,df,df['tail'],cv=splits,method='predict_proba')[:,1]
    cv_results[name]={'speaker_grouped_oof_auc':float(roc_auc_score(df['tail'],pred))}
summary={'definition':{'tail':'bonafide cosine to learned OC center <0.9','severe':'cosine<0.2','not_error_rate':True},'n':len(df),'tail_n':int(df['tail'].sum()),'severe_n':int(df['severe'].sum()),
    'source':rates(df,'source'),'codec':rates(df,'codec'),'trim':rates(df,'trim'),'duration':rates(df,'duration_group'),
    'source_codec':rates(df,'source_codec'),'speakers_n_ge10':rates(df,'speaker_key',10),'numeric':stats,'cv':cv_results,
    'provenance':{'checkpoint':'BestCLS_80_21DF.pt','protocol_sha256':hashlib.sha256(protocol.read_bytes()).hexdigest(),'sample':'same random balanced21DF sample; 1000 bonafide only','acoustics':'first min(original_length,64600) samples before padding, 16kHz'},
    'limitations':['Associations on evaluation sample, not causes or final model-selection evidence.','Tail uses training margin 0.9, not calibrated detection threshold.','Low energy is a frame-RMS proxy, not a speech detector or MOS.','No human listening or MOS inference performed.','Source, speaker, codec, duration may be confounded.','Wilson intervals treat trials as independent; correlated source recordings can make them optimistic.','CV splits exclude shared speaker IDs but duplicate source-recording structure is not fully verified.']}
(out/'summary.json').write_text(json.dumps(summary,indent=2))
fig,axes=plt.subplots(2,2,figsize=(15,10),layout='constrained')
for ax,col,title in [(axes[0,0],'source','By source corpus'),(axes[0,1],'codec','By codec')]:
    rr=rates(df,col); xx=np.arange(len(rr)); p=np.array([r['tail_pct'] for r in rr]); ci=np.array([r['wilson95_pct'] for r in rr])
    ax.barh(xx,p,color='#2563A6',alpha=.8); ax.errorbar(p,xx,xerr=[p-ci[:,0],ci[:,1]-p],fmt='none',color='#333333',capsize=3)
    ax.set_yticks(xx,[r['group']+f" (n={r['n']})" for r in rr]); ax.invert_yaxis(); ax.set(xlabel='Bonafide below cosine 0.9 (%)',title=title,xlim=(0,max(35,ci[:,1].max()+8)))
    for j,r in enumerate(rr): ax.text(r['tail_pct']+.5,j,f"{r['tail']}/{r['n']}",va='center',fontsize=8)
for source,color in zip(sorted(df.source.unique()),['#2563A6','#D87527','#5D8C45']):
    g=df[df.source==source]; axes[1,0].scatter(g.duration_s,g.score,s=12,alpha=.4,color=color,label=source)
axes[1,0].axhline(.9,c='gray',linestyle='--'); axes[1,0].axvline(4.0375,c='gray',linestyle=':'); axes[1,0].set(title='Duration vs center score',xlabel='Original duration (s)',ylabel='Cosine to learned OC center'); axes[1,0].legend(fontsize=8)
counts=df.pivot_table(index='source',columns='codec',values='tail',aggfunc='size')
heat=df.pivot_table(index='source',columns='codec',values='tail',aggfunc='mean')*100
axes[1,1].imshow(heat.to_numpy(),cmap='YlOrRd',vmin=0,vmax=100,aspect='auto')
axes[1,1].set_xticks(range(len(heat.columns)),heat.columns,rotation=45,ha='right',fontsize=8); axes[1,1].set_yticks(range(len(heat.index)),heat.index)
for i in range(len(heat.index)):
    for j in range(len(heat.columns)):
        value=heat.iloc[i,j]; n=counts.iloc[i,j]
        if np.isfinite(value): axes[1,1].text(j,i,f'{value:.0f}%\nn={n:.0f}',ha='center',va='center',fontsize=8,color='black' if value<60 else 'white')
axes[1,1].set_title('Source x codec: tail percentage (cell counts shown)')
fig.suptitle('What forms the bonafide tail? | epoch80 | 1,000 sampled 21DF bonafide trials\nTail = cosine < 0.9 (training margin, not an error threshold); bars show descriptive Wilson intervals',fontsize=13)
fig.savefig(out/'tail_diagnosis.png',dpi=170); fig.savefig(out/'tail_diagnosis.pdf'); plt.close(fig)
print(json.dumps({k:v for k,v in summary.items() if k not in ['source_codec','speakers_n_ge10']},indent=2),flush=True)
