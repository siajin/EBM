"""Low-dimensional cue probes; train on raw 19LA train, evaluate held-out data."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'; os.environ['OMP_NUM_THREADS']='1'
os.environ['MPLCONFIGDIR']='/home/mentee/.cache/matplotlib'
from pathlib import Path
import json, random, time, hashlib
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import welch, fftconvolve
from scipy.stats import spearmanr
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

assert Path('/.dockerenv').exists()
root=Path('/results/analysis'); out=root/'EBM80_21DF_cue_audit'; out.mkdir(exist_ok=False)
data19=Path('/data/ASVspoof2019/LA'); data21=Path('/data/ASVspoof2021_DF')
protocols=[]
def trials19(split,n=1000):
    end='train.trn' if split=='train' else 'dev.trl'
    path=data19/f'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{end}.txt'; protocols.append(path)
    groups={0:[],1:[]}
    for line in path.read_text().splitlines():
        f=line.split(); y=int(f[4]=='bonafide')
        audio=data19/f'ASVspoof2019_LA_{split}/flac/{f[1]}.flac'
        if not audio.exists():audio=audio.with_suffix('.wav')
        assert audio.is_file()
        groups[y].append({'id':f[1],'speaker':f[0],'attack':f[3],'label':y,'source':'asvspoof','split':split,'path':str(audio)})
    rng=random.Random(17 if split=='train' else 18)
    return [t for y in (1,0) for t in rng.sample(groups[y],n)]
def trials21():
    p=data21/'keys/DF/CM/trial_metadata.txt'; protocols.append(p)
    meta={}
    for line in p.read_text().splitlines():
        f=line.split(); meta[f[1]]={'speaker':f[0],'source':f[3],'codec':f[2],'attack':f[4],'label':int(f[5]=='bonafide')}
    manifest=json.loads((root/'EBM80_21DF_centroid/sample_manifest.json').read_text())
    return [dict(meta[t['id']],id=t['id'],split='21df',path=str(data21/f"ASVspoof2021_DF_eval/flac/{t['id']}.wav")) for t in manifest]

def features(path):
    full,sr=sf.read(path,dtype='float32'); assert sr==16000 and full.ndim==1 and len(full)>512
    length=len(full); raw=full[:64600].astype(np.float64)
    a=np.pad(raw,(0,64600-len(raw)),mode='wrap')
    # All signal features see the actual fixed-length evaluation-style input.
    frames=np.pad(a,(0,(-len(a))%320)).reshape(-1,320)
    rms=np.sqrt(np.mean(frames**2,axis=1)+1e-16)
    active=np.flatnonzero(rms>=rms.max()*.1)
    left=int(active[0])*320; right=min(len(a),(int(active[-1])+1)*320)
    overall=np.sqrt(np.mean(a*a)+1e-16); peak=np.max(np.abs(a))+1e-12
    freq,power=welch(a,fs=sr,nperseg=512); power=power+1e-20
    ac=fftconvolve(a,a[::-1],mode='full')[len(a)-1:]
    energy=np.concatenate([[0.],np.cumsum(a*a)])
    lags=np.arange(1600,len(a)-8000)
    corr=ac[lags]/np.sqrt(np.maximum((energy[len(a)-lags])*(energy[-1]-energy[lags]),1e-20))
    best=int(np.argmax(corr))
    return {
      'log_duration':float(np.log(length/sr)), 'padding_fraction':float(max(0,64600-length)/64600),
      'low_energy_fraction':float((rms<rms.max()*.1).mean()),'leading_s':left/sr,'trailing_s':(len(a)-right)/sr,
      'rms_dbfs':float(20*np.log10(overall)), 'peak_dbfs':float(20*np.log10(peak)),
      'low_rms_q10_dbfs':float(20*np.log10(np.quantile(rms,.1))),
      'zero_fraction':float((np.abs(a)<1e-8).mean()),'dc_over_rms':float(abs(a.mean())/overall),
      'crest_db':float(20*np.log10(peak/overall)),
      'zcr':float(np.mean(np.signbit(a[1:])!=np.signbit(a[:-1]))),
      'hf_above4k_fraction':float(power[freq>4000].sum()/power.sum()),
      'spectral_centroid_hz':float((freq*power).sum()/power.sum()),
      'spectral_flatness':float(np.exp(np.log(power[1:]).mean())/power[1:].mean()),
      'step_q99_over_rms':float(np.quantile(np.abs(np.diff(a)),.99)/overall),
      'repeat_peak_corr':float(corr[best]), 'repeat_peak_lag_s':float(lags[best]/sr),
      'wrap_seam_over_rms':float(abs(raw[-1]-raw[0])/overall) if length<64600 else 0.,
    }

trials=trials19('train')+trials19('dev')+trials21()
assert len(trials)==6000 and len({t['id'] for t in trials})==6000
speakers={split:{t['speaker'] for t in trials if t['split']==split} for split in ['train','dev','21df']}
assert not (speakers['train']&speakers['dev']) and not (speakers['train']&speakers['21df'])
(out/'sample_manifest.json').write_text(json.dumps(trials,indent=2))
started=time.time(); rows=[]
for i,t in enumerate(trials):
    row={k:v for k,v in t.items() if k!='path'}; row.update(features(t['path'])); rows.append(row)
    if i%500==0:print(f'features {i}/6000, {time.time()-started:.0f}s',flush=True)
df=pd.DataFrame(rows); feature_cols=list(features(trials[0]['path']))
assert np.isfinite(df[feature_cols].to_numpy()).all()
df.to_csv(out/'cue_features_private.csv',index=False)
train=df[df.split=='train'].copy(); dev=df[df.split=='dev'].copy(); test=df[df.split=='21df'].copy()
length=['log_duration','padding_fraction','wrap_seam_over_rms']
silence=['low_energy_fraction','leading_s','trailing_s']
energy=['rms_dbfs','peak_dbfs','low_rms_q10_dbfs','zero_fraction','dc_over_rms','crest_db']
spectral=['zcr','hf_above4k_fraction','spectral_centroid_hz','spectral_flatness']
boundary=['step_q99_over_rms','repeat_peak_corr','repeat_peak_lag_s']
families={'length_and_pipeline':length,'silence_timing':silence,'energy_floor':energy,'spectral_shape':spectral,
          'repetition_and_steps':boundary,'silence_and_energy':silence+energy,'all_audio':silence+energy+spectral+boundary,
          'all_with_pipeline':feature_cols}
summary={'sample_counts':df.groupby(['split','label']).size().unstack().to_dict(orient='index'),
         'speaker_overlap_train_dev':0,'speaker_overlap_train_21df':0,'models':{},'univariate':{},'feature_summary':{},'bonafide_score_correlation':{}}
def eer(s,y):
    fpr,tpr,th=roc_curve(y,s,drop_intermediate=False); diff=fpr-(1-tpr); i=np.flatnonzero(diff>=0)[0]
    if i==0:return float(fpr[0]*100),float(th[0])
    w=-diff[i-1]/(diff[i]-diff[i-1]); return float(100*(fpr[i-1]+w*(fpr[i]-fpr[i-1]))),float(th[i])
def metrics(y,p,threshold):
    e,_=eer(p,y)
    return {'n':len(y),'bona_n':int(y.sum()),'spoof_n':int((1-y).sum()),'auc':float(roc_auc_score(y,p)),
            'sample_eer_pct':e,'bona_reject_at_dev_threshold_pct':float(100*np.mean(p[y==1]<threshold)),
            'spoof_accept_at_dev_threshold_pct':float(100*np.mean(p[y==0]>=threshold))}
predictions={}
for name,cols in list(families.items())+[('all_audio_tree',families['all_audio'])]:
    if name.endswith('_tree'):model=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=7,max_depth=3,min_samples_leaf=30,learning_rate=.05,l2_regularization=1,random_state=17)
    else:model=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000))
    model.fit(train[cols],train.label)
    pdev=model.predict_proba(dev[cols])[:,1]; _,th=eer(pdev,dev.label.to_numpy())
    ptest=model.predict_proba(test[cols])[:,1]
    predictions[name]=ptest
    entry={'features':cols,'dev_threshold':th,'train_resubstitution':metrics(train.label.to_numpy(),model.predict_proba(train[cols])[:,1],th),
           '19LA_dev':metrics(dev.label.to_numpy(),pdev,th),'21DF_all':metrics(test.label.to_numpy(),ptest,th)}
    for src in sorted(test.source.unique()):
        m=test.source.to_numpy()==src; entry['21DF_'+src]=metrics(test.label.to_numpy()[m],ptest[m],th)
    if not name.endswith('_tree'):entry['standardized_logistic_coefficients']=dict(zip(cols,model[-1].coef_[0].tolist()))
    summary['models'][name]=entry
for col in feature_cols:
    trauc=roc_auc_score(train.label,train[col]); sign=1 if trauc>=.5 else -1
    summary['univariate'][col]={'direction_from_train':sign,'train_auc':float(max(trauc,1-trauc)),
       '19LA_dev_auc':float(roc_auc_score(dev.label,sign*dev[col])), '21DF_auc':float(roc_auc_score(test.label,sign*test[col]))}
    for src in sorted(test.source.unique()):
        g=test[test.source==src]; summary['univariate'][col]['21DF_'+src+'_auc']=float(roc_auc_score(g.label,sign*g[col]))

old=np.load(root/'EBM80_21DF_centroid/embeddings.npz')['cosine']
assert test.id.tolist()==[t['id'] for t in json.loads((root/'EBM80_21DF_centroid/sample_manifest.json').read_text())]
test=test.copy(); test['ebm_cosine']=old
for key,g in [('train',train),('19LA_dev',dev)]+[('21DF_'+src,test[test.source==src]) for src in sorted(test.source.unique())]:
    summary['feature_summary'][key]={str(y):{col:float(v[col].median()) for col in feature_cols} for y,v in g.groupby('label')}
for key,g in [('all',test)]+[(src,test[test.source==src]) for src in sorted(test.source.unique())]:
    g=g[g.label==1]
    summary['bonafide_score_correlation'][key]={col:float(spearmanr(g[col],g.ebm_cosine).statistic) if g[col].nunique()>1 else None for col in feature_cols}

np.savez_compressed(out/'cue_predictions.npz',**predictions,labels=test.label.to_numpy(),sources=test.source.to_numpy(),ebm_cosine=old)
summary['provenance']={'seed_train':17,'seed_dev':18,'per_class_per_split':1000,'features_input':'head 64600, wrap if short; raw audio before training augmentations',
 'protocol_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in protocols},
 'probe_selection':'Feature families and model settings fixed before seeing test results; no 21DF tuning.'}
summary['limitations']=['Small cue probes measure availability of label information, not proof that EBM uses each cue.',
 'Train audio is raw before codec/RawBoost augmentation; these probes do not reconstruct the full training stream.',
 '19LA dev uses unseen speakers but related attack families; 21DF has different source and attack mixtures.',
 'Low energy proxy is relative 20 ms frame RMS, not VAD or MOS.',
 'Pipeline family includes original duration and known wrap boundary, which EBM does not receive explicitly.',
 'Spectral shape can contain both genuine spoof artifacts and recording/channel cues; predictive power alone cannot distinguish them.',
 'EERs are exploratory on sampled trials, not official full-set EER; no multi-centroid model was trained.']
(out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
names=list(summary['models']); conditions=['19LA_dev','21DF_all','21DF_asvspoof','21DF_vcc2018','21DF_vcc2020']
matrix=np.array([[summary['models'][n][c]['auc'] for c in conditions] for n in names])
fig,ax=plt.subplots(figsize=(11,6),layout='constrained'); im=ax.imshow(matrix,vmin=0,vmax=1,cmap='RdYlBu')
ax.set_xticks(range(len(conditions)),['19LA dev','21DF all','21DF ASVspoof','21DF VCC2018','21DF VCC2020'],rotation=20,ha='right')
ax.set_yticks(range(len(names)),names)
for i in range(len(names)):
    for j in range(len(conditions)):ax.text(j,i,f'{matrix[i,j]:.3f}',ha='center',va='center',color='white' if matrix[i,j]>.83 or matrix[i,j]<.17 else 'black')
fig.colorbar(im,ax=ax,label='ROC AUC (0.5 = chance ranking)')
ax.set_title('Can simple cues predict the label?\nFit on 19LA train only; all displayed columns are held-out evaluation',pad=14)
fig.savefig(out/'cue_transfer.png',dpi=170); fig.savefig(out/'cue_transfer.pdf'); plt.close(fig)
print(json.dumps({'models':{k:{c:{m:v[c][m] for m in ['auc','sample_eer_pct']} for c in conditions} for k,v in summary['models'].items()},
 'univariate':summary['univariate'],'correlation':summary['bonafide_score_correlation']},indent=2),flush=True)
