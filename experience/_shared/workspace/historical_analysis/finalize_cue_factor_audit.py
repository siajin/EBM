import os
os.environ['MPLCONFIGDIR']='/home/mentee/.cache/matplotlib'
from pathlib import Path
import json,hashlib,numpy as np
from sklearn.metrics import roc_curve,roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
assert Path('/.dockerenv').exists()
root=Path('/results/analysis/EBM80_cue_factor_audit');out=root/'analysis';r=json.loads((out/'summary.json').read_text());tr=json.loads((root/'sample_manifest.json').read_text());y=np.array([t['label'] for t in tr]);src=np.array([t['source'] for t in tr]);sp=np.array([t['speaker'] for t in tr]);th=r['threshold'];eps=r['direction_epsilon']
def merge(path,fields):
 ss=[np.load(path/f'shard_{i}.npz') for i in range(2)];ix=np.concatenate([a['indices'] for a in ss]);assert np.array_equal(np.sort(ix),np.arange(2000));order=np.argsort(ix)
 return {k:np.concatenate([a[k] for a in ss])[order] for k in fields}
assert json.loads((root/'gain_followup/COMPLETED.json').read_text())['n']==2000
g=merge(root/'gain_followup',['scores'])['scores'].astype(float);s=merge(root,['scores'])['scores'].astype(float)
assert np.max(abs(g[:,0]-s[:,0]))<1e-3 and np.max(abs(g[:,3]-s[:,11]))<1e-3
def stat(a,m):
 v=a[m];return dict(n=len(v),mean=float(v.mean()),median=float(np.median(v)),mean_abs=float(abs(v).mean()),toward_n=int((v>eps).sum()),away_n=int((v<-eps).sum()),abs_ge001_n=int((abs(v)>=.01).sum()))
def metrics(v):
 fpr,tpr,_=roc_curve(y,v,drop_intermediate=False);d=fpr-(1-tpr);i=np.flatnonzero(d>=0)[0];eer=fpr[0] if i==0 else fpr[i-1]+(-d[i-1]/(d[i]-d[i-1]))*(fpr[i]-fpr[i-1])
 return dict(sample_eer_pct=float(100*eer),auc=float(roc_auc_score(y,v)),bona_reject_n=int(((y==1)&(v<th)).sum()),spoof_accept_n=int(((y==0)&(v>=th)).sum()),bona_mean_score=float(v[y==1].mean()),spoof_mean_score=float(v[y==0].mean()),mean_class_gap=float(v[y==1].mean()-v[y==0].mean()))
gain={'scope':'Exploratory attenuation dose follow-up; same2000 files; no resampling/cropping/quantization. No new confirmatory tests.',
 'checks':{'baseline_reproduction_max':float(np.max(abs(g[:,0]-s[:,0]))),'gain01_reproduction_max':float(np.max(abs(g[:,3]-s[:,11])))},'conditions':[]}
for j,scale in enumerate([1.,.5,.25,.1]):
 a={'gain':scale,'db':float(20*np.log10(scale)),'metrics':metrics(g[:,j]),'effects':{word:stat(g[:,j]-g[:,0],y==lab) for word,lab in [('bonafide',1),('spoof',0)]}}
 a['by_source']={str(so):{word:stat(g[:,j]-g[:,0],(src==so)&(y==lab)) for word,lab in [('bonafide',1),('spoof',0)]} for so in np.unique(src)}
 gain['conditions'].append(a)
(out/'gain_summary.json').write_text(json.dumps(gain,indent=2,allow_nan=False));r['gain_followup']=gain
# Repair terminology without changing any numerical estimates.
for en in r['effects'].values():
 for a in en['labels'].values():
  for old,new in [('modified_roi_n','selected_roi_n'),('modified_roi_mean_s','selected_roi_mean_s'),('affected_files_only','selected_roi_files_only')]:
   if old in a:a[new]=a.pop(old)
e=merge(Path('/results/analysis/EBM80_native_quiet_strict/evaluation_view'),['scores','eligible']);es=e['scores'].astype(float);nn=np.load('/results/analysis/EBM80_native_quiet_strict/analysis/merged_private.npz');ns=nn['scores'][nn['cohort']=='new_files'].astype(float)
selected=['native_vs_fixed','added_zero_position','added_texture_front','zero_both','shuffle_both','phase_both','quiet_gain_01','quiet_gain_10','global_gain_01']
vectors=[ns[:,0]-s[:,0],es[:,2]-es[:,3],es[:,4]-es[:,2],s[:,4]-s[:,0],s[:,7]-s[:,0],s[:,8]-s[:,0],s[:,9]-s[:,0],s[:,10]-s[:,0],s[:,11]-s[:,0]]
r['common_eligible_comparison']={name:{word:stat(v,(y==lab)&e['eligible']) for word,lab in [('bonafide',1),('spoof',0)]} for name,v in zip(selected,vectors)}
r['interpretation_repairs']=['ROI counts describe selected regions, not guaranteed nonzero waveform modifications.','Primary comparison figure now uses the same own-eligible files for all contrasts: B931/S887. All-file condition results remain separately reported.','20ms phase randomization preserves frame-local power only; no global-spectrum or boundary invariance claim.']
(out/'summary.json').write_text(json.dumps(r,indent=2,allow_nan=False))
short=['Native vs crop/wrap','Front vs back (+1s zero)','Own vs zero (+1s front)','Zero quiet edges','Shuffle quiet edges','20ms phase randomization','Quiet gain x0.1','Quiet gain x10','Whole waveform x0.1']
fig,ax=plt.subplots(1,2,figsize=(13,6.4),layout='constrained')
for k,word in enumerate(['bonafide','spoof']):
 xx=np.arange(len(selected));vals=[r['common_eligible_comparison'][n][word] for n in selected]
 ax[0].barh(xx+(-.17 if k==0 else .17),[v['mean_abs'] for v in vals],height=.32,label=word,color=['#317BA9','#CB8038'][k]);ax[1].barh(xx+(-.17 if k==0 else .17),[100*v['abs_ge001_n']/v['n'] for v in vals],height=.32,label=word,color=['#317BA9','#CB8038'][k])
for a in ax:a.set_yticks(range(len(selected)),short);a.invert_yaxis();a.grid(axis='x',alpha=.15);a.legend()
ax[0].set(xlabel='Mean absolute change in cosine',title='Magnitude of paired score change');ax[1].set(xlabel='Files with |change| >= 0.01 (%)',title='Fraction changing by at least 0.01')
fig.suptitle('Same eligible files for all contrasts: B931 / S887 | frozen epoch80\nSensitivity at these intervention strengths; not universal causal importance shares',fontsize=12)
fig.savefig(out/'factor_effects.png',dpi=170);plt.close(fig)
fig,ax=plt.subplots(1,2,figsize=(10,4.2),layout='constrained');x=[-a['db'] for a in gain['conditions']]
for word,color in [('bonafide','#317BA9'),('spoof','#CB8038')]:ax[0].plot(x,[a['effects'][word]['mean'] for a in gain['conditions']],marker='o',label=word,color=color)
ax[0].axhline(0,c='gray',lw=1);ax[0].set(xlabel='Whole-waveform attenuation (dB)',ylabel='Mean cosine change',title='Both classes move toward reduced separation');ax[0].legend()
ax[1].plot(x,[a['metrics']['sample_eer_pct'] for a in gain['conditions']],marker='o',color='#76579C');ax[1].set(xlabel='Whole-waveform attenuation (dB)',ylabel='Sample EER (%)',title='Descriptive ranking effect on the same2000files')
for a in ax:a.grid(alpha=.2)
fig.savefig(out/'gain_response.png',dpi=170);plt.close(fig)
print(json.dumps(gain,indent=2),flush=True)
