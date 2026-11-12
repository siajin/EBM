import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2';os.environ['MPLCONFIGDIR']='/home/mentee/.cache/matplotlib'
from pathlib import Path
import json,hashlib
import numpy as np
from sklearn.metrics import roc_curve,roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
assert Path('/.dockerenv').exists()
root=Path('/results/analysis');out=root/'EBM80_cue_factor_audit';dest=out/'analysis';dest.mkdir(exist_ok=False)
assert json.loads((out/'COMPLETED.json').read_text())['n']==2000
tr=json.loads((out/'sample_manifest.json').read_text());y=np.array([t['label'] for t in tr]);src=np.array([t['source'] for t in tr]);sp=np.array([t['speaker'] for t in tr]);length=np.array([t['samples'] for t in tr])
def merge(path,fields):
 ss=[np.load(path/f'shard_{i}.npz') for i in range(2)];ix=np.concatenate([q['indices'] for q in ss]);assert np.array_equal(np.sort(ix),np.arange(2000));order=np.argsort(ix)
 return {k:np.concatenate([q[k] for q in ss])[order] for k in fields},ss[0]['names'].tolist()
d,names=merge(out,['scores','features','checks']);s=d['scores'].astype(float)
e,_=merge(root/'EBM80_native_quiet_strict/evaluation_view',['scores','eligible']);es=e['scores'].astype(float)
nd=np.load(root/'EBM80_native_quiet_strict/analysis/merged_private.npz');nm=nd['cohort']=='new_files';ns=nd['scores'][nm].astype(float);ne=nd['eligible'][nm]
nt=[t for t in json.loads((root/'EBM80_native_quiet_strict/sample_manifest.json').read_text()) if t['cohort']=='new_files'];assert tr==nt
assert np.isfinite(s).all() and np.max(abs(s[:,0]-es[:,0]))<1e-3 and np.max(d['checks'][:,:2])<1e-5
assert np.max(d['checks'][:,2])<1e-8 and np.all(d['checks'][:,3]==1)
threshold_info=json.loads((root/'EBM80_21DF_leading_quiet_paired/summary.json').read_text());th=threshold_info['threshold']
eps=max(.000121,10*float(np.max(abs(s[:,1]-s[:,0]))));rng=np.random.default_rng(30453);R=10000
def metrics(v,m):
 yy=y[m];v=v[m];fpr,tpr,_=roc_curve(yy,v,drop_intermediate=False);q=fpr-(1-tpr);i=np.flatnonzero(q>=0)[0]
 eer=fpr[0] if i==0 else fpr[i-1]+(-q[i-1]/(q[i]-q[i-1]))*(fpr[i]-fpr[i-1])
 b=yy==1;err=(v>=th)!=b
 return dict(n=len(v),bona_n=int(b.sum()),spoof_n=int((~b).sum()),sample_eer_pct=float(100*eer),auc=float(roc_auc_score(yy,v)),
   bona_reject_n=int(err[b].sum()),spoof_accept_n=int(err[~b].sum()),bona_reject_pct=float(100*err[b].mean()),spoof_accept_pct=float(100*err[~b].mean()),balanced_error_pct=float(50*(err[b].mean()+err[~b].mean())))
def stat(v,m):
 a=v[m];return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),mean_abs=float(abs(a).mean()),p90_abs=float(np.quantile(abs(a),.9)),
    toward_n=int((a>eps).sum()),away_n=int((a<-eps).sum()),small_n=int((abs(a)<=eps).sum()),abs_ge001_n=int((abs(a)>=.01).sum()),abs_ge01_n=int((abs(a)>=.1).sum()))
def ci(v,m):
 b=np.zeros(R);ix=np.flatnonzero(m);clusters={}
 for ss in np.unique(src[m]):
  ids=ix[src[ix]==ss];gg=[ids[sp[ids]==x] for x in np.unique(sp[ids])];counts=np.array([len(g) for g in gg]);sums=np.array([v[g].sum() for g in gg]);w=rng.multinomial(len(gg),np.full(len(gg),1/len(gg)),size=R)
  b+=(w@sums)/(w@counts)*(len(ids)/len(ix));clusters[str(ss)]=len(gg)
 return dict(mean_cluster_ci95=np.quantile(b,[.025,.975]).tolist(),clusters=clusters)
contrasts=[]
def add(name,a,b,mask=None,scope='fixed-input edge intervention'):
 contrasts.append((name,a,b,np.ones(2000,dtype=bool) if mask is None else mask,scope))
for j,name in enumerate(names[2:],2):add(name,s[:,j],s[:,0])
add('native_vs_fixed',ns[:,0],s[:,0],scope='Composite preprocessing: remove wrap/crop AND change total input length')
add('native_vs_fixed_short',ns[:,0],s[:,0],length<64600,'Remove waveform wrap and shorten input jointly')
add('native_vs_fixed_long',ns[:,0],s[:,0],length>=64600,'Restore previously cropped right content and extend input jointly')
for name,a,b,m in [('added_zero_front',2,0,None),('added_zero_back',3,0,None),('added_zero_position',2,3,None),('added_own_front',4,0,e['eligible']),('added_own_back',5,0,e['eligible']),('added_own_position',4,5,e['eligible']),('added_texture_front',4,2,e['eligible']),('added_texture_back',5,3,e['eligible'])]:add(name,es[:,a],es[:,b],m,'Added1s, conventional original input preserved; own/zero matched eligibility')
add('shuffle_vs_phase',s[:,7],s[:,8],scope='Contrast of two transforms, not isolated causal spectral coefficient')
res={'plan':json.loads((out/'plan.json').read_text()),'threshold':th,'threshold_source':threshold_info['threshold_source'],'direction_epsilon':eps,
 'checks':{'baseline_reproduction_max':float(np.max(abs(s[:,0]-es[:,0]))),'identity_repeat_max':float(np.max(abs(s[:,1]-s[:,0]))),'phase_energy_relative_max':float(d['checks'][:,0].max()),'phase_power_relative_max':float(d['checks'][:,1].max()),'phase_mean_error_max':float(d['checks'][:,2].max()),'preservation_assertions_all_true':True},
 'performance':{},'effects':{},'source_effects':{},'lead_tail_interaction':{},'factor_decomposition':{},'quiet_statistics':{},'limitations':[
 'Follow-up same corpus/model and already analyzed new-file cohort. All intervals descriptive95% without multiplicity control; no confirmatory significance claims.',
 'Effects are tied to these specific interventions; they are not universal percentages of model decision responsibility.',
 'Threshold is fixed from an existing19LAdev sample;21DF error counts are a transferred operating-point stress test, not official performance.',
 'Quiet proxy is relative -40dB, not certified nonspeech. Genuine synthesis/channel traces may be present; sensitivity does not prove spuriousness.',
 'Per-frame phase preserves frame power spectra, energy and mean, but changes phases and cross-frame boundary context.',
 'Native-vs-fixed jointly changes content repetition/cropping and input length. Front-back also changes joins and positional context.',
 'Speaker-cluster percentile intervals approximate; some source-label groups contain only4speakerclusters.',
 'No retraining, external-domain verification or causal proof of historical learning origin.']}
for j,name in enumerate(names):res['performance'][name]=metrics(s[:,j],np.ones(2000,dtype=bool))
res['performance']['native']=metrics(ns[:,0],np.ones(2000,dtype=bool))
for name,j in [('add_zero_front',2),('add_zero_back',3)]:res['performance'][name]=metrics(es[:,j],np.ones(2000,dtype=bool))
for word,lab in [('bonafide',1),('spoof',0)]:
 m=y==lab;ff=d['features'][m];res['quiet_statistics'][word]=dict(n=int(m.sum()),lead_affected=int((ff[:,0]>0).sum()),tail_affected=int((ff[:,1]>0).sum()),either_affected=int((ff[:,4]>0).sum()),lead_median_s=float(np.median(ff[:,0])),tail_median_s=float(np.median(ff[:,1])),lead_mean_s=float(np.mean(ff[:,0])),tail_mean_s=float(np.mean(ff[:,1])),median_fraction=float(np.median(ff[:,4])),mean_fraction=float(np.mean(ff[:,4])),mean_quiet_energy_over_total_energy=float(np.mean(ff[:,3]*ff[:,4]/ff[:,2])))
for name,a,b,m,scope in contrasts:
 v=a-b;entry={'scope':scope,'labels':{}};res['source_effects'][name]={}
 for word,lab in [('bonafide',1),('spoof',0)]:
  mm=m&(y==lab);z=stat(v,mm);z.update(ci(v,mm));before=(b[mm]>=th);after=(a[mm]>=th);truth=y[mm]==1
  z.update(prediction_flips_n=int((before!=after).sum()),new_error_n=int(((before==truth)&(after!=truth)).sum()),corrected_error_n=int(((before!=truth)&(after==truth)).sum()))
  z['robust_prediction_flips_n']=int(((before!=after)&(abs(a[mm]-th)>eps)&(abs(b[mm]-th)>eps)).sum())
  if name in names[2:11]:
   duration=d['features'][:,0] if name.endswith('_lead') else (d['features'][:,1] if name.endswith('_tail') else d['features'][:,0]+d['features'][:,1])
   affected=mm&(duration>0);unaffected=mm&(duration==0)
   z['modified_roi_n']=int(affected.sum());z['modified_roi_mean_s']=float(duration[mm].mean())
   if affected.any():z['affected_files_only']=stat(v,affected)
   if unaffected.any():
    z['unmodified_score_max_delta']=float(abs(v[unaffected]).max());assert z['unmodified_score_max_delta']<1e-3
  entry['labels'][word]=z
  res['source_effects'][name][word]={str(so):stat(v,mm&(src==so)) for so in np.unique(src[mm])}
 res['effects'][name]=entry
for family,l,t,b in [('zero',2,3,4),('shuffle',5,6,7)]:
 inter=s[:,b]-s[:,l]-s[:,t]+s[:,0];lead=.5*((s[:,l]-s[:,0])+(s[:,b]-s[:,t]));tail=.5*((s[:,t]-s[:,0])+(s[:,b]-s[:,l]));assert np.max(abs(lead+tail-(s[:,b]-s[:,0])))<1e-12
 res['lead_tail_interaction'][family]={word:{'interaction':stat(inter,y==lab),'lead_symmetric_contribution':stat(lead,y==lab),'tail_symmetric_contribution':stat(tail,y==lab)} for word,lab in [('bonafide',1),('spoof',0)]}
# Exact algebraic decomposition of a chosen comparison; not unique attribution.
for word,lab in [('bonafide',1),('spoof',0)]:
 m=(y==lab)&e['eligible'];position=es[:,2]-es[:,3];texture_front=es[:,4]-es[:,2];texture_back=es[:,5]-es[:,3];interaction=texture_front-texture_back
 assert np.max(abs((position+interaction-(es[:,4]-es[:,5]))[m]))<1e-12
 res['factor_decomposition'][word]={key:stat(v,m) for key,v in [('zero_position',position),('texture_front',texture_front),('texture_back',texture_back),('texture_by_position_interaction',interaction),('own_position',es[:,4]-es[:,5])]}
res['cue_availability_prior']=json.loads((root/'EBM80_21DF_cue_audit/summary.json').read_text())
res['pooling_prior']=json.loads((root/'EBM80_21DF_pooling_audit/summary.json').read_text())
# Independent direct-array sanity checks: signed differences and confusion counts.
for j,name in enumerate(names[2:],2):
 for word,lab in [('bonafide',1),('spoof',0)]:
  m=y==lab;v=[float(s[i,j])-float(s[i,0]) for i in np.flatnonzero(m)];a=res['effects'][name]['labels'][word]
  assert abs(sum(v)/len(v)-a['mean'])<1e-12
  assert a['toward_n']+a['away_n']+a['small_n']==a['n']
  olderr=res['performance']['baseline']['bona_reject_n' if lab else 'spoof_accept_n'];newerr=res['performance'][name]['bona_reject_n' if lab else 'spoof_accept_n'];assert newerr-olderr==a['new_error_n']-a['corrected_error_n']
res['independent_arithmetic_checks_passed']=True
(dest/'summary.json').write_text(json.dumps(res,indent=2,allow_nan=False))
selected=['native_vs_fixed','added_zero_position','added_texture_front','zero_both','shuffle_both','phase_both','quiet_gain_01','quiet_gain_10','global_gain_01']
short=['Native vs crop/wrap','Front vs back (+1s zero)','Own vs zero (+1s front)','Zero quiet edges','Shuffle quiet edges','Phase-only quiet edges','Quiet gain x0.1','Quiet gain x10','Whole waveform x0.1']
fig,ax=plt.subplots(1,2,figsize=(13,6),layout='constrained')
for k,word in enumerate(['bonafide','spoof']):
 xx=np.arange(len(selected));vals=[res['effects'][n]['labels'][word] for n in selected]
 ax[0].barh(xx+(-.17 if k==0 else .17),[v['mean_abs'] for v in vals],height=.32,label=word,color=['#317BA9','#CB8038'][k])
 ax[1].barh(xx+(-.17 if k==0 else .17),[100*v['abs_ge001_n']/v['n'] for v in vals],height=.32,label=word,color=['#317BA9','#CB8038'][k])
for a in ax:a.set_yticks(range(len(selected)),short);a.invert_yaxis();a.grid(axis='x',alpha=.15);a.legend()
ax[0].set(xlabel='Mean absolute change in cosine',title='Magnitude of paired score change');ax[1].set(xlabel='Files with |change| >= 0.01 (%)',title='Fraction with a nontrivial score change')
fig.suptitle('Frozen epoch80 | same 2000 follow-up files | intervention-specific sensitivity, not causal feature-importance shares',fontsize=12)
fig.savefig(dest/'factor_effects.png',dpi=170);plt.close(fig)
print(json.dumps({'checks':res['checks'],'performance':res['performance'],'quiet_statistics':res['quiet_statistics']},indent=2),flush=True)
