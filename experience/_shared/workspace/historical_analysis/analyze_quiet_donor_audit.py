import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2';os.environ['MPLCONFIGDIR']='/home/mentee/.cache/matplotlib'
from pathlib import Path
import json,hashlib,numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
assert Path('/.dockerenv').exists()
root=Path('/results/analysis/EBM80_quiet_donor_audit');out=root/'analysis';out.mkdir(exist_ok=False)
plan=json.loads((root/'plan.json').read_text());assert json.loads((root/'inference/COMPLETED.json').read_text())['n']==2000
tr=json.loads((root/'recipient_manifest_private.json').read_text());dm=json.loads((root/'donor_manifest_private.json').read_text());pairs=np.load(root/'assignment_private.npz')['pairs']
ss=[np.load(root/f'inference/shard_{i}.npz') for i in range(2)];ix=np.concatenate([a['indices'] for a in ss]);assert np.array_equal(np.sort(ix),np.arange(2000));order=np.argsort(ix)
d={k:np.concatenate([a[k] for a in ss])[order] for k in ['scores','checks','gains','jumps','counts']};s=d['scores'].astype(float)
y=np.array([t['label'] for t in tr]);src=np.array([t['source'] for t in tr]);sp=np.array([t['speaker'] for t in tr]);codec=np.array([t['codec'] for t in tr]);eligible=d['counts'][:,0]>0
assert np.isfinite(s[:,:2]).all() and np.isfinite(s[eligible]).all() and np.isnan(s[~eligible,2:]).all()
assert np.max(d['checks'][:,0])<1e-5 and np.max(d['checks'][:,1])<1e-8 and np.all(d['checks'][eligible,2]==1)
assert np.array_equal(eligible,[bool(t['eligible_frame_indices']) for t in tr])
for i in range(2000):
 cs=set()
 for si,so in enumerate(plan['sources']):
  for rep in [0,1]:
   a,b=[dm[int(j)] for j in pairs[i,si,rep]];assert a['label']==1 and b['label']==0
   assert (a['source'],a['speaker'],a['codec'])==(b['source'],b['speaker'],b['codec']);assert a['source']==so and a['speaker'] in plan['donor_speakers_by_disjoint_bank'][so][rep];cs.add(a['codec'])
 assert len(cs)==1
for so,sets in plan['donor_speakers_by_disjoint_bank'].items():assert not(set(sets[0])&set(sets[1]))
oldtr=json.loads(Path('/results/analysis/EBM80_cue_factor_audit/sample_manifest.json').read_text());assert [t['id'] for t in tr]==[t['id'] for t in oldtr]
oldss=[np.load(f'/results/analysis/EBM80_cue_factor_audit/shard_{i}.npz') for i in range(2)];oo=np.argsort(np.concatenate([a['indices'] for a in oldss]));old=np.concatenate([a['scores'] for a in oldss])[oo,0]
assert np.max(abs(s[:,0]-old))<1e-3
eps=max(.000121,10*float(np.max(abs(s[:,1]-s[:,0]))));th=json.loads(Path('/results/analysis/EBM80_21DF_leading_quiet_paired/summary.json').read_text())['threshold']
ds=s[:,5:].reshape(2000,3,2,2);delta=ds[:,:,:,0]-ds[:,:,:,1];gg=d['gains'][:,5:].reshape(2000,3,2,2);jj=d['jumps'][:,5:].reshape(2000,3,2,2)
gainmax=np.max(gg,axis=(2,3));jumpmax=np.max(jj,axis=(2,3));bounded=eligible[:,None]&(gainmax<=10);common=bounded.all(1);rng=np.random.default_rng(316204);R=10000
recipient_codec_match=np.array([dm[int(pairs[i,0,0,0])]['codec']==tr[i]['codec'] for i in range(2000)])
assert int(recipient_codec_match.sum())==plan['recipient_codec_matched_n']
def stat(v,m):
 a=v[m]
 if len(a)==0:return {'n':0,'available':False}
 return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),mean_abs=float(abs(a).mean()),toward_n=int((a>eps).sum()),away_n=int((a<-eps).sum()),small_n=int((abs(a)<=eps).sum()),abs_ge001_n=int((abs(a)>=.01).sum()),abs_ge01_n=int((abs(a)>=.1).sum()))
def ci(v,m):
 if m.sum()<5:return {'conditional_mean_ci95':None}
 boot=np.zeros(R);clusters={}
 for so in np.unique(src[m]):
  ids=np.flatnonzero(m&(src==so));groups=[ids[sp[ids]==v] for v in np.unique(sp[ids])];cnt=np.array([len(g) for g in groups]);sm=np.array([v[g].sum() for g in groups]);w=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=R);boot+=(w@sm)/(w@cnt)*(len(ids)/m.sum());clusters[str(so)]=len(groups)
 return {'conditional_mean_ci95':np.quantile(boot,[.025,.975]).tolist(),'recipient_speaker_clusters':clusters}
res={'plan':plan,'n':2000,'eligible':{'bonafide':int((eligible&(y==1)).sum()),'spoof':int((eligible&(y==0)).sum())},
 'frame_reuse':{'unique_frame_count_histogram':{str(int(k)):int(((d['counts'][:,1]==k)&eligible).sum()) for k in np.unique(d['counts'][eligible,1])},'recipients_reusing_frames_n':int((eligible&(d['counts'][:,0]>d['counts'][:,1])).sum()),'eligible_recipient_frames_median':float(np.median(d['counts'][eligible,0]))},
 'threshold':th,'direction_epsilon':eps,'checks':{'baseline_reproduction_max':float(np.max(abs(s[:,0]-old))),'identity_max':float(np.max(abs(s[:,1]-s[:,0]))),'maximum_inner_energy_relative_error':float(d['checks'][:,0].max()),'maximum_inner_mean_error':float(d['checks'][:,1].max()),'all_pair_matching_assertions':True,'all_bank_speakers_disjoint':True,'preservation_assertions':True},
 'primary':{},'all_gain_secondary':{},'source_common_mask':{},'controls':{},'donor_influence':{},'limits':[
 'Donor B-minus-S identifies residual waveform properties associated with donor labels in this finite pool, not a causal effect of labels themselves or proof of historical shortcut learning.',
 'Recipient bootstrap intervals condition on realized donor pool/assignments. They do not capture donor-population uncertainty or shared speaker roles. No confirmatory p-values.',
 'BankA/B donor-speaker disjoint, but only2speakers perbank forVCC2018. Leave-one-donor-speaker ranges are sensitivity checks, not confidence intervals.',
 'Primary gain masks are source-specific. Direct source comparisons use intersection across all sources.',
 'Replacement preserves leadingquiet location/duration, all original frame borders, and inner mean/energy. New joins at internal boundaries can still be artificial.',
 'Original repeated-codec utterances and dataset/model-development dependence may remain; no untouched external validation or retraining.',
 'Lowenergy frames may contain real synthesis/channel traces; transfer does not by itself prove the traces are spurious.']}
for si,so in enumerate(plan['sources']):
 res['primary'][so]={};res['all_gain_secondary'][so]={};res['donor_influence'][so]={}
 for word,lab in [('bonafide',1),('spoof',0)]:
  for key,m in [('primary',bounded[:,si]&(y==lab)),('all_gain_secondary',eligible&(y==lab))]:
   mean_delta=delta[:,si].mean(1);entry=stat(mean_delta,m);entry.update(ci(mean_delta,m));entry['banks']={bank:stat(delta[:,si,rep],m) for rep,bank in enumerate(['A','B'])}
   entry['both_banks_toward_n']=int(((delta[:,si,0]>eps)&(delta[:,si,1]>eps)&m).sum());entry['both_banks_away_n']=int(((delta[:,si,0]<-eps)&(delta[:,si,1]<-eps)&m).sum())
   entry['recipient_sources']={str(ss):int((m&(src==ss)).sum()) for ss in np.unique(src)}
   entry['donorB_minus_original']=stat(ds[:,si,:,0].mean(1)-s[:,0],m);entry['donorS_minus_original']=stat(ds[:,si,:,1].mean(1)-s[:,0],m)
   entry['low_join_subset']=stat(mean_delta,m&(jumpmax[:,si]<=.05))
   entry['recipient_codec_matched_subset']=stat(mean_delta,m&recipient_codec_match)
   entry['multiple_unique_frames_subset']=stat(mean_delta,m&(d['counts'][:,1]>=2))
   entry['no_frame_repetition_subset']=stat(mean_delta,m&(d['counts'][:,0]==d['counts'][:,1]))
   if m.sum():
    jdiff=jj[:,si,:,0].mean(1)-jj[:,si,:,1].mean(1);entry['join_B_minus_S_mean']=float(jdiff[m].mean())
    corr=float(spearmanr(jdiff[m],mean_delta[m]).statistic) if len(np.unique(jdiff[m]))>=2 and len(np.unique(mean_delta[m]))>=2 else float('nan')
    entry['join_effect_spearman']=corr if np.isfinite(corr) else None
    entry['max_amplification_in_subset']=float(gainmax[m,si].max())
    # Count decision changes per bank; mean-of-scores is not presented as an actual classifier.
    entry['decision_flips_by_bank']={}
    for rep,bank in enumerate(['A','B']):
     before=ds[m,si,rep,1]>=th;after=ds[m,si,rep,0]>=th;truth=y[m]==1
     entry['decision_flips_by_bank'][bank]={'n':int(m.sum()),'new_errors':int(((before==truth)&(after!=truth)).sum()),'corrected_errors':int(((before!=truth)&(after==truth)).sum())}
   res[key][so][word]=entry
  m=bounded[:,si]&(y==lab);rows=[];leave=[]
  for rep,bank in enumerate(['A','B']):
   dsp=np.array([dm[int(j)]['speaker'] for j in pairs[:,si,rep,0]])
   for donor in np.unique(dsp[m]):
    mm=m&(dsp==donor);a=stat(delta[:,si,rep],mm);a.update(bank=bank,donor_speaker=str(donor));rows.append(a)
    kept=m&(dsp!=donor)
    if kept.any() and m.any():leave.append({'removed_bank':bank,'removed_speaker':str(donor),'remaining_in_bank':int(kept.sum()),'mean':float(.5*(delta[kept,si,rep].mean()+delta[m,si,1-rep].mean()))})
  res['donor_influence'][so][word]={'by_donor':rows,'leave_one':leave,'leave_one_mean_range':[min(v['mean'] for v in leave),max(v['mean'] for v in leave)] if leave else None,'limitation':'Leave-one keeps banks at equal weight but changes recipient composition within the removed donor bank; not a standardized causal contrast or CI.'}
for word,lab in [('bonafide',1),('spoof',0)]:
 m=common&(y==lab);res['source_common_mask'][word]={'n':int(m.sum()),'donor_label_effects':{so:stat(delta[:,si].mean(1),m) for si,so in enumerate(plan['sources'])},'source_differences':{}}
 for a,b in [(0,1),(0,2),(1,2)]:
  for label,li in [('donorB',0),('donorS',1)]:
   diff=ds[:,a,:,li]-ds[:,b,:,li];entry=stat(diff.mean(1),m);entry.update(ci(diff.mean(1),m));entry['banks']={bank:stat(diff[:,rep],m) for rep,bank in enumerate(['A','B'])};entry['recipient_codec_matched_subset']=stat(diff.mean(1),m&recipient_codec_match)
   entry['decision_flips_by_bank']={}
   for rep,bank in enumerate(['A','B']):
    before=ds[m,b,rep,li]>=th;after=ds[m,a,rep,li]>=th;truth=y[m]==1
    entry['decision_flips_by_bank'][bank]={'n':int(m.sum()),'new_disagreements_with_original_label':int(((before==truth)&(after!=truth)).sum()),'corrected_disagreements_with_original_label':int(((before!=truth)&(after==truth)).sum())}
   res['source_common_mask'][word]['source_differences'][f'{plan["sources"][a]}_minus_{plan["sources"][b]}_{label}']=entry
 res['controls'][word]={name:stat(s[:,j]-s[:,0],eligible&(y==lab)) for j,name in [(2,'self_reuse'),(3,'self_shuffle'),(4,'gaussian')]}
for si,so in enumerate(plan['sources']):
 for word,lab in [('bonafide',1),('spoof',0)]:
  m=bounded[:,si]&(y==lab);entry=res['primary'][so][word]
  if m.any():
   direct=sum((float(s[i,5+si*4])-float(s[i,6+si*4])+float(s[i,7+si*4])-float(s[i,8+si*4]))/2 for i in np.flatnonzero(m))/m.sum();assert abs(direct-entry['mean'])<1e-12
   assert entry['toward_n']+entry['away_n']+entry['small_n']==entry['n']
res['independent_effect_checks_passed']=True
(out/'summary.json').write_text(json.dumps(res,indent=2,allow_nan=False))
fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
for ax,word in zip(axes,['bonafide','spoof']):
 for si,so in enumerate(plan['sources']):
  en=res['primary'][so][word]
  for j,bank in enumerate(['A','B']):ax.bar(si+(-.18 if j==0 else .18),en['banks'][bank].get('mean',np.nan),width=.32,color=['#317BA9','#CB8038'][j],label=bank if si==0 else None)
  ax.text(si,.98,f'n={en["n"]}',ha='center',va='top',transform=ax.get_xaxis_transform())
 ax.axhline(0,color='gray',lw=1);ax.set_xticks(range(3),plan['sources']);ax.set(title=f'Recipient {word}',ylabel='Mean cosine: donor B minus donor S');ax.legend(title='Disjoint donor-speaker bank');ax.grid(axis='y',alpha=.2)
fig.suptitle('Quiet waveform transplantation | source/speaker/codec matched donors\nRecipient body, duration and frame mean/energy preserved | primary max gain<=10',fontsize=12)
fig.savefig(out/'donor_label_effect.png',dpi=170);plt.close(fig)
print(json.dumps({'checks':res['checks'],'primary':res['primary'],'controls':res['controls']},indent=2),flush=True)
