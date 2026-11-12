import os
os.environ['MPLCONFIGDIR']='/home/mentee/.cache/matplotlib'
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('/results/analysis/EBM80_native_quiet_strict')
assert Path('/.dockerenv').exists()
assert json.loads((root/'evaluation_view/COMPLETED.json').read_text())['n']==2000
out=root/'final';out.mkdir(exist_ok=False)
ts=json.loads((root/'evaluation_view/sample_manifest.json').read_text())
ss=[np.load(root/f'evaluation_view/shard_{r}.npz') for r in range(2)]
ix=np.concatenate([a['indices'] for a in ss]);order=np.argsort(ix)
assert np.array_equal(np.sort(ix),np.arange(2000))
d={k:np.concatenate([a[k] for a in ss])[order] for k in ['scores','embeddings','eligible','prefixes']}
assert np.array_equal(ss[0]['center'],ss[1]['center'])
s=d['scores'].astype(float);z=d['embeddings'].astype(float);c=ss[0]['center'].astype(float)
norm=np.linalg.norm(z,axis=-1);calc=np.sum(z*c,axis=-1)/(norm*np.linalg.norm(c))
assert np.nanmax(abs(calc-s))<1e-6
assert np.max(abs(norm[:,:4]-1))<1e-6
assert np.all(np.isfinite(s[:,:4])) and np.all(np.isfinite(s[d['eligible'],4:]))
assert np.all(np.isnan(s[~d['eligible'],4:]))
native=np.load(root/'analysis/merged_private.npz')
nm=native['cohort']=='new_files';nt=[t for t in json.loads((root/'sample_manifest.json').read_text()) if t['cohort']=='new_files']
assert [t['id'] for t in nt]==[t['id'] for t in ts]
ns=native['scores'][nm].astype(float);ne=native['eligible'][nm]
y=np.array([t['label'] for t in ts]);src=np.array([t['source'] for t in ts])
summary=json.loads((root/'analysis/summary.json').read_text());eps=summary['provenance']['direction_epsilon']
def effect(v):
 return dict(n=len(v),mean=float(v.mean()),median=float(np.median(v)),toward_n=int((v>eps).sum()),away_n=int((v<-eps).sum()),within_epsilon_n=int((abs(v)<=eps).sum()))
r={'scope':'Secondary evaluation-input sensitivity, same new 2000 files. Original 64600-sample head/wrap input preserved exactly, then add1s; transformed input is80600 samples. Descriptive, not new confirmatory tests.',
   'checks':{'n':2000,'cosine_max_error':float(np.nanmax(abs(calc-s))),'identity_max_delta':float(np.max(abs(s[:,1]-s[:,0]))),'all_passed':True},
   'results':{},'matched_native_comparison':{},'limits':['Not the conventional fixed-length pipeline after intervention; no additional crop is performed.','Own-prefix eligibility and filler can change with preprocessing; matched eligibility controls population, not identical filler.','No EER comparison to published benchmark.']}
for word,lab in [('bonafide',1),('spoof',0)]:
 r['results'][word]={};r['matched_native_comparison'][word]={}
 for fam,fi,bi,nfi,nbi in [('zero',2,3,10,11),('own40',4,5,12,13)]:
  mask=y==lab
  if fam=='own40':mask &= d['eligible']
  r['results'][word][fam]={key:effect((s[:,a]-s[:,b])[mask]) for key,a,b in [('front_original',fi,0),('back_original',bi,0),('front_back',fi,bi)]}
  mm=(y==lab)&(ne&d['eligible'] if fam=='own40' else True)
  r['matched_native_comparison'][word][fam]={
    'native':effect((ns[:,nfi]-ns[:,0])[mm]),'evaluation_view':effect((s[:,fi]-s[:,0])[mm])}
(out/'evaluation_view_summary.json').write_text(json.dumps(r,indent=2,allow_nan=False))
# Summarize the primary endpoints without hiding negative medians or confidence-method disagreement.
fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
labs=[];rows=[]
for fam in ['zero','own40']:
 for word in ['bonafide','spoof']:
  e=next(a for a in summary['primary'] if a['family']==fam and a['label']==word and a['contrast']=='front_original');rows.append(e);labs.append(('Zero' if fam=='zero' else 'Own quiet')+' / '+('B' if word=='bonafide' else 'S')+f" (n={e['n']})")
yy=np.arange(4)
for j,e in enumerate(rows):
 axs[0].barh(j,e['toward_n']/e['n']*100,color='#287BAD',label='Toward OC center' if j==0 else None)
 axs[0].barh(j,e['within_epsilon_n']/e['n']*100,left=e['toward_n']/e['n']*100,color='#CCCCCC',label='Tiny change' if j==0 else None)
 axs[0].barh(j,e['away_n']/e['n']*100,left=(e['toward_n']+e['within_epsilon_n'])/e['n']*100,color='#D68A37',label='Away from OC center' if j==0 else None)
 axs[0].text(101,j,f"{e['toward_n']}/{e['away_n']}/{e['within_epsilon_n']}",va='center',fontsize=9)
axs[0].set(yticks=yy,yticklabels=labs,xlim=(0,128),xlabel='Share of files (%)',title='Add 1s at front vs native original\nCounts at right: toward / away / tiny')
axs[0].invert_yaxis();axs[0].legend(loc='lower left',bbox_to_anchor=(0,-.27),ncol=1,fontsize=9)
rr=json.loads((root/'analysis/robustness.json').read_text())
for k,contrast in enumerate(['front_original','front_back']):
 for j,base in enumerate(rows):
  e=next(a for a in summary['primary'] if a['family']==base['family'] and a['label']==base['label'] and a['contrast']==contrast)
  lo,hi=e['mean_simultaneous_ci'];ys=j+(-.16 if k==0 else .16)
  axs[1].errorbar(e['mean'],ys,xerr=[[e['mean']-lo],[hi-e['mean']]],fmt='o' if k==0 else 's',color='#287BAD' if k==0 else '#7D529C',capsize=3,label=('Front minus original' if k==0 else 'Front minus back') if j==0 else None)
axs[1].axvline(0,color='#666',lw=1);axs[1].set(yticks=yy,yticklabels=labs,xlabel='Mean change in cosine to OC center',title='8 prespecified paired contrasts\n99.375% speaker-cluster bootstrap intervals');axs[1].invert_yaxis();axs[1].legend(loc='lower left',bbox_to_anchor=(0,-.27),fontsize=9)
fig.suptitle('Full native waveform preserved | new files only | positive = toward trained bonafide OC center',fontsize=12)
fig.savefig(out/'strict_primary_results.png',dpi=180,bbox_inches='tight');plt.close(fig)
print(json.dumps(r,indent=2),flush=True)
