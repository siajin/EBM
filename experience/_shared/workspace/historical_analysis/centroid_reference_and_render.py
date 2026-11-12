import os
os.environ['MPLCONFIGDIR']='/home/mentee/.cache/matplotlib'
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('/results/analysis/EBM80_native_quiet_strict');out=root/'final'
assert Path('/.dockerenv').exists()
d=np.load(root/'analysis/merged_private.npz');z=d['embeddings'].astype(float)
z=z/np.linalg.norm(z,axis=-1,keepdims=True)
ref=z[(d['cohort']=='previous')&(d['labels']==1),0].mean(0);ref/=np.linalg.norm(ref)
trained=d['center'].astype(float);trained/=np.linalg.norm(trained)
sr=z@ref;so=z@trained;eps=json.loads((root/'analysis/summary.json').read_text())['provenance']['direction_epsilon']
r={'reference':'Fixed normalized mean of original native embeddings from previous 1000 bonafide files; applied to disjoint new-file cohort. Never recomputed after intervention.',
   'reference_cosine_to_trained_OC':float(ref@trained),'results':[],'scope':'Secondary descriptive reference sensitivity. No claim of an alternative trained classifier or independent development holdout.'}
for fam,f,b in [('zero',10,11),('own40',12,13)]:
 for word,lab in [('bonafide',1),('spoof',0)]:
  m=(d['cohort']=='new_files')&(d['labels']==lab)
  if fam=='own40':m &= d['eligible']
  for name,base in [('front_original',0),('front_back',b)]:
   v=(sr[:,f]-sr[:,base])[m];ov=(so[:,f]-so[:,base])[m]
   classify=lambda a:np.where(a>eps,1,np.where(a<-eps,-1,0))
   r['results'].append(dict(family=fam,label=word,contrast=name,n=len(v),mean=float(v.mean()),median=float(np.median(v)),toward_n=int((v>eps).sum()),away_n=int((v<-eps).sum()),tiny_n=int((abs(v)<=eps).sum()),direction_category_agreement_with_OC_n=int((classify(v)==classify(ov)).sum())))
(out/'empirical_reference_summary.json').write_text(json.dumps(r,indent=2,allow_nan=False))
summary=json.loads((root/'analysis/summary.json').read_text())
fig,axs=plt.subplots(1,2,figsize=(13,5.8));fig.subplots_adjust(left=.14,right=.98,top=.77,bottom=.29,wspace=.55)
rows=[];labs=[]
for fam in ['zero','own40']:
 for word in ['bonafide','spoof']:
  e=next(a for a in summary['primary'] if a['family']==fam and a['label']==word and a['contrast']=='front_original');rows.append(e);labs.append(('Zero' if fam=='zero' else 'Own quiet')+' / '+('B' if word=='bonafide' else 'S')+f" (n={e['n']})")
yy=np.arange(4)
for j,e in enumerate(rows):
 left=0
 for key,color,label in [('toward_n','#287BAD','Toward OC center'),('within_epsilon_n','#CCCCCC','Tiny change'),('away_n','#D68A37','Away from OC center')]:
  width=e[key]/e['n']*100;axs[0].barh(j,width,left=left,color=color,label=label if j==0 else None);left+=width
 axs[0].text(102,j,f"{e['toward_n']}/{e['away_n']}/{e['within_epsilon_n']}",va='center',fontsize=9)
axs[0].set(yticks=yy,yticklabels=labs,xlim=(0,130),xticks=[0,25,50,75,100],xlabel='Share of files (%)',title='Front +1s vs native original\nCounts: toward / away / tiny');axs[0].invert_yaxis();axs[0].legend(loc='upper left',bbox_to_anchor=(0,-.24),fontsize=9)
for k,contrast in enumerate(['front_original','front_back']):
 for j,base in enumerate(rows):
  e=next(a for a in summary['primary'] if a['family']==base['family'] and a['label']==base['label'] and a['contrast']==contrast)
  lo,hi=e['mean_simultaneous_ci'];axs[1].errorbar(e['mean'],j+(-.16 if k==0 else .16),xerr=[[e['mean']-lo],[hi-e['mean']]],fmt='o' if k==0 else 's',color='#287BAD' if k==0 else '#7D529C',capsize=3,label=('Front minus original' if k==0 else 'Front minus back') if j==0 else None)
axs[1].axvline(0,color='#666',lw=1);axs[1].set(yticks=yy,yticklabels=labs,xlabel='Mean change in cosine to OC center',title='8 primary paired contrasts\n99.375% cluster-bootstrap intervals');axs[1].invert_yaxis();axs[1].legend(loc='upper left',bbox_to_anchor=(0,-.24),fontsize=9)
fig.suptitle('Native waveform preserved | new files only | positive = toward trained bonafide OC center',fontsize=13,y=.98)
fig.text(.5,.02,'Small speaker clusters: alternative cluster-t intervals include zero for Zero/B front-back and Own/B front-original.\nMean effects do not imply a majority of files move toward the center.',ha='center',fontsize=9)
fig.savefig(out/'strict_primary_results.png',dpi=180);plt.close(fig)
print(json.dumps(r,indent=2),flush=True)
