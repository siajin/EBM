import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MPLCONFIGDIR']='/home/mentee/.cache/matplotlib'
from pathlib import Path
import json
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
assert Path('/.dockerenv').exists()
root=Path('/results/analysis'); out=root/'EBM80_21DF_tail_diagnosis'
df=pd.read_csv(out/'bonafide_sample_descriptors.csv')
trials=json.loads((root/'EBM80_21DF_centroid/sample_manifest.json').read_text())
d=np.load(root/'EBM80_21DF_centroid/embeddings.npz'); ids=[t['id'] for t in trials if t['label']==1]
assert ids==df.id.tolist()
pca=PCA(n_components=2,svd_solver='full').fit(d['embeddings']); xy=pca.transform(d['embeddings']); centers=pca.transform(d['learned_center'][None])
bxy=xy[d['labels']==1]; sxy=xy[d['labels']==0]
colors={'asvspoof':'#2563A6','vcc2018':'#D87527','vcc2020':'#5D8C45'}
fig,axes=plt.subplots(1,2,figsize=(15,6),layout='constrained')
axes[0].scatter(*sxy.T,c='#aaaaaa',s=7,alpha=.2,marker='^',label='Spoof (context)')
for source in ['asvspoof','vcc2020','vcc2018']:
    m=(df.source==source).to_numpy(); axes[0].scatter(*bxy[m].T,c=colors[source],s=15,alpha=.6,label=f'Bonafide: {source}')
axes[0].scatter(*centers.T,c='black',marker='*',s=180,label='Learned OC center'); axes[0].set(title='Same embedding PCA, bonafide colored by source',xlabel='PC1',ylabel='PC2'); axes[0].legend(fontsize=8)
hist=[]; labs=[]
for src in colors:
    groups=df[df.source==src].groupby('speaker')
    for speaker,g in groups:
        if len(g)>=10: hist.append((src,speaker,len(g),g['tail'].mean()*100))
hist=sorted(hist,key=lambda r:r[3],reverse=True)[:12]
axes[1].barh(range(len(hist)),[r[3] for r in hist],color=[colors[r[0]] for r in hist]); axes[1].set_yticks(range(len(hist)),[f'{r[1]} (n={r[2]})' for r in hist],fontsize=9); axes[1].invert_yaxis(); axes[1].set(xlabel='Bonafide below cosine 0.9 (%)',title='Highest tail rates by speaker ID (n >= 10)',xlim=(0,100))
fig.suptitle('The bonafide tail is concentrated in VCC source data | epoch80\nSource and speaker associations do not establish recording quality or causality',fontsize=14)
fig.savefig(out/'tail_source_geometry.png',dpi=170); fig.savefig(out/'tail_source_geometry.pdf'); plt.close(fig)
report={'within_source':{},'source_duration':[]}
df['duration_bin']=pd.cut(df.duration_s,[0,2,3.0275,4.0375,np.inf],labels=['<=2','2-3.03','3.03-4.04','>4.04'])
for src,g in df.groupby('source'):
    report['within_source'][src]={}
    for t,h in g.groupby('tail'):
        report['within_source'][src]['tail' if t else 'core']={'n':len(h),'duration_median':float(h.duration_s.median()),'low_energy_median':float(h.low_energy_fraction.median()),'rms_median':float(h.rms_dbfs.median()),'padding_median':float(h.padding_fraction.median())}
    for duration,h in g.groupby('duration_bin',observed=True):
        report['source_duration'].append({'source':src,'duration':str(duration),'n':len(h),'tail':int(h['tail'].sum()),'pct':float(h['tail'].mean()*100)})
report['source_codec_ranges']={src:{'min_tail_pct':float(g.groupby('codec')['tail'].mean().min()*100),'max_tail_pct':float(g.groupby('codec')['tail'].mean().max()*100),'codec_count':int(g.codec.nunique())} for src,g in df.groupby('source')}
(out/'stratified_summary.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
