from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
assert Path('/.dockerenv').exists()
root=Path('/results/analysis/EBM80_21DF_cue_audit')
df=pd.read_csv(root/'cue_features_private.csv'); d=df[df.split=='21df'].reset_index(drop=True)
p=np.load(root/'cue_predictions.npz',allow_pickle=True)
assert np.array_equal(d.label,p['labels']) and np.array_equal(d.source,p['sources'])
d['cosine']=p['ebm_cosine']; rng=np.random.default_rng(601)
summary={'bootstrap':{},'partial_spearman':{},'feature_medians':{}}
for src,g in d.groupby('source'):
    positions=g.index.to_numpy(); speakers=g.speaker.unique(); groups=[g.index[g.speaker==v].to_numpy() for v in speakers]
    entry={}
    samples=[np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))]) for _ in range(1000)]
    for model in ['silence_timing','silence_and_energy']:
        vals=[]
        for ix in samples:
            if len(np.unique(p['labels'][ix]))==2:vals.append(roc_auc_score(p['labels'][ix],p[model][ix]))
        entry[model]={'auc':float(roc_auc_score(g.label,p[model][positions])),
           'speaker_cluster_bootstrap_95pct':[float(x) for x in np.quantile(vals,[.025,.975])],
           'speaker_groups':len(groups),'bona_n':int(g.label.sum()),'spoof_n':int((1-g.label).sum())}
    summary['bootstrap'][src]=entry
    b=g[g.label==1]
    # Conditional association: ranks, adjusted for duration and speaker identity.
    controls=np.column_stack([np.ones(len(b)),rankdata(b.log_duration),pd.get_dummies(b.speaker,drop_first=True,dtype=float).to_numpy()])
    def residual(v):
        x=rankdata(v); return x-controls@np.linalg.lstsq(controls,x,rcond=None)[0]
    yy=residual(b.cosine)
    summary['partial_spearman'][src]={f:float(np.corrcoef(residual(b[f]),yy)[0,1]) for f in ['leading_s','low_energy_fraction','rms_dbfs','low_rms_q10_dbfs','repeat_peak_corr']}
for scope,g in [('19LA_train',df[df.split=='train']),('19LA_dev',df[df.split=='dev'])]+[(f'21DF_{src}',g) for src,g in d.groupby('source')]:
    summary['feature_medians'][scope]={str(label):{f:float(v[f].median()) for f in ['leading_s','trailing_s','low_energy_fraction','rms_dbfs','log_duration']} for label,v in g.groupby('label')}
summary['limitations']=['Bootstrap resamples speaker groups within source; it does not cover new unseen sources, different sampling or model training uncertainty.',
 'Partial Spearman adjusts for rank duration and observed speaker IDs only; residual associations remain non-causal.']
(root/'followup_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
q=json.loads((root/'summary.json').read_text())
models=['length_and_pipeline','silence_timing','energy_floor','spectral_shape','repetition_and_steps','silence_and_energy','all_audio','all_with_pipeline','all_audio_tree']
labels=['Length / padding metadata','Low-energy duration','Energy / noise floor','Spectral shape','Repetition / waveform steps','Duration + energy','All audio cues (linear)','All cues + metadata (linear)','All audio cues (small tree)']
conditions=['19LA_dev','21DF_all','21DF_asvspoof','21DF_vcc2018','21DF_vcc2020']
matrix=np.array([[q['models'][m][c]['auc'] for c in conditions] for m in models])
fig,ax=plt.subplots(figsize=(11.5,6.2),layout='constrained')
im=ax.imshow(matrix,vmin=0,vmax=1,cmap='RdYlBu',aspect='auto')
ax.set_yticks(range(len(models)),labels,fontsize=10)
ax.set_xticks(range(len(conditions)),['19LA dev','21DF all','21DF ASVspoof','21DF VCC2018','21DF VCC2020'],rotation=15,ha='right',fontsize=10)
for i in range(len(models)):
    for j in range(len(conditions)):ax.text(j,i,f'{matrix[i,j]:.3f}',ha='center',va='center',fontsize=10,color='white' if matrix[i,j]>.83 else 'black')
fig.colorbar(im,ax=ax,label='ROC AUC (0.5 = chance ranking)',shrink=.9)
fig.suptitle('Simple cue classifiers trained only on 19LA train\nEvaluation uses held-out speakers; 21DF sources are shown separately',fontsize=13)
fig.savefig(root/'cue_transfer_reviewed.png',dpi=170); fig.savefig(root/'cue_transfer_reviewed.pdf'); plt.close(fig)
