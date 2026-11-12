"""Check label-correlated low-energy context in raw training inputs, no inference."""
from pathlib import Path
import json, random
import numpy as np
import soundfile as sf
from sklearn.metrics import roc_auc_score
assert Path('/.dockerenv').exists()
root=Path('/data/ASVspoof2019/LA')
out=Path('/results/analysis/EBM80_21DF_tail_train_context'); out.mkdir(exist_ok=False)
protocol=root/'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'
groups={0:[],1:[]}
for line in protocol.read_text().splitlines():
    f=line.split(); groups[int(f[4]=='bonafide')].append({'id':f[1],'speaker':f[0],'attack':f[3],'label':int(f[4]=='bonafide')})
rng=random.Random(17); trials=[t for k in [1,0] for t in rng.sample(groups[k],1000)]
(out/'sample_manifest.json').write_text(json.dumps(trials,indent=2))
rows=[]
for i,t in enumerate(trials):
    path=root/'ASVspoof2019_LA_train/flac'/(t['id']+'.flac')
    if not path.exists(): path=path.with_suffix('.wav')
    a,sr=sf.read(path,dtype='float32'); assert sr==16000 and a.ndim==1
    duration=len(a)/sr; a=a[:64600]
    frames=np.pad(a,(0,(-len(a))%320)).reshape(-1,320)
    rms=np.sqrt(np.mean(frames**2,axis=1)+1e-16)
    active=np.flatnonzero(rms>=rms.max()*.1)
    left=int(active[0])*320; right=min(len(a),(int(active[-1])+1)*320)
    rows.append({'label':t['label'],'duration_s':duration,'rms_dbfs':float(20*np.log10(np.sqrt(np.mean(a*a)+1e-16))),
                 'low_energy_fraction':float((rms<rms.max()*.1).mean()),'edge_low_energy_fraction':float((left+len(a)-right)/len(a)),
                 'leading_s':left/sr,'trailing_s':(len(a)-right)/sr})
    if i%500==0: print('processed',i,flush=True)
y=np.array([r['label'] for r in rows]); fields=[k for k in rows[0] if k!='label']
summary={'population':{str(k):len(v) for k,v in groups.items()},'sample_per_class':1000,'seed':17,
 'by_label':{},'univariate_descriptive_auc':{},'limitations':['Original raw training audio before random crops and codec augmentation; not the exact input stream seen in training.',
 'Same 20 ms relative RMS proxy as 21DF analysis, evaluated on first 64,600 samples before padding.',
 'These are label associations, not proof that a particular learned unit uses the cue.']}
for label in [0,1]:
    subset=[r for r in rows if r['label']==label]
    summary['by_label']['bonafide' if label else 'spoof']={f:{'median':float(np.median([r[f] for r in subset])),
      'mean':float(np.mean([r[f] for r in subset]))} for f in fields}
for f in fields: summary['univariate_descriptive_auc'][f]=float(roc_auc_score(y,[r[f] for r in rows]))
(out/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2),flush=True)
