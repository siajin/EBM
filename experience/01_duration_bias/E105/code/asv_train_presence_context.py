import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import json,hashlib
import numpy as np
import soundfile as sf
from sklearn.metrics import roc_auc_score
assert Path('/.dockerenv').exists()
base=Path('/results/analysis/EBM80_21DF_tail_train_context')
manifest=json.loads((base/'sample_manifest.json').read_text())
out=Path('/results/analysis/EBM80_ASV_quiet_presence_hypothesis_20261030')
assert out.exists() and not (out/'train_context.json').exists()
y=np.array([m['label'] for m in manifest]);assert len(y)==2000 and y.sum()==1000
lengths={'full_native':[],'first_64600_before_padding':[]}
for j,m in enumerate(manifest):
    p=Path('/data/ASVspoof2019/LA/ASVspoof2019_LA_train/flac')/(m['id']+'.flac')
    if not p.exists():p=p.with_suffix('.wav')
    a,sr=sf.read(p,dtype='float32');assert sr==16000 and a.ndim==1
    for mode in lengths:
        aa=a if mode=='full_native' else a[:64600]
        rms=np.array([np.sqrt(np.mean(aa[k:k+320].astype(float)**2)) for k in range(0,len(aa),320)])
        lengths[mode].append(float(np.flatnonzero(rms>=.1*rms.max())[0])*.02)
    if j%500==0:print('train raw features',j,flush=True)
summary={'scope':'1000 bonafide and 1000 spoof sampled from 19LA training with existing seed17 manifest; original audio before random cropping or augmentation. Not an exact training-time input distribution.',
 'definition':'20ms relative frame RMS <10% of maximum RMS (-20dB)',
 'training_labels_checked':True,'manifest_sha256':hashlib.sha256((base/'sample_manifest.json').read_bytes()).hexdigest(),'views':{}}
for mode,values in lengths.items():
    v=np.array(values);r={'median_seconds':{'bonafide':float(np.median(v[y==1])),'spoof':float(np.median(v[y==0]))},
      'length_only_AUC':float(roc_auc_score(y,v)),'rules':[]}
    for cut in [.02,.1,.5]:
        b=int(((y==1)&(v>=cut)).sum());s=int(((y==0)&(v>=cut)).sum())
        r['rules'].append({'cutoff_s':cut,'bonafide_with_prefix_n':b,'spoof_with_prefix_n':s,'per_class_n':1000,
          'length_only_balanced_accuracy':(b/1000+(1-s/1000))/2})
    summary['views'][mode]=r
(out/'train_context.json').write_text(json.dumps(summary,indent=2))
np.savez_compressed(out/'private/train_prefix_features.npz',labels=y,**{k:np.array(v) for k,v in lengths.items()})
print(json.dumps(summary,indent=2),flush=True)
