"""Measure wrap-join discontinuities, not silence durations/proportions."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import csv,json,hashlib,time
from pathlib import Path
import numpy as np
import soundfile as sf
import normal_acoustic_features_v2 as base

root=Path('/home/mentee/analysis/embedding_geometry/20261105_normal_cause_01')
assert not (root/'padding_features.csv').exists()
os.nice(10)
rows=list(csv.DictReader((root/'selected_ids.csv').open()));results=[];start=time.monotonic()
for i,row in enumerate(rows):
    audio,sr=sf.read(str(base.path_for(row,Path('/proc/352729/root/data'))),dtype='float32')
    x=audio[:64600].astype(float);n=len(x);rms=np.sqrt(np.mean(x*x));eps=1e-12
    dif=np.diff(x);jump=abs(x[-1]-x[0]);d99=np.quantile(np.abs(dif),.99)
    seam=np.r_[x[-200:],x[:200]]
    seam_spec=np.abs(np.fft.rfft((seam-seam.mean())*np.hanning(400),n=512))**2
    freq=np.fft.rfftfreq(512,1/sr);hf=float(seam_spec[freq>=4000].sum()/max(seam_spec.sum(),1e-20))
    internal=base.frames(x,400,160);power=np.abs(np.fft.rfft((internal-internal.mean(1,keepdims=True))*np.hanning(400),n=512,axis=1))**2
    ihf=power[:,freq>=4000].sum(1)/np.maximum(power.sum(1),1e-20)
    record=dict(case_id=row['case_id'],utterance_id=row['utterance_id'],
        wrap_joins=int((64600-1)//n) if n<64600 else 0,
        join_jump_to_rms=float(jump/(rms+eps)),join_jump_to_diff99=float(jump/(d99+eps)),
        join_HF_fraction=hf,join_HF_excess_over_internal_median_db=float(10*np.log10((hf+1e-20)/(np.median(ihf)+1e-20))),
        internal_HF_max_fraction=float(np.max(ihf)))
    results.append(record)
with (root/'padding_features.csv').open('x',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
base.save(root/'padding_complete.json',dict(status='pass',rows=len(results),seconds=time.monotonic()-start,
    sha256=hashlib.sha256((root/'padding_features.csv').read_bytes()).hexdigest(),
    interpretation='Boundary discontinuity measurement only; no model input was changed or inferred; a boundary with no actual wrap is hypothetical.',
    no_silence_ratio_features=True))
print('padding descriptors complete',len(results),flush=True)
