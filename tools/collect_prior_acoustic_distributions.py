"""Read existing server measurements and return aggregates only; no audio access."""
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'artifacts/geometry_experiments'))
from run_padding_causal import connect

REMOTE = r'''
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
import hashlib,json
from pathlib import Path
import numpy as np
p=Path('/home/mentee/results/analysis/ASV_quiet_BPL_distribution_20261030_v1')
plan=json.loads((p/'plan.json').read_text())
manifest=json.loads((p/'native_manifest_private.json').read_text())
a=np.load(p/'native_metrics_private.npz')['features']
assert len(manifest)==len(a) and a.shape[1:]==(2,15)
features=plan['features']; bins={}
for feature in features:
 if feature in ['fraction','all_zero']: bins[feature]=np.linspace(0,1,51)
 elif feature=='exactzero_fraction':
  maximum=float(a[:,:,features.index(feature)].max());bins[feature]=np.linspace(0,maximum*1.00001,81)
 elif feature=='quiet_energy_fraction':
  maximum=float(a[:,:,features.index(feature)].max());bins[feature]=np.r_[0,np.geomspace(1e-10,maximum*1.00001,81)]
 elif feature=='max_rms_dbfs':
  v=a[:,:,features.index(feature)].astype(float)
  bins[feature]=np.linspace(min(-100,float(v.min())),max(0,float(v.max())),101)
 else:
  v=a[:,:,features.index(feature)].astype(float);assert np.isfinite(v).all()
  bins[feature]=np.linspace(0,max(float(v.max()),1),61)
result={'plan':plan,'source_sha256':{n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in ['plan.json','native_metrics_private.npz','native_manifest_private.json']},'groups':{}}
for dataset in sorted({r['dataset'] for r in manifest}):
 result['groups'][dataset]={}
 for label in [1,0]:
  ix=np.array([i for i,r in enumerate(manifest) if r['dataset']==dataset and r['label']==label]);assert len(ix)>0
  group={'n':len(ix),'features':{}}
  for threshold_index,db in enumerate([20,40]):
   for j,feature in enumerate(features):
    v=a[ix,threshold_index,j].astype(float);assert np.isfinite(v).all()
    counts,edges=np.histogram(v,bins=bins[feature]);assert counts.sum()==len(v),(feature,v.min(),v.max())
    group['features'][str(db)+'_'+feature]={'edges':edges.tolist(),'counts':counts.tolist(),'mean':float(v.mean()),'quantiles':dict(zip(['min','p05','p25','median','p75','p95','max'],np.quantile(v,[0,.05,.25,.5,.75,.95,1]).tolist())),'zero_n':int((v==0).sum())}
  result['groups'][dataset][str(label)]=group
print(json.dumps(result,allow_nan=False))
'''

def main():
    out = PROJECT / 'artifacts/class_distributions_20261108'
    out.mkdir(parents=True, exist_ok=True)
    c = connect()
    try:
        _, stdout, stderr = c.exec_command("python3 - <<'PY'\n" + REMOTE + "\nPY", timeout=50)
        raw = stdout.read().decode()
        errors = stderr.read().decode()
        assert stdout.channel.recv_exit_status() == 0, errors[-2000:]
        result = json.loads(raw)
        (out / 'prior_acoustic_histograms.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps({'status': 'aggregated_existing_measurements', 'groups': {k: {y: g['n'] for y, g in v.items()} for k, v in result['groups'].items()}}))
    finally:
        c.close()

if __name__ == '__main__':
    main()
