from pathlib import Path
import sys,json
root=Path(sys.argv[1]);sys.path.insert(0,str(root/'code/tests'))
from gpu_positive_smoke import worker
if __name__=='__main__':
 import torch.multiprocessing as mp
 mp.spawn(worker,args=(str(root),'checkpoint_oc'),nprocs=2,join=True)
 a=json.loads((root/'matched_independent_gpu_check.json').read_text())
 b=json.loads((root/'matched_same_source_gpu_check.json').read_text())
 for key in ['initial_backend_sha256','policy_sha256','source_fingerprint','evaluation_protocol','OC_input_hashes']:
  assert a[key]==b[key],key
 assert json.loads((root/'checkpoint_oc_gpu_check.json').read_text())['status']=='passed'
 (root/'gpu_checks_complete.json').write_text(json.dumps({'status':'passed','training_checks_from':'v2','conditions':[a,b]},indent=2))
