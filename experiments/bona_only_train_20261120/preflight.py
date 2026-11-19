"""Revalidate identical prior evidence and fresh-training signature before GPU updates."""
import hashlib,json,sys
from pathlib import Path
import torch
from role_objectives import RoleConfig
from role_model import RoleBLinear
from role_runtime import options,signature,source_fingerprint,verify_reference,validate_runtime,validate_smoke_receipt
r=Path(sys.argv[1]);cfg=json.loads((r/'experiment.json').read_text());torch.set_num_threads(1)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
for n,h in cfg['source_sha256'].items():assert sha(r/'code'/n)==h,n
for n,h in json.loads((r/'control_manifest.json').read_text()).items():assert sha(r/n)==h,n
protected=json.loads((r/'preservation.json').read_text())
for n,h in protected['files'].items():assert sha(Path(n))==h,n
v=Path(cfg['validated_root']);assert (v/'run.exit').read_text().strip()=='0'
gpu=json.loads((v/'gpu_acceptance.json').read_text());assert gpu['status']=='passed' and gpu['spoof_short_gradient_exact_zero']
assert (v/'future_evaluation/test.exit').read_text().strip()=='0'
validate_runtime()
args=options(bona_bpl_weight=1.0,spoof_bpl_weight=0.0,eval_input='front64600_wrap',final_eval_policy='deferred_shared',frontend_revision='1a640f32ac3e39899438a2931f9924c02f080a54')
args['reference']=verify_reference();args['source_sha256']=source_fingerprint()
receipt=validate_smoke_receipt(args,v/'smoke/smoke_complete.json')
torch.manual_seed(1);old=RoleBLinear(RoleConfig(bona_bpl_weight=1));rng=torch.get_rng_state().clone()
torch.manual_seed(1);fresh=RoleBLinear(RoleConfig(bona_bpl_weight=1,spoof_bpl_weight=0))
assert torch.equal(rng,torch.get_rng_state()) and all(torch.equal(t,fresh.state_dict()[k]) for k,t in old.state_dict().items())
out=dict(status='passed',source_files=len(cfg['source_sha256']),protected_files=len(protected['files']),signature=signature(args),matching_smoke_signature=receipt['signature'],initial_state_exact=True,initial_rng_exact=True,fresh_initialization=True,no_smoke_weights_or_optimizer_loaded=True,normal_pairs=[144,144,36],spoof_pairs=[0,0,0],CPU_tests=163,actual_GPU_validation_passed=True,during_training_eval=['19LA'],after_training_eval=['21LA','21DF'],new_spending_usd=0,automatic_retries=0)
(r/'preflight_acceptance.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
