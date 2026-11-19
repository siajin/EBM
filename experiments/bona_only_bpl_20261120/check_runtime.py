"""Check immutable runtime, exact seed1 initialization and old v1-v5 model readers."""
import hashlib,json,sys
from pathlib import Path
import torch
from role_model import RoleBLinear
from role_objectives import RoleConfig
from role_runtime import validate_runtime
from role_lifecycle import load_scoring_head
r=Path(sys.argv[1]);torch.set_num_threads(1)
for n,h in json.loads((r/'source_manifest.json').read_text()).items():assert hashlib.sha256((r/'code'/n).read_bytes()).hexdigest()==h,n
for n,h in json.loads((r/'control_manifest.json').read_text()).items():assert hashlib.sha256((r/n).read_bytes()).hexdigest()==h,n
validate_runtime()
torch.manual_seed(1);old=RoleBLinear(RoleConfig(bona_bpl_weight=1));rng=torch.get_rng_state().clone()
torch.manual_seed(1);new=RoleBLinear(RoleConfig(bona_bpl_weight=1,spoof_bpl_weight=0))
assert torch.equal(rng,torch.get_rng_state())
assert all(torch.equal(v,new.state_dict()[k]) for k,v in old.state_dict().items())
assert sum(p.numel() for p in new.parameters())==3744384
checks=[]
for m in json.loads((r/'protected_roles.json').read_text()):
 head,_=load_scoring_head(Path(m['container_path']).parent.parent,m['epoch'])
 assert head.role.metadata()==m['metadata']['role'] and head.role.spoof_bpl_weight==1
 checks.append(dict(family=m['family'],epoch=m['epoch'],schema=head.role.metadata()['schema_version']))
out=dict(status='passed',initial_state_exact=True,initial_rng_exact=True,parameters=3744384,old_checkpoint_readers=checks,frontend_downloads=0,full_training_started=False)
(r/'runtime_acceptance.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
