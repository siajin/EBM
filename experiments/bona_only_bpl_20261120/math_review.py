"""Independent embedding-space feasibility and exact single-factor configuration audit."""
import hashlib,json,math,sys
from pathlib import Path
import torch
import torch.nn.functional as F
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[1];sys.path.insert(0,str(PROJECT/'EBM'))
from role_objectives import RoleConfig,RoleOCSoftmax,RoleBPL
from role_runtime import options
torch.set_num_threads(1);role=RoleConfig(bona_bpl_weight=1,spoof_bpl_weight=0)
checks=[]
for b,s in [(12,108),(12,60),(6,108)]:
 oc=RoleOCSoftmax(6,role).double().requires_grad_(False)
 with torch.no_grad():oc.weight.zero_();oc.weight[0,0]=1
 z=torch.zeros(b+s,6,dtype=torch.float64);z[:b,0]=-1;z[b:,0]=1;short=z.clone();labels=torch.tensor([1]*b+[0]*s)
 objective=oc.backend_loss(z,labels)+RoleBPL(30,1)(z[:b],short[:b])
 lower=(b*5*F.softplus(torch.tensor(-24.,dtype=torch.float64))+s*(5/9)*F.softplus(torch.tensor(-2.,dtype=torch.float64)))/(b+s)+F.softplus(torch.tensor(0.,dtype=torch.float64))-.3
 assert abs(float(objective-lower))<1e-12
 center=oc.center_loss(z[b:],labels[b:]);assert abs(float(center-5*F.softplus(torch.tensor(-2.,dtype=torch.float64))))<1e-12
 checks.append(dict(bona_rows=b,spoof_rows=s,bona_pairs=b*b,spoof_pairs=0,backend_loss=float(objective),embedding_space_lower_bound=float(lower),gap=float(objective-lower),center_loss=float(center)))
base=options(bona_bpl_weight=1,eval_input='front64600_wrap',final_eval_policy='deferred_shared')
new=options(bona_bpl_weight=1,spoof_bpl_weight=0,eval_input='front64600_wrap',final_eval_policy='deferred_shared')
assert {k:v for k,v in base.items() if k!='role'}=={k:v for k,v in new.items() if k!='role'}
old_role=RoleConfig.from_metadata(base['role']);new_role=RoleConfig.from_metadata(new['role'])
assert [k for k in vars(old_role) if getattr(old_role,k)!=getattr(new_role,k)]==['spoof_bpl_weight']
out=dict(status='passed',reference='dual all-cross v3',changed_hyperparameters=['spoof_bpl_weight:1->0'],all_other_runtime_options_exact=True,
 objective='Original spoof-centered OC long mean + original normal long x short BPL class mean; no spoof BPL; two original Adam phases',
 feasibility='Antipodal normal/spoof normalized embedding assignment attains individual angular minima in embedding space; does not assert realizability or convergence of the shared network on real data',
 checks=checks,primary_comparison='v3 dual all-cross, same seed/budget/input/loader/RNG/optimizer/evaluation; v5 is secondary because its spoof pairing differs',
 risks=['Spoof OC still contracts spoofs, full point collapse remains possible','No guarantee shared-parameter OC/BPL gradients align or EER improves','Removal lowers loss scale by deleting a term; lower logged loss is not improved recognition','Post-hoc target EER minima are not unbiased checkpoint selection'],
 test_evidence='12 new tests plus147 existing tests; independent value/grad/finite difference/Adam/DDP/metadata/checkpoint/RNG checks',new_spending_usd=0,full_training_started=False)
(HERE/'mathematical_review.json').write_text(json.dumps(out,indent=2));print(json.dumps(dict(status=out['status'],checks=len(checks),changed=out['changed_hyperparameters'])))
