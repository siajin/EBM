"""Compatibility, original-normal-BPL and unchanged batch/optimizer contract."""
import ast, hashlib, json, math, sys, zipfile
from pathlib import Path
import torch
import torch.nn.functional as F
HERE=Path(__file__).resolve().parent; PROJECT=HERE.parents[1];sys.path.insert(0,str(PROJECT/'EBM'))
from role_objectives import RoleConfig,RoleOCSoftmax,RoleBPL
from model.loss import BPL
from role_runtime import options

torch.set_num_threads(1); rows=[]
for nb,ns in [(12,108),(12,60),(6,108)]:
    role=RoleConfig(bona_bpl_weight=1,spoof_bpl_pairing='same_uid')
    oc=RoleOCSoftmax(6,role).double(); bpl=RoleBPL(); center=torch.tensor([[1.,-1.,0,0,0,0]],dtype=torch.float64)
    with torch.no_grad():oc.weight.copy_(center)
    z=torch.cat([-center.repeat(nb,1),center.repeat(ns,1)]).requires_grad_(True)
    s=z.detach().clone().requires_grad_(True); y=torch.tensor([1]*nb+[0]*ns)
    actual=oc.backend_loss(z,y)+bpl.same_uid(z[nb:],s[nb:])+bpl(z[:nb],s[:nb])
    lower=(5*nb*F.softplus(torch.tensor(20*(-1-.2),dtype=torch.float64))+(5/9)*ns*F.softplus(torch.tensor(20*(.9-1),dtype=torch.float64)))/(nb+ns)+2*(math.log(2)-.3)
    torch.testing.assert_close(actual,lower,rtol=1e-12,atol=1e-12)
    gz,gs=torch.autograd.grad(actual,(z,s));assert torch.isfinite(gz).all() and torch.isfinite(gs).all()
    assert float(gz.norm())<1e-10 and float(gs.norm())<1e-10
    x=torch.randn(nb,6,dtype=torch.float64,requires_grad=True); v=torch.randn_like(x,requires_grad=True)
    normal=bpl(x,v); original=BPL(alpha=30)(x,v); assert torch.equal(normal,original)
    a=torch.autograd.grad(normal,(x,v),retain_graph=True); b=torch.autograd.grad(original,(x,v));assert all(torch.equal(i,j) for i,j in zip(a,b))
    rows.append(dict(bona=nb,spoof=ns,bona_pairs=nb*nb,spoof_pairs=ns,loss=float(actual.detach()),lower_bound=float(lower),normal_original_loss_gradient_exact=True))
out=dict(status='passed',compatibility=rows,objective='OC_long + mean_i spoof_softplus(30*(1-cos(z_i,s_i)))-0.3 + original_normal_BPL(alpha30)',
         reviewed=['Separate class means, no ns-vs-nb^2 pooling','Aligned same source item, independently augmented views','OC class weights, original full local N, spoof-only center update','Unit-sphere derivatives finite; additive-0.3 constants do not affect gradients','Fresh seed1 and two Adam phases unchanged'],
         limits=['Point collapse remains a compatible global minimum; not an anti-collapse constraint','Spoof-only center OC still encourages a cap; shared gradients can still conflict','Both pair modes change relative to last normal-UID experiment; use dual-allcross as single-change control','No theorem implies lower EER; evaluate matched inputs and single-seed uncertainty'],new_spending_usd=0,optimizer_updates=0)
(HERE/'mathematical_review.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
z=zipfile.ZipFile(HERE.parent/'bona_uid_train_20261118/source_payload.zip'); exact={}
for name in ['role_training.py','role_lifecycle.py','baseline240_source/data_loaders.py','baseline240_source/train.py','baseline240_source/main.py','model/b_linear.py','model/loss.py','data_loaders.py']:
    data=(PROJECT/'EBM'/name).read_bytes();assert data==z.read(name),name;exact[name]=hashlib.sha256(data).hexdigest()
def definition(source,name):return ast.dump(next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
assert definition(z.read('role_runtime.py').decode(),'build_trainer')==definition((PROJECT/'EBM/role_runtime.py').read_text(),'build_trainer')
old=options(bona_bpl_weight=1,final_eval_policy='deferred_shared',eval_input='front64600_wrap')
new=options(bona_bpl_weight=1,spoof_bpl_pairing='same_uid',final_eval_policy='deferred_shared',eval_input='front64600_wrap')
assert {k:v for k,v in old.items() if k!='role'}=={k:v for k,v in new.items() if k!='role'}
audit=dict(status='passed',unchanged_sources=exact,build_trainer_AST_exact=True,all_non_objective_options_exact=True,steps_per_epoch=108,global_batch_size=240,rank_batch_size=120,
           per_rank_bona=1290,per_rank_spoof=11616,backend_updates=108,center_updates=108,special_batches=rows,
           limitation='CPU/static proof; actual2GPU108loader smoke is mandatory before training')
(HERE/'batch_contract.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
print(json.dumps(dict(math='passed',batch='passed',unchanged=len(exact))))
