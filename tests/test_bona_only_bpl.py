"""Bona-only BPL ablation: independent formulas, gradients, Adam, DDP and artifacts."""
import copy,datetime,importlib.util,json,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
import torch.nn.functional as F
from torch import nn
from role_objectives import RoleConfig
from role_model import RoleBLinear
from role_training import step_hidden,phase_rng
from model.loss import BPL as OriginalBPL
import role_runtime as runtime
import role_lifecycle as life
from test_role_experiment import make_trainer,fake_features,assert_nested_equal

ROOT=Path(__file__).resolve().parents[1]
def fixture(name):
 spec=importlib.util.spec_from_file_location(name,Path(__file__).parent/'fixtures'/(name+'.py'))
 m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
old_config=fixture('role_objectives_before_bona_only')
old_model=fixture('role_model_before_bona_only')

def head(weight=0.):return RoleBLinear(RoleConfig(bona_bpl_weight=1,spoof_bpl_weight=weight),2,4,6,4)
class OldHead(RoleBLinear):forward=old_model.RoleBLinear.forward
class ReferenceHead(RoleBLinear):
 def forward(self,x,x_short=None,label=None,*,phase='score',mask=None,mask_short=None):
  if phase!='backend':return super().forward(x,x_short,label,phase=phase,mask=mask,mask_short=mask_short)
  z=self.encode(x,mask);short=self.encode(x_short,mask_short);bona=label==1
  return self.loss.backend_loss(z,label)+OriginalBPL(30)(z[bona],short[bona]),{}

def ddp_worker(rank,rendezvous,folder):
 torch.set_num_threads(1)
 torch.distributed.init_process_group('gloo',init_method=rendezvous,rank=rank,world_size=2,timeout=datetime.timedelta(seconds=90))
 try:
  torch.manual_seed(51);t=make_trainer(folder)
  t.classifier=nn.parallel.DistributedDataParallel(head(),find_unused_parameters=True,static_graph=False)
  t.optimizer=torch.optim.Adam(t.classifier.parameters(),lr=1e-5,weight_decay=1e-4)
  t.lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(t.optimizer,150)
  for step,(b,s) in enumerate([(12,108),(12,60),(6,108)] if rank==0 else [(12,108),(6,108),(12,60)]):
   torch.manual_seed(70+rank*10+step)
   r=step_hidden(t.classifier,t.optimizer,torch.randn(b+s,2,5,4),torch.randn(b+s,2,4,4),torch.tensor([1]*b+[0]*s))
   assert r['pair_count']==0 and r['pair_spoof']==0 and r['bona_pair_count']==b*b
  for p in t.classifier.parameters():
   values=[torch.empty_like(p) for _ in range(2)];torch.distributed.all_gather(values,p)
   assert torch.equal(*values);assert t.optimizer.state[p]['step'].item()==3
  t.checkpoints=life.RoleCheckpoints(t);t.checkpoints.save(5)
  loaded,_=life.load_scoring_head(folder,5);assert loaded.role==RoleConfig(bona_bpl_weight=1,spoof_bpl_weight=0)
 finally:torch.distributed.destroy_process_group()

class Checks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):torch.set_num_threads(1)

 def test_v1_to_v5_exact_and_v6_strict_roundtrip(self):
  for kw in [{},{'compact_label':1},{'pair_target_cosine':.9,'r_pull':.8},{'bona_bpl_weight':1},{'bona_bpl_weight':1,'bona_bpl_pairing':'same_uid'},{'spoof_bpl_pairing':'same_uid'},{'bona_bpl_weight':1,'spoof_bpl_pairing':'same_uid'}]:
   metadata=old_config.RoleConfig(**kw).metadata()
   self.assertEqual(RoleConfig(**kw).metadata(),metadata);self.assertEqual(RoleConfig.from_metadata(metadata),RoleConfig(**kw))
  r=RoleConfig(bona_bpl_weight=1,spoof_bpl_weight=0);m=r.metadata()
  self.assertEqual(m['schema_version'],'ebm_role_v6');self.assertEqual(m['pair_reduction'],'disabled');self.assertEqual(RoleConfig.from_metadata(m),r)
  for k,v in [('schema_version','ebm_role_v3'),('spoof_bpl_weight',1),('pair_reduction','mean_all_local_cross_pairs'),('score_bona_transform','center_cosine')]:
   bad=copy.deepcopy(m);bad[k]=v
   with self.assertRaises(ValueError):RoleConfig.from_metadata(bad)
  bad=copy.deepcopy(m);del bad['spoof_bpl_weight']
  with self.assertRaises(ValueError):RoleConfig.from_metadata(bad)

 def test_invalid_weights_and_ambiguous_center_pairing_rejected(self):
  for v in [-1,float('nan'),float('inf'),True,None,'0']:
   with self.assertRaises(ValueError):RoleConfig(spoof_bpl_weight=v)
  with self.assertRaises(ValueError):RoleConfig(compact_label=1,spoof_bpl_weight=0)
  with self.assertRaises(ValueError):RoleConfig(spoof_bpl_pairing='same_uid',spoof_bpl_weight=0)

 def test_full_partial_formula_original_normal_BPL_and_gradients_exact(self):
  for dtype in [torch.float32,torch.float64]:
   for b,s in [(12,108),(12,60),(6,108)]:
    m=head().to(dtype=dtype);m.encode=lambda x,mask=None:x
    x=torch.randn(b+s,6,dtype=dtype,requires_grad=True);y=torch.randn_like(x,requires_grad=True);labels=torch.tensor([1]*b+[0]*s)
    with patch.object(m.loss_bpl,'forward',wraps=m.loss_bpl.forward) as pair:
     loss,stats=m(x,y,labels,phase='backend')
     self.assertEqual(pair.call_count,1);self.assertEqual(len(pair.call_args.args[0]),b)
    ref=m.loss.backend_loss(x,labels)+OriginalBPL(30)(x[:b],y[:b])
    self.assertTrue(torch.equal(loss,ref));self.assertEqual(stats['pair_count'],0);self.assertEqual(float(stats['pair_spoof']),0);self.assertEqual(stats['bona_pair_count'],b*b)
    actual=torch.autograd.grad(loss,(x,y),retain_graph=True);expected=torch.autograd.grad(ref,(x,y))
    self.assertTrue(all(torch.equal(a,e) for a,e in zip(actual,expected)))
    self.assertGreater(float(actual[1][:b].norm()),0);self.assertEqual(float(actual[1][b:].norm()),0)
    self.assertGreater(float(actual[0][b:].norm()),0)  # Spoof long still learns through OC.

 def test_finite_difference_of_bona_only_complete_objective(self):
  m=head().double();m.encode=lambda x,mask=None:x
  x=torch.randn(6,6,dtype=torch.float64,requires_grad=True);y=torch.randn_like(x,requires_grad=True);labels=torch.tensor([1,1,0,0,0,0])
  self.assertTrue(torch.autograd.gradcheck(lambda a,b:m(a,b,labels,phase='backend')[0],(x,y),eps=1e-6,atol=1e-5,rtol=1e-4))

 def test_real_backend_zero_spoof_short_and_nonzero_bona_short_gradient(self):
  for b,s in [(12,108),(12,60),(6,108)]:
   m=head();x=torch.randn(b+s,2,5,4);y=torch.randn(b+s,2,4,4,requires_grad=True)
   loss,_=m(x,y,torch.tensor([1]*b+[0]*s),phase='backend');g=torch.autograd.grad(loss,y)[0]
   self.assertGreater(float(g[:b].norm()),0);self.assertEqual(float(g[b:].norm()),0)

 def test_weight1_is_exact_old_forward_gradients_Adam_and_rng(self):
  for role in [RoleConfig(),RoleConfig(bona_bpl_weight=1),RoleConfig(bona_bpl_weight=1,spoof_bpl_pairing='same_uid'),RoleConfig(bona_bpl_weight=1,bona_bpl_pairing='same_uid')]:
   torch.manual_seed(1);old=OldHead(role,2,4,6,4);state=torch.get_rng_state().clone()
   torch.manual_seed(1);new=RoleBLinear(role,2,4,6,4);self.assertTrue(torch.equal(state,torch.get_rng_state()))
   a=torch.optim.Adam(old.parameters(),lr=1e-5,weight_decay=1e-4);b=torch.optim.Adam(new.parameters(),lr=1e-5,weight_decay=1e-4)
   for i,(nb,ns) in enumerate([(12,108),(12,60),(6,108)]):
    x=torch.randn(nb+ns,2,5,4);y=torch.randn(nb+ns,2,4,4);labels=torch.tensor([1]*nb+[0]*ns)
    results=[step_hidden(m,opt,x,y,labels,rng=lambda p:phase_rng('cpu',1,1,i,0,p)) for m,opt in [(old,a),(new,b)]]
    assert_nested_equal(self,*results);assert_nested_equal(self,old.state_dict(),new.state_dict())
   for p,q in zip(old.parameters(),new.parameters()):assert_nested_equal(self,a.state[p],b.state[q])

 def test_zero_weight_matches_independent_reference_three_Adam_steps(self):
  role=RoleConfig(bona_bpl_weight=1,spoof_bpl_weight=0)
  torch.manual_seed(1);a=head();state=torch.get_rng_state().clone()
  torch.manual_seed(1);b=ReferenceHead(role,2,4,6,4);self.assertTrue(torch.equal(state,torch.get_rng_state()))
  opts=[torch.optim.Adam(m.parameters(),lr=1e-5,weight_decay=1e-4) for m in [a,b]]
  for i,(nb,ns) in enumerate([(12,108),(12,60),(6,108)]):
   x=torch.randn(nb+ns,2,5,4);y=torch.randn(nb+ns,2,4,4);labels=torch.tensor([1]*nb+[0]*ns)
   rows=[]
   for m,opt in zip([a,b],opts):
    phases=[]
    def audit(phase,h,loss):
     phases.append(phase)
     if phase=='backend':self.assertIsNone(h.loss.weight.grad)
     else:self.assertTrue(all(p.grad is None for n,p in h.named_parameters() if n!='loss.weight'))
    rows.append(step_hidden(m,opt,x,y,labels,rng=lambda p:phase_rng('cpu',1,1,i,0,p),observe=audit));self.assertEqual(phases,['backend','center'])
   for k in ['loss1','loss2','backend_gradient_norm','center_gradient_norm','center_update_norm']:self.assertEqual(rows[0][k],rows[1][k])
   assert_nested_equal(self,a.state_dict(),b.state_dict());self.assertEqual(rows[0]['optimizer_steps'],2);self.assertEqual(rows[0]['center_updates'],1)
  for p,q in zip(a.parameters(),b.parameters()):assert_nested_equal(self,opts[0].state[p],opts[1].state[q])

 def test_disabled_objective_preserves_initial_features_OC_bona_and_RNG(self):
  torch.manual_seed(1);old=head(1);state=torch.get_rng_state().clone();torch.manual_seed(1);new=head()
  self.assertTrue(torch.equal(state,torch.get_rng_state()));assert_nested_equal(self,old.state_dict(),new.state_dict())
  labels=torch.tensor([1]*12+[0]*108);x=torch.randn(120,2,5,4);y=torch.randn(120,2,4,4)
  values=[];states=[]
  for m in [old,new]:
   with phase_rng('cpu',1,1,1,0,'backend'):
    values.append(m(x,y,labels,phase='backend'));states.append(torch.get_rng_state().clone())
  self.assertTrue(torch.equal(*states));self.assertTrue(torch.equal(values[0][1]['oc'],values[1][1]['oc']))
  self.assertTrue(torch.equal(values[0][1]['pair_bona'],values[1][1]['pair_bona']))
  torch.testing.assert_close(values[0][0]-values[1][0],values[0][1]['pair_spoof'],rtol=1e-6,atol=1e-5)

 def test_v6_save_restore_Adam_and_all_three_scoring_phases(self):
  with tempfile.TemporaryDirectory() as folder:
   t=make_trainer(folder);t.classifier=head();t.optimizer=torch.optim.Adam(t.classifier.parameters(),lr=1e-5)
   t.lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(t.optimizer,150)
   step_hidden(t.classifier,t.optimizer,torch.randn(8,2,5,4),torch.randn(8,2,4,4),torch.tensor([1]*3+[0]*5))
   t.checkpoints=life.RoleCheckpoints(t);t.checkpoints.save(5);loaded,_=life.load_scoring_head(folder,5)
   self.assertEqual(loaded.role,RoleConfig(bona_bpl_weight=1,spoof_bpl_weight=0));assert_nested_equal(self,loaded.state_dict(),t.classifier.state_dict())
   wrong=copy.deepcopy(t.checkpoints.expected);wrong['role']=RoleConfig(bona_bpl_weight=1).metadata()
   with self.assertRaises(ValueError):life.read_checkpoint(folder,5,wrong)
   for p in ['19LA','21LA','21DF']:life.evaluate(t,p,5,feature_fn=fake_features)

 def test_smoke_receipt_demands_zero_spoof_and_positive_bona_short_gradients(self):
  args=runtime.options(bona_bpl_weight=1,spoof_bpl_weight=0,final_eval_policy='deferred_shared')
  receipt=dict(status='passed',signature=runtime.signature(args),world_size=2,trained_batches=[1,106,108],eval_phases=['19LA'],iterator_policy=args['iterator_policy'],loader_batches_seen=108,data_fingerprint='x',dual_bpl_graph_checked=True,spoof_bpl_disabled_graph_checked=True,ranks=[dict(values=[dict(bona_pair_count=b,pair_count=0,pair_spoof=0.,bona_short_gradient_norm=1.,spoof_short_gradient_norm=0.) for b in [144,144,36]]) for _ in range(2)])
  with tempfile.TemporaryDirectory() as folder:
   p=Path(folder)/'receipt.json';p.write_text(json.dumps(receipt));runtime.validate_smoke_receipt(args,p)
   for k,v in [('pair_count',11664),('pair_spoof',1.),('spoof_short_gradient_norm',1e-20),('bona_short_gradient_norm',0.)]:
    bad=copy.deepcopy(receipt);bad['ranks'][0]['values'][0][k]=v;p.write_text(json.dumps(bad))
    with self.assertRaises(ValueError):runtime.validate_smoke_receipt(args,p)
   bad=copy.deepcopy(receipt);del bad['spoof_bpl_disabled_graph_checked'];p.write_text(json.dumps(bad))
   with self.assertRaises(ValueError):runtime.validate_smoke_receipt(args,p)

 def test_CLI_explicit_loss_weights_and_fixed_comparison_contract(self):
  p=subprocess.run([sys.executable,str(ROOT/'tools/train_role_experiment.py'),'--spoof-bpl-weight','0','--bona-bpl-weight','1','--bona-bpl-pairing','all_cross','--pair-target-cosine','1','--r-pull','.9','--eval-input','front64600_wrap','--final-eval-policy','deferred_shared'],capture_output=True,text=True,timeout=60)
  self.assertEqual(p.returncode,0,p.stderr);d=json.loads(p.stdout);self.assertFalse(d['launched']);self.assertEqual(d['args']['role'],RoleConfig(bona_bpl_weight=1,spoof_bpl_weight=0).metadata())
  old=runtime.options(bona_bpl_weight=1,final_eval_policy='deferred_shared',eval_input='front64600_wrap')
  self.assertEqual({k:d['args'][k] for k in old if k!='role'},{k:v for k,v in old.items() if k!='role'})

 def test_two_CPU_ranks_asymmetric_full_partial_and_checkpoint(self):
  with tempfile.TemporaryDirectory() as folder:
   torch.multiprocessing.spawn(ddp_worker,args=((Path(folder)/'rendezvous').as_uri(),str(Path(folder)/'out')),nprocs=2,join=True)

if __name__=='__main__':unittest.main(verbosity=2)
