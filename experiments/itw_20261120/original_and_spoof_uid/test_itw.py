import copy,importlib.util,os,sys,unittest
from pathlib import Path
import numpy as np
import torch
from role_model import RoleBLinear
from role_objectives import RoleConfig,RoleOCSoftmax
from role_training import phase_rng
from evaluate_itw import wrap,legacy_head,score,independent_eer,calculate_eer

SOURCE=Path(os.environ.get('EBM_ITW_SOURCE_ROOT',str(Path(__file__).resolve().parents[2]/'EBM')))
spec=importlib.util.spec_from_file_location('original_blinear',SOURCE/'baseline240_source/model/b_linear.py')
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)

class Checks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):torch.set_num_threads(1)
 def test_original_wrap_front_and_very_short_inputs(self):
  for n in [1,16,719,64600,64601,100000]:
   a=np.arange(n,dtype=np.float64);out=wrap(a)
   self.assertEqual(len(out),64600);np.testing.assert_array_equal(out,np.resize(a,64600) if n<64600 else a[:64600])
  for a in [np.array([]),np.array([np.nan]),np.zeros((2,2))]:
   with self.assertRaises(AssertionError):wrap(a)
 def test_legacy_adapter_matches_original_BLinear_on_same_noise(self):
  arch=dict(num_layer=2,hidden_size=4,output_size=6,agg_size=4)
  torch.manual_seed(1);new=RoleBLinear(RoleConfig(compact_label=1),**arch).eval().requires_grad_(False)
  old=original.B_Linear(**arch,loss=RoleOCSoftmax(6,RoleConfig(compact_label=1))).eval().requires_grad_(False);old.load_state_dict(new.state_dict(),strict=True)
  for n in [1,2,8]:
   x=torch.randn(n,2,7,4)
   with torch.no_grad(),phase_rng('cpu',1729,0,3,0,'matched_wrap:ITW'):
    z=old.layer_norm(old.linear(old.AHP_ASP(old.bridge_module(x))))
    expected=torch.nn.functional.normalize(z,dim=1)@torch.nn.functional.normalize(old.loss.weight,dim=1).T
   with torch.no_grad(),phase_rng('cpu',1729,0,3,0,'matched_wrap:ITW'):actual=new(x,phase='score')['score_bona']
   self.assertTrue(torch.equal(expected,actual))
   wrapped={'classifier':{'module.'+k:v for k,v in new.state_dict().items()}}
   loaded=legacy_head(wrapped,arch,'cpu')
   self.assertTrue(all(torch.equal(v,loaded.state_dict()[k]) for k,v in new.state_dict().items()))
   with torch.no_grad(),phase_rng('cpu',1729,0,3,0,'matched_wrap:ITW'):got=loaded(x,phase='score')['score_bona']
   torch.testing.assert_close(actual,got,rtol=0,atol=0,msg=lambda msg:f'batch={n}; maxdiff={float((actual-got).abs().max())}; '+msg)
 def test_legacy_loader_rejects_missing_or_extra_weights(self):
  h=RoleBLinear(RoleConfig(compact_label=1),2,4,6,4)
  for state in [{k:v for k,v in h.state_dict().items() if k!='loss.weight'},dict(h.state_dict(),unexpected=torch.ones(1))]:
   with self.assertRaises(RuntimeError):legacy_head(state,h.architecture,'cpu')
 def test_order_independence_and_score_signs_all_role_versions(self):
  roles=[RoleConfig(compact_label=1),RoleConfig(),RoleConfig(r_pull=.8,pair_target_cosine=.9),RoleConfig(bona_bpl_weight=1),RoleConfig(bona_bpl_weight=1,bona_bpl_pairing='same_uid'),RoleConfig(bona_bpl_weight=1,spoof_bpl_pairing='same_uid')]
  torch.manual_seed(8);heads={str(i):RoleBLinear(r,2,4,6,4).eval().requires_grad_(False) for i,r in enumerate(roles)}
  x=torch.randn(8,2,7,4);before=torch.get_rng_state().clone()
  a=score(heads,x,'cpu',0,2);b=score(dict(reversed(list(heads.items()))),x.clone(),'cpu',0,2)
  self.assertEqual(a,b);self.assertTrue(torch.equal(before,torch.get_rng_state()))
  for key,h in heads.items():
   with phase_rng('cpu',1729,0,2,0,'matched_wrap:ITW'):
    z=h(x,phase='embedding');expected=(torch.nn.functional.normalize(z,dim=1)@torch.nn.functional.normalize(h.loss.weight,dim=1).T).flatten()*h.role.score_sign
   np.testing.assert_array_equal(a[key],expected.tolist())
 def test_independent_ROC_ties_and_imbalanced_classes(self):
  for n in [9,100,1001]:
   torch.manual_seed(n);labels=[int(i%3==0) for i in range(n)];scores=torch.randint(-3,4,(n,)).tolist()
   self.assertAlmostEqual(independent_eer(labels,scores),calculate_eer(scores,labels),places=10)
 def test_no_checkpoint_parameter_changes_during_scoring(self):
  h=RoleBLinear(RoleConfig(bona_bpl_weight=1,spoof_bpl_pairing='same_uid'),2,4,6,4).eval().requires_grad_(False)
  before=copy.deepcopy(h.state_dict());score({'m':h},torch.randn(1,2,7,4),'cpu',1,19)
  for k,v in before.items():self.assertTrue(torch.equal(v,h.state_dict()[k]))
  self.assertTrue(all(p.grad is None and not p.requires_grad for p in h.parameters()))

if __name__=='__main__':unittest.main(verbosity=2)
