"""Future v6 shared-evaluation preparation and final result labeling integration."""
import csv,hashlib,json,shutil,tempfile,unittest
from pathlib import Path
import torch
from role_objectives import RoleConfig
from role_model import RoleBLinear
from shared_eval import score_heads
import finalize_bona_only as final
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def make_finished_training(folder,role):
 root=folder/'analysis';result=folder/'result';root.mkdir();(result/'model').mkdir(parents=True);(result/'stages').mkdir()
 (root/'shared_eval.py').write_text('# synthetic shared evaluator fixture\n')
 (root/'experiment.json').write_text(json.dumps(dict(source_sha256={'role_objectives.py':'fake-source-hash'})))
 (result/'stages/training_complete.json').write_text(json.dumps(dict(status='complete',saved_epochs=list(range(5,51,5)),results=[dict(phase='19LA',status='complete',epoch=e) for e in range(5,51,5)])))
 for e in range(5,51,5):
  model=result/'model'/f'CLS_{e}.pt';model.write_bytes(b'synthetic checkpoint fixture')
  (result/'stages'/f'checkpoint_{e}.complete.json').write_text(json.dumps(dict(status='committed',files={model.name:dict(sha256=sha(model))})))
  (result/'model'/f'Model_{e}.json').write_text(json.dumps(dict(role=role.metadata())))
  (result/'stages'/f'eval_{e}_19LA.json').write_text(json.dumps(dict(status='complete',rows=71237,scope='full')))
  (result/'stages'/f'eval_{e}_19LA_scores.csv').write_text('synthetic score fixture\n')
 return root,result

class Checks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):torch.set_num_threads(1)

 def test_v6_prepare_all_ten_heads_and_both_deferred_datasets(self):
  with tempfile.TemporaryDirectory() as folder:
   role=RoleConfig(bona_bpl_weight=1,spoof_bpl_weight=0);root,result=make_finished_training(Path(folder),role)
   final.prepare_shared(root,result);m=json.loads((root/'shared_eval/manifest.json').read_text())
   self.assertEqual(m['expected_role'],role.metadata());self.assertEqual(m['source_root'],str(root/'code'));self.assertEqual(len(m['preserved']),10)
   self.assertEqual(m['pending'],{p:['e'+str(e) for e in range(5,51,5)] for p in ['21LA','21DF']})
   self.assertEqual(len(m['checkpoint_sha256']),10);self.assertEqual(m['frontend_revision'],'1a640f32ac3e39899438a2931f9924c02f080a54')

 def test_wrong_v3_v5_or_normal_UID_model_cannot_enter_this_evaluation(self):
  for role in [RoleConfig(bona_bpl_weight=1),RoleConfig(bona_bpl_weight=1,spoof_bpl_pairing='same_uid'),RoleConfig(bona_bpl_weight=1,bona_bpl_pairing='same_uid',spoof_bpl_weight=0)]:
   with tempfile.TemporaryDirectory() as folder:
    root,result=make_finished_training(Path(folder),role)
    with self.assertRaises(AssertionError):final.prepare_shared(root,result)

 def test_real_v6_shared_scorer_keeps_head_order_and_RNG(self):
  torch.manual_seed(1);a=RoleBLinear(RoleConfig(bona_bpl_weight=1,spoof_bpl_weight=0),2,4,6,4).eval().requires_grad_(False)
  torch.manual_seed(2);b=RoleBLinear(RoleConfig(bona_bpl_weight=1,spoof_bpl_weight=0),2,4,6,4).eval().requires_grad_(False)
  x=torch.randn(8,2,5,4);state=torch.get_rng_state().clone()
  first=score_heads({'e5':a,'e10':b},x,'cpu',0,3,'21DF');other=score_heads({'e10':b,'e5':a},x,'cpu',0,3,'21DF')
  self.assertEqual(first,other);self.assertTrue(torch.equal(state,torch.get_rng_state()))

 def test_full_finalizer_labels_new_model_without_overwriting_prior_UID_results(self):
  # Synthetic scores remain inside a temporary folder; never used as EER evidence.
  with tempfile.TemporaryDirectory() as folder:
   root=Path(folder)/'analysis';result=Path(folder)/'result';root.mkdir();(result/'stages').mkdir(parents=True)
   shutil.copyfile(HERE/'baseline_metrics.json',root/'baseline_metrics.json');full=root/'shared_eval/full';full.mkdir(parents=True)
   (root/'shared_eval/run.exit').write_text('0\n');(full/'complete.json').write_text(json.dumps(dict(status='complete',results=[{}]*30)))
   source=Path(folder)/'synthetic_scores.csv'
   with source.open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['index','uid','label','score_bona'])
    for i in range(71237):w.writerow([i,'synthetic'+str(i),i%2,float(i%2)])
   for e in range(5,51,5):
    shutil.copyfile(source,result/'stages'/f'eval_{e}_19LA_scores.csv')
    (result/'stages'/f'eval_{e}_19LA.json').write_text(json.dumps(dict(status='complete',scope='full',rows=71237,eer_percent=0.)))
   for p,count in [('21LA',148176),('21DF',533928)]:
    (full/f'metrics_{p}.json').write_text(json.dumps(dict(status='complete',scope='full',rows=count,score_files={},metrics={'e'+str(e):dict(EER_percent=0.,independent_error_pp=0.) for e in range(5,51,5)})))
   final.complete(root,result);d=json.loads((root/'comparison.json').read_text());self.assertEqual(len(d['rows']),30)
   for row in d['rows']:
    self.assertEqual(row['bona_only'],0.);self.assertGreater(row['spoof_uid'],0.)
    self.assertEqual(row['bona_only_minus_allcross_pp'],-row['bona_allcross'])
   self.assertEqual(set(d['minima']['19LA']),{'legacy240','spoof_only','relaxed','bona_allcross','bona_uid','spoof_uid','bona_only'})

if __name__=='__main__':unittest.main(verbosity=2)
