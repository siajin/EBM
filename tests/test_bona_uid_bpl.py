"""Bona-only aligned BPL; spoof all-cross and original Adam phases unchanged."""
import copy,datetime,importlib.util,json,random,subprocess,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
import torch.nn.functional as F
from torch import nn
from role_objectives import RoleConfig,RoleBPL
from role_model import RoleBLinear
from role_training import step_hidden,phase_rng
import role_lifecycle as life
import role_runtime as runtime
from test_role_experiment import make_trainer,assert_nested_equal
from test_role_experiment import fake_features
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('pre_uid_objectives',Path(__file__).parent/'fixtures/role_objectives_before_uid.py')
oracle=importlib.util.module_from_spec(spec);sys.modules[spec.name]=oracle;spec.loader.exec_module(oracle)

def head(uid=True):return RoleBLinear(RoleConfig(bona_bpl_weight=1,bona_bpl_pairing='same_uid' if uid else 'all_cross'),2,4,6,4)

def worker(rank,rendezvous,folder):
    torch.set_num_threads(1)
    torch.distributed.init_process_group('gloo',init_method=rendezvous,rank=rank,world_size=2,timeout=datetime.timedelta(seconds=90))
    try:
        torch.manual_seed(12);t=make_trainer(folder);t.classifier=nn.parallel.DistributedDataParallel(head(),find_unused_parameters=True,static_graph=False)
        t.optimizer=torch.optim.Adam(t.classifier.parameters(),lr=1e-5,weight_decay=1e-4)
        t.lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(t.optimizer,150)
        for step,(b,s) in enumerate([(12,108),(12,60),(6,108)] if rank==0 else [(12,108),(6,108),(12,60)]):
            torch.manual_seed(100+rank*10+step)
            r=step_hidden(t.classifier,t.optimizer,torch.randn(b+s,2,5,4),torch.randn(b+s,2,4,4),torch.tensor([1]*b+[0]*s))
            assert r['bona_pair_count']==b and r['pair_count']==s*s
        for p in t.classifier.parameters():
            copies=[torch.empty_like(p) for _ in range(2)];torch.distributed.all_gather(copies,p);assert torch.equal(copies[0],copies[1])
        t.checkpoints=life.RoleCheckpoints(t);t.checkpoints.save(5)
        loaded,_=life.load_scoring_head(folder,5);assert loaded.role.bona_bpl_pairing=='same_uid'
    finally:torch.distributed.destroy_process_group()

class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_old_metadata_exact_and_v4_roundtrip(self):
        for kw in [{},{'pair_target_cosine':.9,'r_pull':.8},{'bona_bpl_weight':1}]:
            old=oracle.RoleConfig(**kw).metadata();self.assertEqual(RoleConfig(**kw).metadata(),old)
            self.assertEqual(RoleConfig.from_metadata(old),RoleConfig(**kw))
        r=RoleConfig(bona_bpl_weight=1,bona_bpl_pairing='same_uid');m=r.metadata()
        self.assertEqual(m['schema_version'],'ebm_role_v4');self.assertEqual(RoleConfig.from_metadata(m),r)
        self.assertEqual(m['pair_reduction'],'mean_all_local_cross_pairs')
        self.assertEqual(m['additional_bona_bpl']['reduction'],'mean_aligned_same_uid')
        for field,value in [('bona_bpl_pairing','all_cross'),('schema_version','ebm_role_v3')]:
            bad=copy.deepcopy(m);bad[field]=value
            with self.assertRaises(ValueError):RoleConfig.from_metadata(bad)
        bad=copy.deepcopy(m);del bad['bona_bpl_pairing']
        with self.assertRaises(ValueError):RoleConfig.from_metadata(bad)
        bad=copy.deepcopy(m);bad['additional_bona_bpl']['reduction']='mean_all_local_cross_pairs'
        with self.assertRaises(ValueError):RoleConfig.from_metadata(bad)

    def test_invalid_pairing_and_disabled_uid(self):
        for value in ['other',None,[],1]:
            with self.assertRaises(ValueError):RoleConfig(bona_bpl_weight=1,bona_bpl_pairing=value)
        with self.assertRaises(ValueError):RoleConfig(bona_bpl_pairing='same_uid')

    def test_uid_formula_gradcheck_and_permutation(self):
        torch.manual_seed(3);loss=RoleBPL();x=torch.randn(4,6,dtype=torch.float64,requires_grad=True);y=torch.randn_like(x,requires_grad=True)
        matrix=F.normalize(x,dim=1)@F.normalize(y,dim=1).T
        expected=(F.softplus(30*(1-matrix.diag()))-.3).mean()
        torch.testing.assert_close(loss.same_uid(x,y),expected,rtol=1e-12,atol=1e-12)
        self.assertTrue(torch.autograd.gradcheck(loss.same_uid,(x,y)))
        self.assertGreater(float((loss.same_uid(x,y)-loss.same_uid(x,y.roll(1,0))).abs().detach()),1e-3)
        torch.testing.assert_close(loss(x,y),loss(x,y.roll(1,0)),rtol=1e-12,atol=1e-12)
        for a,b in [(torch.zeros(2,3),torch.ones(2,3)),(torch.ones(2,3),torch.ones(3,3))]:
            with self.assertRaises(ValueError):loss.same_uid(a,b)

    def test_identical_initial_tensors_rng_and_only_bona_loss_changes(self):
        torch.manual_seed(1);old=head(False);state=torch.get_rng_state().clone()
        torch.manual_seed(1);new=head();self.assertTrue(torch.equal(state,torch.get_rng_state()))
        assert_nested_equal(self,old.state_dict(),new.state_dict())
        for b,s in [(12,108),(12,60),(6,108)]:
            x=torch.randn(b+s,2,5,4);short=torch.randn(b+s,2,4,4);labels=torch.tensor([1]*b+[0]*s)
            with phase_rng('cpu',1,1,1,0,'backend'):a,sa=old(x,short,labels,phase='backend')
            with phase_rng('cpu',1,1,1,0,'backend'):c,sc=new(x,short,labels,phase='backend')
            self.assertTrue(torch.equal(sa['oc'],sc['oc']));self.assertTrue(torch.equal(sa['pair'],sc['pair']))
            self.assertEqual(sc['bona_pair_count'],b);self.assertEqual(sc['pair_count'],s*s)
            torch.testing.assert_close(c,sc['oc']+sc['pair']+sc['pair_bona'])
            with phase_rng('cpu',1,1,1,0,'backend'):
                z=new.encode(x);zs=new.encode(short);normal=new.loss_bpl.same_uid(z[:b],zs[:b])
                grad=torch.autograd.grad(normal,zs)[0]
            self.assertGreater(float(grad[:b].norm()),0);self.assertEqual(float(grad[b:].norm()),0)

    def test_two_phase_updates_and_checkpoint_restore(self):
        with tempfile.TemporaryDirectory() as folder:
            t=make_trainer(folder);t.classifier=head();t.optimizer=torch.optim.Adam(t.classifier.parameters(),lr=1e-5)
            t.lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(t.optimizer,150)
            for b,s in [(12,108),(12,60),(6,108)]:
                phases=[]
                def observe(phase,h,loss):
                    if phase=='backend':self.assertIsNone(h.loss.weight.grad)
                    else:self.assertTrue(all(p.grad is None for n,p in h.named_parameters() if n!='loss.weight'))
                    phases.append(phase)
                r=step_hidden(t.classifier,t.optimizer,torch.randn(b+s,2,5,4),torch.randn(b+s,2,4,4),torch.tensor([1]*b+[0]*s),observe=observe)
                self.assertEqual(phases,['backend','center']);self.assertEqual(r['optimizer_steps'],2);self.assertEqual(r['center_updates'],1)
            self.assertTrue(all(int(s['step'])==3 for s in t.optimizer.state.values()))
            t.checkpoints=life.RoleCheckpoints(t);t.checkpoints.save(5)
            loaded,_=life.load_scoring_head(folder,5);assert_nested_equal(self,loaded.state_dict(),t.classifier.state_dict())
            self.assertEqual(loaded.role.bona_bpl_pairing,'same_uid')

    def test_uid_smoke_receipt_counts(self):
        args=runtime.options(bona_bpl_weight=1,bona_bpl_pairing='same_uid',final_eval_policy='deferred_shared')
        d=dict(status='passed',signature=runtime.signature(args),world_size=2,trained_batches=[1,106,108],eval_phases=['19LA'],iterator_policy=args['iterator_policy'],loader_batches_seen=108,data_fingerprint='x',dual_bpl_graph_checked=True,
               ranks=[dict(values=[dict(bona_pair_count=n,bona_short_gradient_norm=1.) for n in [12,12,6]]) for _ in range(2)])
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'r.json';p.write_text(json.dumps(d));runtime.validate_smoke_receipt(args,p)
            d['ranks'][0]['values'][0]['bona_pair_count']=144;p.write_text(json.dumps(d))
            with self.assertRaises(ValueError):runtime.validate_smoke_receipt(args,p)

    def test_deferred_scope_only_evaluates_19LA_during_training(self):
        with tempfile.TemporaryDirectory() as folder:
            t=make_trainer(folder);t.args.update(epoch=10,final_eval_policy='deferred_shared')
            visits=dict(batches_seen=108,updates=108,bona_rows=1290,spoof_rows=11616)
            seen=[]
            with patch('role_training.train_epoch',return_value=visits),patch.object(life.RoleCheckpoints,'save'),patch.object(life,'evaluate',side_effect=lambda t,p,e:seen.append((p,e)) or dict(phase=p,epoch=e,eer_percent=1)):
                status=life.run_training(t)
            self.assertEqual(status,'training_complete');self.assertEqual(seen,[('19LA',5),('19LA',10)])
            self.assertEqual(json.loads((Path(folder)/'stages/complete.json').read_text())['status'],'awaiting_shared_evaluation')

    def test_unknown_evaluation_policy_rejected(self):
        with self.assertRaises(ValueError):runtime.options(final_eval_policy='other')

    def test_actual_CLI_v4(self):
        p=subprocess.run([sys.executable,str(ROOT/'tools/train_role_experiment.py'),'--bona-bpl-weight','1','--bona-bpl-pairing','same_uid','--final-eval-policy','deferred_shared','--eval-input','front64600_wrap'],capture_output=True,text=True,timeout=60)
        self.assertEqual(p.returncode,0,p.stderr);d=json.loads(p.stdout);self.assertFalse(d['launched']);self.assertEqual(d['args']['role']['schema_version'],'ebm_role_v4')


    def test_three_evaluations_do_not_change_next_Adam_update_or_RNG(self):
        with tempfile.TemporaryDirectory() as folder:
            torch.manual_seed(71);t=make_trainer(Path(folder)/'eval');t.classifier=head()
            t.optimizer=torch.optim.Adam(t.classifier.parameters(),lr=1e-5,weight_decay=1e-4)
            labels=torch.tensor([1]*3+[0]*5);x=torch.randn(8,2,5,4);s=torch.randn(8,2,4,4)
            step_hidden(t.classifier,t.optimizer,x,s,labels)
            reference=copy.deepcopy(t.classifier);reference_optimizer=torch.optim.Adam(reference.parameters(),lr=1e-5,weight_decay=1e-4)
            reference_optimizer.load_state_dict(copy.deepcopy(t.optimizer.state_dict()))
            t.checkpoints=life.RoleCheckpoints(t);t.checkpoints.save(5)
            random.seed(1);np.random.seed(2);torch.manual_seed(3)
            state=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
            parameters=copy.deepcopy(t.classifier.state_dict());moments=copy.deepcopy(t.optimizer.state_dict())
            for phase in ['19LA','21LA','21DF']:life.evaluate(t,phase,5,feature_fn=fake_features)
            assert_nested_equal(self,t.classifier.state_dict(),parameters);assert_nested_equal(self,t.optimizer.state_dict(),moments)
            self.assertEqual(random.getstate(),state[0]);np.testing.assert_equal(np.random.get_state(),state[1]);self.assertTrue(torch.equal(torch.get_rng_state(),state[2]))
            # Omit phase_rng deliberately: equality also proves evaluation did
            # not shift the next ordinary dropout/ASP random draws.
            step_hidden(t.classifier,t.optimizer,x,s,labels)
            torch.set_rng_state(state[2]);step_hidden(reference,reference_optimizer,x,s,labels)
            assert_nested_equal(self,t.classifier.state_dict(),reference.state_dict())
            assert_nested_equal(self,t.optimizer.state_dict(),reference_optimizer.state_dict())


    def test_two_CPU_ranks_asymmetric_partials(self):
        with tempfile.TemporaryDirectory() as folder:
            torch.multiprocessing.spawn(worker,args=((Path(folder)/'rendezvous').as_uri(),str(Path(folder)/'out')),nprocs=2,join=True)

if __name__=='__main__':unittest.main(verbosity=2)
