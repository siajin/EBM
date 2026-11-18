"""Spoof aligned-pair BPL with bona all-cross: formula, gradients and artifacts."""
import copy, datetime, importlib.util, json, subprocess, sys, tempfile, unittest
from pathlib import Path
from torch import nn
import torch
import torch.nn.functional as F
from role_objectives import RoleConfig
from model.loss import BPL as OriginalBPL
from role_model import RoleBLinear
from role_training import step_hidden, phase_rng
import role_lifecycle as life
import role_runtime as runtime
from test_role_experiment import make_trainer, assert_nested_equal

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('pre_spoof_uid_role', Path(__file__).parent/'fixtures/role_objectives_before_spoof_uid.py')
oracle = importlib.util.module_from_spec(spec); sys.modules[spec.name] = oracle; spec.loader.exec_module(oracle)

def head(uid=True):
    return RoleBLinear(RoleConfig(bona_bpl_weight=1, spoof_bpl_pairing='same_uid' if uid else 'all_cross'),2,4,6,4)

def distributed_worker(rank,rendezvous,folder):
    torch.set_num_threads(1)
    torch.distributed.init_process_group('gloo',init_method=rendezvous,rank=rank,world_size=2,timeout=datetime.timedelta(seconds=90))
    try:
        torch.manual_seed(23); t=make_trainer(folder)
        t.classifier=nn.parallel.DistributedDataParallel(head(),find_unused_parameters=True,static_graph=False)
        t.optimizer=torch.optim.Adam(t.classifier.parameters(),lr=1e-5,weight_decay=1e-4)
        t.lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(t.optimizer,150)
        for step,(b,s) in enumerate([(12,108),(12,60),(6,108)] if rank==0 else [(12,108),(6,108),(12,60)]):
            torch.manual_seed(100+rank*10+step)
            r=step_hidden(t.classifier,t.optimizer,torch.randn(b+s,2,5,4),torch.randn(b+s,2,4,4),torch.tensor([1]*b+[0]*s))
            assert r['bona_pair_count']==b*b and r['pair_count']==s
        for p in t.classifier.parameters():
            copies=[torch.empty_like(p) for _ in range(2)]; torch.distributed.all_gather(copies,p)
            assert torch.equal(copies[0],copies[1]); assert t.optimizer.state[p]['step'].item()==3
        t.checkpoints=life.RoleCheckpoints(t); t.checkpoints.save(5)
        loaded,_=life.load_scoring_head(folder,5)
        assert loaded.role==RoleConfig(bona_bpl_weight=1,spoof_bpl_pairing='same_uid')
    finally:
        torch.distributed.destroy_process_group()

class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_v1_to_v4_metadata_exact_and_v5_strict_roundtrip(self):
        for kw in [{},{'pair_target_cosine':.9,'r_pull':.8},{'bona_bpl_weight':1},{'bona_bpl_weight':1,'bona_bpl_pairing':'same_uid'}]:
            old=oracle.RoleConfig(**kw).metadata()
            self.assertEqual(RoleConfig(**kw).metadata(),old)
            self.assertEqual(RoleConfig.from_metadata(old),RoleConfig(**kw))
        for extra in [{},{'bona_bpl_weight':1},{'bona_bpl_weight':1,'bona_bpl_pairing':'same_uid'}]:
            role=RoleConfig(spoof_bpl_pairing='same_uid',**extra); m=role.metadata()
            self.assertEqual(m['schema_version'],'ebm_role_v5'); self.assertEqual(RoleConfig.from_metadata(m),role)
            self.assertEqual(m['pair_reduction'],'mean_aligned_same_uid')
            for key,value in [('schema_version','ebm_role_v4'),('spoof_bpl_pairing','all_cross'),('pair_reduction','mean_all_local_cross_pairs')]:
                bad=copy.deepcopy(m); bad[key]=value
                with self.assertRaises(ValueError): RoleConfig.from_metadata(bad)
            bad=copy.deepcopy(m); del bad['spoof_bpl_pairing']
            with self.assertRaises(ValueError): RoleConfig.from_metadata(bad)
        m=RoleConfig(bona_bpl_weight=1,spoof_bpl_pairing='same_uid').metadata()
        self.assertEqual(m['additional_bona_bpl']['reduction'],'mean_all_local_cross_pairs')

    def test_invalid_spoof_pairing_and_wrong_center_rejected(self):
        for value in ['different_uid',None,[],1,True]:
            with self.assertRaises(ValueError): RoleConfig(spoof_bpl_pairing=value)
        with self.assertRaises(ValueError): RoleConfig(compact_label=1,spoof_bpl_pairing='same_uid')

    def test_independent_formula_and_gradients_for_full_and_partial_batches(self):
        for b,s in [(12,108),(12,60),(6,108)]:
            m=head().double(); m.encode=lambda x,mask=None:x
            x=torch.randn(b+s,6,dtype=torch.float64,requires_grad=True); y=torch.randn_like(x,requires_grad=True)
            labels=torch.tensor([1]*b+[0]*s); loss,stats=m(x,y,labels,phase='backend')
            a=F.normalize(x,dim=1); c=F.normalize(y,dim=1)
            ls=(F.softplus(30*(1-(a[b:]*c[b:]).sum(1)))-.3).mean()
            lb=(F.softplus(30*(1-a[:b]@c[:b].T))-.3).mean()
            reference=m.loss.backend_loss(x,labels)+ls+lb
            torch.testing.assert_close(loss,reference,rtol=1e-13,atol=1e-13)
            actual=torch.autograd.grad(loss,(x,y),retain_graph=True)
            expected=torch.autograd.grad(reference,(x,y))
            for v,w in zip(actual,expected): torch.testing.assert_close(v,w,rtol=1e-12,atol=1e-12)
            self.assertEqual(stats['pair_count'],s); self.assertEqual(stats['bona_pair_count'],b*b)

    def test_complete_objective_finite_differences(self):
        m=head().double(); m.encode=lambda x,mask=None:x
        x=torch.randn(6,6,dtype=torch.float64,requires_grad=True); y=torch.randn_like(x,requires_grad=True)
        labels=torch.tensor([1]*2+[0]*4)
        self.assertTrue(torch.autograd.gradcheck(lambda a,b:m(a,b,labels,phase='backend')[0],(x,y),eps=1e-6,atol=1e-5,rtol=1e-4))

    def test_normal_term_exact_original_EBM_value_and_gradients(self):
        for dtype in [torch.float32,torch.float64]:
            for b in [12,6]:
                m=head().to(dtype=dtype)
                x=torch.randn(b,6,dtype=dtype,requires_grad=True); y=torch.randn_like(x,requires_grad=True)
                actual=m.loss_bpl(x,y); original=OriginalBPL(alpha=30)(x,y)
                self.assertTrue(torch.equal(actual,original))
                a=torch.autograd.grad(actual,(x,y),retain_graph=True); o=torch.autograd.grad(original,(x,y))
                self.assertTrue(all(torch.equal(v,w) for v,w in zip(a,o)))

    def test_cross_uid_spoofs_remain_distinct_at_zero_alignment_gradient(self):
        # Distinct spoof vectors can attain the aligned BPL minimum. All-cross
        # still penalizes this construction; this is the removed coupling.
        m=head().double(); m.encode=lambda x,mask=None:x
        z=torch.eye(6,dtype=torch.float64,requires_grad=True)
        short=z.detach().clone().requires_grad_(); labels=torch.tensor([1,1,0,0,0,0])
        _,stats=m(z,short,labels,phase='backend')
        spoof=m.loss_bpl.same_uid(z[2:],short[2:]); cross=m.loss_bpl(z[2:],short[2:])
        self.assertAlmostEqual(float(spoof.detach()),float(F.softplus(torch.tensor(0.)).item()-.3),places=6)
        self.assertGreater(float(cross.detach()),float(spoof.detach())+1)
        gradients=torch.autograd.grad(spoof,(z,short))
        self.assertTrue(all(float(g.norm())==0 for g in gradients))
        self.assertEqual(stats['pair_count'],4)
        self.assertGreater(float((m.loss_bpl.same_uid(z[2:],short[2:].roll(1,0))-spoof).detach()),1)
        torch.testing.assert_close(m.loss_bpl(z[:2],short[:2]),m.loss_bpl(z[:2],short[:2].roll(1,0)))

    def test_initialization_rng_bona_and_oc_exact_against_allcross(self):
        torch.manual_seed(1); old=head(False); state=torch.get_rng_state().clone()
        torch.manual_seed(1); new=head(); self.assertTrue(torch.equal(state,torch.get_rng_state()))
        assert_nested_equal(self,old.state_dict(),new.state_dict())
        for b,s in [(12,108),(12,60),(6,108)]:
            x=torch.randn(b+s,2,5,4); y=torch.randn(b+s,2,4,4); labels=torch.tensor([1]*b+[0]*s)
            with phase_rng('cpu',1,1,1,0,'backend'): _,a=old(x,y,labels,phase='backend')
            with phase_rng('cpu',1,1,1,0,'backend'): _,c=new(x,y,labels,phase='backend')
            self.assertTrue(torch.equal(a['oc'],c['oc'])); self.assertTrue(torch.equal(a['pair_bona'],c['pair_bona']))
            self.assertEqual(c['pair_count'],s); self.assertEqual(c['bona_pair_count'],b*b)
            with phase_rng('cpu',1,1,1,0,'backend'):
                z=new.encode(x); zs=new.encode(y)
                spoof=new.loss_bpl.same_uid(z[b:],zs[b:]); bona=new.loss_bpl(z[:b],zs[:b])
                gs=torch.autograd.grad(spoof,zs,retain_graph=True)[0]; gb=torch.autograd.grad(bona,zs)[0]
            self.assertGreater(float(gs[b:].norm()),0); self.assertEqual(float(gs[:b].norm()),0)
            self.assertGreater(float(gb[:b].norm()),0); self.assertEqual(float(gb[b:].norm()),0)

    def test_two_phases_and_v5_checkpoint_with_Adam_restore(self):
        with tempfile.TemporaryDirectory() as folder:
            t=make_trainer(folder); t.classifier=head(); t.optimizer=torch.optim.Adam(t.classifier.parameters(),lr=1e-5)
            t.lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(t.optimizer,150)
            for b,s in [(12,108),(12,60),(6,108)]:
                phases=[]
                def observe(phase,h,loss):
                    if phase=='backend': self.assertIsNone(h.loss.weight.grad)
                    else: self.assertTrue(all(p.grad is None for n,p in h.named_parameters() if n!='loss.weight'))
                    phases.append(phase)
                r=step_hidden(t.classifier,t.optimizer,torch.randn(b+s,2,5,4),torch.randn(b+s,2,4,4),torch.tensor([1]*b+[0]*s),observe=observe)
                self.assertEqual(phases,['backend','center']); self.assertEqual(r['optimizer_steps'],2); self.assertEqual(r['center_updates'],1)
            t.checkpoints=life.RoleCheckpoints(t); t.checkpoints.save(5)
            loaded,_=life.load_scoring_head(folder,5); assert_nested_equal(self,loaded.state_dict(),t.classifier.state_dict())
            self.assertEqual(loaded.role,RoleConfig(bona_bpl_weight=1,spoof_bpl_pairing='same_uid'))
            self.assertTrue(all(int(s['step'])==3 for s in t.optimizer.state.values()))

    def test_smoke_requires_new_pair_counts_and_both_short_gradients(self):
        args=runtime.options(bona_bpl_weight=1,spoof_bpl_pairing='same_uid',final_eval_policy='deferred_shared')
        d=dict(status='passed',signature=runtime.signature(args),world_size=2,trained_batches=[1,106,108],eval_phases=['19LA'],iterator_policy=args['iterator_policy'],loader_batches_seen=108,data_fingerprint='x',dual_bpl_graph_checked=True,spoof_uid_graph_checked=True,
               ranks=[dict(values=[dict(bona_pair_count=b,pair_count=s,bona_short_gradient_norm=1.,spoof_short_gradient_norm=1.) for b,s in [(144,108),(144,60),(36,108)]]) for _ in range(2)])
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'receipt.json'; p.write_text(json.dumps(d)); runtime.validate_smoke_receipt(args,p)
            for field,value in [('pair_count',11664),('bona_pair_count',12),('spoof_short_gradient_norm',0.),('bona_short_gradient_norm',0.)]:
                bad=copy.deepcopy(d); bad['ranks'][0]['values'][0][field]=value; p.write_text(json.dumps(bad))
                with self.assertRaises(ValueError): runtime.validate_smoke_receipt(args,p)
            bad=copy.deepcopy(d); del bad['spoof_uid_graph_checked']; p.write_text(json.dumps(bad))
            with self.assertRaises(ValueError): runtime.validate_smoke_receipt(args,p)

    def test_CLI_records_both_class_pairing_modes(self):
        p=subprocess.run([sys.executable,str(ROOT/'tools/train_role_experiment.py'),'--spoof-bpl-pairing','same_uid','--bona-bpl-weight','1','--bona-bpl-pairing','all_cross','--final-eval-policy','deferred_shared','--eval-input','front64600_wrap'],capture_output=True,text=True,timeout=60)
        self.assertEqual(p.returncode,0,p.stderr); d=json.loads(p.stdout); self.assertFalse(d['launched'])
        self.assertEqual(d['args']['role'],RoleConfig(bona_bpl_weight=1,spoof_bpl_pairing='same_uid').metadata())

    def test_two_CPU_ranks_asymmetric_partials(self):
        with tempfile.TemporaryDirectory() as folder:
            torch.multiprocessing.spawn(distributed_worker,args=((Path(folder)/'rendezvous').as_uri(),str(Path(folder)/'out')),nprocs=2,join=True)

if __name__=='__main__': unittest.main(verbosity=2)
