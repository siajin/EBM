"""Additional bona all-cross BPL: isolated class means, phases and artifacts."""
import copy,datetime,importlib.util,json,os
from pathlib import Path
import subprocess,sys,tempfile,unittest
import torch
import torch.nn.functional as F
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from role_objectives import RoleConfig
from role_model import RoleBLinear,config_from_model
from role_training import step_hidden,phase_rng
import role_lifecycle as life
import role_runtime as runtime
from test_role_experiment import make_trainer,fake_features,assert_nested_equal

spec=importlib.util.spec_from_file_location('dual_legacy_oracle',Path(__file__).parent/'fixtures/role_model_before_dual_bpl.py')
oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)
class OracleHead(RoleBLinear):
    forward=oracle.RoleBLinear.forward

def head(weight=1.):return RoleBLinear(RoleConfig(bona_bpl_weight=weight),2,4,6,4)

def distributed_worker(rank,rendezvous,folder):
    torch.set_num_threads(1)
    torch.distributed.init_process_group('gloo',init_method=rendezvous,rank=rank,world_size=2,timeout=datetime.timedelta(seconds=90))
    try:
        torch.manual_seed(19);t=make_trainer(folder,distributed=False)
        torch.manual_seed(19);t.classifier=nn.parallel.DistributedDataParallel(head(),find_unused_parameters=True,static_graph=False)
        t.optimizer=torch.optim.Adam(t.classifier.parameters(),lr=1e-5,weight_decay=1e-4)
        t.lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(t.optimizer,150)
        for step,(b,s) in enumerate(([(12,108),(12,60),(6,108)] if rank==0 else [(12,108),(6,108),(12,60)])):
            torch.manual_seed(100+rank*10+step)
            result=step_hidden(t.classifier,t.optimizer,torch.randn(b+s,2,5,4),torch.randn(b+s,2,4,4),torch.tensor([1]*b+[0]*s))
            assert result['bona_pair_count']==b*b and result['pair_count']==s*s
        for p in t.classifier.parameters():
            copies=[torch.empty_like(p) for _ in range(2)];torch.distributed.all_gather(copies,p)
            assert torch.equal(copies[0],copies[1]);assert t.optimizer.state[p]['step'].item()==3
        t.checkpoints=life.RoleCheckpoints(t);t.checkpoints.save(5)
        loaded,_=life.load_scoring_head(folder,5);assert loaded.role.bona_bpl_weight==1
    finally:torch.distributed.destroy_process_group()

class DualBPLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_schema_v3_roundtrip_and_legacy_defaults_exact(self):
        for target,version in [(1.,'ebm_role_v1'),(.9,'ebm_role_v2')]:
            old=RoleConfig(pair_target_cosine=target).metadata()
            self.assertEqual(old['schema_version'],version);self.assertNotIn('bona_bpl_weight',old);self.assertNotIn('additional_bona_bpl',old)
            self.assertEqual(RoleConfig.from_metadata(old).bona_bpl_weight,0.)
        m=RoleConfig(bona_bpl_weight=1).metadata();self.assertEqual(m['schema_version'],'ebm_role_v3')
        self.assertEqual(RoleConfig.from_metadata(m),RoleConfig(bona_bpl_weight=1))
        for field,value in [('bona_bpl_weight',0),('schema_version','ebm_role_v1'),('pair_class',1)]:
            bad=copy.deepcopy(m);bad[field]=value
            with self.assertRaises(ValueError):RoleConfig.from_metadata(bad)
        bad=copy.deepcopy(m);bad['additional_bona_bpl']['aggregation']='pool_all_pairs'
        with self.assertRaises(ValueError):RoleConfig.from_metadata(bad)

    def test_invalid_weight_or_duplicate_bona_role_rejected(self):
        for v in [-1,float('inf'),float('nan'),True]:
            with self.assertRaises(ValueError):RoleConfig(bona_bpl_weight=v)
        with self.assertRaises(ValueError):RoleConfig(compact_label=1,bona_bpl_weight=1)

    def test_both_class_means_and_gradients_match_independent_formula(self):
        for b,s in [(12,108),(12,60),(6,108)]:
            m=head(.7).double();m.encode=lambda x,mask=None:x
            x=torch.randn(b+s,6,dtype=torch.float64,requires_grad=True);y=torch.randn_like(x,requires_grad=True);labels=torch.tensor([1]*b+[0]*s)
            loss,stats=m(x,y,labels,phase='backend')
            a=F.normalize(x,dim=1);c=F.normalize(y,dim=1)
            ls=(F.softplus(30*(1-a[b:]@c[b:].T))-.3).mean()
            lb=(F.softplus(30*(1-a[:b]@c[:b].T))-.3).mean()
            ref=m.loss.backend_loss(x,labels)+ls+.7*lb
            self.assertTrue(torch.equal(loss,ref));self.assertEqual(stats['bona_pair_count'],b*b);self.assertEqual(stats['pair_count'],s*s)
            actual=torch.autograd.grad(loss,(x,y),retain_graph=True);expected=torch.autograd.grad(ref,(x,y))
            # Independent normalization/reduction order differs by <=1e-16 in float64.
            for actual_grad,expected_grad in zip(actual,expected):torch.testing.assert_close(actual_grad,expected_grad,rtol=1e-14,atol=1e-14)
            pooled=(s*s*ls+b*b*lb)/(s*s+b*b)
            self.assertFalse(torch.isclose(ls+.7*lb,pooled))

    def test_both_embedding_gradients_pass_finite_differences(self):
        m=head().double();m.encode=lambda x,mask=None:x
        x=torch.randn(8,6,dtype=torch.float64,requires_grad=True);y=torch.randn_like(x,requires_grad=True)
        labels=torch.tensor([1]*3+[0]*5)
        self.assertTrue(torch.autograd.gradcheck(lambda a,b:m(a,b,labels,phase='backend')[0],(x,y),eps=1e-6,atol=1e-5,rtol=1e-4))

    def test_zero_weight_matches_frozen_forward_gradients_parameters_Adam(self):
        torch.manual_seed(1);old=OracleHead(RoleConfig(),2,4,6,4)
        torch.manual_seed(1);new=head(0)
        self.assertTrue(all(torch.equal(v,new.state_dict()[k]) for k,v in old.state_dict().items()))
        optimizers=[torch.optim.Adam(m.parameters(),lr=1e-5,weight_decay=1e-4) for m in [old,new]]
        for step,(b,s) in enumerate([(12,108),(12,60),(6,108)]):
            long=torch.randn(b+s,2,5,4);short=torch.randn(b+s,2,4,4);labels=torch.tensor([1]*b+[0]*s)
            results=[]
            for model,opt in zip([old,new],optimizers):
                results.append(step_hidden(model,opt,long,short,labels,rng=lambda p:phase_rng('cpu',1,1,step,0,p)))
            assert_nested_equal(self,results[0],results[1])
            self.assertTrue(all(torch.equal(v,new.state_dict()[k]) for k,v in old.state_dict().items()))
        for p,q in zip(old.parameters(),new.parameters()):assert_nested_equal(self,optimizers[0].state[p],optimizers[1].state[q])

    def test_initialization_and_score_do_not_depend_on_added_loss(self):
        torch.manual_seed(1);a=head(0);torch.manual_seed(1);b=head(1)
        self.assertTrue(all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items()))
        x=torch.randn(8,2,5,4)
        a.eval();b.eval()
        with phase_rng('cpu',1,0,0,0,'test'):s=a(x,phase='score')['score_bona']
        with phase_rng('cpu',1,0,0,0,'test'):t=b(x,phase='score')['score_bona']
        self.assertTrue(torch.equal(s,t))

    def test_bona_short_receives_gradient_only_when_enabled(self):
        labels=torch.tensor([1]*12+[0]*108);a=torch.randn(120,2,5,4)
        for weight in [0,1]:
            torch.manual_seed(1);m=head(weight);b=torch.randn(120,2,4,4,requires_grad=True)
            loss,_=m(a,b,labels,phase='backend');g=torch.autograd.grad(loss,b)[0]
            self.assertEqual(float(g[:12].norm())==0,weight==0)

    def test_two_phase_contract_and_partial_counts(self):
        for b,s in [(12,108),(12,60),(6,108)]:
            m=head();opt=torch.optim.Adam(m.parameters(),lr=1e-5);observed=[]
            def observe(phase,model,loss):
                observed.append(phase)
                for p in model.parameters():
                    if (p is model.loss.weight)==(phase=='backend'):self.assertIsNone(p.grad)
            r=step_hidden(m,opt,torch.randn(b+s,2,5,4),torch.randn(b+s,2,4,4),torch.tensor([1]*b+[0]*s),observe=observe)
            self.assertEqual(observed,['backend','center']);self.assertEqual(r['bona_pair_count'],b*b)
            self.assertTrue(all(v['step'].item()==1 for v in opt.state.values()))
        with self.assertRaises(ValueError):m(torch.randn(2,2,5,4),label=torch.ones(2,dtype=torch.long),phase='center')

    def test_new_checkpoint_roundtrip_and_wrong_objective_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            t=make_trainer(folder);t.classifier=head();t.optimizer=torch.optim.Adam(t.classifier.parameters(),lr=1e-5)
            t.lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(t.optimizer,150);t.checkpoints=life.RoleCheckpoints(t);t.checkpoints.save(5)
            loaded,_=life.load_scoring_head(folder,5)
            self.assertEqual(loaded.role.bona_bpl_weight,1)
            self.assertTrue(all(torch.equal(v,loaded.state_dict()[k]) for k,v in t.classifier.state_dict().items()))
            wrong=copy.deepcopy(t.checkpoints.expected);wrong['role']=RoleConfig().metadata()
            with self.assertRaises(ValueError):life.read_checkpoint(folder,5,wrong)
            for phase in ['19LA','21LA','21DF']:life.evaluate(t,phase,5,feature_fn=fake_features)

    def test_smoke_requires_bona_graph_receipt(self):
        args=runtime.options(bona_bpl_weight=1)
        d=dict(status='passed',signature=runtime.signature(args),world_size=2,trained_batches=[1,106,108],eval_phases=['19LA','21LA','21DF'],iterator_policy=args['iterator_policy'],loader_batches_seen=108,data_fingerprint='test')
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'receipt.json';p.write_text(json.dumps(d))
            with self.assertRaises(ValueError):runtime.validate_smoke_receipt(args,p)
            d['dual_bpl_graph_checked']=True;d['ranks']=[dict(values=[dict(bona_pair_count=n,bona_short_gradient_norm=1.) for n in [144,144,36]]) for _ in range(2)]
            p.write_text(json.dumps(d));runtime.validate_smoke_receipt(args,p)
            d['ranks'][1]['values'][2]['bona_short_gradient_norm']=0;p.write_text(json.dumps(d))
            with self.assertRaises(ValueError):runtime.validate_smoke_receipt(args,p)

    def test_actual_CLI_validate_only(self):
        p=subprocess.run([sys.executable,str(ROOT/'tools/train_role_experiment.py'),'--bona-bpl-weight','1','--pair-target-cosine','1','--r-pull','.9','--eval-input','front64600_wrap'],capture_output=True,text=True,timeout=60)
        self.assertEqual(p.returncode,0,p.stderr);d=json.loads(p.stdout);self.assertFalse(d['launched']);self.assertEqual(d['args']['role']['schema_version'],'ebm_role_v3')

    def test_two_CPU_ranks_asymmetric_partials_sync_and_save(self):
        with tempfile.TemporaryDirectory() as folder:
            rendezvous=(Path(folder)/'rendezvous').as_uri()
            torch.multiprocessing.spawn(distributed_worker,args=(rendezvous,str(Path(folder)/'out')),nprocs=2,join=True)

if __name__=='__main__':unittest.main()
