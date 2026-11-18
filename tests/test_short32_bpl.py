"""Synthetic CPU checks for the unchanged original240 loaders/train + L32."""
import ast
import argparse
import contextlib
import copy
import datetime
import io
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import soundfile as sf
import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler
import short32_bpl as reference
import short32_original240 as short32
import native_steps

ROOT = Path(__file__).resolve().parents[1]


class SilentProgress:
    def __init__(self, *a, **kw): pass
    def __enter__(self): return self
    def __exit__(self, *a): pass


class Frontend(nn.Module):
    def __init__(self):
        super().__init__(); self.scale = nn.Parameter(torch.ones(())); self.inputs = []
    def forward(self, wave, output_hidden_states=True):
        self.inputs.append(wave.detach().clone())
        x = wave[:, ::8000, None] * self.scale
        x = torch.cat([x, x.square(), x.sin(), x.cos()], dim=-1)
        return SimpleNamespace(hidden_states=(x, x*.7))


class PrefixWrapper(nn.Module):
    def __init__(self, module): super().__init__(); self.module = module
    def forward(self, *a, **kw): return self.module(*a, **kw)


def runtime():
    load = short32.load_source
    def intercept(name, namespace=None, member=None):
        if name == 'main.py' and member == 'run': return lambda *a: namespace
        return load(name, namespace, member)
    with patch.object(short32, 'load_source', side_effect=intercept):
        return short32.run(0, {'epoch':1}, {}, {'torch':torch, 'tqdm':SilentProgress})


def head():
    m = short32.load_source('model/b_linear.py'); l = short32.load_source('model/loss.py')
    return m.B_Linear(2,4,6,4,l.OCSoftmax(6,class_weight=[5/9,5]),l.BPL(30))


def options():
    return dict(batch_size=6,ratio_bona=1/3,ratio_spoof=2/3,num_workers=0,
        num_train_frames=404,num_train_frames_short=303,num_test_frames=404,
        num_seg=1,eval_batch_size=2,epoch=1,T_0=150,device='cpu',flag_parent=False,world_size=1,
        lr=1e-4,weight_decay=1e-4,short32_bpl=True)


def sources(folder,lengths=(32000,32001,16000,64000,80000),spoof=9):
    items=[]
    for i,(label,n) in enumerate([(1,n) for n in lengths]+[(0,16000)]*spoof):
        path=Path(folder)/f'{i}.wav'
        sf.write(path,.1*np.sin(np.arange(n)*(.013+i*.001)),16000,subtype='FLOAT')
        items.append(SimpleNamespace(path=str(path),label=label,type='synthetic',speaker=str(i)))
    return items


def trainer(items,model=None,adapter=True,ddp=False):
    a=options(); data=short32.load_source('data_loaders.py'); dummy=SimpleNamespace(trials=items[:2])
    loaders=data.get_loaders(a,items,items[:2],{s:dummy for s in ['19LA','21LA','21DF']})
    cls=runtime()['train'].ModelTrainer if adapter else short32.load_source(
        'train.py',{'torch':torch,'tqdm':SilentProgress},'ModelTrainer')
    t=cls(); t.args=a;t.da=None;t.logger=None
    t.train_loader_bona,t.sampler_bona,t.train_loader_spoof,t.sampler_spoof,t.val_loader,t.eval_loader=loaders
    t.plm=Frontend().eval()
    t.classifier=nn.parallel.DistributedDataParallel(model or head(),find_unused_parameters=True) if ddp else PrefixWrapper(model or head())
    t.optimizer=torch.optim.Adam(list(t.plm.parameters())+list(t.classifier.parameters()),lr=a['lr'],weight_decay=a['weight_decay'])
    t.lr_step='epoch'
    t.lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(t.optimizer,T_0=150,eta_min=1e-5)
    return t


def ddp_worker(rank,rendezvous):
    import torch.distributed as dist
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=rendezvous,rank=rank,world_size=2,timeout=datetime.timedelta(seconds=45))
    try:
        with tempfile.TemporaryDirectory() as folder:
            items=sources(folder,lengths=(16000,64000,64000,64000),spoof=8)
            torch.manual_seed(77);t=trainer(items,ddp=True);c=short32.Short32Hooks(t)
            stages=[]
            t.optimizer.register_step_post_hook(lambda *a:stages.append(t.classifier.module.loss.weight.requires_grad))
            try:
                with patch('random.randint',return_value=0):t.train(1)
                counts=[None,None];dist.all_gather_object(counts,c.last_stats['eligible'])
                assert sorted(counts)==[0,1]
                assert stages==[False,True]
                assert all(p.grad is None for p in t.plm.parameters())
                for p in t.classifier.parameters():
                    values=[torch.empty_like(p) for _ in range(2)]
                    dist.all_gather(values,p.detach())
                    torch.testing.assert_close(values[0],values[1],rtol=0,atol=0)
            finally:c.close()
    finally:dist.destroy_process_group()


class Original240Short32Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.group=tempfile.TemporaryDirectory()
        torch.distributed.init_process_group('gloo',init_method=Path(cls.group.name,'group').as_uri(),rank=0,world_size=1)
    @classmethod
    def tearDownClass(cls):torch.distributed.destroy_process_group();cls.group.cleanup()
    def setUp(self):
        self.folder=tempfile.TemporaryDirectory();self.addCleanup(self.folder.cleanup)

    def test_frozen_sources_identical_to_git(self):
        manifest=short32.verify_source()
        for name in manifest['files']:
            original=subprocess.check_output(['git','show',manifest['commit']+':'+name],cwd=os.environ.get('EBM_REFERENCE_GIT_ROOT',ROOT))
            self.assertEqual(original,(short32.SOURCE/name).read_bytes())

    def test_runtime_exact_original_train_function(self):
        cls=runtime()['train'].ModelTrainer
        original=short32.load_source('train.py',{'torch':torch,'tqdm':SilentProgress},'ModelTrainer').train
        self.assertEqual(cls.train.__code__,original.__code__)
        self.assertEqual(cls.train.__code__.co_filename,str(short32.SOURCE/'train.py'))

    def test_real_original_loader_classes_triplet_and_rng(self):
        t=trainer(sources(self.folder.name));data=short32.load_source('data_loaders.py')
        loaders=(t.train_loader_bona,t.train_loader_spoof)
        rng=torch.get_rng_state().clone(),random.getstate(),np.random.get_state()
        c=short32.Short32Hooks(t);self.addCleanup(c.close)
        self.assertEqual(loaders,(t.train_loader_bona,t.train_loader_spoof))
        self.assertTrue(torch.equal(rng[0],torch.get_rng_state()));self.assertEqual(rng[1],random.getstate())
        np.testing.assert_array_equal(rng[2][1],np.random.get_state()[1])
        self.assertEqual(c.lengths,[32000,32001,16000,64000,80000])
        with patch('random.randint',return_value=0):
            for loader in loaders:
                self.assertIs(type(loader),DataLoader);self.assertIs(type(loader.dataset),data.TrainSet)
                self.assertIs(type(loader.sampler),DistributedSampler)
                row=next(iter(loader));self.assertEqual(len(row),3)
                self.assertEqual((row[0].shape[1],row[1].shape[1]),(64600,48440))

    def test_zero_eligible_exact_full_epoch_updates_optimizer_and_rng(self):
        items=sources(self.folder.name,lengths=(32001,64000,80000,64000,64000))
        torch.manual_seed(33);initial=head()
        baseline=trainer(items,copy.deepcopy(initial),adapter=False);revised=trainer(items,copy.deepcopy(initial))
        c=short32.Short32Hooks(revised);self.addCleanup(c.close);rngs=[]
        for t in [baseline,revised]:
            torch.manual_seed(51);random.seed(51);np.random.seed(51)
            with patch('random.randint',return_value=0):t.train(1)
            rngs.append(torch.get_rng_state().clone())
        for p,q in zip(baseline.classifier.parameters(),revised.classifier.parameters()):self.assertTrue(torch.equal(p,q))
        self.assertTrue(torch.equal(*rngs))
        for p,q in zip(baseline.optimizer.state.values(),revised.optimizer.state.values()):
            for key in p:self.assertTrue(torch.equal(p[key],q[key]))
        self.assertEqual(c.last_stats['eligible'],0)
        self.assertEqual([x.shape for x in baseline.plm.inputs],[x.shape for x in revised.plm.inputs])

    def test_mixed_original_full_epoch_two_stage_ragged_batches(self):
        t=trainer(sources(self.folder.name));c=short32.Short32Hooks(t);self.addCleanup(c.close)
        stats=[];stages=[]
        def step(*a):
            stages.append(t.classifier.module.loss.weight.requires_grad)
            if len(stages)%2:stats.append(dict(c.last_stats))
        t.optimizer.register_step_post_hook(step)
        frontend=copy.deepcopy(t.plm.state_dict());center=t.classifier.module.loss.weight.detach().clone()
        with patch('random.randint',return_value=0):t.train(1)
        self.assertEqual(stages,[False,True]*3)
        self.assertEqual([s['bona'] for s in stats],[2,2,1])
        self.assertEqual(sum(s['eligible'] for s in stats),2)
        self.assertEqual(sum(s['bpl43_pairs'] for s in stats),9)
        self.assertTrue(all(p.grad is None for p in t.plm.parameters()))
        self.assertTrue(all(torch.equal(v,t.plm.state_dict()[k]) for k,v in frontend.items()))
        self.assertFalse(torch.equal(center,t.classifier.module.loss.weight))

    def test_active_extra_branch_preserves_original_views_and_global_rng(self):
        items=sources(self.folder.name)
        torch.manual_seed(33);initial=head()
        baseline=trainer(items,copy.deepcopy(initial),adapter=False);revised=trainer(items,copy.deepcopy(initial))
        c=short32.Short32Hooks(revised);self.addCleanup(c.close);rngs=[]
        for t in [baseline,revised]:
            torch.manual_seed(51);random.seed(51);np.random.seed(51)
            with patch('random.randint',return_value=0):t.train(1)
            rngs.append(torch.get_rng_state().clone())
        self.assertTrue(torch.equal(*rngs))
        original_views=[x for x in revised.plm.inputs if x.shape[1]!=32000]
        self.assertEqual(len(original_views),len(baseline.plm.inputs))
        for x,y in zip(original_views,baseline.plm.inputs):self.assertTrue(torch.equal(x,y))

    def test_outer_monitoring_saves_then_19la_and_restores_classifier_rng(self):
        from positive_bpl import oc_evaluation_protocol
        t=trainer(sources(self.folder.name,lengths=(16000,32001),spoof=4))
        t.args.update(num_layers=2,hidden_size=4,output_size=6,agg_size=4,alpha_cl=30,
                      process_id=0,evaluation_mode='controlled_oc',evaluation_seed=1729)
        original=t.classifier
        with patch('short32_lifecycle.run_training') as lifecycle:
            t.run()
        lifecycle.assert_called_once_with(t)
        self.assertIs(t.classifier,original)
        expected=oc_evaluation_protocol(t.args,None)
        expected['selection']='19LA monitoring every5 epochs; no dev; fixed training budget'
        self.assertEqual(t.evaluation_protocol,expected)
        self.assertEqual(len(original.module._forward_hooks),0)
        t.checkpoints=object()
        with patch('short32_lifecycle.evaluate',return_value=(0.,0.,0.,0.)) as evaluate:
            t.test('19LA',1)
        evaluate.assert_called_once_with(t,'19LA',1,t.checkpoints)

    def test_loss_weights_and_gradients_independent_pair_matrix(self):
        loss=short32.load_source('model/loss.py').BPL(30)
        matrix=lambda x,y:torch.nn.functional.softplus(30*(1-torch.nn.functional.normalize(x,dim=1)@torch.nn.functional.normalize(y,dim=1).T))-.3
        for k in [0,1,2,3]:
            torch.manual_seed(17)
            x=torch.randn(3,6,dtype=torch.float64,requires_grad=True)
            y=torch.randn(3,6,dtype=torch.float64,requires_grad=True)
            z=torch.randn(k,6,dtype=torch.float64,requires_grad=True)
            actual=loss(x,y) if not k else loss(x,y)+.5*k/3*(loss(y[:k],z)-loss(x[:k],y))
            weights=torch.tensor([.5]*k+[1.]*(3-k),dtype=torch.float64)[:,None]
            expected=(matrix(x,y)*weights).mean()
            if k:expected=expected+.5*k/3*matrix(y[:k],z).mean()
            torch.testing.assert_close(actual,expected,rtol=1e-13,atol=1e-13)
            aa=torch.autograd.grad(actual,(x,y,z),allow_unused=True,retain_graph=True)
            bb=torch.autograd.grad(expected,(x,y,z),allow_unused=True)
            for a,b in zip(aa,bb):
                if a is None:self.assertIsNone(b)
                else:torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)

    def test_actual_cli_validation_exits_before_spawn(self):
        import arguments,positive_bpl
        tree=ast.parse((ROOT/'main.py').read_text(encoding='utf-8'))
        block=next(n for n in tree.body if isinstance(n,ast.If) and ast.unparse(n.test)=="__name__ == '__main__'")
        scope=dict(__name__='__main__',arguments=arguments,argparse=argparse,Path=Path,
                   validate_options=positive_bpl.validate_options,
                   validate_evaluation_options=positive_bpl.validate_evaluation_options,
                   evaluation_mode=positive_bpl.evaluation_mode)
        with patch.object(sys,'argv',['main.py','--short32-bpl','--validate-only']),contextlib.redirect_stdout(io.StringIO()),patch.object(torch.multiprocessing,'spawn',side_effect=AssertionError('no worker launch')):
            with self.assertRaises(SystemExit) as error:
                exec(compile(ast.Module(body=[block],type_ignores=[]),str(ROOT/'main.py'),'exec'),scope)
        self.assertEqual(error.exception.code,0);self.assertTrue(scope['args']['short32_bpl'])

    def test_small_bona_loader_cycles_and_two_view_matches_augmented_prefix(self):
        t=trainer(sources(self.folder.name,lengths=(16000,64000),spoof=9))
        c=short32.Short32Hooks(t);self.addCleanup(c.close);stats=[]
        t.optimizer.register_step_post_hook(lambda *a:stats.append(c.last_stats.copy()) if not t.classifier.module.loss.weight.requires_grad else None)
        with patch('random.randint',return_value=0):t.train(1)
        self.assertEqual([s['eligible'] for s in stats],[1,1,1])
        self.assertEqual([s['bona'] for s in stats],[2,2,2])
        for i in range(0,len(t.plm.inputs),3):
            _,short,two=t.plm.inputs[i:i+3]
            ids=list(t.sampler_bona);pos=ids.index(0)
            self.assertTrue(torch.equal(two,short[pos:pos+1,:32000]))

    def test_all_eligible_half_weights_and_k_squared(self):
        t=trainer(sources(self.folder.name,lengths=(16000,32000),spoof=4))
        c=short32.Short32Hooks(t);self.addCleanup(c.close)
        with patch('random.randint',return_value=0):t.train(1)
        self.assertEqual(c.last_stats['bpl32_pairs'],4);self.assertEqual(c.last_stats['bpl32_weight'],.5)
        self.assertTrue(torch.equal(t.plm.inputs[-1],t.plm.inputs[-2][:2,:32000]))

    def test_model_hook_loss_matches_formula_symmetric_gradients(self):
        t=trainer(sources(self.folder.name,lengths=(16000,32001),spoof=4))
        c=short32.Short32Hooks(t);self.addCleanup(c.close)
        t.sampler_bona.set_epoch(1);t.sampler_spoof.set_epoch(1);t.classifier.train()
        t.plm(torch.randn(6,48440));t.classifier.module.loss.weight.requires_grad_(False)
        zs=[];h=t.classifier.module.layer_norm.register_forward_hook(lambda m,i,o:zs.append(o));self.addCleanup(h.remove)
        labels=torch.tensor([1,1,0,0,0,0])
        loss,_=t.classifier(torch.randn(6,2,9,4),x_short=torch.randn(6,2,7,4),label=labels,bona_size=2)
        z4,z3,z2=zs;bpl=t.classifier.module.loss_bpl;p=c.positions
        expected=t.classifier.module.loss(z4,labels)[0]+bpl(z4[:2],z3[:2])+.25*(bpl(z3[p],z2)-bpl(z4[p],z3[:2]))
        torch.testing.assert_close(loss,expected)
        z3.retain_grad();z2.retain_grad();loss.backward()
        self.assertGreater(z3.grad.abs().sum(),0);self.assertGreater(z2.grad.abs().sum(),0)

    def test_original240_two_rank_partial_sizes_and_sampler_rng(self):
        for rank in [0,1]:
            groups=[]
            for count,size in [(2580,12),(22800,108)]:
                s=DistributedSampler(range(count),num_replicas=2,rank=rank,shuffle=True);s.set_epoch(1)
                before=torch.get_rng_state().clone();ids=list(s)
                self.assertTrue(torch.equal(before,torch.get_rng_state()));self.assertEqual(ids,list(s))
                groups.append([ids[i:i+size] for i in range(0,len(ids),size)])
            visits=[groups[0][i%108]+groups[1][i%106] for i in range(108)]
            self.assertEqual([len(visits[i]) for i in [0,105,107]],[120,72,114])
            self.assertEqual(sum(len(groups[1][i%106]) for i in range(108)),11616)

    def test_real_two_worker_original_loader(self):
        t=trainer(sources(self.folder.name,lengths=(32001,64000),spoof=4));old=t.train_loader_bona
        loader=DataLoader(old.dataset,sampler=old.sampler,batch_size=old.batch_size,num_workers=2)
        row=next(iter(loader));self.assertEqual(row[0].shape,(2,64600));self.assertEqual(len(row),3)

    def test_original_rawboost_zero_to_eight_arrays_rng_unchanged(self):
        t=trainer(sources(self.folder.name,lengths=(16000,64000),spoof=4));ds=t.train_loader_bona.dataset
        for kind in range(9):
            choose=lambda lo,hi:kind if (lo,hi)==(0,8) else lo
            random.seed(17);np.random.seed(17)
            with patch('random.randint',side_effect=choose):before=ds[0]
            state=random.getstate(),np.random.get_state();c=short32.Short32Hooks(t)
            try:
                random.seed(17);np.random.seed(17)
                with patch('random.randint',side_effect=choose):after=ds[0]
                for x,y in zip(before[:2],after[:2]):np.testing.assert_array_equal(x,y)
                self.assertEqual(state[0],random.getstate());np.testing.assert_array_equal(state[1][1],np.random.get_state()[1])
            finally:c.close()

    def test_incompatible_options_and_replay_rejected(self):
        for extra in [{'bpl_pairing':'matched_independent'},{'duration_policy_path':'x'},{'short_oc_weight':.1},{'num_train_frames':400},{'epoch':151,'T_0':150}]:
            with self.assertRaises(ValueError):reference.configure(extra)
        t=SimpleNamespace(classifier=head(),args={'short32_bpl':True})
        with self.assertRaisesRegex(ValueError,'never gradient replay'):native_steps.train_batch(t,{})

    def test_default_off_main_dispatch_and_cleanup_parameter_keys(self):
        import arguments
        self.assertFalse(arguments.get_args()[0]['short32_bpl'])
        self.assertNotIn('short32_bpl',(ROOT/'data_loaders.py').read_text(encoding='utf-8'))
        tree=ast.parse((ROOT/'main.py').read_text(encoding='utf-8'))
        run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
        self.assertTrue(any('short32_original240' in ast.unparse(n) for n in run.body[:2]))
        t=trainer(sources(self.folder.name));keys=list(t.classifier.state_dict());params=[id(p) for p in t.classifier.parameters()]
        c=short32.Short32Hooks(t);c.close()
        self.assertEqual(keys,list(t.classifier.state_dict()));self.assertEqual(params,[id(p) for p in t.classifier.parameters()])
        self.assertEqual(len(t.plm._forward_pre_hooks),0);self.assertEqual(len(t.classifier.module._forward_hooks),0)

    def test_original_full_train_cpu_ddp_two_ranks_different_k(self):
        with tempfile.TemporaryDirectory() as folder:
            torch.multiprocessing.spawn(ddp_worker,args=(Path(folder,'ddp').as_uri(),),nprocs=2,join=True)


if __name__=='__main__':unittest.main()
