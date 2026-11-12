"""Synthetic CPU regression of class-role EBM, original training and commits."""
import ast
import contextlib
import copy
import datetime
import importlib.util
import importlib
import json
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
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model import BPL
from role_objectives import RoleConfig, RoleOCSoftmax
from role_model import RoleBLinear
from role_training import logical_batches, phase_rng, step_hidden, train_epoch
import role_lifecycle as life
import role_runtime as runtime


def reference_module(filename):
    spec = importlib.util.spec_from_file_location('reference_'+filename.replace('/', '_'),
                                                 ROOT/'baseline240_source'/filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tiny_head(compact=0):
    return RoleBLinear(RoleConfig(compact_label=compact), num_layer=2, hidden_size=4, output_size=6, agg_size=4)


class PrefixWrapper(nn.Module):
    def __init__(self, module):
        super().__init__(); self.module = module
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class FakeFrontend(nn.Module):
    def __init__(self):
        super().__init__(); self.scale = nn.Parameter(torch.ones(()))
    def forward(self, x, output_hidden_states=True):
        return SimpleNamespace(hidden_states=tuple(x[:, i]*self.scale for i in range(x.shape[1])))


class QuietProgress:
    def __init__(self, *args, **kwargs): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass


class SizedBatches(list):
    """Match DataLoader iterator length, which the original trainer inspects."""
    def __iter__(self):
        base = list.__iter__(self)
        count = len(self)
        class Iterator:
            def __iter__(self): return self
            def __next__(self): return next(base)
            def __len__(self): return count
        return Iterator()


def synthetic_batches():
    bona, spoof = [], []
    for p, n in [(12, 108), (12, 60), (6, 108)]:
        for rows, label, destination in [(p, 1, bona), (n, 0, spoof)]:
            destination.append((torch.randn(rows, 2, 5, 4), torch.randn(rows, 2, 4, 4),
                                torch.full((rows,), label, dtype=torch.long)))
    return bona, spoof


def optimizer_state_by_name(optimizer, model):
    return {name: optimizer.state[p] for name, p in model.named_parameters()}


def assert_nested_equal(case, a, b):
    if torch.is_tensor(a):
        case.assertTrue(torch.equal(a, b))
    elif isinstance(a, dict):
        case.assertEqual(set(a), set(b))
        for key in a: assert_nested_equal(case, a[key], b[key])
    elif isinstance(a, (list, tuple)):
        case.assertEqual(len(a), len(b))
        for x, y in zip(a, b): assert_nested_equal(case, x, y)
    else:
        case.assertEqual(a, b)


class EvalDataset(Dataset):
    def __init__(self):
        generator = torch.Generator().manual_seed(923)
        self.items = [SimpleNamespace(label=y, path=f'synthetic_{i}') for i, y in enumerate([1, 0, 1, 0, 0])]
        self.hidden = torch.randn(5, 2, 5, 4, generator=generator)
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.hidden[i], self.items[i].label, i


def eval_collate(rows):
    x, y, i = zip(*rows)
    return dict(waves=list(x), labels=torch.tensor(y), indices=torch.tensor(i))


def fake_features(plm, waves, device):
    hidden = torch.stack(waves).to(device)
    return hidden, torch.ones(len(waves), hidden.shape[2], dtype=torch.bool, device=device)


def make_trainer(folder, compact=0, distributed=False):
    head = tiny_head(compact)
    if distributed:
        head = nn.parallel.DistributedDataParallel(head, find_unused_parameters=True, static_graph=False)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-5, weight_decay=1e-4)
    ds = EvalDataset()
    indices = list(range(life.rank(), len(ds), torch.distributed.get_world_size())) if distributed else None
    loader = DataLoader(ds, batch_size=2, sampler=indices, collate_fn=eval_collate)
    return SimpleNamespace(args=dict(run_root=str(folder), device='cpu', epoch=1, rand_seed=1,
                                    rank=life.rank(), eval_batch_size=2, evaluation_seed=1729),
                           classifier=head, optimizer=optimizer, plm=FakeFrontend(),
                           lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 150),
                           eval_loader={p: loader for p in ['19LA', '21LA', '21DF']})


def distributed_worker(process_id, rendezvous, folder):
    torch.set_num_threads(1)
    torch.distributed.init_process_group('gloo', init_method=rendezvous, rank=process_id, world_size=2,
                                        timeout=datetime.timedelta(seconds=45))
    try:
        for compact in (0, 1):
            torch.manual_seed(31)
            t = make_trainer(Path(folder)/str(compact), compact, distributed=True)
            initial = {k: v.clone() for k, v in t.classifier.module.state_dict().items()}
            for i, (p, n) in enumerate([(12,108), (12,60), (6,108)]):
                torch.manual_seed(1000+process_id*50+i)
                labels = torch.tensor([1]*p+[0]*n)
                values = step_hidden(t.classifier, t.optimizer,
                    torch.randn(p+n,2,5,4), torch.randn(p+n,2,4,4), labels)
                assert values['pair_count'] == (n if compact==0 else p)**2
            for parameter in t.classifier.parameters():
                copies = [torch.empty_like(parameter) for _ in range(2)]
                torch.distributed.all_gather(copies, parameter)
                assert torch.equal(copies[0], copies[1])
                assert t.optimizer.state[parameter]['step'].item() == 3
            assert any(not torch.equal(initial[k], v) for k,v in t.classifier.module.state_dict().items())
            t.checkpoints = life.RoleCheckpoints(t); t.checkpoints.save(1)
            result = life.evaluate(t, '19LA', 1, feature_fn=fake_features)
            assert result['rows']==5 and 0<=result['eer_percent']<=100
            def failure(plm, waves, device):
                if process_id==1: raise RuntimeError('injected rank1 inference failure')
                return fake_features(plm,waves,device)
            failed=False
            try: life.evaluate(t, '21LA',1,feature_fn=failure)
            except RuntimeError as error:
                assert 'injected rank1' in str(error);failed=True
            assert failed
            life.evaluate(t,'21DF',1,feature_fn=fake_features)
        life.write_json(Path(folder)/f'rank{process_id}.json',dict(status='passed', roles=[0,1],
            per_role_steps=3, parameter_sync=True, adam_steps=3, uneven_eval_rows=5, shared_failure=True))
    finally:
        torch.distributed.destroy_process_group()


class ObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_metadata_roundtrip_and_mismatch_rejected(self):
        for label in (0,1):
            role=RoleConfig(compact_label=label)
            self.assertEqual(RoleConfig.from_metadata(role.metadata()),role)
            bad=role.metadata();bad['score_bona_transform']='center_cosine' if label==0 else 'negative_center_cosine'
            with self.assertRaises(ValueError): RoleConfig.from_metadata(bad)
            bad=role.metadata();bad['pair_class']=1-label
            with self.assertRaises(ValueError): RoleConfig.from_metadata(bad)
        for value in [None,{}, {'compact_label':0}]:
            with self.assertRaises(ValueError): RoleConfig.from_metadata(value)
        for kw in [dict(compact_label=True),dict(compact_label=2),dict(r_pull=.1),
                   dict(alpha_oc=float('nan')),dict(center_loss_multiplier=0)]:
            with self.assertRaises(ValueError): RoleConfig(**kw)

    def test_original_OC_value_and_embedding_center_gradients(self):
        original=reference_module('model/loss.py')
        for p,n in [(12,108),(12,60),(6,108)]:
            torch.manual_seed(17)
            old=original.OCSoftmax(7,class_weight=[5/9,5]).double()
            new=RoleOCSoftmax(7,RoleConfig(compact_label=1)).double()
            new.load_state_dict(old.state_dict())
            z=torch.randn(p+n,7,dtype=torch.float64,requires_grad=True)
            labels=torch.tensor([1]*p+[0]*n)
            a=old(z,labels)[0]; b=new.backend_loss(z,labels)
            self.assertEqual(a.item(),b.item())
            ga=torch.autograd.grad(a,(z,old.weight),retain_graph=True)
            gb=torch.autograd.grad(b,(z,new.weight))
            for x,y in zip(ga,gb): torch.testing.assert_close(x,y,rtol=0,atol=0)

    def test_reverse_gradient_sign_and_class_weight(self):
        m=RoleOCSoftmax(3,RoleConfig()).double()
        with torch.no_grad():m.weight.copy_(torch.tensor([[1.,0.,0.]]))
        z=torch.tensor([[.3,.8,.5]]*6,dtype=torch.float64,requires_grad=True)
        labels=torch.tensor([0,0,0,1,1,1])
        loss=m.backend_loss(z,labels)
        q=m.cosine(z).detach()
        explicit=torch.nn.functional.softplus(20*torch.where(labels[:,None]==0,.9-q,q-.2))
        weights=q.new_tensor([5/9]*3+[5]*3)[:,None]
        self.assertEqual(loss.item(),(explicit*weights).mean().item())
        g=torch.autograd.grad(loss,z)[0]
        after=m.cosine(z.detach()-.001*g)
        self.assertTrue(bool((after[:3]>q[:3]).all()))
        self.assertTrue(bool((after[3:]<q[3:]).all()))

    def test_center_loss_multiplier_and_wrong_class(self):
        for compact,n in [(0,108),(1,12),(1,6)]:
            m=RoleOCSoftmax(7,RoleConfig(compact_label=compact)).double()
            z=torch.randn(n,7,dtype=torch.float64)
            labels=torch.full((n,),compact)
            value=m.center_loss(z,labels)
            expected=(5*torch.nn.functional.softplus(20*(.9-m.cosine(z)))).mean()
            self.assertEqual(value.item(),expected.item())
            with self.assertRaises(ValueError):m.center_loss(z,1-labels)

    def test_inference_is_immutable_and_sign_is_applied_once(self):
        for compact in (0,1):
            m=RoleOCSoftmax(7,RoleConfig(compact_label=compact))
            z=torch.randn(9,7);before=z.clone();center=m.weight.detach().clone()
            raw=m.cosine(z);loss,score=m(z)
            torch.testing.assert_close(score,raw*(1 if compact else -1),rtol=0,atol=0)
            self.assertEqual(loss.item(),0);self.assertTrue(torch.equal(z,before))
            self.assertTrue(torch.equal(m.weight,center))

    def test_invalid_inputs_and_empty_classes_fail(self):
        head=tiny_head()
        x=torch.randn(6,2,5,4)
        for labels in [torch.ones(6),torch.full((6,),2),torch.ones(5,dtype=torch.long),torch.ones(6,dtype=torch.long)]:
            with self.assertRaises(ValueError):head(x,x,label=labels,phase='backend')
        for z in [torch.zeros(6,6),torch.full((6,6),float('nan')),torch.empty(0,6)]:
            with self.assertRaises(ValueError):head.loss.cosine(z)
        with self.assertRaises(ValueError):head(x,phase='unknown')

    def test_pair_gradient_class_routing_all_pairs_and_permutation(self):
        for compact in (0,1):
            m=RoleOCSoftmax(7,RoleConfig(compact_label=compact))
            for p,n in [(12,108),(12,60),(6,108)]:
                labels=torch.tensor([1]*p+[0]*n);mask=labels==compact
                a=torch.randn(p+n,7,requires_grad=True);b=torch.randn(p+n,7,requires_grad=True)
                pair=BPL(30)(a[mask],b[mask]);ga,gb=torch.autograd.grad(pair,(a,b))
                self.assertTrue(torch.equal(ga[~mask],torch.zeros_like(ga[~mask])))
                self.assertTrue(torch.equal(gb[~mask],torch.zeros_like(gb[~mask])))
                self.assertGreater(ga[mask].norm().item(),0);self.assertGreater(gb[mask].norm().item(),0)
                torch.testing.assert_close(pair,BPL(30)(a[mask].flip(0),b[mask]))

    def test_zero_short_embedding_is_rejected_before_pair_gradient(self):
        head=tiny_head();x=torch.randn(6,2,5,4)
        with patch.object(head,'encode',side_effect=[torch.randn(6,6),torch.zeros(6,6)]):
            with self.assertRaisesRegex(ValueError,'Zero-norm'):
                head(x,x,label=torch.tensor([1,0,0,0,0,0]),phase='backend')


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_original_architecture_initialization_and_state_keys(self):
        original=reference_module('model/b_linear.py');losses=reference_module('model/loss.py')
        torch.manual_seed(11)
        a=original.B_Linear(2,4,6,4,losses.OCSoftmax(6,class_weight=[5/9,5]),losses.BPL(30))
        torch.manual_seed(11);b=tiny_head(1)
        self.assertEqual(set(a.state_dict()),set(b.state_dict()))
        for key,value in a.state_dict().items():self.assertTrue(torch.equal(value,b.state_dict()[key]),key)

    def test_actual_original_train_three_batches_losses_parameters_and_Adam(self):
        original=reference_module('model/b_linear.py');losses=reference_module('model/loss.py')
        torch.manual_seed(19)
        ref=original.B_Linear(2,4,6,4,losses.OCSoftmax(6,class_weight=[5/9,5]),losses.BPL(30))
        new=tiny_head(1);new.load_state_dict(ref.state_dict())
        bona,spoof=map(SizedBatches,synthetic_batches())
        tree=ast.parse((ROOT/'baseline240_source/train.py').read_text(encoding='utf-8'))
        node=next(n for n in tree.body if getattr(n,'name',None)=='ModelTrainer')
        scope={'torch':torch,'tqdm':QuietProgress}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'original_train','exec'),scope)
        args=dict(T_0=150,device='cpu',flag_parent=False,epoch=1,rand_seed=1,rank=0)
        def make(model):
            head=PrefixWrapper(model);opt=torch.optim.Adam(head.parameters(),lr=1e-5,weight_decay=1e-4)
            return SimpleNamespace(args=args.copy(),classifier=head,plm=FakeFrontend(),optimizer=opt,
                lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,150,eta_min=1e-5),
                train_loader_bona=bona,train_loader_spoof=spoof,sampler_bona=SimpleNamespace(set_epoch=lambda e:None),
                sampler_spoof=SimpleNamespace(set_epoch=lambda e:None),da=None,logger=None,lr_step='epoch')
        a,b=make(ref),make(new);recorded=[]
        handle=a.classifier.register_forward_hook(lambda module,args,out:recorded.append(float(out[0].detach())))
        state=torch.get_rng_state()
        scope['ModelTrainer'].train(a,1);handle.remove()
        torch.set_rng_state(state);actual=[];b.record_train=actual.append
        with patch('role_training.phase_rng',side_effect=lambda *a:contextlib.nullcontext()):totals=train_epoch(b,1)
        self.assertEqual(totals['updates'],3)
        for i,row in enumerate(actual):
            self.assertEqual(row['loss1'],recorded[2*i]);self.assertEqual(row['loss2'],recorded[2*i+1])
        for key,value in ref.state_dict().items():torch.testing.assert_close(value,new.state_dict()[key],rtol=0,atol=0)
        assert_nested_equal(self,optimizer_state_by_name(a.optimizer,a.classifier),optimizer_state_by_name(b.optimizer,b.classifier))
        self.assertEqual(a.lr_scheduler.state_dict(),b.lr_scheduler.state_dict())

    def test_reverse_steps_isolate_gradients_recompute_and_Adam_counters(self):
        for compact in (0,1):
            m=tiny_head(compact);m.train();opt=torch.optim.Adam(m.parameters(),lr=1e-5,weight_decay=1e-4)
            center=m.loss.weight.detach().clone();backend=m.linear.weight.detach().clone();events=[]
            def observe(phase,head,loss):
                center_only=phase=='center';events.append(phase)
                for p in head.parameters():
                    active=(p is head.loss.weight)==center_only
                    self.assertEqual(p.grad is not None,active)
                if center_only:
                    self.assertFalse(torch.equal(backend,head.linear.weight))
                    self.assertTrue(torch.equal(center,head.loss.weight))
            for p,n in [(12,108),(12,60),(6,108)]:
                center=m.loss.weight.detach().clone();backend=m.linear.weight.detach().clone()
                result=step_hidden(m,opt,torch.randn(p+n,2,5,4),torch.randn(p+n,2,4,4),
                                   torch.tensor([1]*p+[0]*n),observe=observe)
                self.assertEqual(result['pair_count'],(n if compact==0 else p)**2)
                for name in ['backend_gradient_norm','center_gradient_norm','center_norm','center_update_norm']:
                    self.assertGreater(result[name],0)
                for name in ['bona_unit_variance_trace','spoof_unit_variance_trace']:
                    self.assertGreaterEqual(result[name],0)
                self.assertTrue(-1<=result['bona_score_mean']<=1)
                self.assertTrue(-1<=result['spoof_score_mean']<=1)
                self.assertFalse(torch.equal(center,m.loss.weight))
            self.assertEqual(events,['backend','center']*3)
            for p in m.parameters():
                self.assertIsNone(p.grad);self.assertTrue(p.requires_grad)
                self.assertEqual(opt.state[p]['step'].item(),3)

    def test_phase_rng_restores_and_separates_streams(self):
        torch.manual_seed(25);initial=torch.get_rng_state();draws=[]
        for phase in ['backend','center','backend']:
            with phase_rng('cpu',1,3,2,0,phase):draws.append(torch.rand(8))
        self.assertTrue(torch.equal(initial,torch.get_rng_state()))
        self.assertTrue(torch.equal(draws[0],draws[2]));self.assertFalse(torch.equal(draws[0],draws[1]))

    def test_loader_cycle_preserves_class_order_and_does_not_hide_failure(self):
        def row(label,n):return torch.ones(n,2)*label,torch.ones(n,2)*label,torch.full((n,),label)
        batches=list(logical_batches([row(1,2),row(1,1)],[row(0,3)]*4))
        self.assertEqual([len(x[2]) for x in batches],[5,4,5,4])
        for _,_,y in batches:self.assertEqual(y.tolist(),sorted(y.tolist(),reverse=True))
        class Broken:
            def __len__(self):return 1
            def __iter__(self):raise RuntimeError('worker failed')
        with self.assertRaisesRegex(RuntimeError,'worker failed'):list(logical_batches(Broken(),[row(0,3)]))

    def test_real_original_DistributedSampler_and_loader_ragged_schedule(self):
        original=importlib.import_module('baseline240_source.data_loaders')
        items=[SimpleNamespace(path=f'not_read_{i}',label=int(i<2580),type='synthetic',speaker='s')
               for i in range(25380)]
        args=dict(batch_size=120,ratio_bona=.1,ratio_spoof=.9,num_workers=0,
                  num_train_frames=404,num_train_frames_short=303,num_test_frames=404,num_seg=1)
        def sampler(dataset,shuffle=True):
            return torch.utils.data.DistributedSampler(dataset,num_replicas=2,rank=0,shuffle=shuffle)
        def item(dataset,i):
            row=dataset.items[i];x=torch.tensor([i+100000*row.label],dtype=torch.float32)
            return x,x.clone(),row.label
        dummy=SimpleNamespace(trials=[])
        with patch.object(original,'DistributedSampler',side_effect=sampler),patch.object(original.TrainSet,'__getitem__',item):
            loaders=original.get_loaders(args,items,[],{p:dummy for p in ['19LA','21LA','21DF']})
            a,sa,b,sb=loaders[:4];sa.set_epoch(1);sb.set_epoch(1)
            self.assertEqual((a.batch_size,b.batch_size,len(a),len(b)),(12,108,108,106))
            schedule=list(logical_batches(a,b))
        self.assertEqual(len(schedule),108)
        self.assertEqual([len(row[2]) for row in schedule][105:],[72,120,114])
        self.assertEqual(sum(int((y==1).sum()) for _,_,y in schedule),1290)
        self.assertEqual(sum(int((y==0).sum()) for _,_,y in schedule),11616)

    def test_full_108_step_synthetic_epoch_save_and_all_evaluations(self):
        torch.manual_seed(4)
        with tempfile.TemporaryDirectory() as folder:
            t=make_trainer(folder)
            def row(n,label):
                return torch.randn(n,2,5,4),torch.randn(n,2,4,4),torch.full((n,),label)
            t.train_loader_bona=SizedBatches([row(12,1)]*107+[row(6,1)])
            t.train_loader_spoof=SizedBatches([row(108,0)]*105+[row(60,0)])
            t.sampler_bona=SimpleNamespace(set_epoch=lambda e:None)
            t.sampler_spoof=SimpleNamespace(set_epoch=lambda e:None)
            with patch('native_steps.features',side_effect=fake_features):status=life.run_training(t)
            self.assertEqual(status,'complete')
            report=json.loads((Path(folder)/'stages/complete.json').read_text())
            self.assertEqual(report['saved_epochs'],[1]);self.assertEqual(len(report['results']),3)
            for parameter in t.classifier.parameters():
                self.assertEqual(t.optimizer.state[parameter]['step'].item(),108)
            self.assertIsNone(t.plm.scale.grad);self.assertEqual(t.plm.scale.item(),1)
            t.checkpoints.verify(1)

    def test_center_exception_clears_gradients_and_stops_without_second_step(self):
        m=tiny_head();opt=torch.optim.Adam(m.parameters(),lr=1e-5)
        original=m.forward
        def broken(*a,**kw):
            if kw.get('phase')=='center':raise RuntimeError('injected center failure')
            return original(*a,**kw)
        with patch.object(m,'forward',side_effect=broken):
            with self.assertRaisesRegex(RuntimeError,'center failure'):
                step_hidden(m,opt,torch.randn(6,2,5,4),torch.randn(6,2,4,4),torch.tensor([1,0,0,0,0,0]))
        self.assertTrue(all(p.grad is None and p.requires_grad for p in m.parameters()))
        self.assertNotIn(m.loss.weight,opt.state)


class CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.addCleanup(self.directory.cleanup)
        self.t=make_trainer(self.directory.name);self.t.checkpoints=life.RoleCheckpoints(self.t)

    def test_atomic_role_roundtrip_rejects_missing_or_wrong_role_and_overwrite(self):
        t=self.t;t.checkpoints.save(1)
        head,manifest=life.load_scoring_head(self.directory.name,1)
        self.assertEqual(head.role,RoleConfig())
        for key,value in t.classifier.state_dict().items():self.assertTrue(torch.equal(value,head.state_dict()[key]))
        with self.assertRaisesRegex(RuntimeError,'overwrite'):t.checkpoints.save(1)
        wrong=life.model_metadata(t.classifier);wrong['role']=RoleConfig(compact_label=1).metadata()
        with self.assertRaises(ValueError):life.read_checkpoint(self.directory.name,1,wrong)
        sidecar=Path(self.directory.name)/'model/Model_1.json';sidecar.unlink()
        with self.assertRaises(FileNotFoundError):life.read_checkpoint(self.directory.name,1)

    def test_corruption_and_partial_save_never_become_valid(self):
        self.t.checkpoints.save(1)
        with patch.object(life,'atomic_write',side_effect=OSError('injected disk full')):
            with self.assertRaisesRegex(RuntimeError,'disk full'):self.t.checkpoints.save(2)
        self.assertFalse((Path(self.directory.name)/'stages/checkpoint_2.complete.json').exists())
        with (Path(self.directory.name)/'model/CLS_1.pt').open('ab') as stream:stream.write(b'corrupt')
        with self.assertRaises(ValueError):life.read_checkpoint(self.directory.name,1)

    def test_eval_restores_RNG_modes_params_optimizer_and_canonical_scores(self):
        t=self.t;t.checkpoints.save(1);t.plm.train();t.classifier.train()
        before=life.rng_state('cpu');params=copy.deepcopy(t.classifier.state_dict());opt=copy.deepcopy(t.optimizer.state_dict())
        result=life.evaluate(t,'19LA',1,feature_fn=fake_features)
        self.assertEqual(result['rows'],5);self.assertTrue(0<=result['eer_percent']<=100)
        self.assertTrue(t.plm.training);self.assertTrue(t.classifier.training)
        self.assertTrue(torch.equal(before['torch'],torch.get_rng_state()))
        self.assertEqual(before['python'],random.getstate());np.testing.assert_array_equal(before['numpy']['keys'],np.random.get_state()[1])
        assert_nested_equal(self,params,t.classifier.state_dict());assert_nested_equal(self,opt,t.optimizer.state_dict())
        for _,raw,canonical,_ in t.last_eval_rows:self.assertEqual(canonical,-raw)
        with self.assertRaises(ValueError):life.evaluate(t,'V',1,feature_fn=fake_features)

    def test_failed_eval_is_not_complete_and_preserves_checkpoint(self):
        self.t.checkpoints.save(1)
        with self.assertRaisesRegex(RuntimeError,'injected'):
            life.evaluate(self.t,'21LA',1,feature_fn=lambda *a:(_ for _ in ()).throw(RuntimeError('injected')))
        marker=json.loads((Path(self.directory.name)/'stages/eval_1_21LA.json').read_text())
        self.assertEqual(marker['status'],'failed');self.t.checkpoints.verify(1)

    def test_EER_sign_ties_and_invalid_input(self):
        labels=[0,0,1,1]
        self.assertEqual(life.calculate_eer([-.9,-.8,.9,.8],labels),0)
        self.assertEqual(life.calculate_eer([.9,.8,-.9,-.8],labels),100)
        self.assertEqual(life.calculate_eer([0,0,0,0],labels),50)
        for scores,y in [([0,1],[0,0]),([float('nan'),1],[0,1]),([],[])]:
            with self.assertRaises(ValueError):life.calculate_eer(scores,y)

    def test_cleanup_failure_cannot_leave_a_complete_eval_status(self):
        self.t.checkpoints.save(1)
        original=life.coordinated
        from short32_lifecycle import StageError
        def fail_cleanup(stage,function):
            if stage.endswith(':cleanup'):raise StageError('injected cleanup failure')
            return original(stage,function)
        with patch.object(life,'coordinated',side_effect=fail_cleanup):
            with self.assertRaisesRegex(RuntimeError,'cleanup'):
                life.evaluate(self.t,'19LA',1,feature_fn=fake_features)
        marker=json.loads((Path(self.directory.name)/'stages/eval_1_19LA.json').read_text())
        self.assertEqual(marker['status'],'failed')

    def test_save_first_eval_failure_continues_and_reports_incomplete(self):
        t=self.t;t.args['epoch']=6;events=[]
        from short32_lifecycle import StageError
        totals=dict(batches_seen=108,updates=108,bona_rows=1290,spoof_rows=11616)
        def train(trainer,epoch):events.append(('train',epoch));return totals
        def evaluation(trainer,phase,epoch):
            self.assertTrue((Path(self.directory.name)/f'stages/checkpoint_{epoch}.complete.json').exists())
            events.append((phase,epoch))
            if phase=='19LA' and epoch==5:raise StageError('injected inference failure')
            return dict(phase=phase,epoch=epoch,eer_percent=float(epoch),rows=5)
        with patch('role_training.train_epoch',side_effect=train),patch.object(life,'evaluate',side_effect=evaluation):
            status=life.run_training(t)
        self.assertEqual(status,'evaluation_incomplete')
        self.assertLess(events.index(('19LA',5)),events.index(('train',6)))
        report=json.loads((Path(self.directory.name)/'stages/complete.json').read_text())
        self.assertEqual(report['minimum_eer']['19LA']['epoch'],6)
        self.assertEqual(report['minimum_eer']['21DF']['epoch'],5)

    def test_save_failure_stops_before_evaluation_and_next_epoch(self):
        t=self.t;t.args['epoch']=6;epochs=[]
        from short32_lifecycle import StageError
        totals=dict(batches_seen=108,updates=108,bona_rows=1290,spoof_rows=11616)
        def train(trainer,epoch):epochs.append(epoch);return totals
        with patch('role_training.train_epoch',side_effect=train),patch.object(life.RoleCheckpoints,'save',
                side_effect=StageError('injected disk failure')),patch.object(life,'evaluate') as evaluate:
            with self.assertRaisesRegex(RuntimeError,'disk failure'):life.run_training(t)
        self.assertEqual(epochs,[1,2,3,4,5]);evaluate.assert_not_called()


class RuntimeTests(unittest.TestCase):
    def test_validate_only_has_no_launch_directory_or_download(self):
        from tools.train_role_experiment import main
        with tempfile.TemporaryDirectory() as folder,patch('builtins.print'),patch('torch.multiprocessing.spawn') as spawn:
            root=Path(folder)/'must_not_exist'
            self.assertEqual(main(['--run-root',str(root)]),0)
            self.assertFalse(root.exists());spawn.assert_not_called()

    def test_validation_from_non_git_snapshot_and_actual_entrypoint(self):
        result=subprocess.run([sys.executable,str(ROOT/'tools/train_role_experiment.py'),
                               '--compact-class','bonafide'],cwd=tempfile.gettempdir(),capture_output=True,text=True,timeout=60)
        self.assertEqual(result.returncode,0,result.stderr)
        receipt=json.loads(result.stdout);self.assertFalse(receipt['launched'])
        self.assertEqual(receipt['args']['role']['compact_label'],1)

    def test_invalid_mixed_options_and_unpinned_revision_rejected(self):
        for kw in [dict(epoch=0),dict(epoch=151),dict(seed=-1),dict(workers=-1),dict(frontend_revision='main')]:
            with self.assertRaises(ValueError):runtime.options(**kw)
        a=runtime.options();a['short32_bpl']=True
        with self.assertRaises(ValueError):runtime.validate_options(a)
        a=runtime.options();a['batch_size']=60
        with self.assertRaises(ValueError):runtime.validate_options(a)

    def test_smoke_receipt_is_tied_to_role_seed_source_and_frontend(self):
        a=runtime.options(frontend_revision='a'*40);a['source_sha256']=runtime.source_fingerprint()
        receipt=dict(status='passed',signature=runtime.signature(a),world_size=2,trained_batches=[1,106,108],
                     eval_phases=['19LA','21LA','21DF'],data_fingerprint='test-only')
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'smoke.json';p.write_text(json.dumps(receipt))
            runtime.validate_smoke_receipt(a,p)
            for key,value in [('rand_seed',2),('frontend_revision','b'*40),('role',RoleConfig(compact_label=1).metadata())]:
                bad=copy.deepcopy(a);bad[key]=value
                with self.assertRaises(ValueError):runtime.validate_smoke_receipt(bad,p)

    def test_frozen_reference_manifest_and_all_existing_hashes(self):
        self.assertEqual(runtime.verify_reference()['commit'],runtime.REFERENCE)

    def test_parent_timeout_cleans_only_its_child_handles(self):
        from tools.train_role_experiment import wait_workers
        from unittest.mock import MagicMock
        process=MagicMock();process.is_alive.side_effect=[True,False]
        context=SimpleNamespace(processes=[process],join=MagicMock(return_value=False))
        with patch('tools.train_role_experiment.time.monotonic',side_effect=[0.,2.]),patch(
                'tools.train_role_experiment.os.name','nt'):
            with self.assertRaises(TimeoutError):wait_workers(context,1)
        process.terminate.assert_called_once();process.join.assert_called_once_with(timeout=5)

    def test_parent_error_signals_only_previously_verified_own_groups(self):
        from tools.train_role_experiment import wait_workers
        from unittest.mock import MagicMock,call
        import signal
        process=MagicMock();process.pid=56789;process.is_alive.side_effect=[True,False,False,False]
        context=SimpleNamespace(processes=[process],join=MagicMock(side_effect=RuntimeError('child failed')))
        with patch('tools.train_role_experiment.time.monotonic',side_effect=[0.,0.,0.]),patch(
                'tools.train_role_experiment.os.name','posix'),patch(
                'tools.train_role_experiment.os.getpgid',return_value=56789,create=True),patch(
                'tools.train_role_experiment.os.killpg',create=True) as killpg,patch.object(signal,'SIGKILL',9,create=True):
            with self.assertRaisesRegex(RuntimeError,'child failed'):wait_workers(context,10)
        self.assertEqual(killpg.call_args_list,[call(56789,signal.SIGTERM),call(56789,9)])
        process.terminate.assert_not_called();process.kill.assert_not_called()


class DistributedTests(unittest.TestCase):
    def test_two_process_Gloo_both_roles_ragged_updates_checkpoint_eval_and_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            rendezvous=(Path(folder)/'rendezvous').as_uri()
            torch.multiprocessing.spawn(distributed_worker,args=(rendezvous,folder),nprocs=2,join=True)
            for process_id in (0,1):
                receipt=json.loads((Path(folder)/f'rank{process_id}.json').read_text())
                self.assertEqual(receipt['status'],'passed')


if __name__=='__main__':unittest.main()
