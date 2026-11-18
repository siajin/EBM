"""Input/RNG identity and process cleanup against the frozen active v1 loop."""
import ast
import contextlib
import gc
import hashlib
import multiprocessing
from pathlib import Path
import pickle
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import role_training as training
import role_runtime as runtime


class AugmentedRows(Dataset):
    def __init__(self,label,n=None,fail=False):
        self.label=label;self.n=n if n is not None else (2580 if label else 22800);self.fail=fail
    def __len__(self):return self.n
    def __getitem__(self,index):
        if self.fail:raise RuntimeError('INJECTED_READ_FAILURE')
        def view(n):
            # Exercise all three worker RNGs and independent long/short draws.
            return torch.tensor([index,random.random(),np.random.rand(),float(torch.rand(()))]+[0.]*(n-4))
        return view(9),view(5),self.label


def legacy_function():
    source=(Path(__file__).parent/'fixtures/role_logical_batches_v1.py').read_text()
    ns={};exec(compile(source,'frozen_role_iterators_v1','exec'),ns)
    return ns['logical_batches']


def loaders(rank,workers,*,generator_mode='separate',sizes=None,fail=False):
    shared=torch.Generator().manual_seed(129)
    result=[]
    for offset,label in enumerate((1,0)):
        ds=AugmentedRows(label,n=sizes[offset] if sizes else None,fail=fail and not label)
        sampler=DistributedSampler(ds,num_replicas=2,rank=rank,shuffle=True)
        gen=(torch.Generator().manual_seed(1+100003*rank+7919*offset) if generator_mode=='separate'
             else shared if generator_mode=='shared' else None)
        result.append(DataLoader(ds,sampler=sampler,batch_size=12 if label else 108,
                                  num_workers=workers,generator=gen,
                                  **({'multiprocessing_context':'spawn'} if workers else {})))
    return result


def run_epochs(function,rank,workers,epochs,generator_mode='separate'):
    random.seed(11);np.random.seed(11);torch.manual_seed(11)
    ls=loaders(rank,workers,generator_mode=generator_mode)
    records=[]
    for epoch in range(1,epochs+1):
        for loader in ls:loader.sampler.set_epoch(epoch)
        h=hashlib.sha256();sizes=[]
        for long,short,labels in function(*ls):
            for value in (long,short,labels):h.update(value.numpy().tobytes())
            sizes.append((int((labels==1).sum()),int((labels==0).sum())))
        assert len(sizes)==108 and sizes[105]==(12,60) and sizes[107]==(6,108)
        records.append((h.hexdigest(),sizes))
    states=[x.generator.get_state().numpy().tobytes() if x.generator is not None else None for x in ls]
    parent=(random.getstate(),np.random.get_state(),torch.get_rng_state().numpy().tobytes())
    gc.collect()
    return records,states,pickle.dumps(parent)


class ObservedIterator:
    def __init__(self,iterator,record):self.iterator=iterator;self.record=record
    def __iter__(self):return self
    def __next__(self):
        try:value=next(self.iterator)
        except StopIteration:
            self.record['exhausted']=True
            raise
        self.record['reads']+=1
        return value
    def __getattr__(self,name):return getattr(self.iterator,name)


class IteratorLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def tearDown(self):
        gc.collect()
        children=multiprocessing.active_children()
        self.assertEqual(children,[],f'Synthetic workers remain: {[p.pid for p in children]}')

    def test_full_inputs_and_all_rng_states_match_v1_two_epochs_two_ranks(self):
        old=legacy_function()
        for rank in (0,1):
            with self.subTest(rank=rank):
                self.assertEqual(run_epochs(old,rank,0,2),run_epochs(training.logical_batches,rank,0,2))

    def test_spawn_workers_inputs_and_cycle_augmentation_match_v1_both_ranks(self):
        old=legacy_function()
        for rank in (0,1):
            with self.subTest(rank=rank):
                self.assertEqual(run_epochs(old,rank,2,1),run_epochs(training.logical_batches,rank,2,1))

    def test_original_four_workers_preserve_all_input_bytes_and_seeds(self):
        self.assertEqual(run_epochs(legacy_function(),1,4,1),run_epochs(training.logical_batches,1,4,1))

    def test_shared_or_global_generator_consumption_matches_v1(self):
        for mode in ('shared','global'):
            with self.subTest(mode=mode):
                self.assertEqual(run_epochs(legacy_function(),0,0,2,mode),
                                 run_epochs(training.logical_batches,0,0,2,mode))

    def test_every_created_cycle_is_required_and_exhausted(self):
        actual=training._cycle_iterator;records=[]
        def observe(loader,count):
            record=dict(label=loader.dataset.label,limit=count,reads=0,exhausted=False)
            records.append(record)
            return ObservedIterator(actual(loader,count),record)
        ls=loaders(0,0)
        with patch.object(training,'_cycle_iterator',side_effect=observe):
            self.assertEqual(len(list(training.logical_batches(*ls))),108)
        self.assertEqual([(r['label'],r['limit'],r['reads'],r['exhausted']) for r in records],
                         [(0,106,106,True),(1,108,108,True),(0,2,2,True)])

    def test_early_close_drains_only_scheduled_cycles_and_joins_workers(self):
        ls=[DataLoader(AugmentedRows(1,5),batch_size=2,num_workers=1,multiprocessing_context='spawn'),
            DataLoader(AugmentedRows(0,6),batch_size=3,num_workers=1,multiprocessing_context='spawn')]
        iterator=training.logical_batches(*ls)
        next(iterator)
        iterator.close()
        del iterator,ls

    def test_active_worker_failure_propagates_and_workers_are_joined(self):
        ls=loaders(0,1,sizes=(8,20),fail=True)
        iterator=training.logical_batches(*ls)
        with self.assertRaisesRegex(RuntimeError,'INJECTED_READ_FAILURE'):
            next(iterator)
        del iterator,ls

    def test_opposite_large_class_and_equal_lengths(self):
        def row(label,n):return torch.ones(n,2)*label,torch.ones(n,2)*label,torch.full((n,),label)
        for a,b in [([row(1,2)]*2,[row(0,3)]*5),([row(1,2)]*3,[row(0,3)]*3)]:
            expected=list(legacy_function()(a,b));actual=list(training.logical_batches(a,b))
            self.assertEqual(len(expected),len(actual))
            for x,y in zip(expected,actual):
                self.assertTrue(all(torch.equal(i,j) for i,j in zip(x,y)))

    def test_fail_closed_on_unsupported_seed_or_worker_policies(self):
        bad=DataLoader(AugmentedRows(1,8),batch_size=2,shuffle=True)
        good=DataLoader(AugmentedRows(0,12),batch_size=3)
        with self.assertRaisesRegex(ValueError,'deterministic sampler'):
            next(training.logical_batches(bad,good))
        args=runtime.options()
        args['iterator_policy']='legacy'
        with self.assertRaisesRegex(ValueError,'iterator policy'):runtime.validate_options(args)

    def test_smoke_exhausts_all_worker_batches_and_updates_only_three_selected_steps(self):
        consumed=[];metrics=[]
        def batches(*unused):
            for step in range(1,109):
                consumed.append(step)
                b,s=(6 if step==108 else 12),(60 if step==106 else 108)
                labels=torch.tensor([1]*b+[0]*s)
                yield torch.ones(1).expand(b+s,64600),torch.ones(1).expand(b+s,48440),labels
        class PLM:
            def eval(self):return self
            def requires_grad_(self,value):return self
            def parameters(self):return []
            def __call__(self,*a,**kw):return SimpleNamespace(hidden_states=(torch.ones(1),))
        def step(model,opt,a,b,labels,**unused):
            return dict(bona_rows=int((labels==1).sum()),spoof_rows=int((labels==0).sum()))
        with tempfile.TemporaryDirectory() as folder:
            t=SimpleNamespace(args=dict(device='cpu',rand_seed=1,rank=0,run_root=folder),plm=PLM(),
                              classifier=SimpleNamespace(train=lambda:None,loss=SimpleNamespace(weight=torch.ones(1)),role=runtime.RoleConfig()),
                              sampler_bona=SimpleNamespace(set_epoch=lambda x:None),sampler_spoof=SimpleNamespace(set_epoch=lambda x:None),
                              train_loader_bona=None,train_loader_spoof=None,optimizer=None,record_train=metrics.append)
            t.args['data_summary']={'fingerprint':'synthetic'}
            t.args['iterator_policy']='bounded_cycles_legacy_seed_v1'
            with patch.object(runtime,'logical_batches',side_effect=batches),patch.object(runtime,'step_hidden',side_effect=step), \
                 patch.object(runtime,'RoleCheckpoints'),patch.object(runtime,'evaluate') as evaluate, \
                 patch.object(runtime.dist,'get_world_size',return_value=1), \
                 patch.object(runtime.dist,'all_gather',side_effect=lambda output,center:output[0].copy_(center)), \
                 patch.object(runtime.torch.cuda,'max_memory_allocated',return_value=0):
                runtime.run_smoke(t)
            self.assertEqual(consumed,list(range(1,109)))
            self.assertEqual([(r['batch'],r['bona_rows'],r['spoof_rows']) for r in metrics],[(1,12,108),(106,12,60),(108,6,108)])
            self.assertEqual(evaluate.call_count,3)
            import json
            receipt=json.loads((Path(folder)/'smoke_complete.json').read_text())
            self.assertEqual(receipt['loader_batches_seen'],108)
            self.assertEqual(receipt['iterator_policy'],t.args['iterator_policy'])

    def test_old_subset_smoke_receipt_cannot_authorize_new_training(self):
        import json
        args=runtime.options()
        receipt=dict(status='passed',signature=runtime.signature(args),world_size=2,
                     trained_batches=[1,106,108],eval_phases=['19LA','21LA','21DF'],data_fingerprint='synthetic')
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'old_smoke.json';path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError,'matching two-GPU smoke'):
                runtime.validate_smoke_receipt(args,path)


if __name__=='__main__':unittest.main()
