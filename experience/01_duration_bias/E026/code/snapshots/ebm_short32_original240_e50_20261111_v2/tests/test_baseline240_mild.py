"""Archived240 view, shuffle/recycling, and exact full-graph update regressions."""
import ast
import copy
import os
from pathlib import Path
import random
import json
import runpy
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset,DistributedSampler
from baseline240_mild import original_views,ConsumedIterator

ROOT=Path(__file__).resolve().parents[1]


def archived(name):
    if os.environ.get('EBM_ARCHIVED240_ROOT'):
        return (Path(os.environ['EBM_ARCHIVED240_ROOT'])/name).read_text()
    return subprocess.check_output(['git','show','10f3bcd:'+name],cwd=ROOT).decode()


class SilentProgress:
    def __init__(self,*a,**kw):pass
    def __enter__(self):return self
    def __exit__(self,*a):pass


class Baseline240Tests(unittest.TestCase):
    def test_actual_entrypoint_reaches_worker_spawn_without_GPU_execution(self):
        # Execute the real __main__ startup, stubbing only the container surface
        # and process launch. This catches unbound names missed by AST checks.
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'experiment.json').write_text(json.dumps({'source_sha256':{},'baseline_source_sha256':{}}))
            is_file=Path.is_file
            with patch.object(sys,'argv',[str(ROOT/'tools/train_baseline240_mild.py'),str(root)]), \
                 patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0,1'}), \
                 patch.object(Path,'is_file',lambda p:True if p.name=='.dockerenv' else is_file(p)), \
                 patch.object(torch.multiprocessing,'set_sharing_strategy') as sharing, \
                 patch.object(torch.multiprocessing,'spawn') as spawn:
                runpy.run_path(str(ROOT/'tools/train_baseline240_mild.py'),run_name='__main__')
            sharing.assert_called_once_with('file_system')
            spawn.assert_called_once()
            self.assertEqual(spawn.call_args.kwargs['nprocs'],2)
            self.assertTrue((root/'train.claim').is_dir())

    def test_original_view_preparation_is_preserved(self):
        tree=ast.parse(archived('data_loaders.py'))
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='TrainSet')
        sf=SimpleNamespace(read=None)
        ns={'Dataset':object,'np':np,'random':random,'sf':sf,'CodecAugmentationLight':lambda:None}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),'original_trainset','exec'),ns)
        ds=ns['TrainSet']({'num_train_frames':404,'num_train_frames_short':303},[SimpleNamespace(path='test.flac',label=1)])
        for size in [12000,48440,64600,80000]:
            wave=np.arange(size,dtype=np.float64)/100000
            sf.read=lambda _: (wave.copy(),16000)
            # Zero selects no RawBoost and also chooses the first short crop.
            with patch.object(random,'randint',return_value=0):
                expected=ds[0];actual=original_views(ds,0,wave)
            self.assertTrue(np.array_equal(expected[0],actual[0]))
            self.assertTrue(np.array_equal(expected[1],actual[1]))
            self.assertEqual((len(actual[0]),len(actual[1])),(64600,48440))

    def test_original_distributed_shuffle_recycling_and_partial_batches(self):
        seen=[set(),set()];counts=[0,0]
        for rank in [0,1]:
            loaders=[]
            for n,size in [(2580,12),(22800,108)]:
                ds=TensorDataset(torch.arange(n));sampler=DistributedSampler(ds,num_replicas=2,rank=rank,shuffle=True)
                sampler.set_epoch(1);loaders.append(DataLoader(ds,batch_size=size,sampler=sampler))
            self.assertEqual([len(x) for x in loaders],[108,106])
            short=iter(loaders[1]);pairs=[];sizes=[]
            for bona in loaders[0]:
                try:spoof=next(short)
                except StopIteration:short=iter(loaders[1]);spoof=next(short)
                a,b=bona[0],spoof[0]
                for role,ids in enumerate([a,b]):seen[role].update(ids.tolist());counts[role]+=len(ids)
                pairs.append(len(a)**2);sizes.append(len(a)+len(b))
            self.assertEqual(pairs,[144]*107+[36])
            self.assertEqual(sizes[105],72);self.assertEqual(sizes[107],114)
        self.assertEqual([len(x) for x in seen],[2580,22800]);self.assertEqual(counts,[2580,23232])

    def test_consumed_iterator_exposes_original_length(self):
        iterator=SimpleNamespace(__len__=lambda:3)
        class Fake:
            def __len__(self):return 3
        self.assertEqual(len(ConsumedIterator(Fake(),None)),3)

    def test_exact_archived_train_matches_direct_two_step_adam(self):
        losses={};exec(compile(archived('model/loss.py'),'original_losses','exec'),losses)
        class Front(nn.Module):
            def __init__(self):super().__init__();self.scale=nn.Parameter(torch.ones(()))
            def forward(self,x,**kw):return SimpleNamespace(hidden_states=((x*self.scale).unsqueeze(1),))
        class Head(nn.Module):
            def __init__(self):
                super().__init__();self.linear=nn.Linear(5,6);self.dropout=nn.Dropout(.25)
                self.loss=losses['OCSoftmax'](6,class_weight=[5/9,5]);self.loss_bpl=losses['BPL'](30)
            def embedding(self,x):return self.dropout(self.linear(x.mean((1,2))))
            def forward(self,x,x_short=None,label=None,bona_size=None):
                z=self.embedding(x);value,scores=self.loss(z,label)
                if x_short is not None:value=value+self.loss_bpl(z[:bona_size],self.embedding(x_short)[:bona_size])
                return value,scores
        class Wrapped(nn.Module):
            def __init__(self):super().__init__();self.module=Head()
            def forward(self,*a,**kw):return self.module(*a,**kw)
        torch.manual_seed(3);head=Wrapped();front=Front();reference=copy.deepcopy(head);ref_front=copy.deepcopy(front)
        # Actual fixed-size logical batches, plus both original ragged shapes.
        data=[]
        for label,sizes in [(1,[12,12,6]),(0,[108,60,108])]:
            count=sum(sizes);ds=TensorDataset(torch.randn(count,5),torch.randn(count,5),torch.full((count,),label))
            batches=[];start=0
            for size in sizes:batches.append(list(range(start,start+size)));start+=size
            data.append(DataLoader(ds,batch_sampler=batches,num_workers=0))
        tree=ast.parse(archived('train.py'));cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ModelTrainer')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='train')
        ns={'torch':torch,'tqdm':SilentProgress}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),'archived_training_method','exec'),ns)
        args={'T_0':150,'device':'cpu','flag_parent':False}
        opt=torch.optim.Adam(list(front.parameters())+list(head.parameters()),lr=1e-3,weight_decay=1e-4)
        opt2=torch.optim.Adam(list(ref_front.parameters())+list(reference.parameters()),lr=1e-3,weight_decay=1e-4)
        trainer=SimpleNamespace(args=args,plm=front,classifier=head,logger=None,optimizer=opt,
            sampler_bona=SimpleNamespace(set_epoch=lambda _:None),sampler_spoof=SimpleNamespace(set_epoch=lambda _:None),
            train_loader_bona=data[0],train_loader_spoof=data[1],da=None,lr_step='epoch',
            lr_scheduler=SimpleNamespace(step=lambda:None))
        state=torch.get_rng_state();ns['train'](trainer,1)
        torch.set_rng_state(state);ref_front.eval();reference.train()
        # Preserve original DataLoader iterator construction/RNG order as well.
        unused_bona=iter(data[0]);spoof_iter=iter(data[1])
        for a in data[0]:
            b=next(spoof_iter);x=torch.cat((a[0],b[0]));short=torch.cat((a[1],b[1]));labels=torch.cat((a[2],b[2]));n=len(a[2])
            with torch.no_grad():x=ref_front(x).hidden_states;short=ref_front(short).hidden_states
            opt2.zero_grad()
            for name,p in reference.named_parameters():p.requires_grad_(name!='module.loss.weight')
            z=reference.module.embedding(torch.stack(x,1));s=reference.module.embedding(torch.stack(short,1))
            value=reference.module.loss(z,labels)[0]+reference.module.loss_bpl(z[:n],s[:n]);value.backward();opt2.step()
            opt2.zero_grad()
            for name,p in reference.named_parameters():p.requires_grad_(name=='module.loss.weight')
            value=reference.module.loss(reference.module.embedding(torch.stack(x,1)[:n]),labels[:n])[0]
            value.backward();opt2.step()
        for a,b in zip(head.parameters(),reference.parameters()):torch.testing.assert_close(a,b,atol=1e-7,rtol=1e-6)
        self.assertTrue(all(p.grad is None for p in front.parameters()))
        self.assertEqual({int(x['step']) for x in opt.state.values()},{3})


if __name__=='__main__':unittest.main()
