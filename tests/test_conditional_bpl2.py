import copy
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from conditional_bpl2 import LONG,SHORT,FIVE,targets,wrap_prefix,short_window,independent_random
import conditional_bpl2_steps as steps
from native_steps import _freeze
from model.loss import BPL,OCSoftmax

class Toy(nn.Module):
    def __init__(self):
        super().__init__();self.linear=nn.Linear(6,4);self.dropout=nn.Dropout(.3)
        self.loss=OCSoftmax(4,class_weight=[5/9,5]);self.loss_bpl=BPL(30)
    def forward(self,x,mask=None,return_embedding=False,**kwargs):
        z=self.dropout(self.linear(x))
        return z if return_embedding else self.loss(z,kwargs.get('label'),kwargs.get('logical_batch_size'))

class ConditionalTests(unittest.TestCase):
    def test_boundary_and_nested_repeat(self):
        self.assertEqual(targets(LONG),(LONG,SHORT));self.assertEqual(targets(LONG+1),(FIVE,LONG))
        for n in [10000,40000,SHORT,LONG,LONG+1,70000,80000,90000]:
            raw=np.arange(n,dtype='float32');ln,sn=targets(n);parent=wrap_prefix(raw,ln)
            child,start=short_window(parent,n,np.random.default_rng(1));self.assertEqual(len(child),sn)
            np.testing.assert_array_equal(child,parent[start:start+sn]);np.testing.assert_array_equal(parent[:min(n,ln)],raw[:ln])
            if n>=sn:self.assertLessEqual(start+sn,min(n,ln))
            if n<ln:self.assertEqual(parent[n],parent[0])
    def test_aux_rng_does_not_change_original_rng(self):
        random.seed(11);np.random.seed(11);py=random.getstate();npstate=np.random.get_state()
        with independent_random(23):random.random();np.random.rand(100)
        self.assertEqual(py,random.getstate());np.testing.assert_array_equal(npstate[1],np.random.get_state()[1])
    def test_paired_not_allpairs_and_stop_gradient(self):
        torch.manual_seed(2);m=Toy();z={k:torch.randn(n,4,requires_grad=True) for k,n in [('long',20),('short',20),('student',12)]}
        teacher=torch.randn(2,4,requires_grad=True);batch={'bona':12,'labels':torch.tensor([1]*12+[0]*8),'teacher5_indices':[0,3]}
        loss,stats=steps.objective(m,z,batch,teacher)
        self.assertAlmostEqual(stats['loss1'],stats['oc']+.5*stats['bpl']+.5*stats['bpl2'],places=5)
        self.assertEqual(stats['legacy_pairs'],144);self.assertEqual(stats['BPL2_pairs'],12)
        g=torch.autograd.grad(loss,[teacher,z['student']],allow_unused=True);self.assertIsNone(g[0]);self.assertGreater(float(g[1].abs().sum()),0)
    def test_replay_equals_full_graph_with_dropout(self):
        torch.manual_seed(7);model=Toy();reference=copy.deepcopy(model)
        batch={'bona':12,'labels':torch.tensor([1]*12+[0]*12),'teacher5_indices':[0,2,4],
               **{k:list(torch.randn(n,6)) for k,n in [('long',24),('short',24),('student',12),('teacher5',3)]}}
        def features(plm,waves,device):return torch.stack(waves),None
        a=SimpleNamespace(classifier=model,plm=nn.Linear(1,1).requires_grad_(False),args={'device':'cpu'},optimizer=torch.optim.Adam(model.parameters(),lr=1e-4))
        opt=torch.optim.Adam(reference.parameters(),lr=1e-4)
        torch.manual_seed(42)
        with patch.object(steps,'features',features):stats=steps.train_step(a,batch)
        torch.manual_seed(42);_freeze(reference,False);opt.zero_grad(set_to_none=True)
        with torch.no_grad():teacher=reference(torch.stack(batch['teacher5']),return_embedding=True)
        z={}
        for name in ['long','short','student']:
            z[name]=torch.cat([reference(torch.stack(batch[name][i:i+8]),return_embedding=True) for i in range(0,len(batch[name]),8)])
        loss,_=steps.objective(reference,z,batch,teacher);loss.backward();opt.step()
        _freeze(reference,True);opt.zero_grad(set_to_none=True)
        for i in range(0,12,8):
            n=min(8,12-i);value,_=reference(torch.stack(batch['long'][i:i+n]),label=torch.ones(n,dtype=torch.long),logical_batch_size=12)
            (value*n/12).backward()
        opt.step()
        for name,v in model.state_dict().items():torch.testing.assert_close(v,reference.state_dict()[name],rtol=1e-5,atol=1e-7)
        self.assertEqual(stats['optimizer_steps'],2);self.assertEqual(stats['center_updates'],1)

if __name__=='__main__':unittest.main()
