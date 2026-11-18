"""Requested0.9 pair/0.8 center margins, legacy artifacts and wrapped evaluation."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from role_objectives import RoleConfig,RoleBPL,RoleOCSoftmax
from role_model import RoleBLinear,config_from_model
from model.loss import BPL
import role_lifecycle as life
import role_runtime as runtime
from test_role_experiment import make_trainer,fake_features


class RelaxedRoleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_pair_loss_and_both_gradients_match_reference(self):
        x=torch.randn(7,6,dtype=torch.float64,requires_grad=True)
        y=torch.randn(5,6,dtype=torch.float64,requires_grad=True)
        value=RoleBPL(30,.9)(x,y)
        ref=(F.softplus(30*(.9-F.normalize(x,dim=1)@F.normalize(y,dim=1).T))-.3).mean()
        self.assertTrue(torch.equal(value,ref))
        a=torch.autograd.grad(value,(x,y),retain_graph=True);b=torch.autograd.grad(ref,(x,y))
        self.assertTrue(all(torch.equal(i,j) for i,j in zip(a,b)))

    def test_original_pair_default_remains_exact(self):
        x=torch.randn(12,6,requires_grad=True);y=torch.randn(12,6,requires_grad=True)
        a,b=BPL(30)(x,y),RoleBPL(30,1)(x,y)
        self.assertTrue(torch.equal(a,b))
        self.assertTrue(all(torch.equal(i,j) for i,j in zip(torch.autograd.grad(a,(x,y),retain_graph=True),torch.autograd.grad(b,(x,y)))))

    def test_pair_force_softens_but_is_not_a_hard_cutoff(self):
        q=torch.tensor([.85,.95,1.],dtype=torch.float64,requires_grad=True)
        new=F.softplus(30*(.9-q))-.3;old=F.softplus(30*(1-q))-.3
        a=torch.autograd.grad(new.sum(),q,retain_graph=True)[0]
        b=torch.autograd.grad(old.sum(),q)[0]
        self.assertTrue(torch.all((a<0)&(a.abs()<b.abs())))
        self.assertLess(new[-1].item(),0) # unchanged additive constant, no clipping

    def test_OC_backend_and_center_use_point8_and_push_remains_point2(self):
        role=RoleConfig(r_pull=.8,pair_target_cosine=.9);m=RoleOCSoftmax(6,role)
        z=torch.randn(12,6);labels=torch.tensor([1]*3+[0]*9);q=m.cosine(z)
        ref=(F.softplus(20*torch.where(labels[:,None]==0,.8-q,q-.2))*torch.where(labels[:,None]==1,5.,5/9)).mean()
        self.assertTrue(torch.equal(m.backend_loss(z,labels),ref))
        self.assertTrue(torch.equal(m.center_loss(z[3:],labels[3:]),(5*F.softplus(20*(.8-m.cosine(z[3:])))).mean()))

    def test_v1_and_v2_metadata_roundtrip_and_fail_closed(self):
        old=RoleConfig().metadata();self.assertEqual(old['schema_version'],'ebm_role_v1')
        self.assertNotIn('pair_target_cosine',old)
        self.assertEqual(RoleConfig.from_metadata(old).pair_target_cosine,1.)
        new=RoleConfig(r_pull=.8,pair_target_cosine=.9);meta=new.metadata()
        self.assertEqual(meta['schema_version'],'ebm_role_v2');self.assertEqual(RoleConfig.from_metadata(meta),new)
        del meta['pair_target_cosine']
        with self.assertRaises(ValueError):RoleConfig.from_metadata(meta)
        for value in [-1,1.1,float('nan'),True]:
            with self.assertRaises(ValueError):RoleConfig(pair_target_cosine=value)

    def test_new_checkpoint_reloads_target_and_rejects_incompatible_role(self):
        with tempfile.TemporaryDirectory() as folder:
            t=make_trainer(folder);t.classifier=RoleBLinear(RoleConfig(r_pull=.8,pair_target_cosine=.9),num_layer=2,hidden_size=4,output_size=6,agg_size=4)
            t.optimizer=torch.optim.Adam(t.classifier.parameters(),lr=1e-5)
            t.lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(t.optimizer,150)
            t.checkpoints=life.RoleCheckpoints(t);t.checkpoints.save(5)
            head,_=life.load_scoring_head(folder,5)
            self.assertEqual(head.role.r_pull,.8);self.assertEqual(head.loss_bpl.target,.9)
            wrong=copy.deepcopy(t.checkpoints.expected);wrong['role']=RoleConfig().metadata()
            with self.assertRaises(ValueError):life.read_checkpoint(folder,5,wrong)

    def test_head_detects_unrecorded_pair_target_change(self):
        head=RoleBLinear(RoleConfig(r_pull=.8,pair_target_cosine=.9),2,4,6,4)
        head.loss_bpl.target=1.
        with self.assertRaises(ValueError):config_from_model(head)

    def test_wrapped_features_use_exact_fixed_input_without_attention_mask(self):
        calls=[]
        class PLM:
            def __call__(self,x,**kw):
                calls.append((x.clone(),kw));return SimpleNamespace(hidden_states=(x[:,:,None],))
        waves=[torch.randn(64600),torch.randn(64600)]
        hidden,mask=life.wrapped_features(PLM(),waves,'cpu')
        self.assertIsNone(mask);self.assertTrue(torch.equal(calls[0][0],torch.stack(waves)))
        self.assertNotIn('attention_mask',calls[0][1]);self.assertEqual(hidden.shape,(2,1,64600,1))
        with self.assertRaises(ValueError):life.wrapped_features(PLM(),[torch.randn(16000)],'cpu')

    def test_wrapped_full_evaluation_records_correct_protocol_and_restores_training(self):
        with tempfile.TemporaryDirectory() as folder:
            t=make_trainer(folder);t.args['eval_input']='front64600_wrap'
            t.checkpoints=life.RoleCheckpoints(t);t.checkpoints.save(1)
            before=copy.deepcopy(t.classifier.state_dict())
            life.evaluate(t,'19LA',1,feature_fn=fake_features)
            r=json.loads((Path(folder)/'stages/eval_1_19LA.json').read_text())
            self.assertEqual(r['protocol']['input'],'wrap64600/no-mask')
            self.assertTrue(all(torch.equal(v,t.classifier.state_dict()[k]) for k,v in before.items()))

    def test_actual_CLI_accepts_requested_margins_and_wrapped_eval_without_launch(self):
        r=subprocess.run([sys.executable,str(ROOT/'tools/train_role_experiment.py'),'--r-pull','.8','--pair-target-cosine','.9','--eval-input','front64600_wrap'],capture_output=True,text=True,timeout=60)
        self.assertEqual(r.returncode,0,r.stderr);d=json.loads(r.stdout);self.assertFalse(d['launched'])
        self.assertEqual(d['args']['role']['r_pull'],.8);self.assertEqual(d['args']['role']['pair_target_cosine'],.9)
        self.assertEqual(d['args']['eval_input'],'front64600_wrap')


if __name__=='__main__':unittest.main()
