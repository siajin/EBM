"""OC+BPL regression and matched-variable-input tests; synthetic audio only."""
import copy
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
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import arguments
import data_loaders
import native_steps
import positive_bpl as matched
import train
from model import BPL,OCSoftmax


def baseline(name):
    if os.environ.get('EBM_BASELINE_ROOT'):
        return (Path(os.environ['EBM_BASELINE_ROOT'])/name).read_text()
    return subprocess.check_output(['git','show','cf199af:'+name],cwd=ROOT,text=True)


class ToyHead(nn.Module):
    def __init__(self,noise=0.):
        super().__init__()
        self.linear=nn.Linear(5,6)
        self.dropout=nn.Dropout(.25)
        self.loss=OCSoftmax(6,class_weight=[5/9,5.])
        self.loss_bpl=BPL(30.)
        self.noise=noise

    def forward(self,x,mask=None,return_embedding=False,label=None,logical_batch_size=None):
        z=torch.nn.functional.layer_norm(self.dropout(self.linear(x)),(6,))
        if self.noise:
            z=z+torch.randn_like(z)*self.noise
        return z if return_embedding else self.loss(z,label,logical_batch_size=logical_batch_size)


def toy_features(plm,waves,device):
    return torch.stack(waves).to(device),None


class MatchedTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'policy.json'
        self.policy={'sample_rate':16000,'step_samples':8000,'target_samples':[8000,16000],
                     'target_probabilities':[.5,.5],'conditional_profiles':[], 'original_relative_weights':[]}
        self.items=[];self.lengths={}
        for label in [1,0]:
            for speaker in ['s0','s1']:
                attack='bonafide' if label else 'A01'
                self.policy['conditional_profiles'].append({'label':label,'speaker':speaker,'attack':attack,
                    'original_count':2,'bucket_counts':[1,1],'conditional_probability':[[1.,0.],[0.,1.]]})
                self.policy['original_relative_weights'].append({'label':label,'speaker':speaker,
                    'original_share':.5,'common_target_share':.5,'original_weight_multiplier':1.})
                for n in [8000,16000]:
                    path=f'{label}_{speaker}_{n}'
                    self.items.append(SimpleNamespace(path=path,speaker=speaker,label=label,type=attack))
                    self.lengths[path]=n
        self.path.write_text(json.dumps(self.policy))
        self.args={'bpl_pairing':'matched_independent','duration_policy_path':str(self.path),
                   'bpl_pair_size':12,'bpl_steps_per_epoch':108,'rank':0,'world_size':2,
                   'rand_seed':1,'native_length':True,'num_workers':0,'micro_batch_size':8,
                   'batch_size':120,'ratio_bona':.1,'ratio_spoof':.9,'evaluation_mode':'controlled_oc',
                   'short_oc_weight':0.,'padding_loss_weight':0.}

    def dataset(self):
        with patch.object(matched.sf,'info',side_effect=lambda p:SimpleNamespace(frames=self.lengths[p],samplerate=16000,channels=1)):
            return matched.MatchedTrainSet(self.args,self.items)

    def test_original_loss_network_defaults_and_loader_classes_preserved(self):
        import ast
        for name in ['model/loss.py','model/b_linear.py']:
            self.assertEqual((ROOT/name).read_text(),baseline(name))
        namespace={'__file__':str(ROOT/'arguments.py')}
        exec(compile(baseline('arguments.py'),'baseline_args','exec'),namespace)
        for before,after in zip(namespace['get_args'](),arguments.get_args()):
            for key,value in before.items():self.assertEqual(after[key],value,key)
        for text in [baseline('data_loaders.py'),(ROOT/'data_loaders.py').read_text()]:
            tree=ast.parse(text)
            classes={node.name:ast.dump(node) for node in tree.body if isinstance(node,ast.ClassDef)}
            if text==baseline('data_loaders.py'):old=classes
            else:
                for name in old:self.assertEqual(old[name],classes[name],name)

    def test_legacy_objective_exact_regression(self):
        namespace={};exec(compile(baseline('native_steps.py'),'baseline_steps','exec'),namespace)
        model=ToyHead();z={name:torch.randn(120,6) for name in ['long','short']}
        batch={'labels':torch.tensor([1]*12+[0]*108)}
        args={'short_oc_weight':0.,'padding_loss_weight':0.}
        old=namespace['_objective'](model,z,batch,args);new=native_steps._objective(model,z,batch,args)
        self.assertTrue(torch.equal(old[0],new[0]));self.assertEqual(old[1:],new[1:])

    def test_matched_objective_is_original_OC_plus_BPL_with_spoof_gradients(self):
        model=ToyHead();z={'long':torch.randn(120,6,requires_grad=True),'bpl_right':torch.randn(12,6,requires_grad=True)}
        labels=torch.tensor([1]*12+[0]*108)
        loss,bona,stats=native_steps._objective(model,z,{'labels':labels},self.args)
        expected=model.loss(z['long'],labels,logical_batch_size=120)[0]+model.loss_bpl(z['long'][:12],z['bpl_right'])
        self.assertTrue(torch.equal(loss,expected));self.assertEqual(bona,12)
        grad=torch.autograd.grad(loss,tuple(z.values()))
        self.assertGreater(float(grad[0][:12].abs().sum()),0)
        self.assertGreater(float(grad[0][12:].abs().sum()),0)
        self.assertGreater(float(grad[1].abs().sum()),0)
        self.assertGreater(stats['oc_long'],0)

    def test_two_stage_replay_matches_direct_OC_BPL_and_centroid_Adam_updates(self):
        torch.manual_seed(7);head=ToyHead().train();direct=copy.deepcopy(head)
        batch={'long':[torch.randn(5) for _ in range(120)],'bpl_right':[torch.randn(5) for _ in range(12)],
               'labels':torch.tensor([1]*12+[0]*108)}
        start=torch.get_rng_state();old_center=head.loss.weight.detach().clone()
        optimizer=torch.optim.Adam(direct.parameters(),lr=1e-3,weight_decay=1e-4)
        native_steps._freeze(direct,False);z={}
        for name in ['long','bpl_right']:
            z[name]=torch.cat([direct(torch.stack(batch[name][i:i+8]),return_embedding=True) for i in range(0,len(batch[name]),8)])
        loss=direct.loss(z['long'],batch['labels'],logical_batch_size=120)[0]+direct.loss_bpl(z['long'][:12],z['bpl_right'])
        loss.backward();optimizer.step()
        native_steps._freeze(direct,True);optimizer.zero_grad(set_to_none=True)
        for i in range(0,12,8):
            stop=min(i+8,12)
            value,_=direct(torch.stack(batch['long'][i:stop]),label=batch['labels'][i:stop],logical_batch_size=12)
            (value*((stop-i)/12)).backward()
        optimizer.step()
        torch.set_rng_state(start)
        actual_optimizer=torch.optim.Adam(head.parameters(),lr=1e-3,weight_decay=1e-4)
        trainer=SimpleNamespace(classifier=head,plm=None,args=dict(self.args,device='cpu'),optimizer=actual_optimizer)
        with patch.object(native_steps,'features',side_effect=toy_features),patch.object(actual_optimizer,'step',wraps=actual_optimizer.step) as step:
            stats=native_steps.train_batch(trainer,batch)
        self.assertEqual(step.call_count,2);self.assertEqual(stats['positive_pairs'],144)
        self.assertEqual(stats['centroid_updates'],1);self.assertGreater(stats['loss2'],0)
        self.assertFalse(torch.equal(old_center,head.loss.weight))
        self.assertIn(head.loss.weight,actual_optimizer.state)
        for a,b in zip(head.parameters(),direct.parameters()):
            torch.testing.assert_close(a,b,atol=3e-7,rtol=1e-6)

    def test_OC_and_BPL_sampler_control_streams_and_rank_sharding(self):
        ds=self.dataset();a=matched.MatchedBatchSampler(ds,self.args)
        b=matched.MatchedBatchSampler(ds,dict(self.args,bpl_pairing='matched_same_source'))
        a.set_epoch(2);b.set_epoch(2);aa,bb=list(a),list(b)
        self.assertEqual(aa,list(a))
        for x,y in zip(aa,bb):
            self.assertEqual(x[:120],y[:120])
            self.assertEqual([q.index for q in y[:12]],[q.index for q in y[120:]])
            self.assertEqual([q.augmentation_seed for q in x],[q.augmentation_seed for q in y])
            self.assertEqual([ds.items[q.index].label for q in x],[1]*12+[0]*108+[1]*12)
        other=list(matched.MatchedBatchSampler(ds,dict(self.args,rank=1)))
        self.assertNotEqual(aa,other)
        self.assertTrue(any(x.index!=y.index for x,y in zip(aa[0][:12],aa[0][120:])))
        for spec in aa[0]:self.assertLessEqual(spec.stop,ds.lengths[spec.index])

    def test_both_classes_have_same_expected_length_distribution(self):
        ds=self.dataset()
        for label in [1,0]:
            ids=ds.class_indices[label]
            q=sum(p*ds.tables[ds.keys[i]][ds.buckets[i]-1] for i,p in zip(ids,ds.class_probability[label]))
            np.testing.assert_allclose(q,[.5,.5])
        self.policy['conditional_profiles'][2]['bucket_counts']=[2,0]
        self.path.write_text(json.dumps(self.policy))
        with self.assertRaisesRegex(ValueError,'buckets differ'):self.dataset()

    def test_class_attack_metadata_and_invalid_options_rejected(self):
        matched.validate_options(self.args)
        for override in [{'training_mode':'bpl_only'},{'bpl_pair_size':8},{'padding_loss_weight':.1},{'duration_policy_path':None}]:
            with self.assertRaises(ValueError):matched.validate_options(dict(self.args,**override))
        self.items[-1].type='A02'
        with self.assertRaisesRegex(ValueError,'groups differ'):self.dataset()
        self.items[-1].type='A01'
        self.policy['conditional_profiles'][0]['conditional_probability']=[[0.,1.],[0.,1.]]
        self.path.write_text(json.dumps(self.policy))
        with self.assertRaisesRegex(ValueError,'impossible'):self.dataset()

    def test_collate_shares_OC_bona_as_BPL_left_and_retains_spoof(self):
        rows=[(torch.full((5,),float(i)),y,i) for i,y in enumerate([1]*12+[0]*108+[1]*12)]
        batch=matched.collate_matched(rows)
        self.assertEqual(len(batch['long']),120);self.assertEqual(len(batch['bpl_right']),12)
        self.assertEqual(batch['labels'].tolist(),[1]*12+[0]*108)
        self.assertTrue(all(batch['long'][i] is rows[i][0] for i in range(120)))
        self.assertNotIn('short',batch)

    def test_rawboost_all_nine_recipes_equal_original_trainset(self):
        args={'native_length':True,'cap_long':False,'num_train_frames':404,'num_train_frames_short':303}
        ds=data_loaders.TrainSet(args,[SimpleNamespace(path='synthetic',label=1)])
        signal=.1*np.sin(np.arange(16000)*.013)
        for kind in range(9):
            np.random.seed(9)
            with patch.object(data_loaders.sf,'read',side_effect=lambda p:(signal.copy(),16000)),patch.object(data_loaders.random,'randint',return_value=kind):
                original,*_=ds[0]
            np.random.seed(9)
            np.testing.assert_array_equal(matched.rawboost(signal.copy(),kind),original)

    def test_partial_read_equals_full_read_slice_and_all_augmentations(self):
        ds=self.dataset()
        signal=.2*np.sin(np.arange(16000)*.013)
        for fmt,subtype in [('WAV','PCM_16'),('FLAC','PCM_24')]:
            path=Path(self.temp.name)/('source.'+fmt.lower())
            matched.sf.write(path,signal,16000,format=fmt,subtype=subtype)
            ds.items[0].path=str(path);ds.lengths[0]=len(signal)
            full,_=matched.sf.read(path)
            for start,stop in [(0,8000),(1234,9234),(8000,16000)]:
                spec=matched.CropSpec(0,start,stop,17)
                expected=torch.as_tensor(full[start:stop].copy(),dtype=torch.float32)
                self.assertTrue(torch.equal(ds.read_crop(spec,augment=False),expected))
            spec=matched.CropSpec(0,1234,9234,17)
            for kind in range(9):
                py_state,np_state=random.getstate(),np.random.get_state()
                try:
                    random.seed(17);np.random.seed(17)
                    expected=matched.rawboost(full[1234:9234].copy(),kind)
                finally:
                    random.setstate(py_state);np.random.set_state(np_state)
                with patch.object(matched.random,'randint',return_value=kind):
                    actual=ds.read_crop(spec)
                self.assertTrue(torch.equal(actual,torch.as_tensor(expected,dtype=torch.float32)))

    def test_partial_read_still_rejects_changed_original_header(self):
        ds=self.dataset();path=Path(self.temp.name)/'changed.wav'
        ds.items[0].path=str(path)
        for frames,sr,channels in [(8001,16000,1),(8000,8000,1),(8000,16000,2)]:
            data=np.zeros(frames) if channels==1 else np.zeros((frames,channels))
            matched.sf.write(path,data,sr)
            with self.assertRaisesRegex(ValueError,'changed after policy'):
                ds.read_crop(matched.CropSpec(0,0,8000,1),augment=False)

    def test_controlled_evaluation_uses_trained_OC_and_preserves_weights_and_rng(self):
        class Loader(list):dataset=range(4)
        head=ToyHead(noise=.01).eval();trainer=train.ModelTrainer()
        trainer.args=dict(self.args,device='cpu');trainer.classifier=head;trainer.plm=nn.Identity()
        trainer.val_loader=Loader([{'waves':[torch.randn(5) for _ in range(4)],'indices':torch.arange(4),'labels':torch.tensor([0,1,0,1])}])
        before={k:v.clone() for k,v in head.state_dict().items()}
        with patch.object(train,'features',side_effect=toy_features):
            trainer.test('V',1);rows=trainer.last_eval_rows.copy()
            torch.randn(17);np.random.random(7);random.random()
            rng=torch.get_rng_state().clone(),np.random.get_state(),random.getstate()
            trainer.args['bpl_pairing']='legacy_long_short';trainer.test('V',999)
            self.assertEqual(rows,trainer.last_eval_rows)
            self.assertTrue(torch.equal(rng[0],torch.get_rng_state()))
            np.testing.assert_equal(rng[1],np.random.get_state());self.assertEqual(rng[2],random.getstate())
            for k,v in head.state_dict().items():self.assertTrue(torch.equal(v,before[k]))
            with torch.no_grad():head.loss.weight.mul_(-1)
            trainer.test('V',1000)
            self.assertNotEqual(rows,trainer.last_eval_rows)
        self.assertFalse(hasattr(trainer,'score_reference'))

    def test_evaluation_options_and_original_EER_preserved(self):
        matched.validate_evaluation_options(dict(self.args,duration_policy_path=None))
        for override in [{'evaluation_mode':'shared_train_mean'},{'eval_batch_size':4},{'num_test_frames':500},{'evaluation_seed':-1}]:
            with self.assertRaises(ValueError):matched.validate_evaluation_options(dict(self.args,**override))
        namespace={};exec(compile(baseline('train.py'),'baseline_train','exec'),namespace)
        scores=[.1,.6,.3,.9,.5,.5];labels=[0,0,1,1,0,1]
        self.assertEqual(namespace['ModelTrainer']().calculate_EER(scores,labels),train.ModelTrainer().calculate_EER(scores,labels))


if __name__=='__main__':unittest.main(verbosity=2)
