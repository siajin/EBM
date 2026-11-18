"""Synthetic-only checkpoint, entrypoint, and distributed failure injection."""
import copy
import datetime
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import short32_lifecycle as life
import short32_original240 as adapter
import controlled_data_loaders as data
from test_short32_bpl import trainer, sources, ROOT


def fake_features(plm, waves, device):
    x = pad_sequence(waves,batch_first=True).to(device)
    hidden = plm(x).hidden_states
    stacked = torch.stack(hidden,dim=1)
    return stacked,torch.ones((len(waves),stacked.shape[2]),device=device,dtype=torch.bool)


def prepare(folder, *, audio_folder=None, count=5):
    audio = Path(audio_folder or Path(folder)/'audio'); audio.mkdir(exist_ok=True)
    items = sources(audio,lengths=(16000,32001),spoof=4)
    t = trainer(items)
    t.args.update(num_layers=2,hidden_size=4,output_size=6,agg_size=4,alpha_cl=30,
                  process_id=life.rank(),rank=life.rank(),world_size=len(life.gather(0)),
                  evaluation_mode='controlled_oc',evaluation_seed=1729,
                  short32_artifact_dir=str(Path(folder)/'output'))
    t.evaluation_protocol = {'score':'cosine to trained OC','no_dev':True}
    evaluation_items = items[:count]
    def loader():
        ds = data.EnrollmentSet(dict(t.args,native_length=True,cap_long=True),evaluation_items)
        return torch.utils.data.DataLoader(ds,batch_size=2,sampler=data.ExactDistributedEvalSampler(ds,t.args),collate_fn=data.collate_eval)
    t.eval_loader = {p:loader() for p in ['19LA','21LA','21DF']}
    return t


def failure_worker(rank,rendezvous,folder):
    import torch.distributed as dist
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=rendezvous,rank=rank,world_size=2,
                            timeout=datetime.timedelta(seconds=30))
    try:
        torch.manual_seed(7)
        audio = Path(folder)/f'audio{rank}'
        t = prepare(folder,audio_folder=audio)
        store = t.checkpoints = life.Checkpoints(t)
        store.save(1)
        def bad_features(*a,**kw):
            if rank == 1:
                raise RuntimeError('injected rank1 inference failure')
            return fake_features(*a,**kw)
        failed = False
        with patch('native_steps.features',side_effect=bad_features):
            try:t.test('19LA',1)
            except life.StageError as error:
                assert 'injected rank1' in str(error); failed=True
        assert failed
        store.verify(1)
        # Uneven shards: rank0 has2 batches and rank1 has1; no DDP forward.
        with patch('native_steps.features',side_effect=fake_features):
            result=t.test('21DF',1)
        assert 0<=result[0]<=100 and len(t.last_eval_rows)==5
        original_write=life.atomic_write
        def fail_save(path,writer):
            if rank==0 and Path(path).suffix=='.pt':raise OSError('injected disk full')
            return original_write(path,writer)
        save_failed=False
        with patch.object(life,'atomic_write',side_effect=fail_save):
            try:store.save(2)
            except life.StageError as error:
                assert 'disk full' in str(error);save_failed=True
        assert save_failed and not (store.root/'checkpoint_2.complete.json').exists()
        store.verify(1)
        life.write_json(Path(folder)/f'rank{rank}.json',dict(eval_failure_coordinated=failed,
            save_failure_coordinated=save_failed,next_stage_completed=True))
    finally:dist.destroy_process_group()


class LifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.group=tempfile.TemporaryDirectory()
        torch.distributed.init_process_group('gloo',init_method=Path(cls.group.name,'group').as_uri(),rank=0,world_size=1)
    @classmethod
    def tearDownClass(cls):torch.distributed.destroy_process_group();cls.group.cleanup()
    def setUp(self):
        self.folder=tempfile.TemporaryDirectory();self.addCleanup(self.folder.cleanup)
        self.t=prepare(self.folder.name)

    def test_atomic_partial_write_or_replace_failure_preserves_destination(self):
        target=Path(self.folder.name)/'prior.pt';target.write_bytes(b'old')
        def fail(stream):stream.write(b'partial');raise OSError('disk full')
        with self.assertRaises(OSError):life.atomic_write(target,fail)
        self.assertEqual(target.read_bytes(),b'old')
        with patch.object(life.os,'replace',side_effect=OSError('rename failed')):
            with self.assertRaises(OSError):life.atomic_write(target,lambda s:s.write(b'new'))
        self.assertEqual(target.read_bytes(),b'old');self.assertFalse(list(target.parent.glob('*.tmp')))

    def test_checkpoint_readback_commit_rng_and_no_overwrite(self):
        s=self.t.checkpoints=life.Checkpoints(self.t)
        before=life.rng_state('cpu');s.save(1);manifest=s.verify(1)
        state=torch.load(s.model_dir/'TrainState_1.pt',weights_only=True)
        self.assertEqual(state['epoch'],1);self.assertIn('optimizer',state);self.assertIn('scheduler',state)
        self.assertEqual(len(state['rng_by_rank']),1)
        self.assertTrue(torch.equal(before['torch'],torch.get_rng_state()))
        self.assertEqual(before['python'],random.getstate())
        self.assertEqual(set(manifest['files']),{'CLS_1.pt','TrainState_1.pt'})
        with self.assertRaises(life.StageError):s.save(1)
        s.verify(1)

    def test_save_failure_never_creates_commit_marker_and_keeps_prior_checkpoint(self):
        s=life.Checkpoints(self.t);s.save(1)
        with patch.object(life,'atomic_write',side_effect=OSError('disk full')):
            with self.assertRaises(life.StageError):s.save(2)
        self.assertFalse((s.root/'checkpoint_2.complete.json').exists());s.verify(1)

    def test_corruption_prevents_evaluation_before_feature_call(self):
        s=self.t.checkpoints=life.Checkpoints(self.t);s.save(1)
        with (s.model_dir/'CLS_1.pt').open('ab') as out:out.write(b'corruption')
        with patch('native_steps.features') as features:
            with self.assertRaises(life.StageError):self.t.test('19LA',1)
        features.assert_not_called()

    def test_eval_success_or_inference_failure_restores_modes_rng_params_optimizer(self):
        t=self.t;s=t.checkpoints=life.Checkpoints(t);s.save(1)
        t.classifier.train();t.plm.train()
        params={k:v.clone() for k,v in t.classifier.state_dict().items()}
        optimizer=copy.deepcopy(t.optimizer.state_dict()); modes=[m.training for m in t.classifier.modules()]
        before=life.rng_state('cpu')
        with patch('native_steps.features',side_effect=fake_features):result=t.test('19LA',1)
        self.assertTrue(0<=result[0]<=100)
        self.assertEqual(len(t.last_eval_rows),5)
        self.assertTrue((s.root/'eval_1_19LA_scores.csv').exists())
        with patch('native_steps.features',side_effect=RuntimeError('injected inference')):
            with self.assertRaises(life.StageError):t.test('21LA',1)
        self.assertTrue(t.plm.training);self.assertEqual(modes,[m.training for m in t.classifier.modules()])
        self.assertTrue(all(torch.equal(v,t.classifier.state_dict()[k]) for k,v in params.items()))
        self.assertEqual(optimizer,t.optimizer.state_dict())
        self.assertTrue(torch.equal(before['torch'],torch.get_rng_state()));self.assertEqual(before['python'],random.getstate())
        np.testing.assert_array_equal(before['numpy']['keys'],np.random.get_state()[1])

    def test_head_construction_failure_restores_training_objects(self):
        s=self.t.checkpoints=life.Checkpoints(self.t);s.save(1)
        original=self.t.classifier;self.t.classifier.train();self.t.plm.train()
        with patch('model.B_Linear',side_effect=RuntimeError('injected head setup')):
            with self.assertRaises(life.StageError):self.t.test('19LA',1)
        self.assertIs(self.t.classifier,original);self.assertTrue(original.training);self.assertTrue(self.t.plm.training)

    def test_save_first_stages_eval_failure_continues_next_epoch_and_minimum(self):
        t=self.t;t.args['epoch']=6;events=[]
        t.train=lambda epoch:events.append(('train',epoch))
        def evaluate(phase,epoch):
            self.assertTrue((t.checkpoints.root/f'checkpoint_{epoch}.complete.json').exists())
            events.append((phase,epoch))
            if phase=='19LA' and epoch==5:raise life.StageError('injected eval failure')
            return (float(epoch),0,0,0)
        t.test=evaluate
        life.run_training(t)
        self.assertLess(events.index(('19LA',5)),events.index(('train',6)))
        self.assertEqual(events[-4:],[('21LA',5),('21DF',5),('21LA',6),('21DF',6)])
        report=json.loads((t.checkpoints.root/'complete.json').read_text())
        self.assertEqual(report['status'],'evaluation_incomplete')
        self.assertEqual(report['minimum_eer']['21LA']['epoch'],5)
        self.assertEqual(report['minimum_eer']['19LA']['epoch'],6)
        self.assertFalse(any(phase=='V' for phase,epoch in events))

    def test_save_failure_stops_before_evaluation_and_next_epoch(self):
        t=self.t;t.args['epoch']=6;events=[]
        t.train=lambda epoch:events.append(epoch)
        with patch.object(life.Checkpoints,'save',side_effect=life.StageError('disk failure')),patch.object(t,'test') as evaluate:
            with self.assertRaises(life.StageError):life.run_training(t)
        self.assertEqual(events,[1,2,3,4,5]);evaluate.assert_not_called()

    def test_missing_module_is_detected_by_preflight(self):
        import importlib
        original=importlib.import_module
        def missing(name,*a,**kw):
            if name=='controlled_data_loaders':raise ModuleNotFoundError(name)
            return original(name,*a,**kw)
        with patch.object(importlib,'import_module',side_effect=missing):
            with self.assertRaises(ModuleNotFoundError):adapter.validate_runtime()

    def test_failed_dynamic_import_does_not_leave_partial_module_cache(self):
        import importlib.machinery
        name='baseline240_source.model.loss';saved=sys.modules.pop(name,None)
        try:
            with patch.object(importlib.machinery.SourceFileLoader,'exec_module',side_effect=RuntimeError('import failed')):
                with self.assertRaises(RuntimeError):adapter.load_source('model/loss.py')
            self.assertNotIn(name,sys.modules)
            self.assertTrue(hasattr(adapter.load_source('model/loss.py'),'BPL'))
        finally:
            if saved is not None:sys.modules[name]=saved

    def test_actual_original_train_to_checkpoint_and_all_three_evaluations(self):
        t=self.t
        original=t.classifier
        with patch('random.randint',return_value=0),patch('native_steps.features',side_effect=fake_features):
            t.run()
        self.assertIs(t.classifier,original)
        report=json.loads((t.checkpoints.root/'complete.json').read_text())
        self.assertEqual(report['status'],'complete')
        self.assertEqual([r['phase'] for r in report['results']],['19LA','21LA','21DF'])
        state=torch.load(t.checkpoints.model_dir/'TrainState_1.pt',weights_only=True)
        self.assertTrue(state['optimizer']['state'])
        self.assertTrue(all(torch.equal(v,t.classifier.state_dict()[k]) for k,v in state['classifier'].items()))
        self.assertEqual(len(original.module._forward_hooks),0)

    def test_controlled_evaluation_real_two_worker_module_import(self):
        t=self.t;t.checkpoints=life.Checkpoints(t);t.checkpoints.save(1)
        old=t.eval_loader['19LA']
        t.eval_loader['19LA']=torch.utils.data.DataLoader(old.dataset,batch_size=2,
            sampler=old.sampler,num_workers=2,collate_fn=data.collate_eval,timeout=30)
        with patch('native_steps.features',side_effect=fake_features):
            result=t.test('19LA',1)
        self.assertTrue(0<=result[0]<=100);self.assertEqual(len(t.last_eval_rows),5)

    def test_real_main_validate_only_no_optional_transformers_or_gpu(self):
        vendor=ROOT.parent/'artifacts/short32_code_validation_20261111/vendor'
        script="import sys,runpy; from pathlib import Path; vendor=sys.argv[1]; sys.argv=sys.argv[2:]; sys.path[:0]=[vendor,str(Path(sys.argv[0]).parent)]; runpy.run_path(sys.argv[0],run_name='__main__')"
        result=subprocess.run([sys.executable,'-c',script,str(vendor),str(ROOT/'main.py'),'--short32-bpl','--validate-only'],capture_output=True,text=True,timeout=45)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('original240 two DataLoaders',result.stdout)

    def test_two_cpu_ranks_failure_coordination_and_uneven_evaluation(self):
        with tempfile.TemporaryDirectory() as folder:
            torch.multiprocessing.spawn(failure_worker,args=(Path(folder,'group').as_uri(),folder),nprocs=2,join=True)
            for rank in [0,1]:
                result=json.loads(Path(folder,f'rank{rank}.json').read_text())
                self.assertTrue(all(result.values()))


if __name__=='__main__':unittest.main()
