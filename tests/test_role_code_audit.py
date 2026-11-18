"""Regression for role-checkpoint interchange and smoke/full result identity."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader
from model import B_Linear, BPL, OCSoftmax
import role_lifecycle as life
from test_role_experiment import EvalDataset, eval_collate, fake_features, make_trainer


class RoleCodeAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.t = make_trainer(self.folder.name)
        self.t.checkpoints = life.RoleCheckpoints(self.t)

    def smoke_trainer(self):
        self.t.args['mode'] = 'smoke'
        ds = EvalDataset()
        ds.items = [type(ds.items[0])(label=int(i >= 32), path=f'synthetic_{i}') for i in range(64)]
        ds.hidden = torch.randn(64, 2, 5, 4, generator=torch.Generator().manual_seed(4))
        loader = DataLoader(ds, batch_size=8, collate_fn=eval_collate)
        self.t.eval_loader = {p: loader for p in ['19LA','21LA','21DF']}
        self.t.checkpoints.save(0)
        return self.t

    def test_new_CLS_rejects_legacy_bonafide_scorer_instead_of_silent_sign_error(self):
        self.t.checkpoints.save(1)
        artifact = torch.load(Path(self.folder.name)/'model/CLS_1.pt', weights_only=True)
        legacy = B_Linear(2,4,6,4,loss=OCSoftmax(6),loss_bpl=BPL(30))
        # Reproduce the old evaluator's state.get('classifier', state) loading.
        with self.assertRaises(RuntimeError):
            legacy.load_state_dict(artifact.get('classifier', artifact), strict=True)
        training = torch.load(Path(self.folder.name)/'model/TrainState_1.pt',weights_only=True)
        with self.assertRaises(RuntimeError):
            legacy.load_state_dict(training.get('classifier',training),strict=True)

    def test_smoke_checkpoint_requires_explicit_opt_in_for_exported_scoring(self):
        self.smoke_trainer()
        with self.assertRaisesRegex(ValueError, 'smoke|Smoke'):
            life.load_scoring_head(self.folder.name, 0)
        head, manifest = life.load_scoring_head(self.folder.name,0,allow_smoke=True)
        self.assertEqual(manifest['purpose'],'smoke')
        self.assertEqual(head.role.compact_label,0)

    def test_smoke_eval_is_marked_subset_even_when_every_loader_row_was_scored(self):
        t = self.smoke_trainer()
        life.evaluate(t, '19LA', 0, feature_fn=fake_features)
        receipt = json.loads((Path(self.folder.name)/'stages/eval_0_19LA.json').read_text())
        self.assertEqual(receipt['rows'],64)
        self.assertEqual(receipt['scope'],'smoke_subset')
        self.assertEqual(receipt['checkpoint_purpose'],'smoke')

    def test_role_aware_loader_keeps_existing_v1_tensor_checkpoints_readable(self):
        self.t.checkpoints.save(1)
        model_dir=Path(self.folder.name)/'model'
        cls_path,train_path=model_dir/'CLS_1.pt',model_dir/'TrainState_1.pt'
        payload=torch.load(cls_path,weights_only=True)
        state=payload['role_state_dict']
        training=torch.load(train_path,weights_only=True)
        training['classifier']=training.pop('role_classifier')
        training.pop('artifact_type');training.pop('purpose')
        torch.save(state,cls_path);torch.save(training,train_path)
        marker=Path(self.folder.name)/'stages/checkpoint_1.complete.json'
        manifest=json.loads(marker.read_text())
        for path in [cls_path,train_path]:
            manifest['files'][path.name]=dict(sha256=life.sha256(path),bytes=path.stat().st_size)
        marker.write_text(json.dumps(manifest))
        head,receipt=life.load_scoring_head(self.folder.name,1)
        self.assertEqual(receipt['artifact_format'],'legacy_role_v1')
        self.assertEqual(head.role.compact_label,0)
        for key,value in self.t.classifier.state_dict().items():
            self.assertTrue(torch.equal(value,head.state_dict()[key]))

    def test_train_mode_rejects_a_subset_before_feature_extraction(self):
        self.t.args['mode']='train';self.t.checkpoints.save(1)
        with patch('native_steps.features') as features:
            with self.assertRaisesRegex(RuntimeError,'official full'):
                life.evaluate(self.t,'19LA',1)
        features.assert_not_called()

    def test_custom_and_partial_eval_are_never_labelled_full(self):
        self.t.checkpoints.save(1)
        result=life.evaluate(self.t,'19LA',1,feature_fn=fake_features)
        self.assertEqual(result['scope'],'custom')
        result=life.evaluate(self.t,'21LA',1,feature_fn=fake_features,max_batches=1)
        self.assertEqual(result['scope'],'partial_custom')
        self.assertEqual(result['rows'],2)

    def test_contradictory_checkpoint_purpose_is_rejected(self):
        self.t.checkpoints.save(1)
        marker=Path(self.folder.name)/'stages/checkpoint_1.complete.json'
        manifest=json.loads(marker.read_text());manifest['purpose']='smoke'
        marker.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError,'purpose'):
            life.load_scoring_head(self.folder.name,1)

    def test_legacy_score_sign_can_change_perfect_toy_EER_to100(self):
        # This is a constructed sign test, not a measured trained-model result.
        role=self.t.classifier.loss
        legacy=OCSoftmax(6)
        legacy.load_state_dict(role.state_dict())
        axis=role.weight.detach().clone()
        z=torch.cat([axis,axis,-axis,-axis])
        labels=[0,0,1,1]
        correct=role.score(z)['score_bona'].detach().flatten().tolist()
        wrong=legacy(z)[1].detach().flatten().tolist()
        self.assertEqual(life.calculate_eer(correct,labels),0)
        self.assertEqual(life.calculate_eer(wrong,labels),100)


if __name__ == '__main__':
    unittest.main()
