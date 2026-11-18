"""Meaningful guards for fair checkpoint selection, padding, ROC and queue gates."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from paired_evaluation import (CAP, EPOCHS, views, metric, checkpoint_minima,
                               common_scores, dependency_ready, save)


class PairedEvaluationTests(unittest.TestCase):
    def test_repeat_preserves_full_short_source_and_long_identity(self):
        short = np.arange(24000, dtype=np.float32)
        native, wrap = views(short)
        self.assertTrue(np.array_equal(native, short))
        self.assertTrue(np.array_equal(wrap[:24000], short))
        self.assertTrue(np.array_equal(wrap[24000:48000], short))
        self.assertEqual(len(wrap), CAP)
        a, b = views(np.arange(80000, dtype=np.float32))
        self.assertIs(a, b)
        self.assertEqual(len(a), CAP)

    def test_ties_and_score_polarity(self):
        self.assertAlmostEqual(metric([0, 0, 1, 1], [.1, .1, .9, .9])['eer_percent'], 0)
        self.assertAlmostEqual(metric([0, 0, 1, 1], [.9, .9, .1, .1])['eer_percent'], 100)
        self.assertAlmostEqual(metric([0, 1, 0, 1], [.5]*4)['eer_percent'], 50)
        self.assertAlmostEqual(metric([0, 1], [.5, .5], [1, 90])['eer_percent'], 50)

    def test_independent_roc_against_sklearn(self):
        from sklearn.metrics import roc_curve
        rng = np.random.default_rng(10)
        for _ in range(20):
            y = rng.integers(0, 2, 200); s = rng.integers(-4, 5, 200)
            fp, tp, _ = roc_curve(y, s, drop_intermediate=False)
            d = fp+tp-1; hi = np.flatnonzero(d >= 0)[0]; lo = max(0, hi-1)
            alpha = -d[lo]/(d[hi]-d[lo]) if hi != lo else 0
            self.assertAlmostEqual(metric(y, s)['eer_percent'], 100*(fp[lo]+alpha*(fp[hi]-fp[lo])), places=10)

    def test_minimum_not_final_and_common_grid(self):
        rows = [dict(phase='21LA', family='B', condition='native', epoch=e,
                     eer_percent=v, key='B_e'+str(e)) for e, v in [(5, 1.), (15, 3.), (20, 2.), (50, 4.)]]
        minima = checkpoint_minima(rows)
        self.assertEqual(minima[0]['epoch'], 5)
        self.assertEqual(minima[1]['epoch'], 20)
        self.assertEqual(minima[0]['candidate_count'], 4)
        self.assertEqual(minima[1]['candidate_count'], 3)

    def test_common_rng_and_caller_rng_restored(self):
        class Head(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.loss = torch.nn.Linear(3, 1, bias=False); self.noise_std = 1e-5
            def forward(self, x, **kw): return x+torch.randn_like(x)*self.noise_std
        a = Head(); b = Head(); b.load_state_dict(a.state_dict())
        rng = torch.get_rng_state().clone()
        values = common_scores({'a': a, 'b': b}, torch.eye(3), None, 1729, .2)
        self.assertTrue(np.array_equal(values['a'], values['b']))
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))

    def test_gate_never_releases_for_checkpoint_alone_or_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); stages = root/'results'/'short32_stages'; stages.mkdir(parents=True)
            cfg = {'wait_for_root': str(root), 'current_result': str(root/'results')}
            for epoch in EPOCHS: save(stages/f'checkpoint_{epoch}.complete.json', dict(epoch=epoch, status='committed'))
            self.assertFalse(dependency_ready(cfg))
            (root/'run.exit').write_text('143')
            with self.assertRaises(AssertionError): dependency_ready(cfg)
            (root/'run.exit').write_text('0')
            save(stages/'complete.json', dict(status='complete'))
            self.assertTrue(dependency_ready(cfg))
            (root/'run.exit').write_text('2')
            save(stages/'complete.json', dict(status='evaluation_incomplete'))
            self.assertTrue(dependency_ready(cfg))
            (stages/'checkpoint_45.complete.json').unlink()
            with self.assertRaises(AssertionError): dependency_ready(cfg)

    def test_21la_metadata_field_alignment(self):
        from tools.run_paired_evaluation import items_for_phase
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); meta = root/'keys/LA/CM'; meta.mkdir(parents=True)
            (meta/'trial_metadata.txt').write_text('LA_0009 id alaw ita_tx A07 spoof notrim eval\nLA_0001 bona ulaw loc_tx bonafide bonafide notrim eval\n')
            items = [SimpleNamespace(path='id.flac', label=0), SimpleNamespace(path='bona.flac', label=1)]
            with patch('tools.evaluate_B_C_checkpoints.make_items', return_value=items):
                out = items_for_phase({'path_21LA': temp}, '21LA')
            self.assertEqual(out[0].type, 'A07'); self.assertEqual(out[0].codec, 'alaw')
            self.assertEqual(out[1].type, 'bonafide')

    def test_analysis_reconciles_uid_noop_and_detects_tampering(self):
        import pandas as pd
        from paired_evaluation import analyze_phase, sha
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = {'B_e15': {'family': 'B', 'epoch': 15}, 'B_e50': {'family': 'B', 'epoch': 50}}
            save(root/'checkpoints.json', manifest)
            for rank in [0, 1]:
                rows = []
                for i in range(rank, 8, 2):
                    label = int(i >= 4); n = 70000 if i % 2 else 30000
                    for condition in ['native', 'wrap4']:
                        rows.append(dict(phase='19LA', index=i, uid='id'+str(i), speaker='spk'+str(i),
                            label=label, original_samples=n, attack='bonafide' if label else 'A07',
                            codec='none', condition=condition, used_samples=CAP if condition == 'wrap4' else min(n, CAP),
                            wave_sha256='same' if n >= CAP else condition,
                            score_B_e15=.9 if label else .1, score_B_e50=.8 if label else .2))
                file = root/f'full_19LA_rank{rank}.csv'; pd.DataFrame(rows).to_csv(file, index=False)
                save(root/f'full_19LA_done_rank{rank}.json', dict(rows=len(rows), sha256=sha(file)))
            with patch.dict('paired_evaluation.EXPECTED', {'19LA': 8}):
                result = analyze_phase(root, '19LA')
                self.assertEqual(result['validation']['long_noop_sources'], 4)
                self.assertEqual(result['minima'][0]['epoch'], 15)
                file = root/'full_19LA_rank1.csv'; f = pd.read_csv(file)
                f.loc[(f.condition == 'wrap4') & (f.uid == 'id1'), 'score_B_e15'] = .8
                f.to_csv(file, index=False)
                save(root/'full_19LA_done_rank1.json', dict(rows=len(f), sha256=sha(file)))
                with self.assertRaises(AssertionError): analyze_phase(root, '19LA')

    def test_ipf_duration_match_preserves_attack_codec_margins(self):
        import pandas as pd
        from tools.review_repetition_results import ipf_weights
        rows = []
        for label in [0, 1]:
            for length_bin in [1, 2]:
                for codec, count in [('alaw', 3 if length_bin == 1 else 7), ('ulaw', 6 if length_bin == 1 else 4)]:
                    rows += [dict(label=label, bin=length_bin, attack='bonafide' if label else 'A07', codec=codec)]*count
        frame = pd.DataFrame(rows)
        target = pd.Series([.2, .8], index=[1, 2])
        weights, residual = ipf_weights(frame, target)
        self.assertLess(residual, 1e-10)
        frame['weight'] = weights
        for label, group in frame.groupby('label'):
            self.assertAlmostEqual(group[group.bin == 1].weight.sum()/group.weight.sum(), .2, places=9)
            before = group[group.codec == 'alaw'].shape[0]/len(group)
            after = group[group.codec == 'alaw'].weight.sum()/group.weight.sum()
            self.assertAlmostEqual(before, after, places=9)


if __name__ == '__main__': unittest.main()
