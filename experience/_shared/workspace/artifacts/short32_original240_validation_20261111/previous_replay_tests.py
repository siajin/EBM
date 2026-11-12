"""Local synthetic-only tests. Never loads lab audio/checkpoints or launches training."""
import argparse
import ast
import contextlib
import copy
import datetime
import io
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import arguments
import data_loaders
import native_steps
import positive_bpl
import short32_bpl as short32
import train
from model import BPL, OCSoftmax, B_Linear


class ToyHead(nn.Module):
    def __init__(self, noise=0.):
        super().__init__()
        self.linear = nn.Linear(5, 6)
        self.dropout = nn.Dropout(.25)
        self.loss = OCSoftmax(6, class_weight=[5/9, 5])
        self.loss_bpl = BPL(30)
        self.noise = noise

    def forward(self, x, mask=None, return_embedding=False, label=None, logical_batch_size=None):
        z = self.dropout(self.linear(x))
        if self.noise:
            z = z + self.noise * torch.randn_like(z)
        return z if return_embedding else self.loss(z, label, logical_batch_size=logical_batch_size)


def make_batch(n=12, k=2, spoof=12):
    total = n + spoof
    native = torch.tensor([32000] * k + [32001] * (n-k) + [10000] * spoof)
    return {'labels': torch.tensor([1] * n + [0] * spoof), 'original_samples': native,
            'short32_indices': torch.arange(k), 'long': list(torch.randn(total, 5)),
            'short': list(torch.randn(total, 5)), 'two': list(torch.randn(k, 5))}


def toy_features(plm, waves, device):
    return torch.stack(waves).to(device), None


def ddp_worker(rank, rendezvous, replay=False):
    """Two CPU ranks, different active counts, gradient only: no optimizer step."""
    import torch.distributed as dist
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2,
                            timeout=datetime.timedelta(seconds=45))
    try:
        torch.manual_seed(17)
        if replay:
            base = ToyHead(.01).double()
            before_center = base.loss.weight.detach().clone()
            before_linear = base.linear.weight.detach().clone()
            ddp = nn.parallel.DistributedDataParallel(base, find_unused_parameters=True)
            torch.manual_seed(200+rank)
            batch = make_batch(6, 0 if rank == 0 else 2, 12)
            for name in ['long', 'short', 'two']:
                batch[name] = [w.double() for w in batch[name]]
            args = {'device': 'cpu', 'short32_bpl': True, 'short_oc_weight': 0.,
                    'padding_loss_weight': 0., 'micro_batch_size': 8}
            trainer = SimpleNamespace(classifier=ddp, plm=nn.Identity(), args=args,
                                      optimizer=torch.optim.Adam(ddp.parameters(), lr=1e-4))
            with patch.object(native_steps, 'features', side_effect=toy_features):
                stats = native_steps.train_batch(trainer, batch)
            assert stats['optimizer_steps'] == 2 and stats['center_updates'] == 1
            assert not torch.equal(before_center, base.loss.weight)
            assert not torch.equal(before_linear, base.linear.weight)
            for parameter in base.parameters():
                gathered = [torch.empty_like(parameter) for _ in range(2)]
                dist.all_gather(gathered, parameter.detach())
                torch.testing.assert_close(gathered[0], gathered[1], rtol=1e-10, atol=1e-10)
            return
        base = ToyHead().double().eval()
        base.loss.weight.requires_grad_(False)
        reference = copy.deepcopy(base)
        ddp = nn.parallel.DistributedDataParallel(base)

        def rank_input(r):
            torch.manual_seed(100+r)
            batch = make_batch(3, 0 if r == 0 else 2, 3)
            arrays = [torch.stack(batch['long']), torch.stack(batch['short'])]
            if batch['two']:
                arrays.append(torch.stack(batch['two']))
            return batch, torch.cat(arrays).double()

        batch, x = rank_input(rank)
        z = ddp(x, return_embedding=True)
        emb = {'long': z[:6], 'short': z[6:12]}
        if len(batch['two']):
            emb['two'] = z[12:]
        short32.objective(base, emb, batch)[0].backward()
        for r in (0, 1):
            rb, rx = rank_input(r)
            rz = reference(rx, return_embedding=True)
            re = {'long': rz[:6], 'short': rz[6:12]}
            if len(rb['two']):
                re['two'] = rz[12:]
            (short32.objective(reference, re, rb)[0] / 2).backward()
        for name, parameter in base.named_parameters():
            if parameter.requires_grad:
                torch.testing.assert_close(parameter.grad, dict(reference.named_parameters())[name].grad,
                                           rtol=1e-10, atol=1e-10)
    finally:
        dist.destroy_process_group()


class Short32Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.args = arguments.get_args()[0]
        short32.configure(self.args)
        self.args.update(rank=0, world_size=1, batch_size=10, ratio_bona=.5,
                         ratio_spoof=.5, num_workers=0, micro_batch_size=8)

    def audio_items(self):
        items = []
        for index, (label, n) in enumerate([(1, 16000), (1, 32000), (1, 32001),
                                           (1, 80000), (0, 16000), (0, 32000)]):
            x = .15 * np.sin(np.arange(n) * .017)
            path = self.folder / f'u{index}.wav'
            sf.write(path, x, 16000, subtype='FLOAT')
            items.append(SimpleNamespace(path=str(path), label=label, speaker=f's{index}', type='test'))
        return items

    def test_inclusive_gate_wrap_prefix_and_spoof_exclusion(self):
        items = self.audio_items()
        ds = short32.Short32TrainSet(self.args, items)
        rows = [ds[i] for i in range(len(ds))]
        self.assertEqual([r['two'] is not None for r in rows], [True, True, False, False, False, False])
        for row in rows:
            self.assertEqual(len(row['long']), 64600)
            self.assertEqual(len(row['short']), 48440)
            if row['two'] is not None:
                np.testing.assert_array_equal(row['two'], row['short'][:32000])
        # With augmentation disabled, prove exact cyclic source preservation.
        with patch.object(data_loaders.random, 'randint', return_value=0):
            row = ds[0]
        raw = sf.read(items[0].path, dtype='float32')[0]
        np.testing.assert_array_equal(row['two'], np.tile(raw, 2))
        batch = short32.collate_train(rows)
        self.assertEqual(batch['short32_indices'].tolist(), [0, 1])
        self.assertEqual(len(batch['two']), 2)

    def test_original_views_and_rng_unchanged_all_rawboost_types(self):
        items = self.audio_items()
        ds = short32.Short32TrainSet(self.args, items)
        baseline = data_loaders.TrainSet(dict(self.args, native_length=False, cap_long=True), items)
        for kind in range(9):
            for index in [0, 2, 4]:
                def randint(lo, hi):
                    if (lo, hi) == (0, 8):
                        return kind
                    return lo
                random.seed(37); np.random.seed(37)
                with patch.object(data_loaders.random, 'randint', side_effect=randint):
                    original = baseline[index]
                state = random.getstate(), np.random.get_state()
                random.seed(37); np.random.seed(37)
                with patch.object(data_loaders.random, 'randint', side_effect=randint):
                    actual = ds[index]
                np.testing.assert_array_equal(actual['long'], np.asarray(original[0], dtype='float32'))
                np.testing.assert_array_equal(actual['short'], np.asarray(original[1], dtype='float32'))
                self.assertEqual(state[0], random.getstate())
                np.testing.assert_equal(state[1], np.random.get_state())

    def test_spawning_two_real_data_workers(self):
        ds = short32.Short32TrainSet(self.args, self.audio_items())
        loader = torch.utils.data.DataLoader(ds, batch_size=len(ds), num_workers=2,
                    multiprocessing_context='spawn', collate_fn=short32.collate_train,
                    timeout=30,
                    generator=torch.Generator().manual_seed(1))
        batches = list(loader)  # Exhaust, join, and clean up all worker processes.
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0]['short32_indices'].tolist(), [0, 1])

    def test_loader_dispatch_and_default_is_opt_in(self):
        self.assertFalse(arguments.get_args()[0]['short32_bpl'])
        items = self.audio_items()
        loaders = data_loaders.get_loaders(self.args, items, [], {})
        self.assertIsInstance(loaders[0].dataset, short32.Short32TrainSet)
        self.assertIsInstance(loaders[1], short32.OriginalClassBatchSampler)
        for batch in loaders[0]:
            self.assertGreater(int(batch['labels'].sum()), 0)
        for bad in [dict(bpl_pairing='matched_independent'), dict(duration_policy_path='other'),
                    dict(short_oc_weight=.25), dict(padding_loss_weight=.1), dict(num_train_frames=500)]:
            args = dict(self.args, **bad)
            with self.assertRaises(ValueError):
                short32.validate_options(args)

    def test_optional_augmentation_exports_preserved(self):
        import data_processing
        self.assertIn('WaveformAugmetation', data_processing.__all__)
        self.assertIn('CodecAugmentationLight', data_processing.__all__)
        self.assertEqual(data_processing.CodecAugmentationLight.__module__,
                         'data_processing.augmentation.codec_aug')

    def test_original_two_rank_sampler_and_partial_batches(self):
        ds = SimpleNamespace(items=[SimpleNamespace(label=1)] * 2580 + [SimpleNamespace(label=0)] * 22800)
        for epoch in [1, 2, 50]:
            coverage = {1: set(), 0: set()}; visits = {1: 0, 0: 0}
            for rank in [0, 1]:
                args = dict(self.args, rank=rank, world_size=2, batch_size=120, ratio_bona=.1, ratio_spoof=.9)
                sampler = short32.OriginalClassBatchSampler(ds, args); sampler.set_epoch(epoch)
                observed = list(sampler)
                self.assertEqual(len(observed), 108)
                self.assertEqual(len(observed[105]), 72)
                self.assertEqual(len(observed[-1]), 114)
                groups = {}
                for y, ids, size in [(1, list(range(2580)), 12), (0, list(range(2580, 25380)), 108)]:
                    original = torch.utils.data.DistributedSampler(ids, num_replicas=2, rank=rank,
                                                                   shuffle=True, seed=0, drop_last=False)
                    original.set_epoch(epoch); order = [ids[i] for i in original]
                    groups[y] = [order[i:i+size] for i in range(0, len(order), size)]
                expected = [groups[1][i % len(groups[1])] + groups[0][i % len(groups[0])] for i in range(108)]
                self.assertEqual(observed, expected)
                for batch in observed:
                    for index in batch:
                        y = ds.items[index].label; coverage[y].add(index); visits[y] += 1
            self.assertEqual({y: len(v) for y, v in coverage.items()}, {1: 2580, 0: 22800})
            self.assertEqual(visits, {1: 2580, 0: 23232})

    def test_no_eligible_is_exact_old_loss_and_gradient(self):
        torch.manual_seed(8); head = ToyHead().double(); batch = make_batch(6, 0, 8)
        z = {name: torch.randn(14, 6, dtype=torch.double, requires_grad=True) for name in ['long', 'short']}
        old = native_steps._objective(head, z, batch, {'short_oc_weight': 0., 'padding_loss_weight': 0.})[0]
        new, _, stats = short32.objective(head, z, batch)
        self.assertTrue(torch.equal(old, new)); self.assertEqual(stats['short32_pairs'], 0)
        go = torch.autograd.grad(old, tuple(z.values()), retain_graph=True)
        gn = torch.autograd.grad(new, tuple(z.values()))
        for a, b in zip(go, gn):
            self.assertTrue(torch.equal(a, b))

    def test_all_eligible_half_half_and_all_pairs(self):
        torch.manual_seed(11); head = ToyHead().double(); batch = make_batch(4, 4, 6)
        z = {name: torch.randn(n, 6, dtype=torch.double, requires_grad=True)
             for name, n in [('long', 10), ('short', 10), ('two', 4)]}
        actual, _, stats = short32.objective(head, z, batch)
        expected = head.loss(z['long'], batch['labels'], logical_batch_size=10)[0]
        expected = expected + .5 * head.loss_bpl(z['long'][:4], z['short'][:4]) + .5 * head.loss_bpl(z['short'][:4], z['two'])
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        self.assertEqual(stats['short32_pairs'], 16)
        permuted = dict(z, two=z['two'].flip(0))
        torch.testing.assert_close(actual, short32.objective(head, permuted, batch)[0], rtol=1e-12, atol=1e-12)
        grad = torch.autograd.grad(head.loss_bpl(z['short'][:4], z['two']), [z['short'], z['two']])
        self.assertTrue(all(float(g.abs().sum()) > 0 for g in grad))

    def test_mixed_batch_matches_explicit_per_anchor_loss(self):
        for n, k in [(12, 1), (12, 3), (6, 2)]:
            head = ToyHead().double(); batch = make_batch(n, k, 5)
            z = {name: torch.randn(size, 6, dtype=torch.double, requires_grad=True)
                 for name, size in [('long', n+5), ('short', n+5), ('two', k)]}
            actual = short32.objective(head, z, batch)[0]
            rows = []
            for i in range(n):
                a = head.loss_bpl(z['long'][i:i+1], z['short'][:n])
                rows.append(.5*a + .5*head.loss_bpl(z['short'][i:i+1], z['two']) if i < k else a)
            expected = head.loss(z['long'], batch['labels'], logical_batch_size=n+5)[0] + torch.stack(rows).mean()
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
            a = torch.autograd.grad(actual, tuple(z.values()), retain_graph=True)
            b = torch.autograd.grad(expected, tuple(z.values()))
            for ga, gb in zip(a, b):
                torch.testing.assert_close(ga, gb, rtol=1e-10, atol=1e-10)

    def test_invalid_masks_and_missing_views_are_rejected(self):
        batch = make_batch(3, 1, 3)
        for updates in [dict(short32_indices=torch.tensor([3])), dict(short32_indices=torch.tensor([0, 0])),
                        dict(two=[]), dict(labels=torch.tensor([0, 1, 1, 1, 0, 0]))]:
            with self.assertRaises(ValueError):
                short32.validate_batch(dict(batch, **updates))

    def test_replay_matches_full_graph_scratch_model(self):
        torch.manual_seed(5); head = ToyHead(.01); ref = copy.deepcopy(head)
        batch = make_batch(6, 2, 12)
        args = dict(self.args, device='cpu')
        optimizer = torch.optim.Adam(head.parameters(), lr=1e-4)
        trainer = SimpleNamespace(classifier=head, plm=nn.Identity(), optimizer=optimizer, args=args)
        reference_optimizer = torch.optim.Adam(ref.parameters(), lr=1e-4)
        torch.manual_seed(91)
        with patch.object(native_steps, 'features', side_effect=toy_features):
            stats = native_steps.train_batch(trainer, batch)
        torch.manual_seed(91); native_steps._freeze(ref, False); reference_optimizer.zero_grad(set_to_none=True)
        z = {name: torch.cat([ref(torch.stack(batch[name][i:i+8]), return_embedding=True)
                             for i in range(0, len(batch[name]), 8)]) for name in ['long', 'short', 'two']}
        short32.objective(ref, z, batch)[0].backward(); reference_optimizer.step()
        native_steps._freeze(ref, True); reference_optimizer.zero_grad(set_to_none=True)
        value, _ = ref(torch.stack(batch['long'][:6]), label=torch.ones(6, dtype=torch.long), logical_batch_size=6)
        value.backward(); reference_optimizer.step()
        for name, value in head.state_dict().items():
            torch.testing.assert_close(value, ref.state_dict()[name], rtol=1e-5, atol=1e-7)
        self.assertEqual(stats['optimizer_steps'], 2); self.assertEqual(stats['center_updates'], 1)

    def test_real_backend_synthetic_gradients_no_parameter_updates(self):
        torch.manual_seed(12)
        head = B_Linear(25, 1024, 160, 128, loss=OCSoftmax(160), loss_bpl=BPL(30))
        before = {k: v.detach().clone() for k, v in head.state_dict().items()}
        batch = make_batch(3, 2, 3)
        z = {name: head(torch.randn(n, 25, t, 1024), return_embedding=True)
             for name, n, t in [('long', 6, 5), ('short', 6, 4), ('two', 2, 3)]}
        loss = short32.objective(head, z, batch)[0]; loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(head.linear.weight.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in head.parameters()))
        for k, v in head.state_dict().items():
            self.assertTrue(torch.equal(v, before[k]))

    def test_checkpoint_before19LA_monitor_and_default_dispatch(self):
        events = []
        trainer = train.ModelTrainer(); trainer.args = {'epoch': 1, 'short32_bpl': True}
        trainer.classifier = nn.Identity()
        trainer.optimizer = trainer.lr_scheduler = SimpleNamespace(state_dict=lambda: {})
        trainer.logger = SimpleNamespace(save_model=lambda name, state: events.append(name))
        trainer.train = lambda epoch: events.append('synthetic_noop_train')
        trainer._synchronize = lambda: None
        trainer.test = lambda phase, epoch: events.append(phase)
        trainer.run()
        self.assertLess(events.index('TrainState_1'), events.index('19LA'))
        self.assertNotIn('V', events)
        trainer.args['short32_bpl'] = False; trainer.evaluation(1)
        self.assertEqual(events[-1], 'V')

    def test_real_cli_validate_only_block_cannot_spawn(self):
        source = (ROOT / 'main.py').read_text()
        tree = ast.parse(source)
        block = next(n for n in tree.body if isinstance(n, ast.If)
                     and isinstance(n.test, ast.Compare) and ast.unparse(n.test) == "__name__ == '__main__'")
        namespace = dict(__name__='__main__', arguments=arguments, argparse=argparse, Path=Path,
                         validate_options=positive_bpl.validate_options,
                         validate_evaluation_options=positive_bpl.validate_evaluation_options,
                         evaluation_mode=positive_bpl.evaluation_mode)
        with patch.object(sys, 'argv', ['main.py', '--short32-bpl', '--validate-only']), \
             patch.object(torch.multiprocessing, 'spawn', side_effect=AssertionError('Must not spawn')), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as exc:
                exec(compile(ast.Module(body=[block], type_ignores=[]), str(ROOT/'main.py'), 'exec'), namespace)
        self.assertEqual(exc.exception.code, 0)
        self.assertTrue(namespace['args']['short32_bpl'])
        self.assertEqual(namespace['args']['short32_train_input_policy'], 'fixed_wrap_64600_48440')

    @unittest.skipUnless(torch.distributed.is_available() and torch.distributed.is_gloo_available(), 'Gloo unavailable')
    def test_two_cpu_ranks_mixed_eligibility_gradients(self):
        # File rendezvous avoids host/network dependencies; only synthetic scratch models.
        rendezvous = (self.folder/'gloo').resolve().as_uri()
        torch.multiprocessing.spawn(ddp_worker, args=(rendezvous,), nprocs=2, join=True)

    @unittest.skipUnless(torch.distributed.is_available() and torch.distributed.is_gloo_available(), 'Gloo unavailable')
    def test_two_cpu_ranks_replay_and_center_updates_with_empty_branch(self):
        rendezvous = (self.folder/'gloo_replay').resolve().as_uri()
        torch.multiprocessing.spawn(ddp_worker, args=(rendezvous, True), nprocs=2, join=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
