"""Full-graph original240 batching and alternating role-aware Adam updates."""
from contextlib import ExitStack, contextmanager, nullcontext
import hashlib
from itertools import islice
import random
import numpy as np
import torch
from role_model import config_from_model


def unwrap(model):
    return model.module if hasattr(model, 'module') else model


@contextmanager
def preserve_rng(device):
    """Evaluation must not advance Python, NumPy, CPU or this rank's CUDA RNG."""
    py_state, np_state = random.getstate(), np.random.get_state()
    device = torch.device(device)
    devices = [device.index] if device.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


@contextmanager
def phase_rng(device, seed, epoch, step, rank, phase):
    """Bounded, deterministic phase stream; preserve loader/global RNG states."""
    value = int.from_bytes(hashlib.sha256(
        f'ebm_role_v1|{seed}|{epoch}|{step}|{rank}|{phase}'.encode()).digest()[:8], 'big') % (2**63-1)
    device = torch.device(device)
    devices = [device.index] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(value)
        if devices:
            torch.cuda.manual_seed(value)
        yield


class _CycleBatches:
    """Do not prefetch beyond the number of batches needed from this cycle."""
    def __init__(self, batches, count):
        self.batches, self.count = batches, count

    def __len__(self):
        return self.count

    def __iter__(self):
        return islice(iter(self.batches), self.count)


def _check_loader(loader):
    if not isinstance(loader, torch.utils.data.DataLoader):
        return
    if isinstance(loader.dataset, torch.utils.data.IterableDataset):
        raise ValueError('Role batching requires the original map-style dataset')
    if loader.persistent_workers or not getattr(loader, 'in_order', True):
        raise ValueError('Role batching requires fresh, ordered workers per cycle')
    # Other samplers may consume shared RNG while prefetching the discarded
    # iterator. Original DistributedSampler and SequentialSampler do not.
    if not isinstance(loader.sampler, (torch.utils.data.DistributedSampler,
                                       torch.utils.data.SequentialSampler)):
        raise ValueError('Seed-compatible role batching requires a deterministic sampler')


def _reserve_legacy_seed(loader):
    if isinstance(loader, torch.utils.data.DataLoader):
        # PyTorch2.6 _BaseDataLoaderIter consumes exactly this draw even for an
        # iterator never read. Preserve subsequent worker augmentation seeds
        # without starting its workers, queues or pin-memory thread.
        torch.empty((), dtype=torch.int64).random_(generator=loader.generator)


def _cycle_iterator(loader, count):
    if not isinstance(loader, torch.utils.data.DataLoader):
        return iter(islice(iter(loader), count))
    options = dict(dataset=loader.dataset, batch_sampler=_CycleBatches(loader.batch_sampler, count),
                   num_workers=loader.num_workers, collate_fn=loader.collate_fn,
                   pin_memory=loader.pin_memory, timeout=loader.timeout,
                   worker_init_fn=loader.worker_init_fn, generator=loader.generator,
                   persistent_workers=False)
    if loader.num_workers:
        options.update(multiprocessing_context=loader.multiprocessing_context,
                       prefetch_factor=loader.prefetch_factor)
    if getattr(loader, 'pin_memory_device', ''):
        options['pin_memory_device'] = loader.pin_memory_device
    if hasattr(loader, 'in_order'):
        options['in_order'] = loader.in_order
    return iter(torch.utils.data.DataLoader(**options))


def _finish_cycle(iterator):
    """Reach StopIteration before destruction, including on early caller exit.

    Normal training has no remaining batches. An abandoned caller drains only
    its finite scheduled cycle; no optimizer updates are made. Genuine input
    or shutdown errors propagate, rather than becoming ignored __del__ errors.
    """
    try:
        for _ in iterator:
            pass
    finally:
        shutdown = getattr(iterator, '_shutdown_workers', None)
        if shutdown is not None:
            shutdown()


def logical_batches(loader_bona, loader_spoof):
    """Original rows and worker seeds, with bounded, fully exhausted iterators."""
    if not len(loader_bona) or not len(loader_spoof):
        raise ValueError('Both original class loaders must be nonempty')
    for loader in (loader_bona, loader_spoof):
        _check_loader(loader)
    if len(loader_bona) < len(loader_spoof):
        big, small, big_label = loader_spoof, loader_bona, 0
    else:
        big, small, big_label = loader_bona, loader_spoof, 1
    with ExitStack() as cleanup:
        def cycle(loader, count):
            iterator = _cycle_iterator(loader, count)
            cleanup.callback(_finish_cycle, iterator)
            return iterator
        # Match the original bona-then-spoof construction order as well as the
        # second large-loader iterator, including shared-generator test cases.
        for loader in (loader_bona, loader_spoof):
            if loader is big:
                _reserve_legacy_seed(loader)
            else:
                small_iter = cycle(small, min(len(small), len(big)))
        big_iter = cycle(big, len(big))
        for step in range(len(big)):
            a = next(big_iter)
            try:
                b = next(small_iter)
            except StopIteration:
                small_iter = cycle(small, min(len(small), len(big)-step))
                b = next(small_iter)
            positive, negative = (a, b) if big_label == 1 else (b, a)
            if len(positive) != 3 or len(negative) != 3:
                raise ValueError('Require original (long, short, label) dataset outputs')
            if not bool((positive[2] == 1).all()) or not bool((negative[2] == 0).all()):
                raise ValueError('Original class loaders contain incorrect protocol labels')
            yield tuple(torch.cat((p, n), dim=0) for p, n in zip(positive, negative))


def freeze_phase(model, center_only):
    base = unwrap(model)
    for parameter in base.parameters():
        parameter.requires_grad_((parameter is base.loss.weight) == center_only)


def _finite_gradients(model, center_only):
    base = unwrap(model)
    found = False
    norms = []
    for parameter in base.parameters():
        active = (parameter is base.loss.weight) == center_only
        if not active and parameter.grad is not None:
            raise RuntimeError('Frozen phase retained a stale gradient')
        if parameter.grad is not None:
            found = True
            norms.append(parameter.grad.detach().norm())
            if not bool(torch.isfinite(parameter.grad).all()):
                raise RuntimeError('Nonfinite role gradient')
    if not found:
        raise RuntimeError('Role phase produced no active parameter gradients')
    return float(torch.stack(norms).norm())


def step_hidden(classifier, optimizer, long, short, labels, *, rng=None, observe=None):
    """Two Adam calls using frozen SSL features; recompute center with new theta.

    ``rng(phase)`` may supply isolated phase RNG or be omitted for strict original
    RNG-state regression. ``observe`` is an optional verification callback.
    """
    role = config_from_model(classifier)
    if long.requires_grad or short.requires_grad:
        raise ValueError('Frontend hidden states must be detached/frozen')
    if len(long) != len(short) or len(labels) != len(long):
        raise ValueError('Logical batch rows are misaligned')
    selected = labels == role.compact_label
    if not bool(selected.any()) or bool(selected.all()):
        raise ValueError('Both classes are required in each original logical batch')
    context = rng or (lambda phase: nullcontext())
    center_before = unwrap(classifier).loss.weight.detach().double().clone()
    try:
        optimizer.zero_grad(set_to_none=True)
        freeze_phase(classifier, center_only=False)
        with context('backend'):
            loss1, stats = classifier(long, short, labels, phase='backend')
            if not bool(torch.isfinite(loss1)):
                raise RuntimeError('Nonfinite backend loss')
            loss1.backward()
        backend_gradient_norm = _finite_gradients(classifier, center_only=False)
        if observe is not None:
            observe('backend', classifier, loss1)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        freeze_phase(classifier, center_only=True)
        with context('center'):
            # Deliberately recompute backend after theta update, never reuse z.
            loss2, _ = classifier(long[selected], label=labels[selected], phase='center')
            if not bool(torch.isfinite(loss2)):
                raise RuntimeError('Nonfinite center loss')
            loss2.backward()
        center_gradient_norm = _finite_gradients(classifier, center_only=True)
        if observe is not None:
            observe('center', classifier, loss2)
        optimizer.step()
        center_after = unwrap(classifier).loss.weight.detach().double()
        cosine = torch.nn.functional.cosine_similarity(center_before, center_after).clamp(-1,1)
        return dict(loss1=float(loss1.detach()), loss2=float(loss2.detach()),
                    **{k: float(v) if torch.is_tensor(v) else v for k, v in stats.items()},
                    bona_rows=int((labels == 1).sum()), spoof_rows=int((labels == 0).sum()),
                    optimizer_steps=2, center_updates=1,
                    backend_gradient_norm=backend_gradient_norm, center_gradient_norm=center_gradient_norm,
                    center_norm=float(center_after.norm()), center_update_norm=float((center_after-center_before).norm()),
                    center_angle_delta_degrees=float(torch.rad2deg(torch.acos(cosine)).item()))
    finally:
        # No stale gradients cross batch/evaluation/error boundaries.
        optimizer.zero_grad(set_to_none=True)
        for parameter in unwrap(classifier).parameters():
            parameter.requires_grad_(True)


def train_epoch(trainer, epoch, selected_steps=None):
    """Use unchanged original datasets/samplers. Optional steps are smoke only."""
    trainer.plm.eval()
    trainer.plm.requires_grad_(False)
    trainer.classifier.train()
    trainer.sampler_bona.set_epoch(epoch)
    trainer.sampler_spoof.set_epoch(epoch)
    device, args = trainer.args['device'], trainer.args
    totals = dict(bona_rows=0, spoof_rows=0, updates=0, batches_seen=0)
    for step, (long, short, labels) in enumerate(logical_batches(
            trainer.train_loader_bona, trainer.train_loader_spoof), 1):
        totals['batches_seen'] += 1
        if selected_steps is not None and step not in selected_steps:
            continue
        long, short = (w.to(device=device, dtype=torch.float32) for w in (long, short))
        labels = labels.to(device=device, dtype=torch.long)
        # Original fixed-length SSL frontend, no attention masks or replay.
        with torch.no_grad():
            a = trainer.plm(long, output_hidden_states=True).hidden_states
            b = trainer.plm(short, output_hidden_states=True).hidden_states
            hidden_long, hidden_short = torch.stack(a, dim=1), torch.stack(b, dim=1)
        rng = lambda phase: phase_rng(device, args['rand_seed'], epoch, step, args.get('rank', 0), phase)
        values = step_hidden(trainer.classifier, trainer.optimizer, hidden_long, hidden_short, labels, rng=rng)
        values.update(epoch=epoch, batch=step)
        for key in ['bona_rows', 'spoof_rows']:
            totals[key] += values[key]
        totals['updates'] += 1
        if getattr(trainer, 'record_train', None):
            trainer.record_train(values)
        del long, short, a, b, hidden_long, hidden_short
    if selected_steps is None:
        trainer.lr_scheduler.step()
    elif totals['updates'] != len(selected_steps):
        raise RuntimeError('Smoke did not reach all requested original partial batches')
    return totals
