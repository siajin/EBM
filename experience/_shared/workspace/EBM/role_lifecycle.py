"""Save-first, role-aware checkpoints and controlled, no-dev evaluation."""
from pathlib import Path
import json

import torch
from role_model import RoleBLinear, config_from_model, validate_model_metadata
from role_training import unwrap
from short32_lifecycle import atomic_write, coordinated, gather, rank, rng_state, sha256, write_json

CLASSIFIER_FORMAT = 'ebm_role_classifier_v2'
TRAIN_STATE_FORMAT = 'ebm_role_train_state_v2'
OFFICIAL_ROWS = {'19LA': 71237, '21LA': 148176, '21DF': 533928}


def checkpoint_purpose(epoch, mode=None):
    purpose = 'smoke' if epoch == 0 else 'train'
    if mode is not None and mode != purpose:
        raise ValueError('Checkpoint epoch and execution mode disagree')
    return purpose


def model_metadata(classifier):
    base = unwrap(classifier)
    return dict(role=config_from_model(base).metadata(), architecture=dict(base.architecture))


def read_checkpoint(root, epoch, expected=None):
    """Require a final commit and role metadata before trusting tensor-only CLS."""
    root = Path(root)
    if type(epoch) is not int or epoch < 0:
        raise ValueError('Invalid checkpoint epoch')
    marker = root/'stages'/f'checkpoint_{epoch}.complete.json'
    manifest = json.loads(marker.read_text(encoding='utf-8'))
    names = {f'CLS_{epoch}.pt', f'TrainState_{epoch}.pt', f'Model_{epoch}.json'}
    if (manifest.get('epoch') != epoch or manifest.get('status') != 'committed'
            or set(manifest.get('files', {})) != names):
        raise ValueError('Missing/incomplete checkpoint commit')
    for name, info in manifest['files'].items():
        path = root/'model'/name
        if path.stat().st_size != info['bytes'] or sha256(path) != info['sha256']:
            raise ValueError('Checkpoint integrity mismatch: '+name)
    metadata = json.loads((root/'model'/f'Model_{epoch}.json').read_text(encoding='utf-8'))
    validate_model_metadata(metadata)
    if expected is not None and metadata != expected:
        raise ValueError('Checkpoint class role or architecture conflicts with the model')
    payload = torch.load(root/'model'/f'CLS_{epoch}.pt', map_location='cpu', weights_only=True)
    training = torch.load(root/'model'/f'TrainState_{epoch}.pt', map_location='cpu', weights_only=True)
    purpose = checkpoint_purpose(epoch, training.get('args', {}).get('mode'))
    if manifest.get('purpose', purpose) != purpose:
        raise ValueError('Checkpoint purpose mismatch')
    if payload.get('artifact_type') == CLASSIFIER_FORMAT:
        if (payload.get('model_metadata') != metadata or payload.get('purpose') != purpose
                or training.get('artifact_type') != TRAIN_STATE_FORMAT
                or training.get('purpose') != purpose):
            raise ValueError('Role artifact metadata/purpose mismatch')
        state, training_state = payload['role_state_dict'], training['role_classifier']
        artifact_format = CLASSIFIER_FORMAT
    else:
        # Read historical v1 role checkpoints only through this metadata-aware
        # loader. Never rewrite an existing model/manifest to upgrade its format.
        if ('artifact_type' in payload or 'artifact_type' in training
                or not all(torch.is_tensor(v) for v in payload.values())):
            raise ValueError('Unsupported/mixed role artifact formats')
        state, training_state = payload, training['classifier']
        artifact_format = 'legacy_role_v1'
    if (training['epoch'] != epoch or training['model_metadata'] != metadata
            or set(state) != set(training_state)):
        raise ValueError('Classifier and TrainState metadata mismatch')
    if any(not torch.is_tensor(v) or not bool(torch.isfinite(v).all())
           or not torch.equal(v, training_state[k]) for k, v in state.items()):
        raise ValueError('Classifier and TrainState tensor mismatch')
    return metadata, state, dict(manifest, purpose=purpose, artifact_format=artifact_format)


def load_scoring_head(root, epoch, device='cpu', expected=None, *, allow_smoke=False):
    metadata, state, manifest = read_checkpoint(root, epoch, expected)
    if manifest['purpose'] == 'smoke' and not allow_smoke:
        raise ValueError('Smoke checkpoint is not a trained model; use allow_smoke=True only for smoke diagnostics')
    role, shape = validate_model_metadata(metadata)
    # Construction must not consume the caller's training/evaluation RNG.
    device = torch.device(device)
    devices = [device.index] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        head = RoleBLinear(role, **shape).to(device)
    head.load_state_dict(state, strict=True)
    head.eval().requires_grad_(False)
    return head, manifest


class RoleCheckpoints:
    def __init__(self, trainer):
        self.trainer = trainer
        self.root = Path(trainer.args['run_root'])
        self.model_dir, self.stages = self.root/'model', self.root/'stages'
        own = model_metadata(trainer.classifier)
        if any(meta != own for meta in gather(own)):
            raise ValueError('Ranks disagree about center class/model architecture')
        self.expected = own
        def prepare():
            self.model_dir.mkdir(parents=True, exist_ok=True)
            self.stages.mkdir(parents=True, exist_ok=True)
        coordinated('checkpoint_directory', prepare)

    def record(self, stage, status, **details):
        def persist():
            if rank() == 0:
                write_json(self.stages/(stage+'.json'), dict(stage=stage, status=status, **details))
        coordinated('record:'+stage, persist)

    def save(self, epoch):
        rng = gather(coordinated('save_rng', lambda: rng_state(self.trainer.args['device'])))
        def persist():
            if rank() != 0:
                return
            t = self.trainer
            purpose = checkpoint_purpose(epoch, t.args.get('mode'))
            metadata = model_metadata(t.classifier)
            if metadata != self.expected:
                raise ValueError('Model role changed during the run')
            files = [self.model_dir/f'CLS_{epoch}.pt', self.model_dir/f'TrainState_{epoch}.pt',
                     self.model_dir/f'Model_{epoch}.json']
            marker = self.stages/f'checkpoint_{epoch}.complete.json'
            if any(p.exists() for p in [*files, marker]):
                raise FileExistsError('Refuse to overwrite an existing checkpoint')
            state = {k: v.detach().cpu().clone() for k, v in unwrap(t.classifier).state_dict().items()}
            args = {k: v for k, v in t.args.items() if not any(
                s in k.lower() for s in ['token', 'password', 'secret', 'api_key'])}
            payload = dict(artifact_type=CLASSIFIER_FORMAT, purpose=purpose,
                           model_metadata=metadata, role_state_dict=state,
                           load_with='role_lifecycle.load_scoring_head')
            training = dict(artifact_type=TRAIN_STATE_FORMAT, purpose=purpose,
                            epoch=epoch, role_classifier=state, optimizer=t.optimizer.state_dict(),
                            scheduler=t.lr_scheduler.state_dict(), model_metadata=metadata,
                            args=args, rng_by_rank=rng, torch_version=str(torch.__version__))
            atomic_write(files[0], lambda stream: torch.save(payload, stream))
            atomic_write(files[1], lambda stream: torch.save(training, stream))
            write_json(files[2], metadata)
            a = torch.load(files[0], map_location='cpu', weights_only=True)
            b = torch.load(files[1], map_location='cpu', weights_only=True)
            if (a['model_metadata'] != metadata or a['purpose'] != purpose
                    or b['model_metadata'] != metadata or b['epoch'] != epoch
                    or b['purpose'] != purpose or set(a['role_state_dict']) != set(state)
                    or any(not torch.equal(a['role_state_dict'][k], v) or not torch.equal(b['role_classifier'][k], v)
                           or not bool(torch.isfinite(v).all()) for k, v in state.items())):
                raise ValueError('Checkpoint readback failed')
            manifest = dict(epoch=epoch, status='committed', purpose=purpose, files={
                p.name: dict(sha256=sha256(p), bytes=p.stat().st_size) for p in files})
            write_json(marker, manifest)  # Publish last; partial output is never a committed model.
        coordinated('save_checkpoint', persist)
        self.verify(epoch)

    def verify(self, epoch):
        return coordinated('verify_checkpoint', lambda: read_checkpoint(self.root, epoch, self.expected)[2])

    def load_model(self, epoch):
        metadata, state, manifest = coordinated('read_checkpoint', lambda:
            read_checkpoint(self.root, epoch, self.expected))
        coordinated('apply_checkpoint', lambda: unwrap(self.trainer.classifier).load_state_dict(state, strict=True))
        return manifest


def calculate_eer(scores, labels):
    from sklearn.metrics import roc_curve
    import numpy as np
    scores, labels = np.asarray(scores), np.asarray(labels)
    if (scores.ndim != 1 or labels.shape != scores.shape or not len(scores)
            or set(labels) != {0, 1} or not np.isfinite(scores).all()):
        raise ValueError('EER requires finite canonical scores and both protocol classes')
    fp, tp, _ = roc_curve(labels, scores, pos_label=1, drop_intermediate=False)
    delta = fp + tp - 1
    hi = int(np.flatnonzero(delta >= 0)[0])
    if hi == 0:
        return float(100 * fp[0])
    fraction = -delta[hi-1] / (delta[hi] - delta[hi-1])
    return float(100 * (fp[hi-1] + fraction * (fp[hi] - fp[hi-1])))


def evaluate(trainer, phase, epoch, *, feature_fn=None, max_batches=None):
    from role_training import preserve_rng
    with preserve_rng(trainer.args['device']):
        return _evaluate(trainer, phase, epoch, feature_fn=feature_fn, max_batches=max_batches)


def _evaluate(trainer, phase, epoch, *, feature_fn=None, max_batches=None):
    """Unwrapped eval head supports uneven shards; every local failure is shared."""
    if phase not in ('19LA', '21LA', '21DF'):
        raise ValueError('Role experiment has no dev evaluation')
    if max_batches is not None and (type(max_batches) is not int or max_batches < 1):
        raise ValueError('max_batches must be a positive integer')
    if feature_fn is None:
        from native_steps import features as feature_fn
    from role_training import phase_rng
    checkpoints = trainer.checkpoints
    name = f'eval_{epoch}_{phase}'
    modes = [(m, m.training) for m in trainer.plm.modules()]
    head = iterator = None
    checkpoints.record(name, 'running', phase=phase, epoch=epoch)
    try:
        def setup():
            nonlocal head, iterator
            trainer.plm.eval()
            mode = trainer.args.get('mode')
            head, manifest = load_scoring_head(checkpoints.root, epoch, trainer.args['device'],
                                               checkpoints.expected, allow_smoke=(mode == 'smoke'))
            loader = trainer.eval_loader[phase]
            if mode == 'smoke':
                if manifest['purpose'] != 'smoke' or len(loader.dataset) != 64:
                    raise ValueError('Smoke evaluation requires its epoch0 checkpoint and64-trial subset')
                scope = 'smoke_subset'
            elif mode == 'train':
                if manifest['purpose'] != 'train' or len(loader.dataset) != OFFICIAL_ROWS[phase]:
                    raise ValueError('Training evaluation does not cover the official full trial set')
                scope = 'full'
            else:
                scope = 'custom'
            if max_batches is not None:
                scope = 'partial_'+scope
            iterator = iter(loader)
            return loader, manifest, scope
        loader, manifest, scope = coordinated(name+':setup', setup)
        count = coordinated(name+':length', lambda: len(loader))
        count = count if max_batches is None else min(count, max_batches)
        steps, rows = max(gather(count)), []
        with torch.no_grad():
            for step in range(steps):
                def infer():
                    if step >= count:
                        return
                    batch = next(iterator)
                    x, mask = feature_fn(trainer.plm, batch['waves'], trainer.args['device'])
                    # Same phase/input/rank consumes the same draws in either class-role arm.
                    with phase_rng(trainer.args['device'], trainer.args.get('evaluation_seed', 1729),
                                   0, step, trainer.args.get('rank', 0), 'evaluation:'+phase):
                        scores = head(x, mask=mask, phase='score')
                    raw, canonical = scores['raw_center_cosine'].flatten(), scores['score_bona'].flatten()
                    if (len(raw) != len(batch['labels']) or not bool(torch.isfinite(canonical).all())):
                        raise ValueError('Invalid evaluation scores')
                    rows.extend((int(i), float(q), float(s), int(y)) for i, q, s, y in zip(
                        batch['indices'], raw.cpu(), canonical.cpu(), batch['labels']))
                coordinated(name+':batch', infer)
        rows = sorted(row for part in gather(rows) for row in part)
        def validate():
            if max_batches is None and [r[0] for r in rows] != list(range(len(loader.dataset))):
                raise ValueError('Missing or duplicated evaluation index')
            if len({r[0] for r in rows}) != len(rows):
                raise ValueError('Duplicate evaluation rows')
            if len({Path(loader.dataset.items[r[0]].path).stem for r in rows}) != len(rows):
                raise ValueError('Duplicate evaluation UID')
            if any(y != loader.dataset.items[i].label for i, _, _, y in rows):
                raise ValueError('Evaluation labels differ from protocol')
            if any(s != config_from_model(head).score_sign * q for _, q, s, _ in rows):
                raise ValueError('Canonical score sign applied incorrectly')
            return calculate_eer([r[2] for r in rows], [r[3] for r in rows])
        eer = coordinated(name+':validate', validate)
        def persist():
            if rank() == 0:
                text = 'index,uid,raw_center_cosine,score_bona,label\n' + ''.join(
                    f'{i},{Path(loader.dataset.items[i].path).stem},{q:.17g},{s:.17g},{y}\n'
                    for i, q, s, y in rows)
                atomic_write(checkpoints.stages/(name+'_scores.csv'), lambda stream: stream.write(text.encode()))
        coordinated(name+':save_scores', persist)
        checkpoints.record(name, 'complete', epoch=epoch, phase=phase, eer_percent=eer, rows=len(rows),
                           scope=scope, checkpoint_purpose=manifest['purpose'],
                           model_metadata=checkpoints.expected,
                           checkpoint_sha256=manifest['files'][f'CLS_{epoch}.pt']['sha256'],
                           protocol=dict(input='front64600/native/mask', batch_size=trainer.args.get('eval_batch_size', 8),
                                         noise_std=1e-5, score='higher=bonafide', no_dev=True,
                                         selection=('smoke diagnostics; not model performance'
                                                    if manifest['purpose'] == 'smoke'
                                                    else 'fixed epoch; minima retrospective only')))
        trainer.last_eval_rows = rows
        return dict(phase=phase, epoch=epoch, eer_percent=eer, rows=len(rows),
                    scope=scope, checkpoint_purpose=manifest['purpose'])
    except Exception as error:
        # Shared StageError reaches every live rank; never label failed inference complete.
        checkpoints.record(name, 'failed', phase=phase, epoch=epoch, error=str(error))
        raise
    finally:
        for module, training in modes:
            module.training = training
        def cleanup():
            if iterator is not None and hasattr(iterator, '_shutdown_workers'):
                iterator._shutdown_workers()
        try:
            coordinated(name+':cleanup', cleanup)
        except Exception as error:
            checkpoints.record(name, 'failed', phase=phase, epoch=epoch, error='cleanup: '+str(error))
            raise
        head = iterator = None


def run_training(trainer):
    """Save first, monitor 19LA, then evaluate all saved points on 21LA/DF."""
    from role_training import train_epoch
    from short32_lifecycle import StageError
    trainer.checkpoints = RoleCheckpoints(trainer)
    saved, results = [], []
    def attempt(phase, epoch):
        try:
            results.append(dict(status='complete', **evaluate(trainer, phase, epoch)))
        except StageError as error:
            results.append(dict(status='failed', phase=phase, epoch=epoch, error=str(error)))
    for epoch in range(1, trainer.args['epoch']+1):
        if getattr(trainer, 'verify_sources', None):
            coordinated('source_before_epoch', trainer.verify_sources)
        totals = train_epoch(trainer, epoch)
        def verify_visits():
            if (totals['batches_seen'] != 108 or totals['updates'] != 108
                    or totals['bona_rows'] != 1290 or totals['spoof_rows'] != 11616):
                raise RuntimeError('Original240 epoch visitation contract changed')
        coordinated('verify_epoch_visits', verify_visits)
        trainer.checkpoints.record('train_'+str(epoch), 'complete', **totals)
        if epoch % 5 == 0 or epoch == trainer.args['epoch']:
            trainer.checkpoints.save(epoch)
            saved.append(epoch)
            attempt('19LA', epoch)
    final = {k: v.detach().cpu().clone() for k, v in unwrap(trainer.classifier).state_dict().items()}
    try:
        for epoch in saved:
            for phase in ('21LA', '21DF'):
                attempt(phase, epoch)  # Eval loads a separate head, never overwrites optimizer/model.
    finally:
        coordinated('preserve_final_classifier', lambda: unwrap(trainer.classifier).load_state_dict(final, strict=True))
    minima = {}
    for phase in ('19LA', '21LA', '21DF'):
        valid = [r for r in results if r['status'] == 'complete' and r['phase'] == phase]
        if valid:
            minima[phase] = min(valid, key=lambda r: (r['eer_percent'], r['epoch']))
    status = 'complete' if all(r['status'] == 'complete' for r in results) else 'evaluation_incomplete'
    trainer.checkpoints.record('complete', status, saved_epochs=saved, results=results, minimum_eer=minima)
    return status
