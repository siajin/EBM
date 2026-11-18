"""Save-first checkpoints and coordinated, separately recorded short32 evaluations.

No training/data-loader changes and no execution side effects at import.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
import traceback
from datetime import datetime, timezone

import numpy as np
import torch
import torch.distributed as dist


class StageError(RuntimeError):
    pass


def rank():
    return dist.get_rank() if dist.is_initialized() else 0


def gather(value):
    if not dist.is_initialized():
        return [value]
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    return values


def coordinated(stage, function):
    """The function contains no collectives; every live rank reports its outcome."""
    value, error = None, None
    try:
        value = function()
    except Exception as exc:
        error = dict(rank=rank(), type=type(exc).__name__, message=str(exc),
                     traceback=traceback.format_exc(limit=8))
    errors = [e for e in gather(error) if e is not None]
    if errors:
        raise StageError(json.dumps(dict(stage=stage, errors=errors), ensure_ascii=False))
    return value


def atomic_write(path, writer):
    """Keep the destination intact if write/fsync fails; publish only a full file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, value):
    content = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n').encode('utf-8')
    atomic_write(path, lambda stream: stream.write(content))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def rng_state(device):
    state = np.random.get_state()
    result = dict(python=random.getstate(), numpy=dict(kind=state[0], keys=state[1].tolist(),
        position=state[2], has_gauss=state[3], cached_gaussian=state[4]), torch=torch.get_rng_state())
    if torch.device(device).type == 'cuda':
        result['cuda'] = torch.cuda.get_rng_state(device)
    return result


class Checkpoints:
    def __init__(self, trainer):
        self.trainer = trainer
        def location():
            if rank() != 0:
                return None
            explicit = trainer.args.get('short32_artifact_dir')
            if explicit:
                model_dir = Path(explicit)/'model'
            else:
                logger = trainer.logger
                components = getattr(logger, 'components', [logger])
                paths = [Path(c.paths['model']) for c in components if hasattr(c, 'paths') and 'model' in c.paths]
                if len(paths) != 1:
                    raise ValueError('Short32 requires exactly one local checkpoint directory')
                model_dir = paths[0]
            model_dir.mkdir(parents=True, exist_ok=True)
            # Check writability before the first training step, not at epoch5.
            fd, probe = tempfile.mkstemp(prefix='short32-write-probe-', dir=model_dir)
            os.close(fd); os.unlink(probe)
            return str(model_dir.resolve())
        own = coordinated('checkpoint_directory_preflight', location)
        self.model_dir = Path(gather(own)[0])
        self.root = self.model_dir.parent/'short32_stages'
        self.record('preflight', 'passed')

    def record(self, stage, status, **details):
        def save():
            if rank() == 0:
                write_json(self.root/(stage+'.json'), dict(stage=stage, status=status,
                    updated_utc=datetime.now(timezone.utc).isoformat(), **details))
        coordinated('record:'+stage, save)

    def save(self, epoch):
        states = gather(coordinated('checkpoint_rng_capture', lambda: rng_state(self.trainer.args['device'])))
        def save_parent():
            if rank() != 0:
                return
            t = self.trainer
            cls_path = self.model_dir/f'CLS_{epoch}.pt'
            train_path = self.model_dir/f'TrainState_{epoch}.pt'
            marker = self.root/f'checkpoint_{epoch}.complete.json'
            if any(p.exists() for p in [cls_path, train_path, marker]):
                raise FileExistsError('Refuse to overwrite checkpoint epoch'+str(epoch))
            classifier = t.classifier.state_dict()
            args = {k:v for k,v in t.args.items() if not any(s in k.lower() for s in ['token','password','secret','api_key'])}
            from short32_original240 import verify_source
            state = dict(epoch=epoch, classifier=classifier, optimizer=t.optimizer.state_dict(),
                         scheduler=t.lr_scheduler.state_dict(), args=args, rng_by_rank=states,
                         source=verify_source(), frontend_revision=t.args.get('frontend_resolved_revision'))
            atomic_write(cls_path, lambda stream: torch.save(classifier, stream))
            atomic_write(train_path, lambda stream: torch.save(state, stream))
            # Both files must be readable and mutually consistent before commit.
            cls = torch.load(cls_path, map_location='cpu', weights_only=True)
            restored = torch.load(train_path, map_location='cpu', weights_only=True)
            if restored['epoch'] != epoch or set(cls) != set(restored['classifier']):
                raise ValueError('Checkpoint readback metadata mismatch')
            for key, value in cls.items():
                if not torch.equal(value, restored['classifier'][key]) or not torch.equal(value, classifier[key].detach().cpu()):
                    raise ValueError('Checkpoint tensor mismatch: '+key)
                if not torch.isfinite(value).all():
                    raise ValueError('Nonfinite checkpoint: '+key)
            manifest = dict(epoch=epoch, status='committed', world_size=len(states),
                            files={p.name:dict(sha256=sha256(p), bytes=p.stat().st_size) for p in [cls_path,train_path]})
            write_json(marker, manifest)  # Last: absent marker means incomplete.
        coordinated('checkpoint_save:'+str(epoch), save_parent)
        self.verify(epoch)
        self.record('checkpoint_'+str(epoch), 'committed')

    def verify(self, epoch):
        def check():
            manifest = json.loads((self.root/f'checkpoint_{epoch}.complete.json').read_text(encoding='utf-8'))
            if manifest['epoch'] != epoch or manifest['status'] != 'committed':
                raise ValueError('Uncommitted checkpoint')
            expected = {f'CLS_{epoch}.pt', f'TrainState_{epoch}.pt'}
            if set(manifest['files']) != expected:
                raise ValueError('Checkpoint manifest file mismatch')
            for name, info in manifest['files'].items():
                path = self.model_dir/name
                if path.stat().st_size != info['bytes'] or sha256(path) != info['sha256']:
                    raise ValueError('Checkpoint hash/size mismatch: '+name)
            return manifest
        return coordinated('checkpoint_verify:'+str(epoch), check)

    def load_model(self, epoch):
        manifest = self.verify(epoch)
        state = coordinated('checkpoint_load:'+str(epoch), lambda: torch.load(
            self.model_dir/f'CLS_{epoch}.pt', map_location='cpu', weights_only=True))
        coordinated('checkpoint_apply:'+str(epoch), lambda: self.trainer.classifier.load_state_dict(state, strict=True))
        self.trainer.args['evaluation_checkpoint_sha256'] = manifest['files'][f'CLS_{epoch}.pt']['sha256']


def evaluate(trainer, phase, epoch, checkpoints):
    """No DDP forwards; coordinate each batch before gathering final scores."""
    from native_steps import features, unwrap
    from positive_bpl import evaluation_rng
    from model import B_Linear, OCSoftmax, BPL
    name = f'eval_{epoch}_{phase}'
    checkpoints.verify(epoch)
    checkpoints.record(name, 'running', epoch=epoch, phase=phase)
    modes = [(m,m.training) for base in [unwrap(trainer.plm),unwrap(trainer.classifier)] for m in base.modules()]
    head = iterator = None
    try:
        with evaluation_rng(trainer.args, 525252), torch.no_grad():
            def setup():
                nonlocal head, iterator
                if phase == 'V':
                    raise ValueError('dev evaluation is disabled for short32')
                loader = trainer.eval_loader[phase]
                trainer.plm.eval(); unwrap(trainer.classifier).eval()
                a = trainer.args
                head = B_Linear(a['num_layers'],a['hidden_size'],a['output_size'],a['agg_size'],
                               loss=OCSoftmax(a['output_size']),loss_bpl=BPL(a['alpha_cl'])).to(a['device']).eval()
                head.load_state_dict(unwrap(trainer.classifier).state_dict(), strict=True)
                iterator = iter(loader)
                return loader
            loader = coordinated(name+':setup', setup)
            count = coordinated(name+':length', lambda: len(loader))
            batches = max(gather(count))
            rows = []
            # Construction/worker seeding must not shift the ASP scoring stream.
            with evaluation_rng(trainer.args, 525252):
                for step in range(batches):
                    def infer():
                        if step >= count:
                            return
                        batch = next(iterator)
                        x, mask = features(trainer.plm, batch['waves'], trainer.args['device'])
                        scores = head(x, mask=mask)[1].flatten()
                        if not torch.isfinite(scores).all() or len(scores) != len(batch['labels']):
                            raise ValueError('Nonfinite/misaligned evaluation scores')
                        rows.extend((int(i),float(s),int(y)) for i,s,y in zip(batch['indices'],scores.cpu(),batch['labels']))
                    coordinated(name+':batch'+str(step), infer)
                    if step % 50 == 0 or step == batches-1:
                        checkpoints.record(name, 'running', epoch=epoch, phase=phase,
                                           completed_batches=step+1, total_batches=batches)
            rows = sorted(row for part in gather(rows) for row in part)
            def validate():
                if [r[0] for r in rows] != list(range(len(loader.dataset))):
                    raise ValueError('Evaluation indices missing or duplicated')
                if any(y != loader.dataset.items[i].label for i, s, y in rows):
                    raise ValueError('Evaluation class labels differ from protocol')
                eer = trainer.calculate_EER([r[1] for r in rows], [r[2] for r in rows])
                if not math.isfinite(eer):
                    raise ValueError('Nonfinite EER')
                return eer
            eer = coordinated(name+':validate', validate)
            trainer.last_eval_rows = rows
            def persist():
                if rank() == 0:
                    scores = 'index,score,label\n'+''.join(f'{i},{s:.17g},{y}\n' for i,s,y in rows)
                    atomic_write(checkpoints.root/(name+'_scores.csv'), lambda out: out.write(scores.encode()))
            coordinated(name+':scores', persist)
            checkpoints.record(name, 'complete', epoch=epoch, phase=phase, rows=len(rows), eer=eer,
                               protocol=trainer.evaluation_protocol,
                               checkpoint_sha256=trainer.args.get('evaluation_checkpoint_sha256'))
            return eer,0,0,0
    finally:
        # Restore actual training objects/modes even if setup/inference/I/O fails.
        for module, training in modes:
            module.training = training
        def cleanup():
            if iterator is not None and hasattr(iterator, '_shutdown_workers'):
                iterator._shutdown_workers()
        coordinated(name+':cleanup', cleanup)
        head = iterator = None


def run_training(trainer):
    """Only epoch boundaries/monitoring change. trainer.train stays original."""
    checkpoints = Checkpoints(trainer)
    trainer.checkpoints = checkpoints
    saved, results = [], []

    def attempt(phase, epoch):
        try:
            result = trainer.test(phase, epoch)
            results.append(dict(phase=phase,epoch=epoch,eer=result[0],status='complete'))
        except StageError as exc:
            checkpoints.record(f'eval_{epoch}_{phase}', 'failed', phase=phase, epoch=epoch, error=str(exc))
            results.append(dict(phase=phase,epoch=epoch,status='failed',error=str(exc)))
            if rank() == 0:
                print(f'Evaluation failed: epoch{epoch} {phase}; checkpoint preserved; continuing other stages. {exc}', flush=True)

    for epoch in range(1,trainer.args['epoch']+1):
        trainer.train(epoch)
        if epoch % 5 == 0 or epoch == trainer.args['epoch']:
            checkpoints.save(epoch)  # Never evaluate or advance after save failure.
            saved.append(epoch)
            manifest = checkpoints.verify(epoch)
            trainer.args['evaluation_checkpoint_sha256'] = manifest['files'][f'CLS_{epoch}.pt']['sha256']
            attempt('19LA', epoch)
    # All committed checkpoints, same fixed schedule: supports minimum-EER audit.
    final = coordinated('snapshot_final_classifier', lambda: {
        k:v.detach().cpu().clone() for k,v in trainer.classifier.state_dict().items()})
    try:
        for epoch in saved:
            for phase in ['21LA','21DF']:
                try:
                    checkpoints.load_model(epoch)
                    attempt(phase, epoch)
                except StageError as exc:
                    checkpoints.record(f'eval_{epoch}_{phase}', 'failed', phase=phase, epoch=epoch, error=str(exc))
                    results.append(dict(phase=phase,epoch=epoch,status='failed',error=str(exc)))
    finally:
        coordinated('restore_final_classifier', lambda: trainer.classifier.load_state_dict(final,strict=True))
    minimum = {}
    for phase in ['19LA','21LA','21DF']:
        valid = [r for r in results if r['phase']==phase and r['status']=='complete']
        if valid:
            minimum[phase] = min(valid,key=lambda r:(r['eer'],r['epoch']))
    checkpoints.record('complete', 'complete' if all(r['status']=='complete' for r in results) else 'evaluation_incomplete',
                       saved_epochs=saved, results=results, minimum_eer=minimum,
                       note='Dataset-specific minima are retrospective; failures are not treated as success')
