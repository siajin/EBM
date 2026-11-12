"""Opt-in experiment construction; importing never downloads or starts work."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import re
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from role_model import RoleBLinear
from role_objectives import RoleConfig
from role_training import phase_rng, step_hidden, unwrap
from role_lifecycle import RoleCheckpoints, evaluate, run_training
from short32_lifecycle import coordinated, gather, rank, sha256, write_json

ROOT = Path(__file__).resolve().parent
REFERENCE = '10f3bcd057dc6a2ca687fbd5389ad6c9e15a3746'


def verify_reference():
    source = ROOT/'baseline240_source'
    manifest = json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    required = {'main.py', 'train.py', 'data_loaders.py', 'model/b_linear.py', 'model/loss.py'}
    if manifest['commit'] != REFERENCE or set(manifest['files']) != required:
        raise ValueError('Unexpected original240 manifest')
    for name, digest in manifest['files'].items():
        if sha256(source/name) != digest:
            raise ValueError('Frozen original240 source changed: '+name)
    return manifest


def source_fingerprint():
    # Include transitive original-data, model, evaluation and checkpoint helpers.
    names = set()
    for folder in ['model', 'dataset', 'data_processing', 'baseline240_source']:
        names.update(p.relative_to(ROOT).as_posix() for p in (ROOT/folder).rglob('*.py'))
    names.update(p.name for p in ROOT.glob('role_*.py'))
    names.update(['tools/train_role_experiment.py', 'data_loaders.py', 'controlled_data_loaders.py',
                  'native_steps.py', 'short32_lifecycle.py'])
    return {name: sha256(ROOT/name) for name in sorted(names)}


def options(compact_class='spoof', *, epoch=50, seed=1, workers=4, frontend_revision=None,
            path_19LA='/data/ASVspoof2019', path_21LA='/data/ASVspoof2021_LA_eval',
            path_21DF='/data/ASVspoof2021_DF', run_root=None):
    if compact_class not in ('spoof', 'bonafide'):
        raise ValueError('Unknown compact class')
    args = dict(role=RoleConfig(compact_label=int(compact_class == 'bonafide')).metadata(),
                epoch=epoch, rand_seed=seed, num_workers=workers, frontend_revision=frontend_revision,
                path_19LA=path_19LA, path_21LA=path_21LA, path_21DF=path_21DF,
                run_root=str(run_root) if run_root else None,
                batch_size=120, global_batch_size=240, world_size=2, ratio_bona=.1, ratio_spoof=.9,
                num_train_frames=404, num_train_frames_short=303, num_test_frames=404, num_seg=1,
                eval_batch_size=8, evaluation_seed=1729, num_layers=25, hidden_size=1024,
                output_size=160, agg_size=128, lr=1e-5, lr_min=1e-5, weight_decay=1e-4,
                T_0=150, T_mult=1, PLM_name='facebook/wav2vec2-xls-r-300m',
                train_input='original240_fixed_wrap', eval_input='front64600_native_mask',
                rng_policy='role_v1_isolated_phase_and_loader_generators', automatic_retries=0)
    validate_options(args)
    return args


def validate_options(args):
    RoleConfig.from_metadata(args['role'])
    if type(args['epoch']) is not int or not 1 <= args['epoch'] <= 50:
        raise ValueError('Choose a fixed training budget of 1..50 epochs')
    if type(args['rand_seed']) is not int or args['rand_seed'] < 0:
        raise ValueError('Training seed must be a nonnegative integer')
    if type(args['num_workers']) is not int or args['num_workers'] < 0:
        raise ValueError('Worker count must be nonnegative')
    fixed = dict(batch_size=120, global_batch_size=240, world_size=2, ratio_bona=.1, ratio_spoof=.9,
                 num_train_frames=404, num_train_frames_short=303, num_test_frames=404, num_seg=1,
                 eval_batch_size=8, evaluation_seed=1729, num_layers=25, hidden_size=1024,
                 output_size=160, agg_size=128, lr=1e-5, lr_min=1e-5, weight_decay=1e-4,
                 T_0=150, T_mult=1, train_input='original240_fixed_wrap', eval_input='front64600_native_mask')
    if any(args.get(k) != v for k, v in fixed.items()):
        raise ValueError('Original240 model/input/optimizer comparison contract changed')
    for name in ['short32_bpl', 'duration_policy_path', 'activity_policy_path', 'mild_transform_cache',
                 'short_oc_weight', 'padding_loss_weight']:
        if args.get(name):
            raise ValueError('Do not combine the role experiment with '+name)
    if args.get('bpl_pairing', 'legacy_long_short') != 'legacy_long_short':
        raise ValueError('Role experiment uses original long/short full cross-pairs')
    revision = args.get('frontend_revision')
    if revision is not None and (not isinstance(revision, str) or not re.fullmatch(r'[0-9a-f]{40}', revision)):
        raise ValueError('Pin frontend_revision to a resolved 40-character commit SHA')


def signature(args):
    keys = [k for k in args if k not in ('run_root', 'rank', 'device', 'mode', 'smoke_receipt',
                                        'data_summary', 'timeout_hours')]
    data = {k: args[k] for k in sorted(keys)}
    return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_smoke_receipt(args, path):
    receipt = json.loads(Path(path).read_text(encoding='utf-8'))
    if (receipt.get('status') != 'passed' or receipt.get('signature') != signature(args)
            or receipt.get('world_size') != 2 or receipt.get('trained_batches') != [1, 106, 108]
            or receipt.get('eval_phases') != ['19LA', '21LA', '21DF']
            or not receipt.get('data_fingerprint')):
        raise ValueError('A successful matching two-GPU smoke receipt is required before training')
    return receipt


def validate_runtime():
    verify_reference()
    for name in ['soundfile', 'scipy', 'sklearn', 'transformers',
                 'baseline240_source.data_loaders', 'controlled_data_loaders', 'role_lifecycle']:
        importlib.import_module(name)
    from transformers import AutoConfig, Wav2Vec2Model  # Resolve lazy model dependencies before GPU work.
    return dict(runtime_imports='passed', frontend_downloads=0)


def data_summary(items):
    import soundfile as sf
    rows, counts = [], {0: 0, 1: 0}
    for item in items:
        info = sf.info(item.path)
        if info.samplerate != 16000 or info.channels != 1 or info.frames < 720 or item.label not in counts:
            raise ValueError('Original Train requires mono16k nonempty sources: '+str(item.path))
        counts[item.label] += 1
        rows.append(f'{item.path}|{item.label}|{item.speaker}|{item.type}|{info.frames}')
    if counts != {0: 22800, 1: 2580}:
        raise ValueError('Expected original Train25380: bona2580/spoof22800')
    return dict(counts={str(k): v for k, v in counts.items()},
                fingerprint=hashlib.sha256('\n'.join(rows).encode()).hexdigest())


def smoke_items(items):
    """64 deterministic trials:16 per protocol class x native<=2s/>2s."""
    import soundfile as sf
    groups = {(y, short): [] for y in [0, 1] for short in [False, True]}
    for item in items:
        length = sf.info(item.path).frames
        key = (int(item.label), length <= 32000)
        if len(groups[key]) < 16:
            groups[key].append(item)
        if all(len(v) == 16 for v in groups.values()):
            break
    if not all(len(v) == 16 for v in groups.values()):
        raise ValueError('Smoke requires both native-length groups in both classes')
    return [item for values in groups.values() for item in values]


def build_trainer(args, mode):
    """Local preparation is coordinated separately from DDP construction."""
    import controlled_data_loaders as evaluation_data
    from torch.utils.data import DataLoader
    from dataset import ASVspoof2019_LA_Eval, ASVspoof2021_LA_Eval, ASVspoof2021_DF_Eval
    from dataset.ASVspoof2019_la_train import load_train_items_only
    from transformers import AutoConfig, Wav2Vec2Model
    original_data = importlib.import_module('baseline240_source.data_loaders')
    def local_setup():
        random.seed(args['rand_seed']); np.random.seed(args['rand_seed']); torch.manual_seed(args['rand_seed'])
        items = load_train_items_only(args['path_19LA'])
        summary = data_summary(items)
        db = dict(zip(['19LA', '21LA', '21DF'], [ASVspoof2019_LA_Eval(args['path_19LA']),
                   ASVspoof2021_LA_Eval(args['path_21LA']), ASVspoof2021_DF_Eval(args['path_21DF'])]))
        trial_hashes = {}
        for phase, expected in [('19LA',71237),('21LA',148176),('21DF',533928)]:
            trials = db[phase].trials
            if (len(trials) != expected or len({Path(item.path).stem for item in trials}) != expected
                    or {item.label for item in trials} != {0,1}):
                raise ValueError('Official evaluation trial coverage mismatch: '+phase)
            content = '\n'.join(f'{i}|{item.path}|{item.label}|{item.type}' for i,item in enumerate(trials))
            trial_hashes[phase] = hashlib.sha256(content.encode()).hexdigest()
        summary['train_fingerprint'] = summary['fingerprint']
        summary['evaluation_trial_sha256'] = trial_hashes
        summary['fingerprint'] = hashlib.sha256(json.dumps(
            dict(train=summary['train_fingerprint'],evaluation=trial_hashes),sort_keys=True).encode()).hexdigest()
        loaders = original_data.get_loaders(args, items, [], db)
        # The original training Dataset/Sampler/DataLoader types and quotas stay intact.
        for offset, loader in enumerate([loaders[0], loaders[2]]):
            loader.generator = torch.Generator().manual_seed(args['rand_seed']+100003*args['rank']+7919*offset)
        def evaluation(trials):
            trials = smoke_items(trials) if mode == 'smoke' else trials
            ds = evaluation_data.EnrollmentSet(dict(args, native_length=True, cap_long=True), trials)
            return DataLoader(ds, batch_size=8, sampler=evaluation_data.ExactDistributedEvalSampler(ds, args),
                              num_workers=args['num_workers'], collate_fn=evaluation_data.collate_eval,
                              generator=torch.Generator().manual_seed(args['rand_seed']+args['rank']+525252),
                              timeout=120 if args['num_workers'] else 0,
                              **({'multiprocessing_context': 'spawn'} if args['num_workers'] else {}))
        eval_loaders = {phase: evaluation(value.trials) for phase, value in db.items()}
        revision = args['frontend_revision']
        config = AutoConfig.from_pretrained(args['PLM_name'], revision=revision, local_files_only=True)
        if getattr(config, '_commit_hash', None) != revision:
            raise ValueError('Frontend resolved revision differs from the pinned SHA')
        plm = Wav2Vec2Model.from_pretrained(args['PLM_name'], config=config, revision=revision,
                                          local_files_only=True, ignore_mismatched_sizes=False).to(args['device'])
        plm.eval().requires_grad_(False)
        # Head initialization is isolated from any frontend construction RNG use.
        torch.manual_seed(args['rand_seed'])
        head = RoleBLinear(RoleConfig.from_metadata(args['role']), num_layer=args['num_layers'],
                           hidden_size=args['hidden_size'], output_size=args['output_size'],
                           agg_size=args['agg_size']).to(args['device'])
        return items, summary, loaders, eval_loaders, plm, head
    items, summary, loaders, evals, plm, head = coordinated('build_local_components', local_setup)
    summaries = gather(summary)
    if any(value != summary for value in summaries):
        raise ValueError('Ranks read different Train metadata')
    if mode == 'train':
        receipt = validate_smoke_receipt(args, args['smoke_receipt'])
        if receipt['data_fingerprint'] != summary['fingerprint']:
            raise ValueError('Train metadata changed since smoke')
    head = torch.nn.parallel.DistributedDataParallel(head, device_ids=[args['rank']],
                                                    find_unused_parameters=True, static_graph=False)
    optimizer = torch.optim.Adam(head.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args['T_0'], T_mult=args['T_mult'], eta_min=args['lr_min'])
    t = SimpleNamespace(args=args, plm=plm, classifier=head, optimizer=optimizer, lr_scheduler=scheduler,
                        train_loader_bona=loaders[0], sampler_bona=loaders[1],
                        train_loader_spoof=loaders[2], sampler_spoof=loaders[3], eval_loader=evals)
    args['data_summary'] = summary
    def verify():
        if source_fingerprint() != args['source_sha256']:
            raise ValueError('Source changed after launch; use a new reviewed snapshot')
    t.verify_sources = verify
    def record(values):
        path = Path(args['run_root'])/f'train_rank{args["rank"]}.jsonl'
        with path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(values, allow_nan=False)+'\n')
    t.record_train = record
    return t


def run_smoke(trainer):
    """Use actual original sampler rows at full,72 and114-input positions."""
    from torch.utils.data import default_collate
    t, args = trainer, trainer.args
    t.plm.eval().requires_grad_(False); t.classifier.train()
    t.sampler_bona.set_epoch(1); t.sampler_spoof.set_epoch(1)
    class_batches = []
    for loader in [t.train_loader_bona, t.train_loader_spoof]:
        indices = list(loader.sampler)
        class_batches.append([indices[i:i+loader.batch_size] for i in range(0, len(indices), loader.batch_size)])
    values = []
    for step, expected in [(1, (12, 108)), (106, (12, 60)), (108, (6, 108))]:
        def inputs():
            rows = [default_collate([loader.dataset[index] for index in batches[(step-1) % len(batches)]])
                    for loader, batches in zip([t.train_loader_bona, t.train_loader_spoof], class_batches)]
            long, short, labels = (torch.cat(parts).to(args['device']) for parts in zip(*rows))
            if (int((labels == 1).sum()), int((labels == 0).sum())) != expected:
                raise ValueError('Original partial batch schedule changed')
            if long.shape[1] != 64600 or short.shape[1] != 48440:
                raise ValueError('Original input shapes changed')
            return long.float(), short.float(), labels.long()
        long, short, labels = coordinated('smoke_inputs', inputs)
        with torch.no_grad():
            a = torch.stack(t.plm(long, output_hidden_states=True).hidden_states, dim=1)
            b = torch.stack(t.plm(short, output_hidden_states=True).hidden_states, dim=1)
        result = step_hidden(t.classifier, t.optimizer, a, b, labels,
                             rng=lambda phase: phase_rng(args['device'], args['rand_seed'], 1, step, args['rank'], phase))
        result.update(epoch=0, batch=step); t.record_train(result); values.append(result)
        del long, short, labels, a, b
    center = unwrap(t.classifier).loss.weight.detach()
    centers = [torch.empty_like(center) for _ in range(dist.get_world_size())]
    dist.all_gather(centers, center)
    if any(not torch.equal(center, other) for other in centers):
        raise RuntimeError('Centers diverged across smoke ranks')
    if any(p.grad is not None or p.requires_grad for p in t.plm.parameters()):
        raise RuntimeError('Frontend was not frozen')
    t.checkpoints = RoleCheckpoints(t); t.checkpoints.save(0)
    for phase in ['19LA', '21LA', '21DF']:
        evaluate(t, phase, 0)
    summaries = gather(dict(values=values, peak_memory_bytes=torch.cuda.max_memory_allocated(args['device'])))
    if rank() == 0:
        write_json(Path(args['run_root'])/'smoke_complete.json', dict(status='passed', signature=signature(args),
            world_size=2, trained_batches=[1, 106, 108], eval_phases=['19LA', '21LA', '21DF'],
            data_fingerprint=args['data_summary']['fingerprint'], ranks=summaries,
            note='Smoke optimizer/checkpoint is not reused for training'))


def worker(process_id, args, mode, rendezvous):
    import datetime
    # A verified, own process group lets the parent clean up DataLoader children
    # as well as rank processes on timeout/error. No unrelated group is signaled.
    if os.name == 'posix' and os.getpgrp() != os.getpid():
        os.setsid()
    args = dict(args, rank=process_id, device=f'cuda:{process_id}')
    torch.cuda.set_device(process_id)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    dist.init_process_group('nccl', init_method=rendezvous, rank=process_id, world_size=2,
                            timeout=datetime.timedelta(seconds=600))
    try:
        trainer = build_trainer(args, mode)
        if mode == 'smoke':
            run_smoke(trainer)
        else:
            result = run_training(trainer)
            if result != 'complete':
                raise RuntimeError('Training complete, but one or more evaluations failed; checkpoints preserved')
    finally:
        dist.destroy_process_group()
