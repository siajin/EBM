"""Evaluate an explicitly fixed coverage checkpoint without training or a logger."""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def audit_checkpoint(root, smoke):
    import torch
    config = json.loads((root / 'evaluation_config.json').read_text())
    for name, expected in config['source_sha256'].items():
        assert sha(Path(config['code_root']) / name) == expected, name
    assert sha(config['checkpoint']) == config['checkpoint_sha256']
    assert sha(config['train_state']) == config['train_state_sha256']
    state = torch.load(config['train_state'], map_location='cpu', weights_only=True)
    head = torch.load(config['checkpoint'], map_location='cpu', weights_only=True)
    assert state['epoch'] == config['epoch'] == 20
    assert head.keys() == state['classifier'].keys()
    assert all(torch.equal(value, state['classifier'][name]) for name, value in head.items())
    assert all(torch.isfinite(value).all() for value in head.values())
    steps = [int(value['step']) for value in state['optimizer']['state'].values()]
    assert min(steps) == max(steps) == 20 * 108
    assert state['args']['frontend_resolved_revision'] == config['frontend_revision']
    save(root / ('smoke_checkpoint_audit.json' if smoke else 'checkpoint_audit.json'), {
        'status': 'passed', 'epoch': 20, 'Adam_steps_per_parameter': 2160,
        'classifier_matches_TrainState': True, 'optimizer_updates_during_evaluation': 0,
        'checkpoint_sha256': config['checkpoint_sha256'],
        'train_state_sha256': config['train_state_sha256'],
        'frozen_source_files_verified': len(config['source_sha256']),
    })


def worker(rank, root_string, phase, smoke):
    import torch
    from transformers import Wav2Vec2Model
    root = Path(root_string)
    config = json.loads((root / 'evaluation_config.json').read_text())
    sys.path.insert(0, config['code_root'])
    import arguments
    import data_loaders
    import model
    import native_steps
    from positive_bpl import evaluation_rng, oc_evaluation_protocol
    from tools.evaluate_B_C_checkpoints import make_items, Waves, collate
    torch.set_num_threads(2)
    torch.cuda.set_device(rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    args, _, _ = arguments.get_args()
    args.update(cap_long=True, rank=rank, world_size=2, device=f'cuda:{rank}',
                evaluation_seed=1729, eval_batch_size=8)
    destination = root / ('smoke' if smoke else 'evaluation')
    plm = Wav2Vec2Model.from_pretrained(args['PLM_name'], local_files_only=True)
    plm.to(args['device']).eval().requires_grad_(False)
    revision = getattr(plm.config, '_commit_hash', None)
    assert revision == config['frontend_revision']
    head = model.B_Linear(args['num_layers'], args['hidden_size'], args['output_size'], args['agg_size'],
        loss=model.OCSoftmax(args['output_size'], r_real=args['r_real'], r_fake=args['r_fake'],
                             alpha=args['alpha'], class_weight=[5/9, 5]),
        loss_bpl=model.BPL(args['alpha_cl'])).to(args['device']).eval().requires_grad_(False)
    state = torch.load(config['checkpoint'], map_location='cpu', weights_only=True)
    head.load_state_dict({name.removeprefix('module.'): value for name, value in state.items()}, strict=True)
    del state
    versions = {name: value._version for name, value in head.named_parameters()}
    assert all(not module.training for module in head.modules())
    assert all(module.noise_std == 1e-5 for module in head.modules() if hasattr(module, 'noise_std'))
    protocol = oc_evaluation_protocol(args, revision)
    protocol.update(epoch=20, checkpoint_sha256=config['checkpoint_sha256'], phase=phase,
                    optimizer_updates=0, actual_torch_seed=1729+525252+rank,
                    selection='Explicit user-fixed epoch20; no dev or target-based model selection',
                    tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                    tf32_cudnn=torch.backends.cudnn.allow_tf32)
    save(destination / f'protocol_{phase}_rank{rank}.json', protocol)
    items = make_items(args, phase)
    dataset = Waves(items)
    indices = list(range(rank, len(items), 2))
    if smoke:
        indices = indices[:32] if phase == '19LA' else sorted(
            index for label in [1, 0] for index in [i for i in indices if items[i].label == label][:16])
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, sampler=indices,
        num_workers=0 if smoke else 4, collate_fn=collate, pin_memory=False,
        generator=torch.Generator().manual_seed(1+rank))
    reference = None
    if smoke and phase == '19LA':
        reference = pd.read_csv(config['reference_19LA_scores']).set_index('index')['score']
    path = destination / f'scores_{phase}_rank{rank}.csv'
    partial = path.with_suffix('.partial.csv')
    assert not path.exists() and not partial.exists()
    fields = ['index', 'uid', 'speaker', 'label', 'original_samples', 'used_samples', 'attack', 'score']
    count = 0
    maximum_reference_difference = 0.
    started = time.monotonic()
    with partial.open('x', newline='') as stream, torch.no_grad(), evaluation_rng(args, 525252):
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for number, (waves, metadata) in enumerate(loader):
            if number == 0:
                original = data_loaders.EnrollmentSet(args, items)
                for wave, row in zip(waves, metadata):
                    assert np.array_equal(wave.numpy(), np.asarray(original[row['index']][0], dtype='float32'))
            features, mask = native_steps.features(plm, waves, args['device'])
            scores = head(features, mask=mask)[1].flatten()
            assert torch.isfinite(scores).all()
            for row, score in zip(metadata, scores.cpu().tolist()):
                if reference is not None:
                    difference = abs(score-float(reference.loc[row['index']]))
                    maximum_reference_difference = max(maximum_reference_difference, difference)
                    assert difference < 1e-6, (row['index'], difference)
                writer.writerow(dict(row, score=score))
                count += 1
            del features, mask, scores
            if number % 50 == 0:
                stream.flush()
                save(destination / f'progress_rank{rank}.json', {
                    'phase': phase, 'rows': count, 'rank_rows': len(indices),
                    'elapsed_seconds': time.monotonic()-started, 'updated_unix': time.time()})
        stream.flush()
        os.fsync(stream.fileno())
    assert count == len(indices)
    partial.replace(path)
    assert versions == {name: value._version for name, value in head.named_parameters()}
    assert sha(config['checkpoint']) == config['checkpoint_sha256']
    save(destination / f'done_{phase}_rank{rank}.json', {
        'rows': count, 'seconds': time.monotonic()-started, 'sha256': sha(path),
        'optimizer_updates': 0, 'parameters_unchanged': True,
        'maximum_saved19LA_score_difference': maximum_reference_difference if reference is not None else None})


def summarize_phase(root, phase):
    config = json.loads((root / 'evaluation_config.json').read_text())
    sys.path.insert(0, config['code_root'])
    from tools.evaluate_B_C_checkpoints import metrics
    frames = []
    for rank in range(2):
        path = root / 'evaluation' / f'scores_{phase}_rank{rank}.csv'
        done = json.loads((path.parent / f'done_{phase}_rank{rank}.json').read_text())
        assert sha(path) == done['sha256']
        frames.append(pd.read_csv(path))
    frame = pd.concat(frames).sort_values('index').reset_index(drop=True)
    assert frame['index'].tolist() == list(range(config['phases'][phase]))
    assert frame.uid.is_unique and not frame.isna().any().any()
    assert np.array_equal(frame.used_samples, np.minimum(frame.original_samples, 64600))
    value = metrics(frame, 'score')
    assert (value['bonafide'], value['spoof']) == tuple(config['class_counts'][phase])
    value = {key: value[key] for key in ['EER_percent', 'n', 'bonafide', 'spoof']}
    value.update(phase=phase, epoch=20, checkpoint_sha256=config['checkpoint_sha256'],
                 evaluation_protocol='controlled_oc_v1', optimizer_updates=0)
    frame.to_parquet(root / 'evaluation' / f'scores_{phase}.parquet', index=False)
    save(root / f'metrics_{phase}.json', value)
    return value


if __name__ == '__main__':
    import torch.multiprocessing as mp
    root = Path(sys.argv[1])
    smoke = '--smoke' in sys.argv
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '0,1' and Path('/.dockerenv').is_file()
    config = json.loads((root / 'evaluation_config.json').read_text())
    assert sha(__file__) == config['evaluator_sha256']
    destination = root / ('smoke' if smoke else 'evaluation')
    destination.mkdir(exist_ok=False)
    audit_checkpoint(root, smoke)
    phases = ['19LA', '21LA', '21DF'] if smoke else list(config['phases'])
    rows = []
    for phase in phases:
        save(root / 'phase.json', {'phase': phase, 'smoke': smoke, 'updated_unix': time.time()})
        mp.spawn(worker, args=(str(root), phase, smoke), nprocs=2, join=True)
        if not smoke:
            rows.append(summarize_phase(root, phase))
    assert sha(config['checkpoint']) == config['checkpoint_sha256']
    assert sha(config['train_state']) == config['train_state_sha256']
    save(root / ('smoke_complete.json' if smoke else 'complete.json'), {
        'status': 'passed' if smoke else 'complete', 'epoch': 20,
        'metrics': rows, 'optimizer_updates': 0, 'finished_unix': time.time(),
        'checkpoint_sha256': config['checkpoint_sha256']})
