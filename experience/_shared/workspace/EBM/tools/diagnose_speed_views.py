"""Frozen-model speed interventions. Derived scores only; never trains or saves audio."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

CODE = Path('/home/mentee/code/EBM')
CAP = 64600
SR = 16000
HEADS = ['legacy240', 'B', 'C']
CONDITIONS = ['native_cap', 'full_native', 'slow_cap', 'fast_cap', 'norm_full4', 'roundtrip_cap']
EXPECTED = {'dev': 24844, '19LA': 71237, '21LA': 148176, '21DF': 533928}
META = ['phase', 'index', 'uid', 'speaker', 'label', 'original_samples', 'attack']
FIELDS = META + ['condition', 'rate', 'used_samples', 'wave_sha256', 'gpu_rank'] + [
    field for name in HEADS for field in [f'score_{name}', f'cos_{name}_native', f'norm_{name}']]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, indent=2)+'\n')
    temporary.replace(path)


def selection(root, smoke=False):
    protocol = json.loads((root/'protocol.json').read_text())
    path = root/'selected.csv'
    if protocol.get('selected_sha256'):
        assert sha(path) == protocol['selected_sha256'], 'Selection changed'
    frame = pd.read_csv(path, dtype={'uid': str, 'speaker': str, 'attack': str})
    assert set(META).issubset(frame.columns), frame.columns.tolist()
    assert not frame[META].isna().any().any()
    assert not frame.duplicated(['phase', 'index']).any()
    assert not frame.duplicated(['phase', 'uid']).any()
    assert set(frame.phase).issubset(EXPECTED)
    assert set(frame.label).issubset({0, 1})
    assert frame.original_samples.between(720, 30*SR).all()
    if smoke:
        frame = pd.concat([frame.iloc[rank::2].head(2) for rank in range(2)]).sort_index()
    return frame.reset_index(drop=True), protocol


def make_items(args, phase):
    from dataset import ASVspoof2019_LA_Train, ASVspoof2019_LA_Eval, ASVspoof2021_LA_Eval, ASVspoof2021_DF_Eval
    if phase == 'dev':
        items = ASVspoof2019_LA_Train(args['path_19LA']).dev_set
    elif phase == '19LA':
        items = ASVspoof2019_LA_Eval(args['path_19LA']).trials
    elif phase == '21LA':
        items = ASVspoof2021_LA_Eval(args['path_21LA']).trials
    else:
        items = ASVspoof2021_DF_Eval(args['path_21DF']).trials
    assert len(items) == EXPECTED[phase]
    if phase in ['21LA', '21DF']:
        base = args['path_21LA'] if phase == '21LA' else args['path_21DF']
        relative = 'keys/LA/CM/trial_metadata.txt' if phase == '21LA' else 'keys/DF/CM/trial_metadata.txt'
        mapping = {}
        for line in (Path(base)/relative).read_text().splitlines():
            tokens = line.split()
            mapping[tokens[1]] = tokens[0]
        for item in items:
            item.speaker = mapping[Path(item.path).stem]
    return items


def stretch(wave, rate):
    import librosa
    assert np.isfinite(rate) and rate > 0
    # Identity must introduce no phase-vocoder change.
    out = wave.copy() if rate == 1. else librosa.effects.time_stretch(
        wave, rate=float(rate), n_fft=2048, hop_length=512)
    out = np.asarray(out, dtype=np.float32)
    assert len(out) == round(len(wave)/rate), (len(wave), rate, len(out))
    assert out.ndim == 1 and np.isfinite(out).all()
    return out


def views(wave):
    capped = wave[:CAP].copy()
    slow = stretch(capped, .8)
    normalized_rate = len(wave)/CAP
    normalized = stretch(wave, normalized_rate)
    roundtrip_rate = len(slow)/len(capped)
    roundtrip = stretch(slow, roundtrip_rate)
    assert len(normalized) == CAP and len(roundtrip) == len(capped)
    return [('native_cap', capped, 1.), ('full_native', wave, 1.),
            ('slow_cap', slow, .8), ('fast_cap', stretch(capped, 1.25), 1.25),
            ('norm_full4', normalized, normalized_rate),
            ('roundtrip_cap', roundtrip, roundtrip_rate)]


def worker(rank, root_string, smoke):
    import librosa
    import soundfile as sf
    import torch
    from transformers import Wav2Vec2Model
    root = Path(root_string)
    sys.path.insert(0, str(CODE))
    import arguments
    import model
    import native_steps
    torch.set_num_threads(2)
    torch.cuda.set_device(rank)
    device = f'cuda:{rank}'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20261109+rank)
    frame, protocol = selection(root, smoke)
    assigned = frame.iloc[rank::2].reset_index(drop=True)
    assert len(assigned) > 0
    suffix = '_smoke' if smoke else ''
    destination = root/f'scores_rank{rank}{suffix}.csv'
    partial = destination.with_suffix('.partial.csv')
    assert not destination.exists() and not partial.exists(), 'Refuse overwrite'
    args, _, _ = arguments.get_args()
    plm = Wav2Vec2Model.from_pretrained(args['PLM_name'], local_files_only=True).to(device).eval().requires_grad_(False)
    revision = getattr(plm.config, '_commit_hash', None)
    assert revision == '1a640f32ac3e39899438a2931f9924c02f080a54', revision
    heads = {}
    checks = {}
    assert set(protocol['checkpoints']) == set(HEADS)
    for name in HEADS:
        source = protocol['checkpoints'][name]
        checkpoint = Path(source['path'])
        assert sha(checkpoint) == source['sha256'], (name, 'checkpoint hash')
        obj = model.B_Linear(args['num_layers'], args['hidden_size'], args['output_size'], args['agg_size'],
            loss=model.OCSoftmax(args['output_size'], r_real=args['r_real'], r_fake=args['r_fake'], alpha=args['alpha'], class_weight=[5/9, 5]),
            loss_bpl=model.BPL(args['alpha_cl'])).to(device).eval().requires_grad_(False)
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        state = {key.removeprefix('module.'): value for key, value in state.items()}
        obj.load_state_dict(state, strict=True)
        for module in obj.modules():
            if hasattr(module, 'noise_std'):
                module.noise_std = 0.
        heads[name] = obj
        checks[name] = {'path': str(checkpoint), 'sha256': source['sha256'], 'strict_load': True}
        del state
    toy = np.sin(2*np.pi*440*np.arange(40960)/SR).astype(np.float32)
    assert np.array_equal(stretch(toy, 1.), toy)
    toy_views = views(toy)
    assert len(toy_views[4][1]) == CAP
    assert len(toy_views[5][1]) == len(toy)
    qc = {'rank': rank, 'device': device, 'librosa_version': librosa.__version__,
          'soundfile_version': sf.__version__, 'torch_version': torch.__version__,
          'phase_vocoder': {'n_fft': 2048, 'hop_length': 512, 'pitch_preserving': True,
                            'identity_rate_is_exact_copy': True},
          'frontend_revision': revision, 'pooling_noise_std': 0., 'checkpoints': checks,
          'toy_identity_array_equal': True, 'toy_normalized_samples': len(toy_views[4][1]),
          'no_output_length_repair': True, 'audio_or_hidden_saved': False,
          'selected_sha256': sha(root/'selected.csv'), 'protocol_sha256': sha(root/'protocol.json')}
    save(root/f'QC_rank{rank}{suffix}.json', qc)
    items = {phase: make_items(args, phase) for phase in assigned.phase.unique()}
    start = time.time()
    rows = 0
    cache_hits = 0
    with partial.open('x', newline='') as stream, torch.inference_mode():
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for number, record in enumerate(assigned.to_dict('records')):
            item = items[record['phase']][int(record['index'])]
            assert Path(item.path).stem == record['uid'] and int(item.label) == int(record['label'])
            assert str(getattr(item, 'speaker', 'unknown')) == str(record['speaker'])
            assert str(getattr(item, 'type', 'unknown')) == str(record['attack'])
            wave, sr = sf.read(item.path, dtype='float32')
            assert sr == SR and wave.ndim == 1 and len(wave) == record['original_samples']
            assert 720 <= len(wave) <= 30*SR and np.isfinite(wave).all()
            baseline = {}
            # Reuse scores for byte-identical conditions; no hidden states are retained.
            output_cache = {}
            for condition, audio, rate in views(wave):
                wave_sha = hashlib.sha256(audio.tobytes()).hexdigest()
                if wave_sha in output_cache:
                    measured = output_cache[wave_sha]
                    cache_hits += 1
                else:
                    x, mask = native_steps.features(plm, [torch.from_numpy(audio)], device)
                    measured = {}
                    for name, head in heads.items():
                        z = head(x, mask=mask, return_embedding=True)
                        assert torch.isfinite(z).all()
                        norm = float(z.norm(dim=1).item())
                        assert norm > 0
                        unit = torch.nn.functional.normalize(z, dim=1)
                        centroid = torch.nn.functional.normalize(head.loss.weight, dim=1)
                        score = float((unit@centroid.T).item())
                        if condition == 'native_cap':
                            baseline[name] = unit.detach().clone()
                        cosine = float((unit*baseline[name]).sum().item())
                        assert -1.00001 <= score <= 1.00001 and -1.00001 <= cosine <= 1.00001
                        measured.update({f'score_{name}': score, f'cos_{name}_native': float(np.clip(cosine, -1, 1)), f'norm_{name}': norm})
                        del z, unit, centroid
                    output_cache[wave_sha] = measured
                    del x, mask
                out = {key: record[key] for key in META}
                out.update(condition=condition, rate=rate, used_samples=len(audio), wave_sha256=wave_sha, gpu_rank=rank)
                out.update(measured)
                writer.writerow(out)
                rows += 1
            if (number+1) % 8 == 0 or number+1 == len(assigned):
                stream.flush()
                save(root/f'progress_rank{rank}{suffix}.json', {
                    'utterances': number+1, 'expected_utterances': len(assigned), 'rows': rows,
                    'elapsed_seconds': time.time()-start, 'updated_unix': time.time(), 'smoke': smoke})
        stream.flush()
        os.fsync(stream.fileno())
    assert rows == len(assigned)*len(CONDITIONS)
    partial.replace(destination)
    save(root/f'done_rank{rank}{suffix}.json', {
        'rank': rank, 'utterances': len(assigned), 'rows': rows, 'sha256': sha(destination),
        'elapsed_seconds': time.time()-start, 'identical_wave_cache_hits': cache_hits,
        'selected_sha256': sha(root/'selected.csv'), 'protocol_sha256': sha(root/'protocol.json'),
        'smoke': smoke})


def summarize(root, smoke):
    frame, _ = selection(root, smoke)
    suffix = '_smoke' if smoke else ''
    parts = []
    hashes = {}
    for rank in range(2):
        path = root/f'scores_rank{rank}{suffix}.csv'
        done = json.loads((root/f'done_rank{rank}{suffix}.json').read_text())
        assert sha(path) == done['sha256']
        part = pd.read_csv(path, dtype={'uid': str, 'speaker': str, 'attack': str})
        assert len(part) == done['rows'] and (part.gpu_rank == rank).all()
        parts.append(part)
        hashes[path.name] = done['sha256']
    scores = pd.concat(parts, ignore_index=True)
    assert len(scores) == len(frame)*len(CONDITIONS)
    assert not scores.isna().any().any()
    assert not scores.duplicated(['phase', 'index', 'condition']).any()
    assert set(scores.condition) == set(CONDITIONS)
    for condition in CONDITIONS:
        observed = scores.loc[scores.condition == condition].sort_values(['phase', 'index'])
        expected = frame.sort_values(['phase', 'index'])
        for field in META:
            assert observed[field].tolist() == expected[field].tolist(), (condition, field)
    assert (scores.loc[scores.condition == 'norm_full4', 'used_samples'] == CAP).all()
    for name in HEADS:
        assert np.isfinite(scores[[f'score_{name}', f'cos_{name}_native', f'norm_{name}']]).all().all()
        assert np.allclose(scores.loc[scores.condition == 'native_cap', f'cos_{name}_native'], 1., atol=1e-6)
    destination = root/f'all_scores{suffix}.csv'
    assert not destination.exists(), 'Refuse overwrite'
    scores.to_csv(destination, index=False)
    hashes[destination.name] = sha(destination)
    save(root/f'complete{suffix}.json', {
        'status': 'inference_complete_analysis_pending', 'smoke': smoke,
        'utterances': len(frame), 'rows': len(scores), 'conditions': CONDITIONS, 'heads': HEADS,
        'sha256': hashes, 'selected_sha256': sha(root/'selected.csv'),
        'protocol_sha256': sha(root/'protocol.json'), 'pooling_noise_std': 0.,
        'audio_or_hidden_saved': False, 'no_target_checkpoint_selection': True})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--smoke', action='store_true')
    cli = parser.parse_args()
    assert Path('/.dockerenv').is_file(), 'Run GPU inference only in existing container'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0,1'
    import torch
    import torch.multiprocessing as mp
    assert torch.cuda.device_count() == 2
    assert cli.root.is_dir()
    selection(cli.root, cli.smoke)
    mp.spawn(worker, args=(str(cli.root), cli.smoke), nprocs=2, join=True)
    summarize(cli.root, cli.smoke)


if __name__ == '__main__':
    main()
