"""Cache all official 2021 LA eval vectors with frozen, archived EBM weights.

Only the SSL frontend is batched. Each classifier call and its explicit ASP
noise use the original per-utterance RNG recipe. QC compares against complete
single-utterance frontend + classifier inference before extraction is allowed.
Original laboratory code is imported on the server, never copied or modified.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace


CORE_DIR = Path('/home/mentee/analysis/embedding_geometry/20261105_df_fast_01')
if CORE_DIR.is_dir():
    sys.path.insert(0, str(CORE_DIR))
    import df_extract_v1 as core
else:
    import df_extract as core

EXPECTED_HASH = '104873703a57cad98b686d770e0860e031a2cf9abef79b72668780c1660e0b24'
EXPECTED_COUNT = 148176


def protocol_records(lines, audio_root):
    records, partitions = [], Counter()
    for line in lines:
        if not line.strip():
            continue
        a = line.split()
        if len(a) != 8 or a[5] not in {'bonafide', 'spoof'}:
            raise ValueError('Unexpected 2021 LA protocol schema')
        partitions[a[7]] += 1
        if a[7] != 'eval':
            continue
        if (a[4] == 'bonafide') != (a[5] == 'bonafide'):
            raise ValueError('Label differs from archived 2021 LA evaluator')
        speaker = a[0] if a[0] not in {'-', 'unknown'} else 'unknown_shared'
        records.append({
            'dataset': 'ASVspoof2021_LA', 'split': 'R21LA',
            'utterance_id': a[1], 'family_id': a[1],
            'family_resolution': 'source_family_unresolved_speaker_group_only',
            'speaker_group': 'asvspoof::' + speaker,
            'dependency_group': 'asvspoof::' + speaker,
            'speaker_id_known': speaker != 'unknown_shared',
            'y': int(a[5] == 'bonafide'), 'attack_id': a[4],
            'codec': a[2], 'transmission': a[3], 'source_corpus': 'asvspoof', 'trim': a[6],
            'path': str(audio_root / 'flac' / (a[1] + '.flac')),
            'condition_id': 'native_21LA_' + a[2] + '_' + a[3], 'crop_id': 'native_first',
        })
    if len({r['utterance_id'] for r in records}) != len(records):
        raise ValueError('Duplicate evaluation utterance IDs')
    return records, dict(partitions)


def prepare(root, reference, front_batch, graph_frontend=False):
    import pandas as pd
    archived, args = core.settings(reference)
    base = Path(args['path_21LA'])
    protocol = base / 'keys/LA/CM/trial_metadata.txt'
    records, partitions = protocol_records(protocol.read_text().splitlines(), base)
    frame = pd.DataFrame(records).sort_values('utterance_id', kind='stable').reset_index(drop=True)
    assert len(frame) == EXPECTED_COUNT
    assert frame.groupby('y').size().to_dict() == {0: 133360, 1: 14816}
    assert all(Path(p).is_file() for p in frame.path), 'Missing source audio'
    checkpoint_hash = core.digest(reference / 'model/BestCLS_50_21DF.pt')
    assert checkpoint_hash == EXPECTED_HASH
    manifest = root / 'manifest.parquet'
    assert not manifest.exists() and not (root / 'input_audit.json').exists()
    frame.to_parquet(manifest, index=False)
    audit = {
        'status': 'pass', 'profile': '21LA_full_eval_batched_SSL_frozen_EBM_v1',
        'extraction_splits': ['R21LA'], 'Q_features_not_extracted': True,
        'manifest_sha256': core.digest(manifest), 'protocol_sha256': core.digest(protocol),
        'checkpoint_sha256': checkpoint_hash, 'script_sha256': core.digest(__file__),
        'core_extractor_sha256': core.digest(core.__file__),
        'archive_code_sha256': {str(p.relative_to(archived)): core.digest(p) for p in archived.rglob('*.py')},
        'partition_counts': partitions, 'sample_count': len(frame),
        'class_counts': frame.groupby('y').size().to_dict(),
        'speaker_groups': int(frame.speaker_group.nunique()),
        'attack_count': int(frame.loc[frame.y == 0, 'attack_id'].nunique()),
        'codec_counts': frame.codec.value_counts().to_dict(),
        'transmission_counts': frame.transmission.value_counts().to_dict(),
        'unknown_speaker_rows': int((~frame.speaker_id_known).sum()),
        'feature_dimension': 160, 'crop_samples': args['num_test_frames'] * 160 - 40,
        'SSL_frontend_batch_size': front_batch, 'classifier_microbatch_size': 1,
        'frontend_execution_mode': 'single_utterance_CUDA_graph' if graph_frontend else 'eager',
        'per_utterance_seed_base': 20261105,
        'frontend_eval_CPU_RNG_replay': 'one torch.rand([]) per encoder layer; actual forward RNG advancement verified',
        'reference_QC_maximum_absolute_z_h_score_error': 1e-5,
        'original_archived_eval_label_agreement': True,
        'source_family_resolved': False, 'full_official_eval': True,
        'new_EBM_training': False, 'original_sources_modified': False,
    }
    core.save_json(root / 'input_audit.json', audit)
    print(json.dumps({k: v for k, v in audit.items() if k != 'archive_code_sha256'}), flush=True)


def worker(rank, root_string, reference_string, stage, phase_name, front_batch, graph_frontend):
    import numpy as np
    import pandas as pd
    import soundfile as sf
    import torch
    from transformers import AutoConfig, Wav2Vec2Model

    root, reference = Path(root_string), Path(reference_string)
    phase = root / phase_name
    torch.set_num_threads(2)
    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    archived, args = core.settings(reference)
    sys.path.insert(0, str(archived))
    import model
    from data_loaders import EnrollmentSet

    frame = pd.read_parquet(root / 'manifest.parquet')
    if stage == 'qc':
        frame = pd.concat([g.sample(n=32 if label == 1 else 96, random_state=20261105)
                           for label, g in frame.groupby('y')]).sort_values('utterance_id').reset_index(drop=True)
    shared = frame.head(8).copy()
    shard = frame.iloc[rank::2].reset_index(drop=True)
    config = AutoConfig.from_pretrained(args['PLM_name'], local_files_only=True)
    plm = Wav2Vec2Model.from_pretrained(args['PLM_name'], config=config, local_files_only=True).to(device).eval()
    classifier = model.B_Linear(
        num_layer=args['num_layers'], hidden_size=args['hidden_size'], output_size=args['output_size'],
        agg_size=args['agg_size'],
        loss=model.OCSoftmax(args['output_size'], r_real=args['r_real'], r_fake=args['r_fake'], alpha=args['alpha']),
        loss_bpl=model.BPL(alpha=args['alpha_cl']),
    ).to(device).eval()
    state = torch.load(reference / 'model/BestCLS_50_21DF.pt', map_location='cpu', weights_only=True)
    classifier.load_state_dict({key.removeprefix('module.'): value for key, value in state.items()}, strict=True)
    for parameter in list(plm.parameters()) + list(classifier.parameters()):
        parameter.requires_grad_(False)
    centroid = classifier.loss.weight.detach().cpu().numpy().reshape(-1).astype('float64')
    centroid /= np.linalg.norm(centroid)
    captured = {}

    def hook(_module, inputs, output):
        captured['h'] = inputs[0].detach().cpu().numpy().copy()[0]
        captured['z'] = output.detach().cpu().numpy().copy()[0]

    handle = classifier.layer_norm.register_forward_hook(hook)

    graph = graph_input = graph_hidden = None
    if graph_frontend:
        assert front_batch == 1
        # Preserve the original single-utterance kernel shapes. Capture only
        # the frozen SSL computation; classifier ASP noise stays eager.
        with torch.random.fork_rng(devices=[rank]), torch.inference_mode():
            graph_input = torch.zeros((1, args['num_test_frames'] * 160 - 40), device=device)
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    plm(graph_input, output_hidden_states=True)
            torch.cuda.current_stream(device).wait_stream(stream)
            torch.cuda.synchronize(device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_hidden = plm(graph_input, output_hidden_states=True).hidden_states
        print(f'rank={rank} single-utterance SSL CUDA graph ready', flush=True)

    def read_audio(row):
        item = SimpleNamespace(path=row.path, type=row.attack_id, label=int(row.y))
        audio, _, label = EnrollmentSet(args, [item])[0]
        info = sf.info(row.path)
        assert int(label) == row.y and info.samplerate == 16000 and info.channels == 1
        assert len(audio) == args['num_test_frames'] * 160 - 40
        return torch.as_tensor(audio, dtype=torch.float32), info

    def classify(hidden, row, info, seed_already_set=False):
        seed = core.utterance_seed(row, 20261105)
        if not seed_already_set:
            torch.random.default_generator.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            # Installed Transformers draws a LayerDrop scalar even in eval.
            # Its value does not skip layers in eval, but the downstream ASP
            # draws CPU noise, so preserve the complete reference RNG state.
            for _ in range(config.num_hidden_layers):
                torch.rand([])
        _, score = classifier(hidden)
        raw_score = float(score.item())
        z, h = captured['z'].copy(), captured['h'].copy()
        assert np.isfinite(z).all() and np.isfinite(h).all() and np.isfinite(raw_score)
        norm = float(np.linalg.norm(z.astype('float64')))
        assert norm > 1e-12
        unit = z.astype('float64') / norm
        cosine = float(unit @ centroid)
        assert abs(cosine - raw_score) <= 1e-5
        tangent = float(np.linalg.norm(unit - cosine * centroid))
        assert abs(tangent**2 - (1 - cosine**2)) < 1e-10
        metadata = {key: row[key] for key in frame.columns if key != 'path'}
        crop_samples = args['num_test_frames'] * 160 - 40
        metadata.update(raw_score=raw_score, cosine_s=cosine, raw_norm=norm, tangent_norm=tangent,
                        direction_valid=int(tangent >= 1e-4), extraction_seed=seed,
                        duration=info.duration, crop_start=0, crop_samples=crop_samples,
                        padding_fraction=max(0, crop_samples - info.frames) / crop_samples)
        return z, h, metadata

    def infer_batch(rows):
        if graph_frontend and len(rows) > 1:
            return [value for j in range(len(rows)) for value in infer_batch(rows.iloc[j:j+1])]
        rows = list(rows.iterrows())
        inputs, infos = zip(*(read_audio(row) for _, row in rows))
        with torch.random.fork_rng(devices=[rank]), torch.inference_mode():
            cpu_rng = torch.random.get_rng_state()
            gpu_rng = torch.cuda.get_rng_state(rank)
            if graph_frontend:
                graph_input.copy_(torch.stack(inputs).to(device))
                graph.replay()
                hidden = graph_hidden
                # Graph replay has no Python-side LayerDrop scalar draws.
                for _ in range(config.num_hidden_layers):
                    torch.rand([])
            else:
                hidden = plm(torch.stack(inputs).to(device), output_hidden_states=True).hidden_states
            cpu_rng_after_frontend = torch.random.get_rng_state()
            torch.random.set_rng_state(cpu_rng)
            for _ in range(config.num_hidden_layers):
                torch.rand([])
            assert torch.equal(cpu_rng_after_frontend, torch.random.get_rng_state()), 'Unexpected frontend CPU RNG advancement'
            assert torch.equal(gpu_rng, torch.cuda.get_rng_state(rank)), 'Frontend consumed GPU RNG in eval'
            output = []
            for j, (_, row) in enumerate(rows):
                features = torch.stack([layer[j:j+1] for layer in hidden], dim=1)
                output.append(classify(features, row, infos[j]))
        return output

    def infer_reference(row):
        audio, info = read_audio(row)
        seed = core.utterance_seed(row, 20261105)
        with torch.random.fork_rng(devices=[rank]), torch.inference_mode():
            torch.random.default_generator.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            hidden = plm(audio.unsqueeze(0).to(device), output_hidden_states=True).hidden_states
            return classify(torch.stack(hidden, dim=1), row, info, seed_already_set=True)

    first, second = infer_batch(shared), infer_batch(shared)
    repeats = [float(np.max(np.abs(a[0] - b[0]))) for a, b in zip(first, second)]
    assert max(repeats) <= 1e-5
    core.save_json(phase / f'rank{rank}_reproducibility.json', {
        'maximum_repeat_z_error': max(repeats),
        'samples': [{'id': r[2]['utterance_id'], 'z': r[0].tolist(), 's': r[2]['cosine_s']} for r in first],
    })
    n = len(shard)
    zs = np.lib.format.open_memmap(phase / f'rank{rank}_z.npy', mode='w+', dtype='float32', shape=(n, 160))
    hs = np.lib.format.open_memmap(phase / f'rank{rank}_h.npy', mode='w+', dtype='float32', shape=(n, 160))
    metadata_rows, reference_errors = [], []
    start = time.monotonic()
    for begin in range(0, n, front_batch):
        part = shard.iloc[begin:begin+front_batch]
        output = infer_batch(part)
        for offset, ((_, row), (z, h, metadata)) in enumerate(zip(part.iterrows(), output)):
            if stage == 'qc':
                ref_z, ref_h, ref_meta = infer_reference(row)
                errors = {'z': float(np.max(np.abs(z - ref_z))), 'h': float(np.max(np.abs(h - ref_h))),
                          'score': abs(metadata['cosine_s'] - ref_meta['cosine_s'])}
                assert max(errors.values()) <= 1e-5, errors
                reference_errors.append(errors)
            i = begin + offset
            zs[i], hs[i] = z, h
            metadata.update(rank=rank, embedding_row=i)
            metadata_rows.append(metadata)
        done = min(begin + front_batch, n)
        if done % (front_batch * 8) == 0 or done == n:
            elapsed = time.monotonic() - start
            core.save_json(phase / f'rank{rank}_progress.json', {
                'completed': done, 'total': n, 'elapsed_seconds': elapsed,
                'utterances_per_second': done / elapsed,
            })
            print(f'rank={rank} {done}/{n} seconds={elapsed:.1f}', flush=True)
        if done % (front_batch * 32) == 0:
            zs.flush(); hs.flush()
    zs.flush(); hs.flush()
    pd.DataFrame(metadata_rows).to_parquet(phase / f'rank{rank}_rows.parquet', index=False)
    np.save(phase / f'rank{rank}_centroid.npy', centroid)
    record = {
        'gpu': torch.cuda.get_device_name(rank), 'torch': torch.__version__,
        'frontend_commit': getattr(config, '_commit_hash', None), 'strict_state_loading': True,
        'sample_count': n, 'elapsed_seconds': time.monotonic() - start,
        'SSL_frontend_batch_size': front_batch, 'classifier_microbatch_size': 1,
        'frontend_CPU_scalar_draws_per_utterance': config.num_hidden_layers,
        'frontend_execution_mode': 'single_utterance_CUDA_graph' if graph_frontend else 'eager',
        'maximum_allocated_GPU_bytes': torch.cuda.max_memory_allocated(rank),
    }
    if reference_errors:
        record['single_utterance_reference_comparison'] = {
            'sample_count': len(reference_errors),
            **{key: max(r[key] for r in reference_errors) for key in ['z', 'h', 'score']},
        }
    core.save_json(phase / f'rank{rank}_model.json', record)
    handle.remove()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'qc', 'extract'])
    parser.add_argument('--root', required=True)
    parser.add_argument('--phase')
    parser.add_argument('--front-batch', type=int, default=16)
    parser.add_argument('--graph-frontend', action='store_true')
    parser.add_argument('--reference', default='/home/mentee/results/CM/LA/EBM_e50_bs160_lr1e5')
    args = parser.parse_args()
    assert Path('/.dockerenv').exists(), 'Use the existing container'
    root, reference = Path(args.root), Path(args.reference)
    if args.stage == 'prepare':
        prepare(root, reference, args.front_batch, args.graph_frontend)
        return
    import torch
    import torch.multiprocessing as mp
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0,1' and torch.cuda.device_count() == 2
    audit = json.loads((root / 'input_audit.json').read_text())
    assert audit['status'] == 'pass' and audit['SSL_frontend_batch_size'] == args.front_batch
    assert audit['frontend_execution_mode'] == ('single_utterance_CUDA_graph' if args.graph_frontend else 'eager')
    assert core.digest(root / 'manifest.parquet') == audit['manifest_sha256']
    assert core.digest(reference / 'model/BestCLS_50_21DF.pt') == audit['checkpoint_sha256'] == EXPECTED_HASH
    assert core.digest(__file__) == audit['script_sha256'], 'Extraction source changed since manifest lock'
    if args.stage == 'extract':
        qc = json.loads((root / 'qc/summary.json').read_text())
        assert qc['status'] == 'pass' and qc['single_utterance_reference_verified']
        assert (root / 'qc.exit').read_text().strip() == '0'
    phase_name = args.phase or args.stage
    phase = root / phase_name
    phase.mkdir(exist_ok=False)
    core.save_json(phase / 'run.json', {
        'stage': args.stage, 'script_sha256': core.digest(__file__),
        'core_extractor_sha256': core.digest(core.__file__), 'manifest_sha256': audit['manifest_sha256'],
        'start_epoch': time.time(), 'SSL_frontend_batch_size': args.front_batch,
        'classifier_microbatch_size': 1,
        'frontend_execution_mode': audit['frontend_execution_mode'],
    })
    mp.spawn(worker, args=(str(root), str(reference), args.stage, phase_name, args.front_batch, args.graph_frontend), nprocs=2, join=True)
    models = [json.loads((phase / f'rank{rank}_model.json').read_text()) for rank in [0, 1]]
    if args.stage == 'qc':
        assert all(m['single_utterance_reference_comparison']['sample_count'] == 64 for m in models)
    core.finalize(root, phase_name, args.stage)
    summary = json.loads((phase / 'summary.json').read_text())
    summary.update(SSL_frontend_batch_size=args.front_batch, classifier_microbatch_size=1,
                   frontend_execution_mode=audit['frontend_execution_mode'],
                   full_official_21LA_eval=(args.stage == 'extract'))
    if args.stage == 'qc':
        summary['single_utterance_reference_verified'] = True
        summary['single_utterance_reference_max_errors'] = {
            key: max(m['single_utterance_reference_comparison'][key] for m in models)
            for key in ['z', 'h', 'score']
        }
    if args.stage == 'extract':
        import pandas as pd
        from fast_analyze_v1 import metrics
        rows = pd.read_parquet(phase / 'scores.parquet')
        summary['native_metrics'] = metrics(rows.y.to_numpy(), rows.cosine_s.to_numpy())
    core.save_json(phase / 'summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
