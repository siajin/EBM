"""Independent server-side validation of native-quiet paired inference.

Read-only for all experimental sources. Writes one aggregate verification.json.
No model inference and no export of audio or per-file embeddings.
"""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/results/analysis/EBM80_native_quiet_strict/inference')
    args = parser.parse_args()
    assert Path('/.dockerenv').exists(), 'Run only in the experiment container'
    root = Path(args.root)
    parent = root.parent
    manifest = json.loads((root / 'sample_manifest.json').read_text())
    provenance = json.loads((root / 'provenance.json').read_text())
    completed = json.loads((root / 'COMPLETED.json').read_text())
    plan = json.loads((parent / 'plan.json').read_text())
    n = len(manifest)
    assert n == 4000 and completed['completed_trials'] == n
    assert provenance['workers'] == completed['workers'] == 2
    assert manifest == json.loads((parent / 'sample_manifest.json').read_text())
    assert len({t['id'] for t in manifest}) == n
    assert len({t['pcm_sha256'] for t in manifest}) == n
    assert not plan['exact_pcm_duplicate_pairs']
    labels = np.array([t['label'] for t in manifest], dtype=int)
    cohorts = np.array([t['cohort'] for t in manifest])
    sources = np.array([t['source'] for t in manifest])
    speakers = np.array([t['speaker'] for t in manifest])
    assert set(labels.tolist()) == {0, 1}
    assert set(cohorts.tolist()) == {'previous', 'new_files'}
    counts = Counter((t['cohort'], t['label'], t['source']) for t in manifest)
    for cohort in ['previous', 'new_files']:
        for label in [0, 1]:
            assert sum(v for (c, y, _), v in counts.items() if c == cohort and y == label) == 1000
    for label in [0, 1]:
        for source in np.unique(sources):
            assert counts['previous', label, source] == counts['new_files', label, source]
            assert counts['previous', label, source] == plan['source_label_counts_per_cohort'][str((int(label), str(source)))]

    data = Path('/data/ASVspoof2021_DF')
    protocol = data / 'keys/DF/CM/trial_metadata.txt'
    metadata = {}
    for line in protocol.read_text().splitlines():
        f = line.split()
        if len(f) >= 9 and f[7] == 'eval':
            metadata[f[1]] = (f[0], f[3], f[2], int(f[5] == 'bonafide'), f[4])
    for t in manifest:
        assert metadata[t['id']] == (t['speaker'], t['source'], t['codec'], t['label'], t['attack'])
        assert isinstance(t['samples'], int) and t['samples'] > 512
        assert len(t['pcm_sha256']) == 64

    names = ['native', 'identity']
    for dose in ['025', '050', '100']:
        names += [f'zero_front_{dose}', f'zero_back_{dose}', f'own_front_{dose}', f'own_back_{dose}']
    names += ['ramp_front_100', 'ramp_back_100']
    shards = [np.load(root / f'shard_{i}.npz', allow_pickle=False) for i in range(2)]
    for shard in shards:
        assert shard['names'].tolist() == names
    indices = np.concatenate([a['indices'] for a in shards]).astype(int)
    assert np.array_equal(np.sort(indices), np.arange(n))
    order = np.argsort(indices)
    z = np.concatenate([a['embeddings'] for a in shards])[order].astype(np.float64)
    scores = np.concatenate([a['scores'] for a in shards])[order].astype(np.float64)
    eligible = np.concatenate([a['eligible'] for a in shards])[order].astype(bool)
    prefixes = np.concatenate([a['prefixes'] for a in shards])[order].astype(int)
    lengths = np.concatenate([a['lengths'] for a in shards])[order].astype(int)
    checks = np.concatenate([a['checks'] for a in shards])[order]
    seams = np.concatenate([a['seams'] for a in shards])[order]
    center = shards[0]['center'].astype(np.float64)
    assert np.array_equal(shards[0]['center'], shards[1]['center'])
    assert z.shape == (n, 16, 160) and scores.shape == (n, 16)
    assert center.shape == (160,) and abs(np.linalg.norm(center) - 1) < 2e-6
    assert checks.shape == (n, 3) and seams.shape == (n, 3, 2, 2)
    assert np.isfinite(checks).all() and np.max(np.abs(checks[:, 0])) < 1e-8
    assert np.all(checks[:, 1:] == 1)
    assert np.array_equal(lengths, np.array([t['samples'] for t in manifest]))
    assert np.all((prefixes >= 0) & (prefixes < lengths) & (prefixes % 320 == 0))
    assert np.array_equal(eligible, prefixes >= 320)
    expected = np.ones((n, 16), dtype=bool)
    for col in [4, 5, 8, 9, 12, 13]:
        expected[:, col] = eligible
    assert np.isfinite(z[expected]).all() and np.isfinite(scores[expected]).all()
    assert np.isnan(z[~expected]).all() and np.isnan(scores[~expected]).all()
    assert np.isfinite(seams[:, :, 0]).all()
    assert np.isfinite(seams[eligible, :, 1]).all() and np.isnan(seams[~eligible, :, 1]).all()
    norm_error = float(np.max(np.abs(np.linalg.norm(z[expected], axis=1) - 1)))
    assert norm_error < 2e-6
    # Use float64 elementwise product/sum, independently of the inference einsum.
    rederived = np.sum(z * center[None, None, :], axis=-1)
    score_error = float(np.max(np.abs(rederived[expected] - scores[expected])))
    assert score_error < 2e-6 and np.max(np.abs(scores[expected])) <= 1.00001
    identity_delta = scores[:, 1] - scores[:, 0]
    epsilon = max(1e-4, 10 * float(np.max(np.abs(identity_delta))))

    hashes = {'plan': sha(parent / 'plan.json'), 'manifest': sha(parent / 'sample_manifest.json'),
              'protocol': sha(protocol), 'checkpoint': sha(provenance['checkpoint'])}
    assert hashes['plan'] == provenance['plan_sha256']
    assert hashes['manifest'] == provenance['manifest_sha256']
    assert hashes['protocol'] == plan['protocol_sha256']
    assert hashes['checkpoint'] == provenance['checkpoint_sha256']
    source = Path('/home/mentee/results/CM/LA/EBM/backup/scripts')
    hashes['model_sources'] = {name: sha(source / name) for name in provenance['source_sha256']}
    assert hashes['model_sources'] == provenance['source_sha256']
    inference_script = Path('/home/mentee/analysis/native_quiet_strict.py')
    if inference_script.exists():
        hashes['inference_script'] = sha(inference_script)
        assert hashes['inference_script'] == provenance['script_sha256']

    def endpoint(mask, before, after):
        count = int(mask.sum())
        if count == 0:
            return {'n': 0}
        delta = scores[mask, after] - scores[mask, before]
        dz = z[mask, after] - z[mask, before]
        projected = np.sum(dz * center[None, :], axis=1)
        vector_error = float(np.max(np.abs(delta - projected)))
        assert vector_error < 4e-6
        norm = np.linalg.norm(dz, axis=1)
        assert np.all(np.abs(projected) <= norm + 3e-6)
        assert np.all(np.sign(projected[np.abs(delta) > 4e-6]) == np.sign(delta[np.abs(delta) > 4e-6]))
        # Orthogonal movement uses a precisely normalized center for decomposition.
        unit = center / np.linalg.norm(center)
        radial = dz @ unit
        orthogonal = np.linalg.norm(dz - radial[:, None] * unit[None, :], axis=1)
        assert np.max(np.abs(norm**2 - radial**2 - orthogonal**2)) < 1e-10
        return {'n': count, 'mean_delta_cosine': float(delta.mean()), 'median_delta_cosine': float(np.median(delta)),
                'raw_positive_n': int((delta > 0).sum()), 'toward_center_n': int((delta > epsilon).sum()),
                'away_from_center_n': int((delta < -epsilon).sum()), 'within_epsilon_n': int((np.abs(delta) <= epsilon).sum()),
                'toward_ge001_n': int((delta >= .01).sum()), 'away_ge001_n': int((delta <= -.01).sum()),
                'toward_ge01_n': int((delta >= .1).sum()), 'away_ge01_n': int((delta <= -.1).sum()),
                'mean_embedding_displacement': float(norm.mean()), 'median_embedding_displacement': float(np.median(norm)),
                'mean_orthogonal_displacement': float(orthogonal.mean()), 'max_cosine_projection_discrepancy': vector_error,
                'speaker_source_cluster_n': len({(sources[k], speakers[k]) for k in np.flatnonzero(mask)})}

    primary = {}
    for family, front, back in [('zero', 10, 11), ('own40', 12, 13)]:
        primary[family] = {}
        for label, word in [(1, 'bonafide'), (0, 'spoof')]:
            mask = (cohorts == 'new_files') & (labels == label)
            if family == 'own40':
                mask &= eligible
            primary[family][word] = {'front_minus_original': endpoint(mask, 0, front),
                                     'front_minus_back': endpoint(mask, back, front),
                                     'back_minus_original_secondary': endpoint(mask, 0, back)}

    # Twelve deterministic PCM spot checks: one file from each cohort/label/source stratum.
    import soundfile as sf
    checked = []
    max_reconstructed_energy_error = 0.
    for cohort in ['previous', 'new_files']:
        for label in [1, 0]:
            for source_name in sorted(np.unique(sources)):
                candidates = [k for k, t in enumerate(manifest) if t['cohort'] == cohort and t['label'] == label and t['source'] == source_name]
                k = min(candidates, key=lambda ix: manifest[ix]['id'])
                t = manifest[k]
                audio, sr = sf.read(data / 'ASVspoof2021_DF_eval/flac' / (t['id'] + '.wav'), dtype='float32')
                assert sr == 16000 and audio.ndim == 1 and np.isfinite(audio).all()
                assert len(audio) == t['samples'] and hashlib.sha256(audio.tobytes()).hexdigest() == t['pcm_sha256']
                frame_energy = np.array([np.mean(np.square(audio[q:q+320].astype(np.float64))) for q in range(0, len(audio), 320)])
                frame_rms = np.sqrt(frame_energy + 1e-20)
                left = int(np.flatnonzero(frame_rms >= frame_rms.max() / 100)[0]) * 320
                assert left == prefixes[k]
                for dose in [4000, 8000, 16000]:
                    fillers = [np.zeros(dose, dtype=np.float32)]
                    if left >= 320:
                        fillers.append(np.tile(audio[:left], int(np.ceil(dose / left)))[:dose])
                    for q in fillers:
                        front = np.r_[q, audio]; back = np.r_[audio, q]
                        assert front.shape == back.shape == (len(audio) + dose,)
                        assert front[dose:].tobytes() == audio.tobytes() == back[:len(audio)].tobytes()
                        assert front[:dose].tobytes() == back[len(audio):].tobytes()
                        energy_error = abs(float(np.dot(front.astype(float), front.astype(float))) - float(np.dot(back.astype(float), back.astype(float))))
                        max_reconstructed_energy_error = max(max_reconstructed_energy_error, energy_error)
                        assert energy_error < 1e-8
                checked.append({'cohort': cohort, 'label': label, 'source': source_name})
    assert len(checked) == 12
    result = {'all_checks_passed': True, 'completed_trials': n, 'source_label_counts': [dict(cohort=k[0], label=k[1], source=k[2], n=v) for k, v in sorted(counts.items())],
              'maximum_embedding_norm_error': norm_error, 'maximum_rederived_cosine_error': score_error,
              'identity_max_abs_delta': float(np.max(np.abs(identity_delta))), 'identity_median_abs_delta': float(np.median(np.abs(identity_delta))),
              'direction_epsilon': epsilon, 'primary_point_estimates': primary, 'hashes_verified': hashes,
              'decoded_pcm_spot_checks': {'n': len(checked), 'strata': checked, 'max_energy_error': max_reconstructed_energy_error},
              'limits': ['PCM reread and independent concatenation reconstruction cover 12 deterministic files; they are not a saved-input audit of all GPU forwards.',
                         'All 4000 inference rows have runtime PCM-hash and content-preservation assertions in the separately reviewed inference implementation.',
                         'This validator does not independently calculate bootstrap intervals or certify a historical learning mechanism.',
                         'Ramp fillers differ at the two ends because their values depend on different native endpoints. They are secondary splice sensitivity controls, not identical-filler pairs.']}
    (root / 'verification.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({'all_checks_passed': True, 'completed_trials': n, 'max_cosine_error': score_error,
                      'identity_max_delta': result['identity_max_abs_delta'], 'primary_point_estimates': primary,
                      'decoded_pcm_spot_checks_n': len(checked)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
