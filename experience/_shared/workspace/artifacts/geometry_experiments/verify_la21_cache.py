"""Validate the completed 21LA derived cache offline, without fitting a model."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from audit_cached_core_results import independent_auc, independent_eer
from geometry_cache import DEFAULT_ROOT, load_embeddings, read_json, resolve_run, validate_run

MODEL_HASH = '104873703a57cad98b686d770e0860e031a2cf9abef79b72668780c1660e0b24'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    start = time.monotonic()
    folder, entry, split = resolve_run('21LA_eval', args.cache_root)
    validation = validate_run('21LA_eval', args.cache_root)
    data = load_embeddings('21LA_eval', args.cache_root, expected_checkpoint_sha256=MODEL_HASH)
    audit = read_json(folder / 'input_audit.json')
    qc = read_json(folder / 'qc/summary.json')
    summary = read_json(folder / 'extract/summary.json')
    rows = data.rows
    assert split == 'R21LA' and set(rows.split) == {'R21LA'}
    assert len(rows) == 148176 and rows.groupby('y').size().to_dict() == {0: 133360, 1: 14816}
    assert set(rows.dataset) == {'ASVspoof2021_LA'} and 'path' not in rows.columns
    assert rows.speaker_group.nunique() == 67 and rows.codec.nunique() == 7
    assert rows.loc[rows.y == 0, 'attack_id'].nunique() == 13
    assert (rows.crop_samples == 64600).all() and (rows.crop_start == 0).all()
    assert (rows.duration > 0).all() and rows.padding_fraction.between(0, 1).all()
    assert audit['full_official_eval'] and summary['full_official_21LA_eval']
    assert qc['single_utterance_reference_verified']
    assert max(qc['single_utterance_reference_max_errors'].values()) <= 1e-5
    assert qc['max_cross_gpu_z_error'] <= 1e-5 and qc['max_repeat_z_error'] <= 1e-5
    assert audit['partition_counts'] == {'eval': 148176, 'progress': 16464, 'hidden': 16926}
    assert audit['protocol_sha256'] == '6cc64a94d74c3c749a4ad8f5e377d0f46ed221477cfe3585996d8b550548bba8'
    for row in rows[['dataset', 'utterance_id', 'condition_id', 'crop_id', 'extraction_seed']].itertuples(index=False):
        key = '|'.join(str(value) for value in row[:4]) + '|20261105'
        expected = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big') % (2**31 - 1)
        assert expected == row.extraction_seed
    geom = data.geometry()
    y, score = rows.y.to_numpy(), geom['cosine']
    metrics = {'eer_percent': independent_eer(y, score), 'auroc': independent_auc(y, score)}
    for key, value in metrics.items():
        assert abs(value - summary['native_metrics'][key]) < 1e-9
    overlap, UID_overlap = {}, {}
    for name in ['T', 'V', 'DF_pilot']:
        other = load_embeddings(name, args.cache_root, expected_checkpoint_sha256=MODEL_HASH)
        assert np.max(np.abs(other.centroid - data.centroid)) < 1e-12
        groups = other.rows.speaker_group.astype(str)
        if name in {'T', 'V'}:
            groups = 'asvspoof::' + groups.str.removeprefix('asvspoof::')
        overlap[name] = len(set(groups) & set(rows.speaker_group))
        UID_overlap[name] = len(set(other.rows.utterance_id) & set(rows.utterance_id))
        if name == 'V':
            bona = np.sort(other.rows.loc[other.rows.y == 1, 'cosine_s'].to_numpy())
            threshold = float(bona[int(np.floor(.01 * len(bona)))])
    assert overlap['T'] == 0 and overlap['V'] == 0
    out = args.out or args.cache_root / 'local_analysis/21la_cache_validation_v1'
    out.mkdir(parents=True, exist_ok=False)
    report = {
        'status': 'pass', 'UTC': datetime.now(timezone.utc).isoformat(),
        'dataset': '21LA_eval', 'rows': len(rows), 'class_counts': {'bonafide': 14816, 'spoof': 133360},
        'z_shape': list(data.z.shape), 'h_shape': list(data.h.shape),
        'cache_fingerprint': entry['cache_fingerprint'], 'checkpoint_sha256': MODEL_HASH,
        'file_hash_validation': validation, 'native_metrics_independent': metrics,
        'raw_forward_score_metrics': {'eer_percent': independent_eer(y, rows.raw_score.to_numpy()),
                                      'auroc': independent_auc(y, rows.raw_score.to_numpy())},
        'V_fixed_threshold': threshold,
        'V_fixed_21LA_errors': {'bona_false_reject': float(np.mean(score[y == 1] < threshold)),
                               'spoof_false_accept': float(np.mean(score[y == 0] >= threshold))},
        'speaker_group_overlap_with_cached_sets': overlap, 'UID_overlap_with_cached_sets': UID_overlap,
        'source_family_map_resolved': False,
        'reference_QC_errors': qc['single_utterance_reference_max_errors'],
        'direction_valid_rows': int(geom['valid'].sum()), 'seed_rows_checked': len(rows),
        'new_training': False, 'new_inference': False, 'offline': True,
        'elapsed_seconds': time.monotonic() - start,
    }
    (out / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
