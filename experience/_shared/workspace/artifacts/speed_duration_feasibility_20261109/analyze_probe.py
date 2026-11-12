"""Analyze locked speed-intervention scores; diagnostic strata, not benchmark EER.

Usage: python artifacts/speed_duration_feasibility_20261109/analyze_probe.py [probe_directory]
No SSH, audio, model fitting, checkpoint loading, or inference.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score

HEADS = ['legacy240', 'B', 'C']
CONDITIONS = ['native_cap', 'full_native', 'slow_cap', 'fast_cap', 'norm_full4', 'roundtrip_cap']
META = ['phase', 'index', 'uid', 'speaker', 'label', 'original_samples', 'attack']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def manual_roc(y, score):
    order = np.argsort(-score, kind='stable')
    labels, sorted_score = y[order], score[order]
    ends = np.r_[np.flatnonzero(sorted_score[1:] != sorted_score[:-1]), len(score)-1]
    tp = np.r_[0, np.cumsum(labels == 1)[ends]]
    fp = np.r_[0, np.cumsum(labels == 0)[ends]]
    return fp/(y == 0).sum(), tp/(y == 1).sum()


def eer_metrics(y, score):
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    assert set(y) == {0, 1} and np.isfinite(score).all()
    fpr, tpr, thresholds = roc_curve(y, score, pos_label=1, drop_intermediate=False)
    mfpr, mtpr = manual_roc(y, score)
    assert len(fpr) == len(mfpr)
    assert np.allclose(fpr, mfpr, atol=1e-14) and np.allclose(tpr, mtpr, atol=1e-14)
    difference = fpr+tpr-1
    crossing = int(np.flatnonzero(difference >= 0)[0])
    if crossing == 0:
        eer = float(fpr[0])
    else:
        fraction = -difference[crossing-1]/(difference[crossing]-difference[crossing-1])
        eer = float(fpr[crossing-1]+fraction*(fpr[crossing]-fpr[crossing-1]))
    # Independent crossing solve on the separately generated tie-group ROC.
    before = np.flatnonzero(mfpr+mtpr <= 1)[-1]
    if mfpr[before]+mtpr[before] == 1:
        checked_eer = float(mfpr[before])
    else:
        dx = mfpr[before+1]-mfpr[before]
        dy = mtpr[before+1]-mtpr[before]
        checked_eer = float(mfpr[before]+dx*(1-mfpr[before]-mtpr[before])/(dx+dy))
    assert abs(eer-checked_eer) < 1e-12
    j = min(range(len(fpr)), key=lambda i: (abs(difference[i]), fpr[i], -thresholds[i]))
    # A constant-score sample can select the all-reject ROC endpoint. Keep an
    # equivalent finite threshold so strict JSON remains portable.
    selected_threshold = float(thresholds[j])
    if np.isposinf(selected_threshold):
        selected_threshold = float(np.nextafter(score.max(), np.inf))
    assert np.isfinite(selected_threshold)
    return {'diagnostic_EER_percent': 100*eer, 'diagnostic_AUROC': float(roc_auc_score(y, score)),
            'sample_dev_EER_point_threshold': selected_threshold,
            'ROC_independent_max_error': float(max(np.max(np.abs(fpr-mfpr)), np.max(np.abs(tpr-mtpr))))}


def errors(y, score, threshold):
    y, score = np.asarray(y), np.asarray(score)
    rejected = score < threshold
    normal, spoof = y == 1, y == 0
    nr, sa = int(rejected[normal].sum()), int((~rejected[spoof]).sum())
    return {'normal_rejected': nr, 'spoof_accepted': sa,
            'normal_reject_percent': float(100*nr/normal.sum()),
            'spoof_accept_percent': float(100*sa/spoof.sum())}


def rate_group(values):
    return pd.cut(values, [0, .5, .8, 1., 1.25, 2., np.inf], right=False,
                  labels=['<0.5', '0.5–<0.8', '0.8–<1', '1–<1.25', '1.25–<2', '>=2']).astype(str)


def validate_lengths(scores, protocol, complete):
    """Preserve exact phase-vocoder checks; allow only declared atempo errors."""
    target = int(protocol['target_samples'])
    capped = np.minimum(scores.original_samples.to_numpy(), target)
    expected = capped.copy()
    full = scores.condition.to_numpy() == 'full_native'
    normalized = scores.condition.to_numpy() == 'norm_full4'
    speed = scores.condition.isin(['slow_cap', 'fast_cap']).to_numpy()
    expected[full] = scores.original_samples.to_numpy()[full]
    expected[normalized] = target
    expected[speed] = np.rint(capped[speed]/scores.rate.to_numpy()[speed]).astype('int64')
    assert (scores.used_samples >= 720).all()
    is_atempo = 'ffmpeg atempo' in protocol.get('algorithm', '').lower()
    error = scores.used_samples.to_numpy()-expected
    if is_atempo:
        assert 'ffmpeg atempo' in complete.get('algorithm', '').lower()
        assert complete['no_output_length_repair'] is True
        assert {'expected_samples', 'length_error_samples'}.issubset(scores.columns)
        assert np.array_equal(scores.expected_samples.to_numpy(), expected)
        assert np.array_equal(scores.length_error_samples.to_numpy(), error)
        assert (error[scores.condition.isin(['native_cap', 'full_native'])] == 0).all()
    else:
        # The original 512-utterance run retains every exact-length invariant.
        assert (scores.loc[scores.condition == 'norm_full4', 'used_samples'] == target).all()
        assert (error == 0).all(), 'Phase-vocoder output must retain its exact requested lengths'
    diagnostics = {}
    for condition in CONDITIONS:
        mask = scores.condition.to_numpy() == condition
        values, differences = scores.used_samples.to_numpy()[mask], error[mask]
        diagnostics[condition] = {
            'N': int(mask.sum()), 'actual_samples_min': int(values.min()),
            'actual_samples_max': int(values.max()), 'actual_seconds_mean': float(values.mean()/16000),
            'max_absolute_error_samples': int(np.abs(differences).max()),
            'max_absolute_error_seconds': float(np.abs(differences).max()/16000),
            'mean_signed_error_samples': float(differences.mean()),
            'mean_absolute_error_samples': float(np.abs(differences).mean())}
        if is_atempo:
            published = complete['output_length_diagnostics'][condition]
            for key in ['actual_samples_min', 'actual_samples_max', 'max_absolute_error_samples', 'mean_signed_error_samples']:
                assert np.isclose(diagnostics[condition][key], published[key], atol=1e-12), (condition, key)
    return diagnostics, is_atempo


def main(root):
    protocol = json.loads((root/'protocol.json').read_text())
    complete = json.loads((root/'complete.json').read_text())
    assert complete['status'] == 'inference_complete_analysis_pending'
    assert complete['selected_sha256'] == protocol['selected_sha256'] == sha(root/'selected.csv')
    assert complete['protocol_sha256'] == sha(root/'protocol.json')
    assert set(complete['conditions']) == set(CONDITIONS) and set(complete['heads']) == set(HEADS)
    score_path = root/'all_scores.csv'
    assert sha(score_path) == complete['sha256']['all_scores.csv']
    selection = pd.read_csv(root/'selected.csv', dtype={'uid': str, 'speaker': str, 'attack': str})
    scores = pd.read_csv(score_path, dtype={'uid': str, 'speaker': str, 'attack': str})
    assert len(selection) == protocol['N'] == complete['utterances']
    assert len(scores) == complete['rows'] == len(selection)*len(CONDITIONS)
    assert not selection.duplicated(['phase', 'index']).any()
    assert not scores.duplicated(['phase', 'index', 'condition']).any()
    assert not scores.isna().any().any()
    assert set(scores.condition) == set(CONDITIONS) and set(scores.gpu_rank) == {0, 1}
    reference = selection.sort_values(['phase', 'index']).reset_index(drop=True)
    for condition in CONDITIONS:
        actual = scores.loc[scores.condition == condition].sort_values(['phase', 'index']).reset_index(drop=True)
        for key in META:
            assert actual[key].tolist() == reference[key].tolist(), (condition, key)
    for rank in range(2):
        path = root/f'scores_rank{rank}.csv'
        done = json.loads((root/f'done_rank{rank}.json').read_text())
        assert sha(path) == done['sha256'] == complete['sha256'][path.name]
    target = int(protocol['target_samples'])
    length_diagnostics, is_atempo = validate_lengths(scores, protocol, complete)
    phase_class_counts = selection.groupby(['phase', 'label']).size()
    for (phase, label), count in phase_class_counts.items():
        strata = selection.loc[(selection.phase == phase) & (selection.label == label)].groupby('length_bin').size()
        assert (strata == protocol['quota_per_phase_class_length_bin']).all()
        assert int(count) == int(strata.sum())
    scores = scores.merge(selection[['phase', 'index', 'length_bin']], on=['phase', 'index'], how='left', validate='many_to_one')
    scores['required_full4_rate'] = scores.original_samples/target
    scores['required_full4_rate_group'] = rate_group(scores.required_full4_rate)
    scores['applied_stage_rate_group'] = rate_group(scores.rate)
    baseline = scores.loc[scores.condition == 'native_cap'].set_index(['phase', 'index'])
    secondary_thresholds = {}
    for name in HEADS:
        secondary_thresholds[name] = {}
        for condition in CONDITIONS:
            dev = scores.loc[(scores.phase == 'dev') & (scores.condition == condition)]
            secondary_thresholds[name][condition] = eer_metrics(dev.label, dev[f'score_{name}'])['sample_dev_EER_point_threshold']

    metrics, pairs = [], []
    for phase in protocol['phases']:
        for name in HEADS:
            fixed_threshold = float(protocol['dev_thresholds'][name])
            for condition in CONDITIONS:
                current = scores.loc[(scores.phase == phase) & (scores.condition == condition)].copy()
                normal_n = int((current.label == 1).sum())
                spoof_n = int((current.label == 0).sum())
                assert normal_n == spoof_n and normal_n > 0
                score = current[f'score_{name}'].to_numpy()
                assert np.isfinite(score).all() and np.all(np.abs(score) <= 1.00001)
                record = {'phase': phase, 'head': name, 'condition': condition,
                          'N': len(current), 'normal_N': normal_n, 'spoof_N': spoof_n,
                          'single_class_error_step_percent': 100/normal_n,
                          'fixed_full_dev_threshold': fixed_threshold,
                          'used_samples_min': int(current.used_samples.min()),
                          'used_samples_max': int(current.used_samples.max()),
                          'used_seconds_mean': float(current.used_samples.mean()/16000)}
                record.update(eer_metrics(current.label, score))
                # On target sets this field is still the threshold selected only on dev.
                record.pop('sample_dev_EER_point_threshold')
                record.update({f'fixed_{k}': v for k, v in errors(current.label, score, fixed_threshold).items()})
                sample_dev_threshold = secondary_thresholds[name][condition]
                record['secondary_same_condition_sample_dev_threshold'] = sample_dev_threshold
                record.update({f'secondary_sample_dev_{k}': v for k, v in errors(current.label, score, sample_dev_threshold).items()})
                metrics.append(record)
                keys = pd.MultiIndex.from_frame(current[['phase', 'index']])
                native = baseline.loc[keys]
                native_score = native[f'score_{name}'].to_numpy()
                current['score_delta'] = score-native_score
                current['absolute_score_delta'] = np.abs(current.score_delta)
                current['centroid_angle_delta_degrees'] = np.degrees(np.arccos(np.clip(score, -1, 1))-np.arccos(np.clip(native_score, -1, 1)))
                current['native_embedding_angle_degrees'] = np.degrees(np.arccos(np.clip(current[f'cos_{name}_native'], -1, 1)))
                current['decision_changed'] = (score >= fixed_threshold) != (native_score >= fixed_threshold)
                current['rejected'] = score < fixed_threshold
                current['native_rejected'] = native_score < fixed_threshold
                current['norm_delta'] = current[f'norm_{name}'].to_numpy()-native[f'norm_{name}'].to_numpy()
                cuts = [('all', pd.Series('all', index=current.index)),
                        ('original_length_bin', current.length_bin),
                        ('required_full4_rate_group', current.required_full4_rate_group),
                        ('applied_stage_rate_group', current.applied_stage_rate_group)]
                for dimension, grouping in cuts:
                    for label in [1, 0]:
                        for group_value in grouping.loc[current.label == label].unique():
                            group = current.loc[(current.label == label) & (grouping == group_value)]
                            if len(group) == 0:
                                continue
                            r = {'phase': phase, 'head': name, 'condition': condition,
                                 'class': 'normal' if label else 'spoof', 'label': label,
                                 'group_dimension': dimension, 'group_value': str(group_value), 'N': len(group),
                                 'mean_score': float(group[f'score_{name}'].mean()),
                                 'mean_score_delta': float(group.score_delta.mean()),
                                 'median_score_delta': float(group.score_delta.median()),
                                 'p10_score_delta': float(group.score_delta.quantile(.1)),
                                 'p90_score_delta': float(group.score_delta.quantile(.9)),
                                 'mean_absolute_score_delta': float(group.absolute_score_delta.mean()),
                                 'mean_centroid_angle_delta_degrees': float(group.centroid_angle_delta_degrees.mean()),
                                 'mean_native_embedding_cosine': float(group[f'cos_{name}_native'].mean()),
                                 'mean_native_embedding_angle_degrees': float(group.native_embedding_angle_degrees.mean()),
                                 'mean_norm_delta': float(group.norm_delta.mean()),
                                 'fixed_threshold_decisions_changed': int(group.decision_changed.sum()),
                                 'fixed_threshold_current_class_errors': int(group.rejected.sum() if label else (~group.rejected).sum()),
                                 'fixed_threshold_native_class_errors': int(group.native_rejected.sum() if label else (~group.native_rejected).sum())}
                            pairs.append(r)
    metric_frame, pair_frame = pd.DataFrame(metrics), pd.DataFrame(pairs)
    assert len(metric_frame) == len(protocol['phases'])*len(HEADS)*len(CONDITIONS)
    assert not metric_frame.isna().any().any() and not pair_frame.isna().any().any()
    metric_frame.to_csv(root/'metrics.csv', index=False, encoding='utf-8-sig')
    pair_frame.to_csv(root/'pair_changes.csv', index=False, encoding='utf-8-sig')
    preservation = None
    observation = root/'last_check.json'
    if observation.exists():
        preservation = json.loads(observation.read_text()).get('preservation')
    roundtrip = pair_frame.loc[(pair_frame.group_dimension == 'all') & (pair_frame.condition == 'roundtrip_cap')]
    normalized = pair_frame.loc[(pair_frame.group_dimension == 'all') & (pair_frame.condition == 'norm_full4')]
    comparisons = []
    for (phase, name, label), group in pair_frame.loc[pair_frame.group_dimension == 'all'].groupby(['phase', 'head', 'label']):
        by_condition = group.set_index('condition')
        comparisons.append({'phase': phase, 'head': name, 'class': 'normal' if label else 'spoof',
            'normalization_mean_absolute_delta': float(by_condition.loc['norm_full4', 'mean_absolute_score_delta']),
            'roundtrip_mean_absolute_delta': float(by_condition.loc['roundtrip_cap', 'mean_absolute_score_delta']),
            'full_native_mean_absolute_delta': float(by_condition.loc['full_native', 'mean_absolute_score_delta']),
            'interpretation': 'Comparison only; normalization also changes content when original exceeds cap. Roundtrip has two phase-vocoder passes and is not an additive artifact correction.'})
    output = {
        'status': 'diagnostic_analysis_complete', 'scope': 'Locked, balanced class-and-length-stratified diagnostic sample; not population benchmark EER.',
        'selection_N': len(selection), 'algorithm': protocol.get('algorithm', 'librosa phase vocoder, exact target length'),
        'sample_class_counts': [{'phase': phase, 'label': int(label), 'N': int(count),
                                'single_class_error_step_percent': 100/int(count)}
                               for (phase, label), count in phase_class_counts.items()],
        'output_length_diagnostics': length_diagnostics,
        'metrics': metrics, 'secondary_sample_dev_thresholds': secondary_thresholds,
        'roundtrip_vs_normalization': comparisons, 'source_checkpoint_preservation': preservation,
        'verification': {'selected_sha256': sha(root/'selected.csv'), 'scores_sha256': sha(score_path),
                         'protocol_sha256': sha(root/'protocol.json'), 'score_completion_hashes_verified': True,
                         'independent_tie_group_ROC_verified': True,
                         'max_ROC_error': float(metric_frame.ROC_independent_max_error.max()),
                         'exact_output_lengths_required': not is_atempo,
                         'atempo_length_errors_independently_recomputed': is_atempo},
        'limitations': protocol['limitations'] + [
            'Diagnostic class sizes and the error-rate step caused by one error are recorded in sample_class_counts; these are not full benchmark populations.',
            'Secondary thresholds use only the same-condition diagnostic dev subset and must not be used as deployment thresholds.',
            'EER can remain stable while all scores shift; fixed-threshold errors and paired class-specific changes must also be inspected.',
            'No confidence intervals or significance claims; bins and extreme sentinels alter the natural length mixture.',
            'Cosine/angle changes are descriptive; no causal separation of time modification from transform artifacts.',
            'Applied rate for roundtrip_cap describes only its second pass; total nominal timing returns to native cap duration.',
            'Preservation is reported only from downloaded last_check.json when present, not independently reread from server.'],
    }
    save(root/'results.json', output)
    columns = ['phase', 'head', 'condition', 'diagnostic_EER_percent', 'fixed_normal_reject_percent', 'fixed_spoof_accept_percent']
    print(metric_frame[columns].to_string(index=False))
    print('Independent ROC check passed; source/checkpoint preservation:', preservation)
    print('Results:', root/'results.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', nargs='?', type=Path, default=Path(__file__).resolve().parent/'probe')
    cli = parser.parse_args()
    main(cli.root)
