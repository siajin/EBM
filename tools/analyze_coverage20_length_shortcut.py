"""CPU-only audit of saved 21LA scores; never imports a model or training code."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, roc_curve


META = ['index', 'uid', 'speaker', 'label', 'original_samples', 'used_samples', 'attack']
BINS = [('le2', 0, 32000), ('2to3', 32000, 48000),
        ('3to4p0375', 48000, 64600), ('over4p0375', 64600, np.inf)]
MODELS = ['coverage20', 'B_e20', 'C_e20', 'B_e45', 'C_e30',
          'B_e50', 'C_e50', 'legacy240_e50_nonrepeat']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def independent_eer(y, score):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    if len(np.unique(y)) != 2:
        return None
    order = np.argsort(-score, kind='stable')
    v, lab = score[order], y[order]
    ends = np.r_[np.flatnonzero(v[1:] != v[:-1]), len(v) - 1]
    tp = np.cumsum(lab)[ends]
    fp = ends + 1 - tp
    fpr = np.r_[0., fp / (y == 0).sum()]
    fnr = np.r_[1., 1 - tp / (y == 1).sum()]
    delta = fpr - fnr
    k = int(np.flatnonzero(delta >= 0)[0])
    w = -delta[k - 1] / (delta[k] - delta[k - 1])
    return float(100 * (fpr[k - 1] + w * (fpr[k] - fpr[k - 1])))


def operating_point(y, score):
    fpr, tpr, thresholds = roc_curve(y, score, drop_intermediate=False)
    delta = fpr - (1 - tpr)
    k = int(np.flatnonzero(delta >= 0)[0])
    w = -delta[k - 1] / (delta[k] - delta[k - 1])
    eer = float(100 * (fpr[k - 1] + w * (fpr[k] - fpr[k - 1])))
    assert abs(eer - independent_eer(y, score)) < 1e-10
    j = min(range(len(delta)), key=lambda i: (abs(delta[i]), fpr[i], -thresholds[i]))
    return {'EER_percent': eer, 'threshold': float(thresholds[j]),
            'FRR_percent': float(100 * (1 - tpr[j])),
            'FAR_percent': float(100 * fpr[j]), 'AUROC': float(roc_auc_score(y, score))}


def read_pair(root, pattern, usecols=None):
    return pd.concat([pd.read_csv(root / pattern.format(r=r), usecols=usecols,
                                  float_precision='round_trip') for r in [0, 1]],
                     ignore_index=True).sort_values('index').reset_index(drop=True)


def summarize(frame, score, threshold, model, masks):
    y = frame.label.to_numpy()
    normal, spoof = y == 1, y == 0
    rejected = score < threshold
    nr = int((normal & rejected).sum())
    sa = int((spoof & ~rejected).sum())
    rows = []
    for name, mask in masks:
        nm, sm = normal & mask, spoof & mask
        bn, sn = int(nm.sum()), int(sm.sum())
        br, sf = int((nm & rejected).sum()), int((sm & ~rejected).sum())
        rows.append({'model': model, 'bin': name, 'normal_n': bn, 'spoof_n': sn,
                     'normal_rejected': br, 'spoof_accepted': sf,
                     'FRR_percent': 100 * br / bn if bn else None,
                     'FAR_percent': 100 * sf / sn if sn else None,
                     'share_normal_errors_percent': 100 * br / nr if nr else None,
                     'share_spoof_errors_percent': 100 * sf / sa if sa else None,
                     'normal_class_share_percent': 100 * bn / normal.sum(),
                     'spoof_class_share_percent': 100 * sn / spoof.sum(),
                     'balanced_error_contribution_pp': 50 * (br / normal.sum() + sf / spoof.sum()),
                     'within_bin_EER_percent': independent_eer(y[mask], score[mask])})
    assert sum(x['normal_n'] for x in rows) == int(normal.sum())
    assert sum(x['spoof_n'] for x in rows) == int(spoof.sum())
    assert sum(x['normal_rejected'] for x in rows) == nr
    assert sum(x['spoof_accepted'] for x in rows) == sa
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    workspace = Path(__file__).resolve().parents[2]
    bcroot = workspace / 'artifacts/ebm_B_C_checkpoints_20261109'
    checks = {}
    for y, s, expected in [([1, 1, 0, 0], [4, 3, 2, 1], 0),
                           ([1, 1, 0, 0], [1, 2, 3, 4], 100),
                           ([1, 1, 0, 0], [1, 1, 1, 1], 50),
                           ([1, 1, 0, 0], [3, 2, 2, 1], 25)]:
        assert abs(operating_point(np.array(y), np.array(s))['EER_percent'] - expected) < 1e-10
    checks['synthetic_tie_aware_EER_cases'] = 4
    cov = read_pair(root, 'coverage20_rank{r}.csv')
    bc = read_pair(bcroot / 'LA_analysis', 'scores_21LA_rank{r}.csv')
    old = read_pair(root, 'legacy50_common_rank{r}.csv')
    source_manifest = json.loads((bcroot / 'LA_analysis/sources.json').read_text())
    hashes = {}
    for r in [0, 1]:
        for path, expected in [
            (root / f'coverage20_rank{r}.csv', json.loads((root / f'done_21LA_rank{r}.json').read_text())['sha256']),
            (root / f'legacy50_common_rank{r}.csv', json.loads((root / f'legacy50_done_rank{r}.json').read_text())['sha256']),
            (bcroot / f'LA_analysis/scores_21LA_rank{r}.csv', source_manifest[f'scores_21LA_rank{r}.csv']['sha256'])]:
            hashes[str(path)] = digest(path)
            assert hashes[str(path)] == expected
    for other in [bc, old]:
        assert cov[META].equals(other[META])
    assert len(cov) == 148176 and cov.uid.is_unique
    assert cov['index'].tolist() == list(range(148176))
    assert set(cov.label) == {0, 1}
    assert int(cov.label.sum()) == 14816
    assert (cov.used_samples == np.minimum(cov.original_samples, 64600)).all()
    assert (cov.original_samples > 0).all()
    assert np.array_equal(old.score_B.to_numpy(), bc.score_B_e50.to_numpy())
    checks.update(full_UID_index_class_length_alignment=True, B50_reference_scores_exact=True)
    frame = cov[META].copy()
    frame['score_coverage20'] = cov.score
    for model in MODELS[1:-1]:
        frame['score_' + model] = bc['score_' + model]
    frame['score_legacy240_e50_nonrepeat'] = old.score_legacy240
    assert np.isfinite(frame.filter(like='score_').to_numpy()).all()
    sample = frame.original_samples.to_numpy()
    y = frame.label.to_numpy()
    masks = [(name, (sample > lo) & (sample <= hi)) for name, lo, hi in BINS]
    detailed_edges = [0, 8000, 16000, 24000, 32000, 40000, 48000, 56000, 64600, np.inf]
    detailed_masks = [(f'({lo / 16000:g},{hi / 16000:g}]', (sample > lo) & (sample <= hi))
                      for lo, hi in zip(detailed_edges[:-1], detailed_edges[1:])]
    expected_bc = json.loads((bcroot / 'metrics.json').read_text())['21LA']
    comparison = json.loads((workspace / 'artifacts/ebm_coverage_e20_eval_20261110/comparison.json').read_text())
    expected_coverage = next(x for x in comparison['same_epoch_comparison'] if x['phase'] == '21LA')['coverage_EER_percent']
    expected_legacy = next(x for x in comparison['fixed50_reference'] if x['phase'] == '21LA')['legacy240_EER_percent']
    global_metrics, rows, fine_rows, cross_rows, associations = {}, [], [], [], []
    metadata = None
    meta_path = root / 'trial_metadata_21LA.txt'
    if meta_path.exists():
        meta = pd.read_csv(meta_path, sep=r'\s+', header=None,
                           names=['speaker', 'uid', 'codec', 'transmission', 'attack', 'class', 'trim', 'subset'])
        meta = meta.loc[meta.subset == 'eval']
        assert len(meta) == 148176 and meta.uid.is_unique
        metadata = frame[['index', 'uid', 'speaker', 'attack', 'label']].merge(
            meta, on='uid', how='left', validate='one_to_one', suffixes=('', '_official'))
        assert not metadata.isna().any().any()
        assert (metadata.speaker == metadata.speaker_official).all()
        assert (metadata.attack == metadata.attack_official).all()
        assert np.array_equal(metadata['class'].eq('bonafide').astype(int).to_numpy(), y)
        checks['official_metadata_UID_speaker_attack_label_join'] = True
        hashes[str(meta_path)] = digest(meta_path)
    for model in MODELS:
        score = frame['score_' + model].to_numpy()
        metric = operating_point(y, score)
        expected = (expected_coverage if model == 'coverage20' else expected_legacy
                    if model == 'legacy240_e50_nonrepeat' else expected_bc[model]['EER_percent'])
        assert abs(metric['EER_percent'] - expected) < 1e-9, (model, metric, expected)
        metric['normal_rejected'] = int(((y == 1) & (score < metric['threshold'])).sum())
        metric['spoof_accepted'] = int(((y == 0) & (score >= metric['threshold'])).sum())
        global_metrics[model] = metric
        rows += summarize(frame, score, metric['threshold'], model, masks)
        fine_rows += summarize(frame, score, metric['threshold'], model, detailed_masks)
        for label, name in [(1, 'bonafide'), (0, 'spoof')]:
            mask = y == label
            associations.append({'model': model, 'class': name,
                                 'spearman_original_duration_score': float(spearmanr(sample[mask], score[mask]).statistic),
                                 'spearman_used_duration_score': float(spearmanr(np.minimum(sample[mask], 64600), score[mask]).statistic)})
        for attack in sorted(frame.loc[frame.label == 0, 'attack'].unique()):
            for bin_name, length_mask in masks:
                mask = (y == 0) & frame.attack.eq(attack).to_numpy() & length_mask
                n = int(mask.sum())
                cross_rows.append({'model': model, 'dimension': 'attack', 'group': attack,
                                   'bin': bin_name, 'class': 'spoof', 'n': n,
                                   'errors': int((mask & (score >= metric['threshold'])).sum()),
                                   'error_percent': 100 * float((score[mask] >= metric['threshold']).mean()) if n else None})
        if metadata is not None:
            for codec in sorted(metadata.codec.unique()):
                for bin_name, length_mask in masks:
                    for label, name in [(1, 'bonafide'), (0, 'spoof')]:
                        mask = (y == label) & metadata.codec.eq(codec).to_numpy() & length_mask
                        n = int(mask.sum())
                        wrong = (score < metric['threshold']) if label == 1 else (score >= metric['threshold'])
                        cross_rows.append({'model': model, 'dimension': 'codec', 'group': codec,
                                           'bin': bin_name, 'class': name, 'n': n,
                                           'errors': int((mask & wrong).sum()),
                                           'error_percent': 100 * float(wrong[mask].mean()) if n else None})
    out = pd.DataFrame(rows)
    contributions = []
    for model in ['B_e20', 'C_e20']:
        new = out.loc[out.model == 'coverage20'].set_index('bin')
        ref = out.loc[out.model == model].set_index('bin')
        for name in new.index:
            contributions.append({'reference': model, 'bin': name,
                                  'normal_error_change': int(new.loc[name, 'normal_rejected'] - ref.loc[name, 'normal_rejected']),
                                  'spoof_error_change': int(new.loc[name, 'spoof_accepted'] - ref.loc[name, 'spoof_accepted']),
                                  'balanced_error_change_pp': float(new.loc[name, 'balanced_error_contribution_pp'] - ref.loc[name, 'balanced_error_contribution_pp'])})
    standardized = []
    if metadata is not None:
        enriched = frame.merge(metadata[['uid', 'codec']], on='uid', validate='one_to_one')
        for model in ['coverage20', 'B_e20', 'C_e20']:
            threshold = global_metrics[model]['threshold']
            for label, cname, groups in [(0, 'spoof', ['attack', 'codec']),
                                         (1, 'bonafide', ['speaker', 'codec'])]:
                z = enriched.loc[enriched.label == label].copy()
                z['short'] = z.original_samples <= 32000
                z['error'] = (z['score_' + model] >= threshold) if label == 0 else (z['score_' + model] < threshold)
                cells = z.groupby(groups + ['short'], observed=True).agg(
                    n=('error', 'size'), errors=('error', 'sum')).reset_index()
                n = cells.pivot(index=groups, columns='short', values='n')
                errors = cells.pivot(index=groups, columns='short', values='errors')
                valid = n.dropna()
                weights = valid.min(axis=1)
                weights = weights / weights.sum()
                short = errors.loc[valid.index, True] / valid[True]
                long = errors.loc[valid.index, False] / valid[False]
                standardized.append({'model': model, 'class': cname, 'strata': 'x'.join(groups),
                                     'both_length_strata': len(valid), 'strata_total': len(n),
                                     'short_covered': int(valid[True].sum()), 'long_covered': int(valid[False].sum()),
                                     'weight_rule': 'min(short_n,long_n), normalized across overlap strata',
                                     'standardized_short_error_percent': float((weights * short).sum() * 100),
                                     'standardized_long_error_percent': float((weights * long).sum() * 100)})
    checks['all8_global_EERs_match_saved_metrics'] = True
    checks['all_length_error_numerators_and_denominators_reconcile'] = True
    result = {'status': 'complete', 'scope': 'Saved21LA scores only; no inference/training',
              'threshold_policy': 'Each model uses ONE global21LA retrospective ROC EER-nearest threshold; ties lowerFPR then higherthreshold. Applied unchanged to every length bin. Diagnostic only, not deployment/model selection.',
              'bin_policy': 'Original waveform duration; right-closed bins, 2seconds includes exactly32000samples; used input capped64600.',
              'global': global_metrics, 'length': rows, 'duration_score_associations': associations,
              'length_only_reference': {'original_duration_EER_percent': independent_eer(y, sample),
                                        'input_capped_duration_EER_percent': independent_eer(y, np.minimum(sample, 64600))},
              'coverage_minus_baseline_contributions': contributions,
              'within_attack_codec_or_speaker_length': standardized,
              'zero_bonafide_at_or_below_1second': int(((y == 1) & (sample <= 16000)).sum()) == 0,
              'exact2second_counts': {'bonafide': int(((y == 1) & (sample == 32000)).sum()),
                                     'spoof': int(((y == 0) & (sample == 32000)).sum())},
              'checks': checks, 'source_sha256': hashes,
              'limits': ['Association does not establish a learnedshortcut or isolate batch/repetition/cropping causality.',
                         'Originalduration4.0375plus inputs are all capped4.0375; those bins reflect source/content/attack differences too.',
                         'B/C saved evaluationRNG differs from coverage20.',
                         'legacy240 is epoch50 while primary comparison isepoch20.',
                         'Per-binEER uses a different threshold per bin and is supplementary; FRR/FAR use globalmodel threshold.',
                         'No statistical independence assumption across codec versions or speakers.']}
    for name, data in [('length_errors.csv', out), ('length_errors_fine.csv', pd.DataFrame(fine_rows)),
                       ('attack_codec_length_errors.csv', pd.DataFrame(cross_rows)),
                       ('error_change_contributions.csv', pd.DataFrame(contributions)),
                       ('duration_score_associations.csv', pd.DataFrame(associations)),
                       ('within_attack_codec_or_speaker_length.csv', pd.DataFrame(standardized))]:
        data.to_csv(root / name, index=False)
    frame.to_parquet(root / 'joined_scores_21LA.parquet', index=False)
    (root / 'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_path = Path('C:/Windows/Fonts/malgun.ttf')
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.3), sharey=True)
    labels = ['2초 이하', '2초 초과~3초', '3초 초과~4.0375초', '4.0375초 초과']
    for ax, metric, title in zip(axes, ['FRR_percent', 'FAR_percent'],
                                  ['정상 음성을 스푸프로 거부', '스푸프를 정상으로 통과']):
        for model, label, color, marker in [('coverage20', '이번 20epoch', '#b23a25', 'o'),
                                           ('B_e20', 'B 20epoch', '#176b91', 's'),
                                           ('C_e20', 'C 20epoch', '#655394', '^')]:
            values = out.loc[out.model == model, metric].to_numpy()
            ax.plot(range(4), values, marker=marker, color=color, label=label, linewidth=2)
            if model == 'coverage20':
                for x, value in enumerate(values):
                    ax.annotate(f'{value:.2f}%', (x, value), xytext=(0, 8),
                                textcoords='offset points', ha='center', fontsize=9)
        ax.set_xticks(range(4), labels, rotation=15, ha='right')
        ax.set_title(title, fontsize=12)
        ax.set_ylim(0, 24)
        ax.grid(axis='y', alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xlabel('원음 길이')
    axes[0].set_ylabel('해당 클래스·길이 구간의 오류율 (%)')
    axes[1].legend(loc='upper right', frameon=False, fontsize=9)
    fig.suptitle('21LA 길이별 오류율 · 각 모델의 전체 EER 근방 단일 진단 임계값', fontsize=13)
    fig.tight_layout()
    fig.savefig(root / 'length_error_rates.png', dpi=190, bbox_inches='tight')
    fig.savefig(root / 'length_error_rates.svg', bbox_inches='tight')
    plt.close(fig)
    print(out.loc[out.model.isin(['coverage20', 'B_e20', 'C_e20', 'legacy240_e50_nonrepeat']),
                  ['model', 'bin', 'normal_n', 'spoof_n', 'normal_rejected', 'spoof_accepted', 'FRR_percent', 'FAR_percent', 'within_bin_EER_percent']].to_string(index=False))
    print('GLOBAL', json.dumps(global_metrics))
    print('CHANGE', json.dumps(contributions))


if __name__ == '__main__':
    main()
