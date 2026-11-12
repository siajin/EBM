"""Metadata-only speed-factor audit; no audio, model inference, SSH, or training.

rate = original_duration / target_duration, matching the convention that rate>1
plays faster and rate<1 plays slower. This is the required transform factor,
not an estimate of syllables/sec or intrinsic speaking speed.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
sys.path.insert(0, str(PROJECT / 'artifacts/geometry_experiments'))
from geometry_cache import resolve_run

TARGET_SAMPLES = 64600
SR = 16000
TARGET_SECONDS = TARGET_SAMPLES / SR
EXPECTED = {'train': (25380, 2580), 'dev': (24844, 2548),
            '19LA': (71237, 7355), '21LA': (148176, 14816), '21DF': (533928, 14869)}
SOURCES = {}
TABLE = []


def load(alias):
    folder, entry, split = resolve_run(alias)
    path = folder / 'extract/scores.parquet'
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    record = next(x for x in entry['files'] if x['path'] == 'extract/scores.parquet')
    assert sha == record['sha256'], path
    SOURCES[alias] = {'path': str(path.relative_to(PROJECT)), 'sha256': sha, 'split': split}
    f = pd.read_parquet(path, columns=['split', 'utterance_id', 'y', 'duration'])
    f = f.loc[f.split == split].copy()
    assert f.utterance_id.is_unique and not f.isna().any().any()
    assert set(f.y.unique()) == {0, 1} and f.duration.gt(0).all()
    f['samples'] = np.rint(f.duration * SR).astype('int64')
    assert np.max(np.abs(f.samples / SR - f.duration)) < 1e-12
    return f


def describe(rate):
    q = np.quantile(rate, [0, .01, .05, .25, .5, .75, .95, .99, 1])
    out = dict(N=len(rate), mean=float(rate.mean()), sd=float(rate.std()),
               quantiles={str(p): float(v) for p, v in zip([0, .01, .05, .25, .5, .75, .95, .99, 1], q)})
    predicates = {
        'slow_rate_lt_1': rate < 1, 'fast_rate_gt_1': rate > 1, 'unchanged_rate_eq_1': rate == 1,
        'rate_lt_0.5': rate < .5, 'rate_lt_0.67': rate < .67,
        'rate_lt_two_thirds': rate < 2/3,
        'rate_in_0.8_to_1.25_inclusive': (rate >= .8) & (rate <= 1.25),
        'rate_gt_1.5': rate > 1.5, 'rate_gt_2': rate > 2, 'rate_gt_3': rate > 3,
    }
    out['thresholds'] = {name: {'count': int(mask.sum()), 'percent': float(100*mask.mean())}
                         for name, mask in predicates.items()}
    assert sum(out['thresholds'][key]['count'] for key in ['slow_rate_lt_1', 'fast_rate_gt_1', 'unchanged_rate_eq_1']) == len(rate)
    return out


def main():
    aliases = {a: load(a) for a in ['T', 'V', 'Q', '19LA_eval', '21LA_eval', 'DF_pilot', 'DF_remaining']}
    frames = {'train': aliases['T'], 'dev': pd.concat([aliases['V'], aliases['Q']], ignore_index=True),
              '19LA': aliases['19LA_eval'], '21LA': aliases['21LA_eval'],
              '21DF': pd.concat([aliases['DF_pilot'], aliases['DF_remaining']], ignore_index=True)}
    output = {
        'scope': 'All unique utterances per official split, class-normalized. Not sampler-exposure weighted.',
        'target_samples': TARGET_SAMPLES, 'sample_rate_hz': SR, 'target_seconds': TARGET_SECONDS,
        'target_4.04_difference_seconds': 4.04 - TARGET_SECONDS,
        'rate_definition': 'original_samples/64600; >1 faster, <1 slower; output length target 64600',
        'meaning_limit': 'Required duration-normalization factor, not measured linguistic speech rate. Includes silence and amount of spoken content.',
        'sources': SOURCES, 'datasets': {}, 'train_to_target_rate_distribution_shift': {},
    }
    for name, f in frames.items():
        assert f.utterance_id.is_unique and len(f) == EXPECTED[name][0] and f.y.sum() == EXPECTED[name][1]
        result = {'classes': {}}
        for y, class_name in [(1, 'bonafide'), (0, 'spoof')]:
            rates = f.loc[f.y == y, 'samples'].to_numpy()/TARGET_SAMPLES
            result['classes'][class_name] = r = describe(rates)
            row = {'dataset': name, 'class': class_name, 'N': len(rates), 'rate_mean': r['mean'], 'rate_sd': r['sd']}
            row.update({'rate_q'+p: value for p, value in r['quantiles'].items()})
            row.update({key+'_count': v['count'] for key, v in r['thresholds'].items()})
            row.update({key+'_percent': v['percent'] for key, v in r['thresholds'].items()})
            TABLE.append(row)
        bona = f.loc[f.y == 1, 'samples'].to_numpy()/TARGET_SAMPLES
        spoof = f.loc[f.y == 0, 'samples'].to_numpy()/TARGET_SAMPLES
        result['class_rate_KS'] = float(ks_2samp(bona, spoof).statistic)
        result['class_rate_Wasserstein1'] = float(wasserstein_distance(bona, spoof))
        output['datasets'][name] = result
        if name != 'train':
            output['train_to_target_rate_distribution_shift'][name] = {}
            for y, class_name in [(1, 'bonafide'), (0, 'spoof')]:
                a = frames['train'].loc[frames['train'].y == y, 'samples'].to_numpy()/TARGET_SAMPLES
                b = f.loc[f.y == y, 'samples'].to_numpy()/TARGET_SAMPLES
                output['train_to_target_rate_distribution_shift'][name][class_name] = {
                    'KS': float(ks_2samp(a, b).statistic), 'Wasserstein1': float(wasserstein_distance(a, b))}
    # Existing independent C input audit owns original train sample totals.
    audit = json.loads((PROJECT/'artifacts/ebm_C_20261108/metadata_validation.json').read_text())
    for y in [1, 0]:
        assert int(frames['train'].loc[frames['train'].y == y, 'samples'].sum()) == audit['class_length_counts'][str(y)]['total_original_samples']
    output['validation'] = {'all_expected_counts_and_unique_utterance_ids': True,
                            'cache_hashes_match_catalog': True,
                            'duration_roundtrip_to_integer_samples': True,
                            'train_samples_match_independent_C_input_audit': True,
                            'full_dev_V_plus_Q_and_full_DF_pilot_plus_remaining': True}
    (HERE/'rate_distribution.json').write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    pd.DataFrame(TABLE).to_csv(HERE/'rate_distribution.csv', index=False, encoding='utf-8-sig')
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
