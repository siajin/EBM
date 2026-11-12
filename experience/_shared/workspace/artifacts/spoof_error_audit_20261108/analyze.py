"""Offline audit of the original bs160 EBM; no server or model execution."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'artifacts/geometry_experiments'))
from geometry_cache import resolve_run

OUT = Path(__file__).resolve().parent
CHECKPOINT = '104873703a57cad98b686d770e0860e031a2cf9abef79b72668780c1660e0b24'


def load(name):
    folder, entry, split = resolve_run(name)
    assert entry['identity']['checkpoint_sha256'] == CHECKPOINT
    path = folder / 'extract/scores.parquet'
    expected = next(x['sha256'] for x in entry['files'] if x['path'] == 'extract/scores.parquet')
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == expected
    f = pd.read_parquet(path)
    return f[f.split.eq(split)].copy(), dict(path=str(path.relative_to(ROOT)), sha256=actual, split=split)


def group_summary(f, keys):
    result = f.groupby(keys, observed=True, dropna=False).wrong.agg(errors='sum', n='count').reset_index()
    result['error_rate_percent'] = 100 * result.errors / result.n
    result['share_of_class_errors_percent'] = 100 * result.errors / f.wrong.sum()
    result['share_of_class_population_percent'] = 100 * result.n / len(f)
    assert int(result.errors.sum()) == int(f.wrong.sum())
    assert int(result.n.sum()) == len(f)
    return result.sort_values(['errors', 'n'], ascending=False)


def main():
    sources = []
    frames = []
    for name in ['DF_pilot', 'DF_remaining']:
        f, provenance = load(name)
        sources.append(provenance)
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    assert not df.utterance_id.duplicated().any()
    assert df.y.value_counts().to_dict() == {0: 519059, 1: 14869}
    assert np.isfinite(df[['cosine_s', 'duration', 'padding_fraction']].to_numpy()).all()
    assert np.array_equal(df.padding_fraction.gt(0), df.duration.lt(64600 / 16000))
    fpr, tpr, thresholds = roc_curve(df.y, df.cosine_s, drop_intermediate=False)
    k = int(np.argmin(abs(fpr - (1 - tpr))))
    threshold = float(thresholds[k])
    assert threshold == -0.9145589682336335
    df['wrong'] = np.where(df.y.eq(0), df.cosine_s.ge(threshold), df.cosine_s.lt(threshold))
    df['repeat'] = df.padding_fraction.gt(0)
    df['duration_bin'] = pd.cut(df.duration, [0, 1, 1.5, 2, 3.0275, 4.0375, 6, np.inf], right=False)
    result = {'checkpoint_sha256': CHECKPOINT, 'model': 'original EBM bs160 epoch50',
              'population': 'ASVspoof2021 DF official eval; 533928 trials',
              'threshold': threshold, 'threshold_policy': 'target EER-nearest ROC vertex, diagnostic only',
              'accept_rule': 'cosine_s >= threshold', 'sources': sources, 'classes': {}}
    repeat_tables = []
    for y, label in [(1, 'bonafide'), (0, 'spoof')]:
        f = df[df.y.eq(y)]
        result['classes'][label] = {'n': len(f), 'errors': int(f.wrong.sum()),
                                    'error_rate_percent': float(100 * f.wrong.mean())}
        for column in ['repeat', 'duration_bin', 'source_corpus', 'attack_id', 'codec', 'vocoder']:
            table = group_summary(f, [column])
            table.to_csv(OUT / f'{label}_{column}.csv', index=False)
            if column == 'repeat':
                table['class'] = label
                repeat_tables.append(table)
            if y == 0 and column in ['attack_id', 'source_corpus', 'vocoder']:
                result[f'spoof_{column}'] = table.to_dict('records')
        # Independent confusion counts against numpy, including ROC polarity.
        scores = f.cosine_s.to_numpy()
        count = np.count_nonzero(scores < threshold) if y else np.count_nonzero(scores >= threshold)
        assert count == f.wrong.sum()
        assert abs(count / len(f) - ((1 - tpr[k]) if y else fpr[k])) < 1e-14
    assert result['classes']['bonafide']['errors'] == 482
    assert result['classes']['spoof']['errors'] == 16827
    pd.concat(repeat_tables, ignore_index=True).to_csv(OUT / 'repeat_by_class.csv', index=False)
    # Sensitivity: one dev-selected threshold, held fixed on DF.
    dev, provenance = load('V')
    fa, tp, th = roc_curve(dev.y, dev.cosine_s, drop_intermediate=False)
    j = int(np.argmin(abs(fa - (1 - tp))))
    dev_threshold = float(th[j])
    result['dev_threshold_sensitivity'] = {'threshold': dev_threshold, 'source': provenance,
        'spoof_accept_percent': float(100 * df.loc[df.y.eq(0), 'cosine_s'].ge(dev_threshold).mean()),
        'bonafide_reject_percent': float(100 * df.loc[df.y.eq(1), 'cosine_s'].lt(dev_threshold).mean())}
    result['verification'] = {'score_file_hashes': 'pass', 'unique_UID': 'pass',
        'class_coverage': 'pass', 'ROC_vs_direct_counts': 'pass', 'group_reconciliation': 'pass',
        'limitations': ['One fixed model; descriptive concentration is not causal attribution.',
                       'Padding groups differ in source, attack and duration.',
                       'Target EER threshold is diagnostic, not a deployment threshold.']}
    (OUT / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps({k: v for k, v in result.items() if not k.startswith('spoof_')}, ensure_ascii=False, indent=2))
    print(pd.concat(repeat_tables).to_string(index=False))


if __name__ == '__main__':
    main()
