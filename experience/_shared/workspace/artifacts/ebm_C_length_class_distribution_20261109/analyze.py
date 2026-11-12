"""Compare B/C Long input lengths from existing metadata; no audio or inference."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

PROJECT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / 'artifacts/geometry_experiments'))
from geometry_cache import resolve_run

folder, entry, split = resolve_run('T')
source = folder / 'extract/scores.parquet'
source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
assert source_sha == next(x['sha256'] for x in entry['files'] if x['path'] == 'extract/scores.parquet')
f = pd.read_parquet(source, columns=['split', 'utterance_id', 'y', 'duration'])
f = f.loc[f.split == split].copy()
assert len(f) == 25380 and f.utterance_id.is_unique
assert not f.isna().any().any() and f.duration.gt(0).all()
labels = f.y.to_numpy()
samples = np.rint(f.duration.to_numpy()*16000).astype('int64')
audit = json.loads((PROJECT / 'artifacts/ebm_C_20261108/metadata_validation.json').read_text())
for y in [1, 0]:
    record = audit['class_length_counts'][str(y)]
    assert (labels == y).sum() == record['N']
    assert samples[labels == y].sum() == record['total_original_samples']
    assert np.minimum(samples[labels == y], 64600).sum() == record['total_B_used_samples']
    assert (samples[labels == y] > 64600).sum() == record['longer_than_B_cap']

result = {
    'scope': 'All unique training utterances, class-normalized; not sampler-exposure weighted. Long input lengths before RawBoost. B/C Short policy unchanged.',
    'source': str(source.relative_to(PROJECT)), 'source_sha256': source_sha,
    'cap_samples': 64600, 'sample_rate': 16000, 'conditions': {},
}
histograms = []
cap = 64600/16000
bin_labels = ['<2', '2–3.0275', '3.0275–4.0375', '>4.0375–5', '>5–6', '>6']
for condition, values in [('B_cap', np.minimum(samples, 64600)/16000), ('C_full', samples/16000)]:
    r = {'classes': {}}
    for y, name in [(1, 'bonafide'), (0, 'spoof')]:
        v = values[labels == y]
        masks = [v < 2, (v >= 2) & (v < 3.0275), (v >= 3.0275) & (v <= cap),
                 (v > cap) & (v <= 5), (v > 5) & (v <= 6), v > 6]
        counts = [int(m.sum()) for m in masks]
        assert sum(counts) == len(v)
        r['classes'][name] = {
            'N': len(v), 'mean_seconds': float(v.mean()), 'sd_seconds': float(v.std()),
            'p50_seconds': float(np.quantile(v, .5)), 'p95_seconds': float(np.quantile(v, .95)),
            'under2_percent': float(100*np.mean(v < 2)), 'over_cap_percent': float(100*np.mean(v > cap)),
            'over5_percent': float(100*np.mean(v > 5)), 'over6_percent': float(100*np.mean(v > 6)),
            'histogram_counts': counts,
        }
        histograms.extend({'condition': condition, 'bin': b, 'class': name, 'count': n,
                           'denominator': len(v), 'fraction': n/len(v)} for b, n in zip(bin_labels, counts))
    b, s = values[labels == 1], values[labels == 0]
    support = np.unique(values)
    difference = np.searchsorted(np.sort(b), support, side='right')/len(b) - np.searchsorted(np.sort(s), support, side='right')/len(s)
    independent_ks = float(np.abs(difference).max())
    independent_w1 = float(np.sum(np.abs(difference[:-1])*np.diff(support)))
    assert np.isclose(independent_ks, ks_2samp(b, s).statistic, atol=1e-12)
    assert np.isclose(independent_w1, wasserstein_distance(b, s), atol=1e-12)
    r['KS_max_CDF_gap'] = independent_ks
    r['Wasserstein1_seconds'] = independent_w1
    r['spoof_minus_bonafide_mean_seconds'] = float(s.mean()-b.mean())
    result['conditions'][condition] = r

(OUT/'analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
pd.DataFrame(histograms).to_csv(OUT/'class_length_histograms.csv', index=False, encoding='utf-8-sig')
charts = []
for condition, title in [('B_cap', 'B · Long 최대 4.0375초'), ('C_full', 'C · Long 원길이 전체')]:
    rows = []
    for i, label in enumerate(bin_labels):
        rows.append([label] + [result['conditions'][condition]['classes'][c]['histogram_counts'][i]/result['conditions'][condition]['classes'][c]['N'] for c in ['bonafide', 'spoof']])
    charts.append({
        'id': 'training-length-'+condition.lower(), 'title': title,
        'description': '학습 정상 2,580개·spoof 22,800개. 클래스 내부 비율. 3.0275–4.0375 구간은 상한 포함. B는 긴 음성이 이 구간에 합산됨.',
        'height': 300,
        'spec': {'type': 'bar', 'x': '길이 구간', 'y': ['정상', 'Spoof'], 'xLabel': 'Long 입력 길이 (초)',
                 'yLabel': '클래스 내부 비율', 'yDomain': [0, .65], 'sortOrder': 'original',
                 'valueFormat': {'style': 'percent', 'maximumFractionDigits': 2}},
        'columns': ['길이 구간', '정상', 'Spoof'], 'rows': rows,
    })
(OUT/'chart-config.json').write_text(json.dumps({'charts': charts}, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(json.dumps(result, ensure_ascii=False, indent=2))
