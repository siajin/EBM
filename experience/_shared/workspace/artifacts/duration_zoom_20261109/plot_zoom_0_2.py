from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CACHE = ROOT / 'artifacts/geometry_cache/runs'
sources = [
    '20261105_fast_p0_01/686e593443178bb8',
    '20261105_frozen_all_01/Q/c288fb7a7ab48dd3',
    '20261105_frozen_all_01/19LA_eval/4b8955da49f12316',
    '20261105_21la_full_03/4110118cb92e35c1',
    '20261105_df_fast_01/1b0a7488e72f4ec2',
    '20261105_frozen_all_01/DF_remaining/096bc3b970f33181',
]
frames, provenance = [], []
validation = json.loads((ROOT / 'artifacts/class_distributions_20261108/validation.json').read_text())
known = {str(Path(x['path'])): x['sha256'] for x in validation['source_files']}
for source in sources:
    p = CACHE / source / 'extract/scores.parquet'
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha == known[str(p.relative_to(ROOT))], p
    frames.append(pd.read_parquet(p, columns=['split', 'utterance_id', 'y', 'duration']))
    provenance.append({'path': str(p), 'sha256': sha})

first = frames[0]
datasets = [
    ('19LA_train', '2019 LA · train', first[first.split == 'T']),
    ('19LA_dev', '2019 LA · dev', pd.concat([first[first.split == 'V'], frames[1]])),
    ('19LA_eval', '2019 LA · eval', frames[2]),
    ('21LA_eval', '2021 LA · eval', frames[3]),
    ('21DF_eval', '2021 DF · eval', pd.concat(frames[4:])),
]
expected = [(2580, 22800), (2548, 22296), (7355, 63882), (14816, 133360), (14869, 519059)]
plt.rcParams.update({'font.family': 'Malgun Gothic', 'font.size': 12,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'svg.fonttype': 'none', 'axes.unicode_minus': False})
fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
colors = ['#2376AB', '#D97813']
edges = np.round(np.linspace(0, 2, 41), 8)
rows, summaries, ymax = [], [], 0
for i, ((key, title, data), counts_expected) in enumerate(zip(datasets, expected)):
    ax = axes.flat[i]
    assert data.utterance_id.is_unique
    assert data.duration.notna().all() and (data.duration > 0).all()
    annotation = []
    for j, (label, name) in enumerate([(1, 'Bona fide'), (0, 'Spoof')]):
        values = data.loc[data.y == label, 'duration'].to_numpy()
        assert len(values) == counts_expected[j]
        short = values[values < 2]
        counts, _ = np.histogram(short, bins=edges)
        assert counts.sum() == len(short)
        pct = counts / len(values) * 100
        ymax = max(ymax, pct.max())
        ax.stairs(pct, edges, color=colors[j], linewidth=2.2,
                  linestyle='-' if label == 1 else '--', label=name)
        share = len(short) / len(values) * 100
        annotation.append(f'{name}: {share:.2f}% ({len(short):,}/{len(values):,})')
        summaries.append({'dataset': key, 'class': name, 'N': len(values),
                          'under2_count': len(short), 'under2_pct': share,
                          'exactly_2s_count': int((values == 2).sum())})
        for lo, hi, count, value in zip(edges[:-1], edges[1:], counts, pct):
            rows.append({'dataset': key, 'class': name, 'bin_left_s': lo,
                         'bin_right_s': hi, 'count': int(count),
                         'denominator': len(values), 'percent': value})
    ax.set_title(title, loc='left', fontweight='bold', pad=14)
    ax.text(.02, .96, '2초 미만 비율\n' + '\n'.join(annotation),
            transform=ax.transAxes, va='top', fontsize=10.5,
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .85})
    ax.set_xlim(0, 2)
    ax.set_xticks(np.arange(0, 2.01, .25))
    ax.tick_params(labelbottom=True)
    ax.set_xlabel('원본 음성 길이 (초)')
    ax.set_ylabel('클래스 전체 발화 중 비율 / bin (%)')
    ax.grid(axis='y', color='#E3E5E7', linewidth=.7)
    ax.set_axisbelow(True)
for ax in axes.flat[:5]:
    ax.set_ylim(0, np.ceil(ymax * 1.15 * 2) / 2)
axes.flat[5].set_visible(False)
fig.suptitle('원본 음성 길이 분포 · 0–2초 확대', x=.065, y=.975,
             ha='left', fontsize=23)
fig.legend(handles=[Line2D([0], [0], color=colors[0], lw=2.2, label='Bona fide'),
                    Line2D([0], [0], color=colors[1], lw=2.2, ls='--', label='Spoof')],
           loc='upper center', bbox_to_anchor=(.5, .92), ncol=2, frameon=False)
fig.text(.065, .025, '0.05초 공통 bin · 분모는 각 데이터셋·클래스 전체 발화 수 · 모든 패널 동일 축 · 표시 구간 0 ≤ 길이 < 2초',
         fontsize=11, color='#555555')
fig.subplots_adjust(left=.065, right=.985, bottom=.12, top=.82, wspace=.27, hspace=.52)
fig.savefig(OUT / 'duration_histogram_0_2s.png', dpi=180, facecolor='white')
fig.savefig(OUT / 'duration_histogram_0_2s.svg', facecolor='white')
pd.DataFrame(rows).to_csv(OUT / 'histogram_0_2s.csv', index=False)
pd.DataFrame(summaries).to_csv(OUT / 'summary_0_2s.csv', index=False)
(OUT / 'validation.json').write_text(json.dumps({'sources': provenance,
    'checks': ['Source SHA256 matches prior validated cache', 'Unique IDs in each dataset',
               'Expected class counts', 'Positive finite durations',
               'Histogram bin totals equal exact under-2s counts'],
    'scope': 'Original full-utterance durations; official eval only for 2021; no audio processing',
    'bins': '[0,2) in 0.05-second bins; class-wide denominator'}, indent=2), encoding='utf-8')
print(pd.DataFrame(summaries).to_string(index=False))
print(OUT / 'duration_histogram_0_2s.png')
