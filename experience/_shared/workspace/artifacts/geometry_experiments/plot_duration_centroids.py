"""Standalone scientific figure from the verified, locked paired contrasts."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/geometry_cache/local_analysis/duration_centroids_20261106_v1'
assert json.loads((OUT / 'independent_verification.json').read_text())['status'] == 'pass'
font = Path('C:/Windows/Fonts/malgun.ttf')
if font.exists():
    font_manager.fontManager.addfont(str(font))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'font.size': 11, 'axes.unicode_minus': False})
record = json.loads((OUT / 'paired_intervals.json').read_text())[0]
data = [r for r in record['paired'] if r['primary']]
labels = ['기존 EBM 대비', '전체 평균 중심 대비', '기존 점수 길이 보정 대비', '전체 평균 점수 길이 보정 대비']
x = np.array([r['improvement_pp'] for r in data])
lo, hi = np.array([r['CI98p75'] for r in data]).T
fig, ax = plt.subplots(figsize=(10.2, 4.5), dpi=150)
fig.subplots_adjust(left=.31, right=.96, top=.76, bottom=.22)
ax.axvline(0, color='#777777', linewidth=1)
ax.errorbar(x, np.arange(4), xerr=np.array([x-lo, hi-x]), fmt='o', color='#245D89',
            linewidth=1.7, capsize=4, markersize=6)
ax.set_yticks(np.arange(4), labels)
ax.invert_yaxis()
ax.set_ylim(3.6, -.6)
ax.set_xlim(min(lo)-.018, max(hi)+.018)
ax.set_xlabel('대조군 EER - 길이별 중심 EER (퍼센트포인트)', labelpad=12)
ax.spines[['top', 'right', 'left']].set_visible(False)
ax.tick_params(axis='y', length=0, pad=10)
ax.grid(axis='x', color='#E5E5E5', linewidth=.7)
ax.set_axisbelow(True)
for i, value in enumerate(x):
    ax.annotate(f'{value:+.5f}', (value, i), xytext=(0, 11), textcoords='offset points', ha='center', va='bottom')
fig.text(.04, .94, 'DF 전체: 길이별 정상 중심의 EER 변화', fontsize=16)
fig.text(.04, .865, '533,928개 · 화자 군집 재표집 10,000회 · 다중 비교 보정 98.75% 구간', fontsize=11)
fig.text(.31, .065, '← 길이별 중심이 악화', fontsize=10)
fig.text(.96, .065, '길이별 중심이 개선 →', ha='right', fontsize=10)
fig.savefig(OUT / 'paired_EER_contrasts.png', dpi=150, facecolor='white')
plt.close(fig)
print(OUT / 'paired_EER_contrasts.png')
