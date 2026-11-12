"""Plot published EBM results and reviewed per-run EER minima."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

folder = Path(__file__).resolve().parent
data = json.loads((folder / 'ebm-three-runs-paper-comparison-20261102.json').read_text(encoding='utf-8'))
plt.rcParams.update({
    'font.family': 'Malgun Gothic', 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.spines.left': False, 'axes.edgecolor': '#6b7280',
    'axes.linewidth': 0.7, 'axes.unicode_minus': False,
})
fig, axes = plt.subplots(3, 1, figsize=(10, 9))
fig.subplots_adjust(left=0.26, right=0.95, top=0.84, bottom=0.17, hspace=0.7)
fig.suptitle('논문 보고값과 실험별 최저 EER 비교', x=0.055, y=0.975,
             ha='left', fontsize=17)
fig.text(0.055, 0.933, '논문: Interspeech 2025 EBM  |  실험: 평가셋별 최저 기록  |  낮을수록 좋음', fontsize=10.5)
labels = ['논문 보고값 (2025)', '80 epoch · 기존 BPL',
          '50 epoch · 기존 BPL', '50 epoch · 전역 BPL']
colors = ['#b8872c', '#c26a31', '#555b66', '#1768ac']
for ax, dataset in zip(axes, ('19LA', '21LA', '21DF')):
    record = data['datasets'][dataset]
    keys = ('old80', 'local50', 'global50')
    values = [record['paper_reported']] + [record['minima'][k]['eer'] for k in keys]
    epochs = [None] + [record['minima'][k]['epoch'] for k in keys]
    bars = ax.barh(range(4), values, height=0.58, color=colors)
    bars[0].set_hatch('///')
    bars[0].set_edgecolor('#735518')
    ax.set_yticks(range(4), labels)
    ax.tick_params(axis='y', length=0, pad=12)
    ax.invert_yaxis()
    ax.set_title(dataset, loc='left', pad=11, fontsize=12.5, fontweight='bold')
    ax.set_xlim(0, max(values) * 1.52)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_xlabel('EER (%)', labelpad=4)
    ax.grid(axis='x', color='#e2e5e9', linewidth=0.7)
    ax.set_axisbelow(True)
    for index, (value, epoch) in enumerate(zip(values, epochs)):
        suffix = '' if epoch is None else f' (epoch {epoch})'
        ax.text(value + max(values) * 0.025, index, f'{value:.4f}{suffix}',
                va='center', fontsize=10.5)
fig.text(0.055, 0.095, '각 평가셋의 가로축 범위는 다릅니다. 최저값은 서로 다른 체크포인트에서 나올 수 있습니다.', fontsize=9.5)
fig.text(0.055, 0.063, '80 epoch 실험: batch 80 · lr 1e-6 / 50 epoch 실험: batch 160 · lr 1e-5', fontsize=9.5)
fig.text(0.055, 0.031, '논문: batch 30 · 150 epoch. 논문 보고값과 테스트셋 내 최저 기록은 동일 조건 비교가 아닙니다.', fontsize=9.5)
path = folder / 'ebm-three-runs-paper-comparison-20261102.png'
fig.savefig(path, dpi=160, facecolor='white')
plt.close(fig)
print(path)
