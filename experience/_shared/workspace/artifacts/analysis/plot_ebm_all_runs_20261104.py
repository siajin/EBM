"""Validate aggregate experiment comparisons and plot reviewed EER records."""
import json
import math
from decimal import Decimal
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

folder = Path(__file__).resolve().parent
source = folder / 'ebm-all-runs-comparison-20261104.json'
data = json.loads(source.read_text(encoding='utf-8'))
datasets = ('19LA', '21LA', '21DF')
keys = ('old80', 'local160', 'global160', 'local240')
checks = 0


def verify_delta(record):
    global checks
    target = Decimal(str(record['target']))
    reference = Decimal(str(record['reference']))
    expected_pp = target - reference
    expected_relative = (reference - target) / reference * 100
    assert abs(expected_pp - Decimal(str(record['delta_pp']))) < Decimal('1e-10')
    assert abs(expected_relative - Decimal(str(record['relative_reduction_percent']))) < Decimal('1e-10')
    checks += 1


for dataset in datasets:
    record = data['datasets'][dataset]
    for key in keys:
        rows = record['series'][key]
        assert len({row['epoch'] for row in rows}) == len(rows)
        assert all(math.isfinite(row['eer_percent']) and 0 <= row['eer_percent'] <= 100 for row in rows)
        assert record['best'][key] == min(rows, key=lambda row: row['eer_percent'])
    for comparison in record['current_best_vs_reference'].values():
        verify_delta(comparison)
    for comparison in record['common_epoch_comparisons'].values():
        assert comparison['epochs'] == [15, 20, 25, 30, 35, 40]
        assert comparison['improved'] + comparison['worsened'] == len(comparison['pairs'])
        for pair in comparison['pairs']:
            verify_delta(pair)
    for name in ('current_at40_vs_local160', 'current_at40_vs_global160',
                 'current_at45_vs_local160', 'current_at45_vs_global160', 'current_40_to_45'):
        if record[name] is not None:
            verify_delta(record[name])
    assert not record['series']['failed320']
    assert record['final']['local240'] is None

assert data['datasets']['21DF']['at_epoch45']['local240'] is None
assert data['datasets']['21DF']['final']['old80'] is None
assert data['datasets']['21DF']['final']['local160'] is None
assert data['source_validation']['current_metrics_agree_with_both_rank_console']
config = data['source_validation']['config_comparison']
assert set(config['EBM_e50_bs240_lr1e5']['differences_from_bs160']) == {'batch_size', 'name'}
assert data['experiments']['local240']['source_differences_from_local160'] == ['arguments.py']

plt.rcParams.update({
    'font.family': 'Malgun Gothic', 'font.size': 10.5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#6b7280', 'axes.linewidth': 0.7,
    'axes.unicode_minus': False,
})
colors = {'old80': '#6c743d', 'local160': '#555b66', 'global160': '#1768ac',
          'local240': '#c26a31', 'paper': '#8b8f97'}
styles = {
    'local160': dict(color=colors['local160'], linestyle='--', marker='s',
                     markerfacecolor='white', markersize=4.5, label='batch 160 · 기존 BPL'),
    'global160': dict(color=colors['global160'], linestyle='-.', marker='^',
                      markersize=5, label='batch 160 · 전역 BPL'),
    'local240': dict(color=colors['local240'], linestyle='-', marker='o',
                     markersize=5, label='batch 240 · 기존 BPL (진행 중)'),
}
fig, axes = plt.subplots(3, 2, figsize=(15.5, 11.5), gridspec_kw={'width_ratios': [1.2, 1]})
fig.subplots_adjust(left=0.07, right=0.98, top=0.84, bottom=0.13, wspace=0.87, hspace=0.61)
fig.suptitle('EBM 실험 비교: batch 240의 현재 기록', x=0.07, y=0.98,
             ha='left', fontsize=19)
fig.text(0.07, 0.942, '2026-11-04 10:32 KST  |  EER (%), 낮을수록 좋음  |  batch 240: 45 epoch 학습 완료, 21DF 평가 중', fontsize=11)
fig.text(0.07, 0.872, '50 epoch 설정의 평가 추이', fontsize=13, fontweight='bold')
fig.text(0.615, 0.872, '논문 보고값과 실험별 전체 최저 기록', fontsize=13, fontweight='bold')

bar_order = ('paper', 'old80', 'local160', 'global160', 'local240')
bar_labels = ['논문 보고값', '80 epoch · batch 80', '50 epoch · batch 160',
              '50 epoch · batch 160 전역', '이번 · batch 240']
for row_index, dataset in enumerate(datasets):
    record = data['datasets'][dataset]
    ax = axes[row_index, 0]
    for key, style in styles.items():
        rows = record['series'][key]
        ax.plot([row['epoch'] for row in rows], [row['eer_percent'] for row in rows],
                linewidth=1.7, **style)
    best = record['best']['local240']
    ax.scatter([best['epoch']], [best['eer_percent']], s=110, facecolor='none',
               edgecolor=colors['local240'], linewidth=1.2, zorder=4)
    ax.axhline(record['paper_reported'], color='#353941', linestyle=':', linewidth=1.1)
    ax.set_title(dataset, loc='left', pad=9, fontsize=12, fontweight='bold')
    ax.set_ylabel('EER (%)')
    ax.set_xlabel('Epoch')
    ax.set_xlim(13, 52)
    ax.set_xticks(range(15, 51, 5))
    ax.margins(y=0.15)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.grid(axis='y', color='#e2e5e9', linewidth=0.7)
    ax.set_axisbelow(True)

    ax = axes[row_index, 1]
    values = [record['paper_reported']] + [record['best'][key]['eer_percent'] for key in keys]
    epochs = [None] + [record['best'][key]['epoch'] for key in keys]
    bars = ax.barh(range(5), values, color=[colors[key] for key in bar_order], height=0.59)
    bars[0].set_hatch('///')
    bars[0].set_edgecolor('#555b66')
    bars[-1].set_edgecolor('#6d3d1e')
    ax.set_yticks(range(5), bar_labels)
    ax.tick_params(axis='y', length=0, pad=9, labelsize=10.2)
    ax.invert_yaxis()
    ax.set_title(dataset, loc='left', pad=9, fontsize=12, fontweight='bold')
    ax.set_xlim(0, max(values) * 1.68)
    ax.set_xlabel('EER (%)')
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.grid(axis='x', color='#e2e5e9', linewidth=0.7)
    ax.set_axisbelow(True)
    for i, (value, epoch) in enumerate(zip(values, epochs)):
        suffix = '' if epoch is None else f' (ep{epoch})'
        ax.text(value + max(values) * 0.025, i, f'{value:.4f}{suffix}', va='center', fontsize=10)

handles, labels = axes[0, 0].get_legend_handles_labels()
handles.append(plt.Line2D([], [], color='#353941', linestyle=':', label='논문 보고값'))
labels.append('논문 보고값')
fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.065, 0.921),
           ncol=4, frameon=False, fontsize=10.3, columnspacing=1.7)
fig.text(0.07, 0.080, '이번 기록: 19LA·21LA는 epoch 45까지, 21DF는 epoch 40까지. 최저값은 평가셋마다 서로 다른 checkpoint에서 나올 수 있습니다.', fontsize=10)
fig.text(0.07, 0.052, '각 패널의 축 범위는 다릅니다. 80 epoch: lr 1e-6 / 50 epoch 설정: lr 1e-5. 논문: batch 30·150 epoch로 직접 통제 비교가 아닙니다.', fontsize=10)
fig.text(0.07, 0.024, 'batch 320은 첫 배치 OOM으로 평가값이 없어 제외했습니다. 빈 구간과 누락된 최종 평가값을 추정하거나 연결하지 않았습니다.', fontsize=10)
output = folder / 'ebm-all-runs-comparison-20261104.png'
fig.savefig(output, dpi=160, facecolor='white')
plt.close(fig)

data['validation'] = {
    'decimal_comparison_records_checked': checks,
    'unique_finite_eer_points_and_minima': True,
    'missing_final_metrics_retained_as_null': True,
    'config_differences_and_current_console_alignment': True,
    'visual_inspection': 'pending',
}
source.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'figure': str(output), 'decimal_records_checked': checks}, ensure_ascii=False))
