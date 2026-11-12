"""Cross-check diagnosis calculations and visualize aggregate training dynamics."""
import json
from decimal import Decimal
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

folder = Path(__file__).resolve().parent
source = folder / 'ebm-performance-diagnosis-20261104.json'
data = json.loads(source.read_text(encoding='utf-8'))

# A loss difference is absolute loss, not percentage points.
for record in data['training_proxy'].values():
    change = record['epoch25_to45']
    if 'delta_pp' in change:
        change['delta_absolute'] = change.pop('delta_pp')
finding_loss = data['findings'][0]['evidence']['current25to45_loss']
if 'delta_pp' in finding_loss:
    finding_loss['delta_absolute'] = finding_loss.pop('delta_pp')

checked = 0


def verify(node):
    global checked
    if isinstance(node, dict):
        if {'target', 'reference', 'relative_change_percent'} <= node.keys():
            target = Decimal(str(node['target']))
            reference = Decimal(str(node['reference']))
            expected = (target - reference) / reference * 100
            assert abs(expected - Decimal(str(node['relative_change_percent']))) < Decimal('1e-9')
            delta = node.get('delta_absolute', node.get('delta_pp'))
            assert abs((target - reference) - Decimal(str(delta))) < Decimal('1e-9')
            checked += 1
        for value in node.values():
            verify(value)
    elif isinstance(node, list):
        for value in node:
            verify(value)


verify(data)
for key, record in data['experiments'].items():
    assert record['iterations_per_epoch'] == (1290 + record['bona_per_gpu'] - 1) // record['bona_per_gpu']
    assert record['lr_log_unique'] == [record['settings']['lr']]
    assert record['settings']['lr'] == record['settings']['lr_min']
    assert record['settings']['alpha_cl'] == 30
    assert record['optimizer_steps_per_iteration'] == 2
for dataset, record in data['datasets'].items():
    for key, rows in record['series'].items():
        assert len(rows) == len({row['epoch'] for row in rows})
        assert record['best'][key] == min(rows, key=lambda row: row['eer_percent'])
    assert record['series']['local240'][-1]['epoch'] == 45

plt.rcParams.update({
    'font.family': 'Malgun Gothic', 'font.size': 10.5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#6b7280', 'axes.linewidth': 0.7,
    'axes.unicode_minus': False,
})
styles = {
    'local160': dict(color='#555b66', linestyle='--', marker='s',
                     markerfacecolor='white', markersize=4.7, label='batch 160 · 기존 BPL'),
    'global160': dict(color='#1768ac', linestyle='-.', marker='^',
                      markersize=5, label='batch 160 · 전역 BPL'),
    'local240': dict(color='#c26a31', linestyle='-', marker='o',
                     markersize=5, label='batch 240 · 기존 BPL (진행 중)'),
}
fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.7))
fig.subplots_adjust(left=0.09, right=0.97, top=0.79, bottom=0.18, hspace=0.56, wspace=0.27)
fig.suptitle('학습 loss 감소와 평가 EER 악화가 함께 관측됩니다',
             x=0.09, y=0.965, ha='left', fontsize=17)
fig.text(0.09, 0.921, '50 epoch 설정 3개 비교  |  학습률 모두 1e-5 고정  |  2026-11-04 12:30 KST 기록', fontsize=10.7)

ax = axes[0, 0]
for key, style in styles.items():
    rows = [row for row in data['training_proxy'][key]['records'] if 15 <= row['epoch'] <= 50]
    ax.plot([row['epoch'] for row in rows], [row['mean'] for row in rows], linewidth=1.7, **style)
ax.set_title('기록된 학습 loss · 구간 평균', loc='left', pad=10, fontsize=11.5)
ax.set_ylabel('Loss (loss1 + loss2)')

for ax, dataset in zip((axes[0, 1], axes[1, 0], axes[1, 1]), ('19LA', '21LA', '21DF')):
    for key, style in styles.items():
        rows = data['datasets'][dataset]['series'][key]
        ax.plot([row['epoch'] for row in rows], [row['eer_percent'] for row in rows],
                linewidth=1.7, **style)
    ax.set_title(dataset + ' 평가', loc='left', pad=10, fontsize=11.5)
    ax.set_ylabel('EER (%) · 낮을수록 좋음')
    current = data['datasets'][dataset]['series']['local240']
    ax.annotate(f"{current[-1]['eer_percent']:.4f}",
                xy=(current[-1]['epoch'], current[-1]['eer_percent']),
                xytext=(7, 5), textcoords='offset points', color='#8a461a', fontsize=9.5)

for ax in axes.flat:
    ax.set_xlim(13, 52)
    ax.set_xticks(range(15, 51, 5))
    ax.set_xlabel('Epoch')
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.margins(y=0.17)
    ax.grid(axis='y', color='#e2e5e9', linewidth=0.7)
    ax.set_axisbelow(True)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.08, 0.881),
           ncol=3, frameon=False, fontsize=10, columnspacing=1.7)
fig.text(0.09, 0.110, 'batch 240은 45 epoch 평가까지 반영했습니다. 최종 50 epoch 성능은 아직 없습니다. 각 패널의 세로축 범위는 다릅니다.', fontsize=9.5)
fig.text(0.09, 0.076, '학습 loss는 epoch당 9개 로그구간 평균입니다. 끝부분 일부 배치는 미포함되며, 두 단계 학습 목적함수를 합친 진단용 값입니다.', fontsize=9.5)
fig.text(0.09, 0.042, '이 패턴은 과적합·고정 학습률에 따른 변동 가설과 양립하지만, 단일 seed 기록만으로 두 원인을 분리하거나 확정할 수 없습니다.', fontsize=9.5)
output = folder / 'ebm-performance-diagnosis-20261104.png'
fig.savefig(output, dpi=160, facecolor='white')
plt.close(fig)
data['validation'] = {
    'decimal_effect_records_checked': checked,
    'unique_eer_epochs_and_minima': True,
    'constant_lr_and_alpha_verified_across_runs': True,
    'training_loss_windows_not_full_epoch_means': True,
    'visual_inspection': 'pending',
}
source.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'figure': str(output), 'decimal_comparisons': checked}, ensure_ascii=False))
