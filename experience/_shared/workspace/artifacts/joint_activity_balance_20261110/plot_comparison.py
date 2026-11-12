"""Plot the inspectable full-candidate expected distributions."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd

root = Path(__file__).resolve().parent
font_manager.fontManager.addfont('C:/Windows/Fonts/malgun.ttf')
plt.rcParams.update({'font.family': 'Malgun Gothic', 'font.size': 10, 'axes.unicode_minus': False})
data = pd.read_csv(root / 'validation/expected_by_length.csv')
diff = pd.read_csv(root / 'validation/distribution_differences.csv')
fig, axes = plt.subplots(1, 2, figsize=(11, 3.7), layout='constrained')
for label, name, color, marker in [(1, '보나', '#1766a1', 'o'), (0, '스푸프', '#bc5726', 's')]:
    for policy, style, suffix in [('before', '--', '전'), ('after', '-', '후')]:
        frame = data[(data.label == label) & (data.policy == policy)]
        axes[0].plot(frame.seconds, frame.mean_speech_fraction * 100, style, color=color,
                     marker=marker, markersize=4, label=f'{name} {suffix}', alpha=.85)
axes[0].set(title='길이별 평균 음성활동 비율', ylabel='VAD 음성 프레임 / 전체 프레임 (%)',
            xlabel='조각 길이 (초)', ylim=(0, 100))
axes[0].legend(ncol=2, frameon=False, fontsize=9)
for policy, color, marker, name in [('before', '#777777', 'o', '전'), ('after', '#1766a1', 's', '후')]:
    frame = diff[diff.policy == policy]
    axes[1].plot(frame.seconds, frame.binned_TV * 100, color=color, marker=marker,
                 label=name, markersize=4)
axes[1].set(title='음성활동 구간 분포 차이 · 작을수록 유사',
            ylabel='TV 거리 × 100 (0 = 같은 구간 확률)', xlabel='조각 길이 (초)')
axes[1].legend(frameon=False)
for ax in axes:
    ax.set_xticks([.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#dddddd', linewidth=.6)
fig.suptitle('Train 기대분포 · 증강 전 WebRTC VAD2 · 무음 별도 + 양수 20% 구간', fontsize=11)
fig.savefig(root / 'activity_before_after.png', dpi=170)
