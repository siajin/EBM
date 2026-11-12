"""Scientific plots of saved legacy240/B/C 21LA duration error profiles."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    root = parser.parse_args().root
    configs = [('legacy240_e50_nonrepeat', '기존240 · 50epoch', '#b23a25', 'o'),
               ('B_e45', 'B · 45epoch', '#176b91', 's'),
               ('C_e30', 'C · 30epoch', '#655394', '^')]
    data = pd.read_csv(root / 'length_errors.csv')
    fine = pd.read_csv(root / 'length_errors_fine.csv')
    models = [x[0] for x in configs]
    fine = fine.loc[fine.model.isin(models) & ((fine.normal_n + fine.spoof_n) > 0)]
    fine.to_csv(root / 'reference_only_length_errors.csv', index=False)
    summary = json.loads((root / 'analysis.json').read_text())
    record = {'status': 'complete', 'spending_ceiling_USD': 0, 'actual_cost_USD': 0,
              'selection': 'B45/C30 are respective observed21LA minimum checkpoints. Legacy24050 is available common-input reference, not claimed minimum over all nonrepeat checkpoints.',
              'evaluation': 'All scores front64600cap/no repetition/masks. Legacy historical repeating evaluation is NOT this table.',
              'threshold': summary['threshold_policy'], 'bin_policy': summary['bin_policy'],
              'global': {m: summary['global'][m] for m in models},
              'checks_reused': summary['checks'],
              'source_files': ['length_errors.csv', 'length_errors_fine.csv', 'analysis.json'],
              'cleanup': 'Local savedscore aggregation and plotting only; no remote resource or model execution.'}
    (root / 'reference_length_profiles.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    font = Path('C:/Windows/Fonts/malgun.ttf')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4), sharey=True)
    for ax, metric, title in zip(axes, ['FRR_percent', 'FAR_percent'],
                                  ['정상 음성을 스푸프로 거부', '스푸프를 정상으로 통과']):
        for model, label, color, marker in configs:
            values = data.loc[data.model == model, metric].to_numpy()
            assert len(values) == 4
            ax.plot(range(4), values, color=color, marker=marker, linewidth=2, label=label)
        ax.set_xticks(range(4), ['2초 이하', '2초 초과~3초', '3초 초과~4.0375초', '4.0375초 초과'],
                      rotation=15, ha='right')
        ax.set_ylim(0, 19)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('원음 길이')
        ax.grid(axis='y', alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('해당 클래스·길이 구간의 오류율 (%)')
    axes[1].legend(frameon=False, fontsize=9)
    fig.suptitle('21LA 기존240·B·C · 비반복 공통 입력, 모델별 전체 EER 근방 단일 진단 임계값', fontsize=12)
    fig.tight_layout()
    fig.savefig(root / 'reference_length_error_rates.png', dpi=190, bbox_inches='tight')
    fig.savefig(root / 'reference_length_error_rates.svg', bbox_inches='tight')
    plt.close(fig)
    print('Saved3 reference profiles and scientific figure; prior figures preserved.')


if __name__ == '__main__':
    main()
