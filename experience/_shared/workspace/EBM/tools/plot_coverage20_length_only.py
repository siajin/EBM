"""Plot only the completed coverage20 model from the verified length audit."""
import argparse
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
    data = pd.read_csv(root / 'length_errors.csv')
    data = data.loc[data.model == 'coverage20'].copy()
    assert len(data) == 4
    assert data.normal_n.sum() == 14816 and data.spoof_n.sum() == 133360
    fine = pd.read_csv(root / 'length_errors_fine.csv')
    fine = fine.loc[(fine.model == 'coverage20') & ((fine.normal_n + fine.spoof_n) > 0)]
    fine.to_csv(root / 'coverage20_only_length_errors.csv', index=False)
    font = Path('C:/Windows/Fonts/malgun.ttf')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(8, 4.2))
    values = data.within_bin_EER_percent.to_numpy()
    ax.plot(range(4), values, color='#176b91', marker='o', linewidth=2)
    for i, value in enumerate(values):
        ax.annotate(f'{value:.2f}%', (i, value), xytext=(0, 9), textcoords='offset points', ha='center')
    ax.set_xticks(range(4), ['2초 이하', '2초 초과~3초', '3초 초과~4.0375초', '4.0375초 초과'])
    ax.set_ylim(0, 17)
    ax.set_ylabel('구간별 EER (%) · 낮을수록 좋음')
    ax.set_xlabel('원음 길이 · 각 구간에서 별도로 계산한 EER')
    ax.set_title('이번 20epoch 모델만 비교: 길이별 정상·스푸프 분리력')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=.2)
    fig.tight_layout()
    fig.savefig(root / 'coverage20_only_length_eer.png', dpi=190, bbox_inches='tight')
    fig.savefig(root / 'coverage20_only_length_eer.svg', bbox_inches='tight')
    plt.close(fig)
    print('Verified class totals; current-model-only figure and table saved.')


if __name__ == '__main__':
    main()
