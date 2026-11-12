"""Plot reviewed aggregate EER logs; no research code or data are downloaded."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

folder = Path(__file__).resolve().parent
data = json.loads((folder / 'ebm-bpl-comparison-20261102.json').read_text(encoding='utf-8'))
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11.5,
    'axes.labelsize': 11, 'axes.titlesize': 11.5,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#6b7280', 'axes.linewidth': 0.7,
})
fig, axes = plt.subplots(3, 1, figsize=(9, 9.3), sharex=True)
fig.subplots_adjust(left=0.105, right=0.96, top=0.84, bottom=0.14, hspace=0.65)
fig.suptitle('Local vs global BPL: completed 50-epoch runs', x=0.105, y=0.975, ha='left', fontsize=16)
fig.text(0.105, 0.939, 'Batch 160, learning rate 1e-5, seed 1  |  EER (%) - lower is better', fontsize=11)

styles = {
    'previous': dict(color='#555b66', linestyle='--', marker='s', markersize=4.5,
                     markerfacecolor='white', label='Previous: local BPL'),
    'current': dict(color='#1768ac', linestyle='-', marker='o', markersize=5,
                    label='Current: global BPL'),
}
for ax, dataset in zip(axes, ('19LA', '21LA', '21DF')):
    values = data['datasets'][dataset]
    for series in ('previous', 'current'):
        rows = values[series]
        ax.plot([row['epoch'] for row in rows], [row['eer'] for row in rows],
                linewidth=1.6, **styles[series])
    best_old = values['previous_best']
    best_new = values['current_best']
    ax.set_title(
        f"{dataset}  |  Lowest recorded EER: "
        f"{best_old['eer']:.4f} (ep{best_old['epoch']}) -> "
        f"{best_new['eer']:.4f} (ep{best_new['epoch']})",
        loc='left', pad=10,
    )
    ax.set_ylabel('EER (%)')
    ax.set_xlim(13, 52)
    ax.set_xticks(range(15, 51, 5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.margins(y=0.18)
    ax.grid(axis='y', color='#e2e5e9', linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis='x', labelbottom=True)
    ax.set_xlabel('Epoch')
    for series, best in (('previous', best_old), ('current', best_new)):
        ax.scatter([best['epoch']], [best['eer']], s=80,
                   facecolors='none', edgecolors=styles[series]['color'],
                   linewidths=1.3, zorder=4)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.097, 0.925),
           ncol=2, frameon=False, fontsize=10.5, columnspacing=2)
fig.text(0.105, 0.046, 'Each panel has its own y scale. Rings mark test-trajectory minima, not a shared best checkpoint.', fontsize=9)
fig.text(0.105, 0.021, 'Previous 21DF ep50 is unavailable. Current 21DF ep50 comes from the console log.', fontsize=9)
output = folder / 'ebm-bpl-eer-comparison-20261102.png'
fig.savefig(output, dpi=160, facecolor='white')
plt.close(fig)
print(output)
