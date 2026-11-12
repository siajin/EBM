"""Readable scientific figures and concise findings from the verified sweep."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/geometry_cache/local_analysis/low_occupancy_ablation_20261111_v1"


def main():
    check = json.loads((OUT / "independent_validation.json").read_text(encoding="utf-8"))
    assert check["status"] == "passed"
    f = pd.read_csv(OUT / "metrics.csv", float_precision="round_trip")
    primary = f[f.method == "low_energy_first"]
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 11})
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), layout="constrained")
    for ax, dataset in zip(axes, ("19LA", "21LA", "DF")):
        b = primary[(primary.dataset == dataset) & (primary.k <= 158)].sort_values("k")
        ax.plot(b.k, b.eer_percent, color="#356AA0", marker="o", markersize=4)
        ax.axhline(float(b.iloc[0].eer_percent), color="#7B838B", linestyle="--", linewidth=1)
        ax.set(title=f"{dataset} 전체 평가", xlabel="낮은 점유 순서로 제거한 잔여 차원 수", ylabel="EER (%)", xlim=(0, 159))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
    fig.suptitle("Train 정상에서 점유가 적은 축부터 제거 · 점선은 제거 전 EER\n보관 batch160 · epoch50 / 159개 전체 제거의 부호 전용 끝점은 별도 전체 그래프에 표시", fontsize=12)
    fig.savefig(OUT / "primary_eer_sweep_detail.png", dpi=180)
    plt.close(fig)
    # Repair observed missing minus glyphs in logarithmic exponent labels.
    spectrum = np.load(OUT / "train_only_bases.npz")["eigenvalues_low_first"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    axes[0].plot(np.arange(1, 160), spectrum, color="#356AA0")
    axes[0].set(xlabel="적은 점유 순서의 잔여 축", ylabel="Train 정상 방향의 평균 제곱 투영", yscale="log")
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0e}"))
    axes[1].plot(np.arange(160), np.r_[0, np.cumsum(spectrum)] * 100, color="#356AA0")
    axes[1].set(xlabel="제거한 잔여 차원 수", ylabel="제거되는 Train 정상 방향 에너지 (%)", ylim=(0, 100))
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
    fig.suptitle("원점 보존 · 평균을 빼지 않은 Train 정상 잔여 방향의 점유율")
    fig.savefig(OUT / "train_direction_spectrum.png", dpi=180)
    plt.close(fig)
    table = primary[(primary.dataset != "V") & primary.k.isin([0, 32, 64, 128, 152, 158, 159])]
    table.to_csv(OUT / "primary_summary.csv", index=False)
    findings = {"status": "complete_and_independently_verified", "model": "archived batch160 epoch50",
                "train_normal_only_ranking_n": 2580,
                "primary_predeclared_k16": "EER unchanged on all three full evaluation datasets",
                "observations": ["Removing 1 through 32 low-occupancy axes leaves all three reported EERs unchanged.",
                                 "Removing 152 axes: 19LA 0.435177 to 0.386650; 21LA 1.053539 to 1.055789; DF 3.241828 to 3.149353 percent EER.",
                                 "Removing 158 axes improves 19LA but worsens DF EER to 5.066091 percent.",
                                 "At fixed original V threshold, removing 152 axes changes DF normal rejection 19.873562 to 19.416235 percent, and spoof acceptance 0.362965 to 0.391285 percent.",
                                 "Retuned V threshold at 152 axes increases DF normal rejection to 21.474208 percent while reducing spoof acceptance to 0.296113 percent.",
                                 "No general attack-independent irrelevance claim is established; no target-optimal setting is selected for deployment."],
                "train_normal_direction_energy_removed": {str(k): float(spectrum[:k].sum()) for k in [16, 32, 64, 128, 152, 158, 159]},
                "basis_fit_and_ordering_used_attack_labels": False,
                "evaluations": {d: {"total": int(g.iloc[0].n), "normal": int(g.iloc[0].normal_n)} for d, g in primary.groupby("dataset")},
                "metrics": table.to_dict(orient="records"), "verification": check,
                "cost_and_cleanup": {"new_spend_usd": 0, "paid_resources_created": 0, "analysis_processes_finished": True,
                                     "training_modified": False, "model_updates": 0, "new_inferences": 0, "remote_operations": 0},
                "limitations": ["Exploratory evaluation sweep; small EER changes have no uncertainty estimate and are not a confirmed generalization gain.",
                                "The original model's current-training checkpoints were not evaluated.",
                                "Removing whole tangent axes and renormalizing differs from one-sided directional removal and from retraining."]}
    (OUT / "findings.json").write_text(json.dumps(findings, ensure_ascii=False, indent=2), encoding="utf-8")
    print(table[["dataset", "k", "eer_percent"]].to_string(index=False))


if __name__ == "__main__":
    main()
