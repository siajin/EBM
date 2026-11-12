"""Research figure from saved core outputs; no fitting or inference."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    root = Path(__file__).resolve().parents[1] / "geometry_cache/local_analysis/cached_core_v1"
    summary = json.loads((root / "summary.json").read_text())
    probes = {r["case"]: r for r in summary["probe_metrics"]}
    colors = {"scalar": "#677387", "noise": "#c2a54d", "direction": "#26806d",
              "metadata": "#7d6da0", "metadata_direction": "#397eb1",
              "preLN_h": "#cf7350", "postLN_z": "#506fab", "preLN_stats": "#93806f"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    fig, axes = plt.subplots(3, 2, figsize=(14, 14), layout="constrained")
    cases = ["scalar", "noise", "direction", "metadata", "metadata_direction"]
    labels = ["s + norm", "+ noise", "+ direction", "+ conditions", "+ cond. + dir."]

    def bar(ax, values, keys, names):
        ax.bar(np.arange(len(values)), values, color=[colors[k] for k in keys], width=.7)
        ax.set_xticks(np.arange(len(values)), names, rotation=12, ha="right")
        ax.set_ylim(0, max(values)*1.22)
        ax.bar_label(ax.containers[0], fmt="%.4f", padding=3, fontsize=9)
        ax.set_ylabel("Class-balanced log-loss (lower is better)")

    bar(axes[0, 0], [probes[k]["balanced_logloss"] for k in cases], cases, labels)
    axes[0, 0].set_title("A  DF-internal nested speaker CV: all 20,000 rows", loc="left")
    match = next(r for r in summary["matched"] if r["name"] == "primary")
    matched_probes = {r["case"]: r for r in match.get("probe_metrics", [])}
    if matched_probes:
        bar(axes[0, 1], [matched_probes[k]["balanced_logloss"] for k in cases], cases, labels)
    axes[0, 1].set_title(f"B  Matched source / codec: {match['pairs']} pairs", loc="left")
    axes[0, 1].text(.02, .98, "|delta cosine| <= 0.01; |delta log norm| <= 0.002\nSubset of saved held-out predictions; descriptive", transform=axes[0, 1].transAxes,
                    va="top", fontsize=8)
    curves = pd.DataFrame(summary["channel_curves"])
    curves = curves[curves.dataset == "DF_pilot"]
    ax = axes[1, 0]
    x = curves.removed.to_numpy()
    ax.axhline(curves.native_EER.iloc[0], color="#677387", linestyle="--", label="Native cosine")
    ax.fill_between(x, curves.random_min.to_numpy(), curves.random_max.to_numpy(), color="#8ca7b8", alpha=.22, label="Random masks: min / max (20)")
    ax.plot(x, curves.random_median, "s-", color="#537a94", label="Random median")
    ax.plot(x, curves.top_EER, "o-", color="#bd674d", label="T-Fisher top channels removed")
    ax.set(xlabel="Channels removed (of 160)", ylabel="DF pilot EER (%)", title="C  Existing cosine after joint z / centroid masking")
    ax.set_xticks(x)
    ax.legend(fontsize=8)
    cases_ln = ["postLN_z", "preLN_h", "preLN_stats"]
    bar(axes[1, 1], [probes[k]["balanced_logloss"] for k in cases_ln], cases_ln,
        ["scalar + z", "scalar + h", "scalar + h mean/std"])
    axes[1, 1].set_title("D  LayerNorm: same L2 probe / nested C grid", loc="left")
    assoc = pd.DataFrame(summary["cluster_associations"])
    assoc = assoc[(assoc.dataset == "DF_pilot") & (assoc.field != "y")]
    fields = ["source_corpus", "codec", "duration_bin", "padding_bin"]
    ax = axes[2, 0]
    for offset, scope, color in [(-.25, "all", "#677387"), (0, "bona", "#397eb1"), (.25, "spoof", "#bd674d")]:
        values = [float(assoc[(assoc.field == f) & (assoc.scope == scope)].NMI.iloc[0]) for f in fields]
        ax.bar(np.arange(4)+offset, values, width=.24, label=scope, color=color)
    ax.set_xticks(range(4), ["Source", "Codec", "Duration bins", "Padding bins"])
    ax.set(ylabel="Normalized mutual information", ylim=(0, 1), title="E  T-fixed direction clusters vs DF conditions")
    ax.legend(fontsize=8)
    ax.text(.02, .98, "Descriptive associations; different category counts", transform=ax.transAxes, va="top", fontsize=8)
    predictions = pd.read_parquet(root / "DF_nested_OOF_predictions.parquet")
    cluster_y = next(r for r in summary["cluster_associations"] if r["dataset"] == "DF_pilot" and r["field"] == "y")
    ax = axes[2, 1]
    counts = pd.DataFrame(summary["cluster_descriptions"])
    counts = counts[counts.dataset == "DF_pilot"]
    ax.bar(counts.cluster, 100*counts.bona_fraction, color="#397eb1", label="Bonafide")
    ax.bar(counts.cluster, 100*(1-counts.bona_fraction), bottom=100*counts.bona_fraction, color="#bd674d", label="Spoof")
    ax.set(xlabel="T-fixed spherical cluster (K=4)", ylabel="Within-cluster class composition (%)", ylim=(0, 112),
           title=f"F  Class composition; cluster/class NMI = {cluster_y['NMI']:.3f}")
    ax.set_xticks(counts.cluster)
    for row in counts.itertuples():
        ax.text(row.cluster, 102, f"n={row.n:,}", ha="center", fontsize=8)
    ax.legend(fontsize=8)
    fig.suptitle("Frozen EBM embedding cache: exploratory CPU diagnostics\nDF pilot only; within-DF probes do not establish T-to-DF transfer", fontsize=15)
    fig.savefig(root / "core_diagnostics.png", dpi=170)
    plt.close(fig)
    print(root / "core_diagnostics.png")


if __name__ == "__main__":
    main()
