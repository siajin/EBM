"""Posthoc ROC/ranking explanation using existing scores; no new detector fit."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score

from channel_keep_sweep import digest, save
from fast_analyze import metrics
from geometry_cache import DEFAULT_ROOT


def main():
    source = DEFAULT_ROOT / "local_analysis/channel_keep_one_v3"
    out = DEFAULT_ROOT / "local_analysis/channel_roc_explanation_v1"
    out.mkdir(parents=True, exist_ok=False)
    assert (source / "complete").read_text().strip() == "0"
    save(out / "analysis_record.json", {
        "scope": "posthoc descriptive explanation of already observed DF scores",
        "script_sha256": digest(Path(__file__)), "predictions_sha256": digest(source / "DF_scores.parquet"),
        "models": ["native", "keep_bottom2"], "targets_bona_false_reject": [.005, .01, .02, .03, .035, .05, .1],
        "not_deployment_threshold_selection": True, "new_inference": False, "new_training": False,
        "pair_counts_are_not_independent_samples": True})
    data = pd.read_parquet(source / "DF_scores.parquet")
    assert not data.utterance_id.duplicated().any()
    y = data.y.to_numpy()
    assert len(y) == 20000 and y.sum() == 2000
    records, curves, model_metrics = [], {}, []
    for name in ["native", "keep_bottom2"]:
        score = data[name].to_numpy()
        fp, tp, threshold = roc_curve(y, score, drop_intermediate=False)
        fn = 1-tp
        curves[name] = (fp, fn)
        model_metrics.append({"case": name, **metrics(y, score)})
        for target in [.005, .01, .02, .03, .035, .05, .1]:
            allowed = fn <= target+1e-12
            best = float(fp[allowed].min())
            records.append({"case": name, "bona_false_reject_cap_percent": 100*target,
                            "minimum_spoof_false_accept_percent": 100*best,
                            "scope": "empirical DF ROC description, not an operating threshold chosen for deployment"})
    pd.DataFrame(records).to_csv(out / "matched_ROC_points.csv", index=False)
    b, sp = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    s0, s2 = data.native.to_numpy(), data.keep_bottom2.to_numpy()
    counts, ties = np.zeros(4, dtype=np.int64), np.zeros(2, dtype=np.int64)
    for ix in np.array_split(b, 20):
        a = s0[ix, None]-s0[sp][None, :]
        c = s2[ix, None]-s2[sp][None, :]
        counts += np.array([np.sum((a > 0) & (c > 0)), np.sum((a <= 0) & (c > 0)),
                            np.sum((a > 0) & (c <= 0)), np.sum((a <= 0) & (c <= 0))])
        ties += np.array([np.sum(a == 0), np.sum(c == 0)])
    assert np.all(ties == 0)
    pairs = len(b)*len(sp)
    assert counts.sum() == pairs
    assert abs((counts[0]+counts[2])/pairs-roc_auc_score(y, s0)) < 1e-12
    assert abs((counts[0]+counts[1])/pairs-roc_auc_score(y, s2)) < 1e-12
    summary = {"status": "complete_descriptive", "model_metrics": model_metrics, "matched_ROC_points": records,
               "bona_spoof_pairs": pairs, "both_correct": int(counts[0]),
               "native_wrong_two_correct": int(counts[1]), "native_correct_two_wrong": int(counts[2]),
               "both_wrong": int(counts[3]), "ties": ties.tolist(),
               "pair_count_is_not_independent_sample_size": True,
               "interpretation": "ROC curves cross; lower EER does not imply better full ROC or lower error at every target",
               "cause_not_identified": "No causal attribution to codec/source/noise from this descriptive comparison"}
    save(out / "summary.json", summary)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for name, color, label in [("native", "#555555", "Native160: EER3.3500%, AUC0.995943"),
                               ("keep_bottom2", "#bd674d", "T-Fisher bottom2: EER2.9222%, AUC0.987659")]:
        fp, fn = curves[name]
        for ax in axes:
            ax.plot(100*fn[::-1], 100*fp[::-1], color=color, label=label)
        for target in [1., 3.]:
            value = next(r["minimum_spoof_false_accept_percent"] for r in records
                         if r["case"] == name and r["bona_false_reject_cap_percent"] == target)
            axes[1].plot(target, value, "o", color=color)
            axes[1].annotate(f"{value:.2f}%", (target, value), xytext=(5, 6), textcoords="offset points", color=color, fontsize=9)
    for ax in axes:
        ax.set(xlabel="Bonafide false reject (%)", ylabel="Spoof false acceptance (%)")
        ax.grid(alpha=.2)
    axes[0].set(xlim=(0, 10), ylim=(0, 100), title="Tails differ substantially")
    axes[1].set(xlim=(.4, 5), ylim=(0, 19), title="Near-EER detail: the curves cross")
    axes[1].plot([.4, 5], [.4, 5], "--", color="#999999", lw=.8, label="Equal-error line")
    axes[1].axvline(1, color="#aaaaaa", lw=.6)
    axes[1].axvline(3, color="#aaaaaa", lw=.6)
    axes[0].legend(fontsize=8)
    fig.suptitle("More channels can improve overall ranking while changing the EER balance point\nAlready-observed DF pilot; empirical ROC description only", fontsize=12)
    fig.savefig(out / "roc_crossing.png", dpi=170)
    plt.close(fig)
    (out / "complete").write_text("0\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
