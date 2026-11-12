"""Exact tied 1-D cosine and an interpretable audit of the prior 2-D readout.

For nonzero scalars cosine is exactly sign(z_j*c_j). Quotient roundoff must
not be used to manufacture ranks within the two mathematically tied scores.
"""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import pandas as pd
from sklearn.metrics import det_curve
from threadpoolctl import threadpool_limits

from audit_cached_core_results import independent_eer, independent_auc
from cached_core_diagnostics import CHECKPOINT, weighted_eer_setup, weighted_eer
from channel_keep_sweep import digest, save, keep_score
from fast_analyze import metrics
from geometry_cache import DEFAULT_ROOT, load_embeddings, validate_run, read_json


PREVIOUS = DEFAULT_ROOT / "local_analysis/channel_keep_tiny_v1"
CORE = DEFAULT_ROOT / "local_analysis/cached_core_v1"


def count_metrics(y, score):
    """Independent closed form from the two score bins, with linear EER ties."""
    assert set(np.unique(score)).issubset({-1., 1.})
    fpr = float(np.mean(score[y == 0] == 1))
    fnr = float(np.mean(score[y == 1] == -1))
    value = fpr/(fpr+1-fnr) if fpr >= fnr else fnr/(1-fpr+fnr)
    return {"eer_percent": 100*value, "auroc": 1-.5*(fpr+fnr),
            "bona_negative": int(np.sum((y == 1) & (score == -1))),
            "bona_positive": int(np.sum((y == 1) & (score == 1))),
            "spoof_negative": int(np.sum((y == 0) & (score == -1))),
            "spoof_positive": int(np.sum((y == 0) & (score == 1))),
            "bona_false_reject_at_zero_percent": 100*fnr,
            "spoof_false_accept_at_zero_percent": 100*fpr}


def endpoint_restored_det_eer(y, score):
    # sklearn may trim DET to a single operating point for binary scores.
    # Restore reject-all and accept-all before the same linear interpolation.
    fp, fn, _ = det_curve(y, score)
    fp, fn = np.r_[0., fp[::-1], 1.], np.r_[1., fn[::-1], 0.]
    difference = fp-fn
    hi = int(np.flatnonzero(difference >= 0)[0])
    if hi == 0:
        return float(50*(fp[0]+fn[0]))
    t = -difference[hi-1]/(difference[hi]-difference[hi-1])
    return float(100*(fp[hi-1]+t*(fp[hi]-fp[hi-1])))


def boundary_checks():
    y = np.array([1, 1, 0, 0])
    for score, expected in [(np.array([1, 1, -1, -1]), 0.), (np.ones(4), 50.),
                            (-np.ones(4), 50.), (np.array([-1, -1, 1, 1]), 100.)]:
        result = count_metrics(y, score)
        assert abs(result["eer_percent"]-expected) < 1e-12
        assert abs(metrics(y, score)["eer_percent"]-expected) < 1e-12
        assert abs(endpoint_restored_det_eer(y, score)-expected) < 1e-12
    rng = np.random.default_rng(71)
    for _ in range(30):
        y = np.r_[np.ones(20), np.zeros(31)].astype(int)
        score = rng.choice([-1., 1.], len(y))
        result = count_metrics(y, score)
        assert abs(result["eer_percent"]-metrics(y, score)["eer_percent"]) < 1e-12
        assert abs(result["auroc"]-independent_auc(y, score)) < 1e-12
        assert abs(result["eer_percent"]-endpoint_restored_det_eer(y, score)) < 1e-12
    return "pass: perfect, constant, inverse and30 random tied binary scores"


def figures(out, data, scores, audit):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = audit["coordinates_zero_based"]
    z, c, y = data.z[:, idx].astype(float), data.centroid[idx], data.rows.y.to_numpy()
    rng = np.random.default_rng(17)
    display = np.concatenate([rng.choice(np.flatnonzero(y == label), 1000, replace=False) for label in [0, 1]])
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), layout="constrained")
    colors = {0: "#bb654d", 1: "#397eb1"}
    names = {0: "Spoof", 1: "Bonafide"}
    for label in [0, 1]:
        ix = display[y[display] == label]
        axes[0, 0].scatter(z[ix, 0], z[ix, 1], s=7, alpha=.35, color=colors[label], label=names[label])
        unit = z[ix]/np.linalg.norm(z[ix], axis=1, keepdims=True)
        axes[0, 1].scatter(unit[:, 0], unit[:, 1], s=7, alpha=.35, color=colors[label], label=names[label])
    axes[0, 0].set(xlabel=f"Final z[{idx[0]}] (full EBM already computed)", ylabel=f"Final z[{idx[1]}]",
                   title="A  Actual learned coordinates; 1,000 displayed / class")
    axes[0, 0].legend()
    theta = np.linspace(0, 2*np.pi, 400)
    axes[0, 1].plot(np.cos(theta), np.sin(theta), color="#999999", lw=.7)
    direction = c/np.linalg.norm(c)
    axes[0, 1].arrow(0, 0, direction[0], direction[1], width=.01, color="#333333", length_includes_head=True)
    axes[0, 1].set(xlabel="Selected pair after L2 normalization", ylabel="", title="B  2-D cosine reads angle to the centroid arrow")
    axes[0, 1].set_aspect("equal")
    axes[0, 1].set_xlim(-1.12, 1.12)
    axes[0, 1].set_ylim(-1.12, 1.12)
    bins = np.array([-1., 1.])
    for offset, label in [(-.17, 0), (.17, 1)]:
        values = [100*np.mean(scores["keep_bottom1"][y == label] == value) for value in bins]
        axes[1, 0].bar(np.arange(2)+offset, values, width=.32, color=colors[label], label=names[label])
    axes[1, 0].set(xticks=[0, 1], xticklabels=["-1: opposite sign", "+1: same sign"], ylim=(0, 107),
                   ylabel="Within-class share (%)", title="C  1-D cosine has exactly two scores")
    axes[1, 0].legend()
    for bars in axes[1, 0].containers:
        axes[1, 0].bar_label(bars, fmt="%.2f%%", padding=3, fontsize=8)
    from sklearn.metrics import roc_curve
    for name, color, label in [("native", "#555555", "Native: all160"),
                               ("keep_bottom2", "#bd674d", "Selected pair:113,95"),
                               ("keep_bottom1", "#26806d", "One coordinate:95 (exact ties)")]:
        fp, tp, _ = roc_curve(y, scores[name], drop_intermediate=False)
        axes[1, 1].plot(fp, tp, color=color, label=label)
    axes[1, 1].set(xlabel="Spoof false acceptance rate", ylabel="Bonafide acceptance rate", title="D  Full ROC can differ despite similar EER", xlim=(0, 1), ylim=(0, 1))
    axes[1, 1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Full audio -> full EBM ->160 learned values -> select2 or1 -> recompute cosine\nDF pilot; display subsampling only; all20,000 rows used for metrics", fontsize=12)
    fig.savefig(out / "how_two_and_one_work.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_ROOT / "local_analysis/channel_keep_one_v2")
    out = parser.parse_args().out
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    rank = np.asarray(read_json(CORE / "channel_masks.json")["ranking_T_macroFisher"], dtype=int)
    masks = {"native": np.arange(160)}
    for k in [2, 1]:
        masks[f"keep_top{k}"] = rank[:k]
        masks[f"keep_bottom{k}"] = rank[-k:]
        for seed in range(200, 220):
            masks[f"keep_random{k}_s{seed}"] = np.random.default_rng(seed).permutation(160)[-k:]
    for family in ["top", "bottom"]:
        assert set(masks[f"keep_{family}1"]).issubset(masks[f"keep_{family}2"])
    for seed in range(200, 220):
        assert set(masks[f"keep_random1_s{seed}"]).issubset(masks[f"keep_random2_s{seed}"])
    lock = {"version": "channel_keep_one_v2", "created_utc": datetime.now(timezone.utc).isoformat(),
            "script_sha256": digest(Path(__file__)), "counts": [2, 1], "new_counts": [1],
            "previous_protocol_sha256": digest(PREVIOUS / "protocol_lock.json"),
            "dependency_sha256": {name: digest(Path(__file__).with_name(name)) for name in
                                  ["channel_keep_sweep.py", "cached_core_diagnostics.py", "geometry_cache.py", "fast_analyze.py", "audit_cached_core_results.py"]},
            "masks": "same T-only top/bottom and nested random seeds200..219; no DF selection",
            "one_dimensional_score": "exact sign(z[j])*sign(c[j]); raw quotient roundoff cannot break mathematical ties",
            "zero_policy": "fail if selected z or c is zero; no silent exclusions",
            "EER": "linear ROC interpolation across entire tied score bins; not a realizable deterministic threshold if rates differ at sign threshold",
            "verification_note": "v1 stopped on T because trimmed single-point DET reference omitted endpoints; v2 restores reject-all/accept-all; primary ROC and count formula unchanged",
            "failed_v1_script_sha256": digest(DEFAULT_ROOT / "local_analysis/channel_keep_one_v1/runner_at_failed_validation.py"),
            "checkpoint_sha256": CHECKPOINT, "GPU_inference_runs": 0, "EBM_retrained": False, "Q_opened": False,
            "bootstrap": "source-prefixed speaker paired2000 seed7301; pointwise95 only",
            "limits": ["already-observed DF pilot", "family dependencies unresolved", "full160-dimension encoder and LN still computed",
                       "no smallest-dimension or deployment improvement claim"]}
    save(out / "protocol_lock.json", lock)
    save(out / "masks.json", {name: kept.tolist() for name, kept in masks.items()})
    math_check = boundary_checks()
    threshold = read_json(DEFAULT_ROOT / "local_analysis/channel_keep_sweep_v1/threshold_lock.json")["native_V_threshold"]
    records, counts, audits, checks, score_error, metric_error, anchor_error = [], [], [], [], 0., 0., 0.
    for dataset in ["T", "V", "DF_pilot"]:
        checks.append(validate_run(dataset))
        data = load_embeddings(dataset, expected_checkpoint_sha256=CHECKPOINT)
        z, c, y = data.z.astype(float), data.centroid, data.rows.y.to_numpy()
        local = {}
        for name, kept in masks.items():
            raw = data.rows.cosine_s.to_numpy().copy() if name == "native" else keep_score(z, c, kept)
            if len(kept) == 1:
                j = int(kept[0])
                assert np.all(z[:, j] != 0) and c[j] != 0
                score = np.sign(z[:, j])*np.sign(c[j])
                independent = np.where(np.signbit(z[:, j]) == np.signbit(c[j]), 1., -1.)
                assert np.array_equal(score, independent)
                closed = count_metrics(y, score)
                counts.append({"dataset": dataset, "case": name, "coordinate_zero_based": j, **closed})
                raw_metric = metrics(y, raw)
                audits.append({"dataset": dataset, "case": name, "coordinate_zero_based": j,
                               "raw_quotient_unique_scores": len(np.unique(raw)), "canonical_unique_scores": len(np.unique(score)),
                               "raw_quotient_EER_numeric_only": raw_metric["eer_percent"], "exact_sign_EER": closed["eer_percent"],
                               "max_quotient_to_sign_error": float(np.max(np.abs(raw-score))),
                               "zero_count": int(np.sum(z[:, j] == 0))})
                score_error = max(score_error, float(np.max(np.abs(raw-score))))
            else:
                score = raw
                removed = np.setdiff1d(np.arange(160), kept)
                zz, cc = z.copy(), c.copy()
                zz[:, removed], cc[removed] = 0., 0.
                independent = np.sum(zz*cc, axis=1)/(np.linalg.norm(zz, axis=1)*np.linalg.norm(cc))
                score_error = max(score_error, float(np.max(np.abs(independent-score))))
            actual = metrics(y, score)
            ref_eer = closed["eer_percent"] if len(kept) == 1 else independent_eer(y, independent)
            ref_auc = closed["auroc"] if len(kept) == 1 else independent_auc(y, independent)
            metric_error = max(metric_error, abs(actual["eer_percent"]-ref_eer), abs(actual["auroc"]-ref_auc))
            assert abs(actual["eer_percent"]-endpoint_restored_det_eer(y, score)) < 1e-9
            local[name] = score
            records.append({"dataset": dataset, "case": name, "kept": len(kept), **actual,
                            "bona_false_reject_percent": float(100*np.mean(score[y == 1] < threshold)),
                            "spoof_false_accept_percent": float(100*np.mean(score[y == 0] >= threshold))})
        if dataset == "DF_pilot":
            df_data, df_scores = data, local
            old = pd.read_parquet(PREVIOUS / "DF_scores.parquet")
            assert np.array_equal(old.utterance_id, data.rows.utterance_id)
            for name in ["native", "keep_top2", "keep_bottom2"]+[f"keep_random2_s{seed}" for seed in range(200, 220)]:
                anchor_error = max(anchor_error, float(np.max(np.abs(local[name]-old[name].to_numpy()))))
            pd.concat([data.rows[["utterance_id", "speaker_group", "y"]], pd.DataFrame(local)], axis=1).to_parquet(out / "DF_scores.parquet", index=False)
        print("one-dimensional dataset complete", dataset, flush=True)
    assert score_error < 1e-12 and metric_error < 1e-9 and anchor_error < 1e-12
    pd.DataFrame(records).to_csv(out / "all_metrics.csv", index=False)
    pd.DataFrame(counts).to_csv(out / "exact_sign_class_counts.csv", index=False)
    save(out / "one_dimensional_roundoff_audit.json", audits)
    frame = pd.DataFrame(records)
    curves = []
    for dataset in ["T", "V", "DF_pilot"]:
        subset = frame[frame.dataset == dataset].set_index("case")
        for k in [2, 1]:
            random = subset.loc[[f"keep_random{k}_s{seed}" for seed in range(200, 220)]]
            curves.append({"dataset": dataset, "kept": k, "native_EER": float(subset.loc["native", "eer_percent"]),
                           "top_EER": float(subset.loc[f"keep_top{k}", "eer_percent"]),
                           "bottom_EER": float(subset.loc[f"keep_bottom{k}", "eer_percent"]),
                           "random_EER_median": float(random.eer_percent.median()), "random_EER_min": float(random.eer_percent.min()),
                           "random_EER_max": float(random.eer_percent.max())})
    pd.DataFrame(curves).to_csv(out / "curves.csv", index=False)
    y, groups = df_data.rows.y.to_numpy(), df_data.rows.speaker_group.to_numpy()
    speakers, codes = np.unique(groups, return_inverse=True)
    draws = np.random.default_rng(7301).multinomial(len(speakers), np.full(len(speakers), 1/len(speakers)), size=2000)
    boot = {}
    for name in ["native", "keep_top1", "keep_bottom1"]:
        prepared = weighted_eer_setup(y, df_scores[name], codes)
        boot[name] = np.array([weighted_eer(prepared, draw) for draw in draws])
    intervals = []
    for name in ["keep_top1", "keep_bottom1"]:
        delta = boot[name]-boot["native"]
        good = np.isfinite(delta)
        intervals.append({"case": name, "EER_impairment_pp": metrics(y, df_scores[name])["eer_percent"]-metrics(y, df_scores["native"])["eer_percent"],
                          "CI95_pointwise": np.quantile(delta[good], [.025, .975]).tolist(), "dropped_replicates": int((~good).sum())})
    save(out / "paired_intervals.json", intervals)
    idx = masks["keep_bottom2"]
    pair_z, pair_c = df_data.z[:, idx].astype(float), df_data.centroid[idx]
    angle_score = np.cos(np.arctan2(pair_z[:, 1], pair_z[:, 0])-np.arctan2(pair_c[1], pair_c[0]))
    pair_audit = {"coordinates_zero_based": idx.tolist(), "centroid_pair": pair_c.tolist(),
                  "max_angle_score_error": float(np.max(np.abs(angle_score-df_scores["keep_bottom2"]))),
                  "overall_pair_correlation": float(np.corrcoef(pair_z.T)[0, 1]),
                  "class_mean_pair": {str(label): pair_z[y == label].mean(0).tolist() for label in [0, 1]},
                  "display_points_per_class": 1000}
    assert pair_audit["max_angle_score_error"] < 1e-12
    save(out / "two_coordinate_explanation_stats.json", pair_audit)
    figures(out, df_data, df_scores, pair_audit)
    audit = {"status": "pass", "cache_validation": checks, "math_boundary_checks": math_check,
             "metrics_independently_checked": len(records), "score_max_error": score_error, "metric_max_error": metric_error,
             "previous2_cases_reproduced": 23, "previous2_max_score_error": anchor_error,
             "exact_1D_ties_preserved": True, "protocol_sha256": digest(out / "protocol_lock.json"),
             "GPU_inference_runs": 0, "Q_opened": False}
    save(out / "verification.json", audit)
    summary = {"status": "complete_exploratory", "protocol_sha256": audit["protocol_sha256"],
               "DF_curves": [r for r in curves if r["dataset"] == "DF_pilot"], "DF_intervals": intervals,
               "random_single_coordinate_unique_count": len({int(masks[f"keep_random1_s{seed}"][0]) for seed in range(200, 220)}),
               "pair_audit": pair_audit, "elapsed_seconds": time.monotonic()-started, "GPU_inference_runs": 0, "EBM_retrained": False, "Q_opened": False}
    save(out / "summary.json", summary)
    (out / "complete").write_text("0\n", encoding="utf-8")
    print(summary, flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
