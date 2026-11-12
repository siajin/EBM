"""Extend the frozen cosine sweep to 5,3,2 channels, reproducing the 10 anchor."""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from audit_cached_core_results import independent_eer, independent_auc
from cached_core_diagnostics import CHECKPOINT, weighted_eer_setup, weighted_eer
from channel_keep_sweep import digest, save, keep_score
from fast_analyze import metrics
from geometry_cache import DEFAULT_ROOT, load_embeddings, validate_run, read_json


COUNTS = [10, 5, 3, 2]
PREVIOUS = DEFAULT_ROOT / "local_analysis/channel_keep_sweep_v1"
CORE = DEFAULT_ROOT / "local_analysis/cached_core_v1"


def plot(out, combined):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = combined[combined.dataset == "DF_pilot"].sort_values("kept")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    for ax, field, ylabel in [(axes[0], "EER", "DF pilot EER (%)"),
                               (axes[1], "BFR", "DF bona false reject (%) at native V threshold")]:
        x = data.kept.to_numpy()
        ax.fill_between(x, data[f"random_{field}_min"].to_numpy(), data[f"random_{field}_max"].to_numpy(),
                        color="#8ca7b8", alpha=.25, label="Random 20: min / max")
        ax.plot(x, data[f"random_{field}_median"], "s-", color="#537a94", label="Random median")
        ax.plot(x, data[f"top_{field}"], "o-", color="#26806d", label="Keep T-Fisher top k")
        ax.plot(x, data[f"bottom_{field}"], "^-", color="#bd674d", label="Keep T-Fisher bottom k")
        ax.axhline(data[f"native_{field}"].iloc[0], color="#555555", linestyle="--", label="Native: all 160")
        ax.set(xlabel="Coordinates retained (of 160)", ylabel=ylabel, xlim=(20.7, 1.3), xticks=[20, 10, 5, 3, 2])
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(fontsize=8)
    axes[0].set_title("Same mask on z and centroid; both re-normalized")
    axes[1].set_title("Existing V threshold unchanged")
    fig.suptitle("Frozen EBM: channel sweep extended to 5, 3, 2\nFixed DF pilot; exploratory; no extraction or retraining", fontsize=13)
    fig.savefig(out / "channel_keep_tiny.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_ROOT / "local_analysis/channel_keep_tiny_v1")
    out = parser.parse_args().out
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    assert (PREVIOUS / "complete").read_text().strip() == "0"
    prior_lock = read_json(PREVIOUS / "protocol_lock.json")
    assert digest(Path(__file__).with_name("channel_keep_sweep.py")) == prior_lock["script_sha256"]
    rank = np.asarray(read_json(CORE / "channel_masks.json")["ranking_T_macroFisher"], dtype=int)
    assert np.array_equal(np.sort(rank), np.arange(160))
    masks = {"native": np.arange(160)}
    for k in COUNTS:
        masks[f"keep_top{k}"] = rank[:k]
        masks[f"keep_bottom{k}"] = rank[-k:]
        for seed in range(200, 220):
            masks[f"keep_random{k}_s{seed}"] = np.random.default_rng(seed).permutation(160)[-k:]
    for big, small in zip(COUNTS, COUNTS[1:]):
        for family in ["top", "bottom"]:
            assert set(masks[f"keep_{family}{small}"]).issubset(masks[f"keep_{family}{big}"])
        for seed in range(200, 220):
            assert set(masks[f"keep_random{small}_s{seed}"]).issubset(masks[f"keep_random{big}_s{seed}"])
    assert all(len(kept) == len(set(kept)) for kept in masks.values())
    lock = {"version": "channel_keep_tiny_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "script_sha256": digest(Path(__file__)), "prior_protocol_sha256": digest(PREVIOUS / "protocol_lock.json"),
            "prior_masks_sha256": digest(CORE / "channel_masks.json"), "checkpoint_sha256": CHECKPOINT,
            "dependency_sha256": {name: digest(Path(__file__).with_name(name)) for name in
                                  ["channel_keep_sweep.py", "cached_core_diagnostics.py", "geometry_cache.py", "fast_analyze.py", "audit_cached_core_results.py"]},
            "counts": COUNTS, "new_counts": [5, 3, 2], "anchor_reproduced": 10,
            "families": "T-only Fisher top/bottom; last k of seed200..219 permutations; nested; all masks reported",
            "score": "cosine(z[K],c[K]), normalize both; no norm feature or probe",
            "selection_on_new_DF_results": False, "EBM_retrained": False, "GPU_inference_runs": 0, "Q_opened": False,
            "intervals": "pointwise95; paired source-prefixed speaker bootstrap2000 seed7301; no simultaneous or noninferiority claim",
            "threshold": "original native V threshold, unchanged",
            "limits": ["already-observed20000-row DF pilot", "family dependencies unresolved",
                       "160-dimension encoder/LayerNorm still computed", "no smallest dimension or best DF mask selected"]}
    save(out / "protocol_lock.json", lock)
    save(out / "masks.json", {name: kept.tolist() for name, kept in masks.items()})
    threshold = read_json(PREVIOUS / "threshold_lock.json")["native_V_threshold"]
    records, curves, checks, score_error, metric_error, anchor_error = [], [], [], 0., 0., 0.
    for dataset in ["T", "V", "DF_pilot"]:
        checks.append(validate_run(dataset))
        data = load_embeddings(dataset, expected_checkpoint_sha256=CHECKPOINT)
        z, c, y = data.z.astype(float), data.centroid, data.rows.y.to_numpy()
        native = data.rows.cosine_s.to_numpy()
        native_eer = metrics(y, native)["eer_percent"]
        local = {}
        for name, kept in masks.items():
            score = native.copy() if name == "native" else keep_score(z, c, kept)
            assert np.isfinite(score).all() and np.max(np.abs(score)) <= 1+1e-12
            removed = np.setdiff1d(np.arange(160), kept)
            zz, cc = z.copy(), c.copy()
            zz[:, removed], cc[removed] = 0., 0.
            independent = np.sum(zz*cc, axis=1)/(np.sqrt(np.sum(zz**2, axis=1))*np.linalg.norm(cc))
            score_error = max(score_error, float(np.max(np.abs(independent-score))))
            actual = metrics(y, score)
            metric_error = max(metric_error, abs(actual["eer_percent"]-independent_eer(y, independent)),
                               abs(actual["auroc"]-independent_auc(y, independent)))
            local[name] = score
            records.append({"dataset": dataset, "case": name, "kept": len(kept), **actual,
                            "EER_impairment_pp": actual["eer_percent"]-native_eer,
                            "fixed_V_threshold": threshold,
                            "bona_false_reject_percent": float(100*np.mean(score[y == 1] < threshold)),
                            "spoof_false_accept_percent": float(100*np.mean(score[y == 0] >= threshold))})
        frame = pd.DataFrame([r for r in records if r["dataset"] == dataset]).set_index("case")
        for k in COUNTS:
            random = frame.loc[[f"keep_random{k}_s{seed}" for seed in range(200, 220)]]
            result = {"dataset": dataset, "kept": k, "random_count": len(random)}
            for short, col in [("EER", "eer_percent"), ("BFR", "bona_false_reject_percent")]:
                for family in ["top", "bottom"]:
                    result[f"{family}_{short}"] = float(frame.loc[f"keep_{family}{k}", col])
                result[f"native_{short}"] = float(frame.loc["native", col])
                result[f"random_{short}_min"] = float(random[col].min())
                result[f"random_{short}_median"] = float(random[col].median())
                result[f"random_{short}_max"] = float(random[col].max())
            curves.append(result)
        if dataset == "DF_pilot":
            df_data, df_scores = data, local
            saved = pd.concat([data.rows[["utterance_id", "speaker_group", "y"]], pd.DataFrame(local)], axis=1)
            saved.to_parquet(out / "DF_scores.parquet", index=False)
            old = pd.read_parquet(PREVIOUS / "DF_scores.parquet")
            assert np.array_equal(old.utterance_id, data.rows.utterance_id)
            for name in ["native", "keep_top10", "keep_bottom10"]+[f"keep_random10_s{seed}" for seed in range(200, 220)]:
                anchor_error = max(anchor_error, float(np.max(np.abs(old[name].to_numpy()-local[name]))))
        print("tiny sweep dataset complete", dataset, flush=True)
    assert score_error < 1e-12 and metric_error < 1e-9 and anchor_error < 1e-12
    pd.DataFrame(records).to_csv(out / "all_metrics.csv", index=False)
    pd.DataFrame(curves).to_csv(out / "curves.csv", index=False)
    previous_curves = pd.read_csv(PREVIOUS / "curves.csv")
    combined = pd.concat([previous_curves[previous_curves.kept == 20], pd.DataFrame(curves)], ignore_index=True)
    combined.to_csv(out / "combined_curves.csv", index=False)
    y, groups = df_data.rows.y.to_numpy(), df_data.rows.speaker_group.to_numpy()
    speakers, codes = np.unique(groups, return_inverse=True)
    draws = np.random.default_rng(7301).multinomial(len(speakers), np.full(len(speakers), 1/len(speakers)), size=2000)
    samples = {}
    for name in ["native"]+[f"keep_{family}{k}" for k in COUNTS for family in ["top", "bottom"]]:
        prepared = weighted_eer_setup(y, df_scores[name], codes)
        assert abs(weighted_eer(prepared, np.ones(len(speakers)))-metrics(y, df_scores[name])["eer_percent"]) < 1e-9
        samples[name] = np.array([weighted_eer(prepared, draw) for draw in draws])
    intervals = []
    for name in list(samples)[1:]:
        delta = samples[name]-samples["native"]
        good = np.isfinite(delta)
        intervals.append({"case": name, "EER_impairment_pp": metrics(y, df_scores[name])["eer_percent"]-metrics(y, df_scores["native"])["eer_percent"],
                          "CI95_pointwise": np.quantile(delta[good], [.025, .975]).tolist(),
                          "groups": len(speakers), "dropped_replicates": int((~good).sum())})
    save(out / "paired_intervals.json", intervals)
    audit = {"status": "pass", "datasets_validated": checks, "masks_per_dataset": len(masks),
             "metrics_independently_checked": len(records), "score_max_error": score_error,
             "metric_max_error": metric_error, "previous10_cases_reproduced": 23, "previous10_max_score_error": anchor_error,
             "nested_masks_checked": True, "protocol_sha256": digest(out / "protocol_lock.json"),
             "GPU_inference_runs": 0, "Q_opened": False}
    save(out / "verification.json", audit)
    plot(out, combined)
    summary = {"status": "complete_exploratory", "protocol_sha256": audit["protocol_sha256"],
               "DF_curves": [r for r in curves if r["dataset"] == "DF_pilot"], "DF_intervals": intervals,
               "elapsed_seconds": time.monotonic()-started, "GPU_inference_runs": 0, "EBM_retrained": False,
               "Q_opened": False, "limits": lock["limits"]}
    save(out / "summary.json", summary)
    (out / "complete").write_text("0\n", encoding="utf-8")
    print(summary, flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
