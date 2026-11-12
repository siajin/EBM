"""CPU-only fixed-cosine channel sweep, keeping 80,70,...,10 coordinates.

Top/bottom masks use T-only Fisher ranks. Random masks continue the exact old
removal permutations. All settings are locked before new DF scores are computed.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

from audit_cached_core_results import independent_eer, independent_auc
from cached_core_diagnostics import CHECKPOINT, weighted_eer_setup, weighted_eer
from fast_analyze import metrics
from geometry_cache import DEFAULT_ROOT, load_embeddings, resolve_run, validate_run


KEEP_COUNTS = [80, 70, 60, 50, 40, 30, 20, 10]
OLD = DEFAULT_ROOT / "local_analysis/cached_core_v1"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def keep_score(z, c, kept):
    zk, ck = z[:, kept].astype(float), c[kept]
    zn, cn = np.linalg.norm(zk, axis=1), np.linalg.norm(ck)
    assert cn > 1e-12 and np.all(zn > 1e-12)
    return (zk@ck)/(zn*cn)


def make_masks(rank):
    masks = {"native": np.arange(160)}
    for k in KEEP_COUNTS:
        masks[f"keep_top{k}"] = rank[:k]
        masks[f"keep_bottom{k}"] = rank[-k:]
        for seed in range(200, 220):
            masks[f"keep_random{k}_s{seed}"] = np.random.default_rng(seed).permutation(160)[-k:]
    for name, kept in masks.items():
        assert len(kept) == len(set(kept)) and np.all((kept >= 0) & (kept < 160))
    for big, small in zip(KEEP_COUNTS, KEEP_COUNTS[1:]):
        for prefix in ["top", "bottom"]:
            assert set(masks[f"keep_{prefix}{small}"]).issubset(masks[f"keep_{prefix}{big}"])
        for seed in range(200, 220):
            assert set(masks[f"keep_random{small}_s{seed}"]).issubset(masks[f"keep_random{big}_s{seed}"])
    return masks


def plot_results(out, curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    df = pd.DataFrame(curves)
    df = df[df.dataset == "DF_pilot"].sort_values("kept")
    x = df.kept.to_numpy()
    for ax, prefix, ylabel in [(axes[0], "EER", "DF pilot EER (%)"),
                                (axes[1], "BFR", "DF bona false reject (%) at native V threshold")]:
        ax.fill_between(x, df[f"random_{prefix}_min"].to_numpy(), df[f"random_{prefix}_max"].to_numpy(),
                        color="#8ca7b8", alpha=.25, label="Random 20: min / max")
        ax.plot(x, df[f"random_{prefix}_median"], "s-", color="#537a94", label="Random median")
        ax.plot(x, df[f"top_{prefix}"], "o-", color="#26806d", label="Keep T-Fisher top k")
        ax.plot(x, df[f"bottom_{prefix}"], "^-", color="#bd674d", label="Keep T-Fisher bottom k (old drop-top path)")
        ax.axhline(df[f"native_{prefix}"].iloc[0], color="#555555", linestyle="--", label="Native: all 160")
        ax.set(xlabel="Coordinates retained (of 160)", ylabel=ylabel, xlim=(82, 8))
        ax.set_xticks(KEEP_COUNTS)
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(fontsize=8)
    axes[0].set_title("Existing cosine with the same mask on z and centroid")
    axes[1].set_title("Threshold unchanged; different from EER threshold sweep")
    fig.suptitle("Frozen EBM: 80 down to 10 coordinates retained\nFixed DF pilot, exploratory; no extraction or retraining", fontsize=13)
    fig.savefig(out / "channel_keep_sweep.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_ROOT / "local_analysis/channel_keep_sweep_v1")
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    assert (OLD / "complete").read_text().strip() == "0"
    old_masks = json.loads((OLD / "channel_masks.json").read_text())
    rank = np.asarray(old_masks["ranking_T_macroFisher"], dtype=int)
    assert np.array_equal(np.sort(rank), np.arange(160))
    masks = make_masks(rank)
    lock = {
        "version": "channel_keep_sweep_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "exploratory_followup_of_observed_DF_pilot", "checkpoint_sha256": CHECKPOINT,
        "script_sha256": digest(Path(__file__)), "prior_masks_sha256": digest(OLD / "channel_masks.json"),
        "dependency_sha256": {name: digest(Path(__file__).with_name(name)) for name in
                              ["geometry_cache.py", "cached_core_diagnostics.py", "fast_analyze.py", "audit_cached_core_results.py"]},
        "keep_counts": KEEP_COUNTS, "baseline": "native all160, original archived cosine",
        "families": {"top": "T-Fisher ranking first k coordinates", "bottom": "T-Fisher ranking last k; continues previous drop-top experiment",
                     "random": "last k of original seed200..219 permutations, nested across k; all20 reported"},
        "score": "cosine(z[kept],c[kept]), separately re-normalize both, no added norm feature",
        "fit_or_selection_on_DF": False, "probe_fit": False, "EBM_retrained": False,
        "GPU_inference_runs": 0, "Q_opened": False,
        "intervals": "DF top/bottom impairment vs native; paired source-prefixed speaker bootstrap2000 seed7301; pointwise95 only",
        "random_spread": "mask design variability, not a sampling CI",
        "threshold": "existing native V threshold, never selected anew on DF",
        "limits": ["fixed already-observed20000-row DF pilot", "family dependencies unresolved",
                   "no simultaneous or noninferiority claim", "160-dimension encoder and LayerNorm still computed before masking",
                   "no minimum dimension or best DF mask selection"]}
    save(out / "protocol_lock.json", lock)
    save(out / "masks.json", {name: kept.tolist() for name, kept in masks.items()})
    folder, _, _ = resolve_run("DF_pilot")
    threshold = json.loads((folder / "registry.json").read_text())["thresholds"]["native_cosine"]["threshold"]
    save(out / "threshold_lock.json", {"native_V_threshold": threshold, "source_registry_sha256": digest(folder / "registry.json")})
    records, curves, checks, df_scores = [], [], [], None
    max_score_error, max_metric_error, previous_reproduction_error = 0., 0., 0.
    for dataset in ["T", "V", "DF_pilot"]:
        checks.append(validate_run(dataset))
        data = load_embeddings(dataset, expected_checkpoint_sha256=CHECKPOINT)
        z, c, y = data.z.astype(float), data.centroid, data.rows.y.to_numpy()
        native = data.rows.cosine_s.to_numpy()
        assert np.max(np.abs(keep_score(z, c, np.arange(160))-native)) < 1e-12
        if dataset == "T":
            bona = z[y == 1]
            attack = data.rows.attack_id.to_numpy()
            fisher = np.mean([(bona.mean(0)-z[attack == a].mean(0))**2 /
                              (bona.var(0, ddof=1)+z[attack == a].var(0, ddof=1)+1e-8)
                              for a in sorted(set(attack)-{"bonafide"})], axis=0)
            assert np.array_equal(np.argsort(-fisher, kind="stable"), rank)
        local = {}
        for name, kept in masks.items():
            score = native.copy() if name == "native" else keep_score(z, c, kept)
            assert np.isfinite(score).all() and np.max(np.abs(score)) <= 1+1e-12
            # Independent calculation keeps full arrays, zeroing removed indices.
            removed = np.setdiff1d(np.arange(160), kept)
            zz, cc = z.copy(), c.copy()
            zz[:, removed], cc[removed] = 0., 0.
            independent = np.sum(zz*cc, axis=1)/(np.sqrt(np.sum(zz**2, axis=1))*np.linalg.norm(cc))
            max_score_error = max(max_score_error, float(np.max(np.abs(independent-score))))
            result = metrics(y, score)
            max_metric_error = max(max_metric_error, abs(independent_eer(y, independent)-result["eer_percent"]),
                                   abs(independent_auc(y, independent)-result["auroc"]))
            local[name] = score
            records.append({"dataset": dataset, "case": name, "kept": len(kept), "removed": 160-len(kept),
                            **result, "EER_impairment_pp": result["eer_percent"]-metrics(y, native)["eer_percent"],
                            "fixed_V_threshold": threshold, "bona_false_reject_percent": float(100*np.mean(score[y == 1] < threshold)),
                            "spoof_false_accept_percent": float(100*np.mean(score[y == 0] >= threshold)),
                            "mean_abs_score_change": float(np.abs(score-native).mean()),
                            "max_abs_score_change": float(np.abs(score-native).max()),
                            "spearman_native": float(spearmanr(score, native).statistic),
                            "centroid_energy_retained": float(np.sum(c[kept]**2))})
        frame = pd.DataFrame([r for r in records if r["dataset"] == dataset]).set_index("case")
        for k in KEEP_COUNTS:
            random = frame.loc[[f"keep_random{k}_s{seed}" for seed in range(200, 220)]]
            curves.append({"dataset": dataset, "kept": k, "native_EER": float(frame.loc["native", "eer_percent"]),
                           "top_EER": float(frame.loc[f"keep_top{k}", "eer_percent"]),
                           "bottom_EER": float(frame.loc[f"keep_bottom{k}", "eer_percent"]),
                           "random_EER_min": float(random.eer_percent.min()), "random_EER_median": float(random.eer_percent.median()),
                           "random_EER_max": float(random.eer_percent.max()),
                           "native_BFR": float(frame.loc["native", "bona_false_reject_percent"]),
                           "top_BFR": float(frame.loc[f"keep_top{k}", "bona_false_reject_percent"]),
                           "bottom_BFR": float(frame.loc[f"keep_bottom{k}", "bona_false_reject_percent"]),
                           "random_BFR_min": float(random.bona_false_reject_percent.min()),
                           "random_BFR_median": float(random.bona_false_reject_percent.median()),
                           "random_BFR_max": float(random.bona_false_reject_percent.max()), "random_count": len(random)})
        if dataset == "DF_pilot":
            df_scores, df_data = local, data
            saved = pd.concat([data.rows[["utterance_id", "speaker_group", "y"]].reset_index(drop=True), pd.DataFrame(local)], axis=1)
            saved.to_parquet(out / "DF_scores.parquet", index=False)
            previous = pd.read_parquet(OLD / "DF_masked_cosine_scores.parquet")
            assert np.array_equal(previous.utterance_id, data.rows.utterance_id)
            compare = {"keep_bottom80": "drop_top80"}
            compare.update({f"keep_random80_s{seed}": f"drop_random80_s{seed}" for seed in range(200, 220)})
            for new, old in compare.items():
                previous_reproduction_error = max(previous_reproduction_error, float(np.max(np.abs(local[new]-previous[old].to_numpy()))))
        print("sweep dataset complete", dataset, flush=True)
    assert max_score_error < 1e-12 and max_metric_error < 1e-9 and previous_reproduction_error < 1e-12
    pd.DataFrame(records).to_csv(out / "all_metrics.csv", index=False)
    pd.DataFrame(curves).to_csv(out / "curves.csv", index=False)
    print("independent score / metric checks and previous80 reproduction passed", flush=True)
    y, groups = df_data.rows.y.to_numpy(), df_data.rows.speaker_group.to_numpy()
    speakers, codes = np.unique(groups, return_inverse=True)
    draws = np.random.default_rng(7301).multinomial(len(speakers), np.full(len(speakers), 1/len(speakers)), size=2000)
    selected = ["native"]+[f"keep_{family}{k}" for k in KEEP_COUNTS for family in ["top", "bottom"]]
    bootstrap = {}
    for name in selected:
        prepared = weighted_eer_setup(y, df_scores[name], codes)
        assert abs(weighted_eer(prepared, np.ones(len(speakers)))-metrics(y, df_scores[name])["eer_percent"]) < 1e-9
        bootstrap[name] = np.asarray([weighted_eer(prepared, count) for count in draws])
    intervals = []
    for name in selected[1:]:
        delta = bootstrap[name]-bootstrap["native"]
        good = np.isfinite(delta)
        intervals.append({"case": name, "EER_impairment_pp": metrics(y, df_scores[name])["eer_percent"]-metrics(y, df_scores["native"])["eer_percent"],
                          "CI95_pointwise": np.quantile(delta[good], [.025, .975]).tolist(),
                          "groups": len(speakers), "dropped_replicates": int((~good).sum())})
    save(out / "paired_intervals.json", intervals)
    audit = {"status": "pass", "datasets_validated": checks, "masks_per_dataset": len(masks),
             "metrics_independently_checked": len(records), "score_max_error": max_score_error,
             "metric_max_error": max_metric_error, "previous80_cases_reproduced": 21,
             "previous80_max_score_error": previous_reproduction_error,
             "T_Fisher_ranking_recomputed_matches": True, "nested_masks_checked": True,
             "protocol_sha256": digest(out / "protocol_lock.json"), "Q_opened": False, "GPU_inference_runs": 0}
    save(out / "verification.json", audit)
    plot_results(out, curves)
    summary = {"status": "complete_exploratory", "protocol_sha256": audit["protocol_sha256"],
               "DF_curves": [r for r in curves if r["dataset"] == "DF_pilot"], "DF_intervals": intervals,
               "elapsed_seconds": time.monotonic()-started, "GPU_inference_runs": 0, "EBM_retrained": False,
               "Q_opened": False, "limits": lock["limits"]}
    save(out / "summary.json", summary)
    (out / "complete").write_text("0\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
