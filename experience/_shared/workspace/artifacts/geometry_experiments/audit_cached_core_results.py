"""Verify saved core results and add a descriptive fixed-threshold mask audit.

This reads the original completed outputs. It never refits or changes a score.
The extra threshold analysis is explicitly posthoc and selects no mask.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import det_curve, log_loss
from threadpoolctl import threadpool_limits

from geometry_cache import DEFAULT_ROOT, load_embeddings, resolve_run


def independent_eer(y, score):
    fp, fn, _ = det_curve(y, score)
    # DET output runs in descending false-positive rate; reverse before crossing.
    fp, fn = fp[::-1], fn[::-1]
    d = fp-fn
    crossing = np.flatnonzero(d >= 0)
    if not len(crossing):
        # sklearn can trim a perfect classifier to the single (0,0) point.
        assert np.all(fp == 0) and np.all(fn == 0)
        return 0.
    hi = int(crossing[0])
    if hi == 0:
        return float(50*(fp[0]+fn[0]))
    t = -d[hi-1]/(d[hi]-d[hi-1])
    return float(100*(fp[hi-1]+t*(fp[hi]-fp[hi-1])))


def independent_auc(y, score):
    n1, n0 = int(y.sum()), int((1-y).sum())
    return float((rankdata(score)[y == 1].sum()-n1*(n1+1)/2)/(n1*n0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_ROOT / "local_analysis/cached_core_v1")
    args = parser.parse_args()
    out = args.out
    assert (out / "complete").read_text().strip() == "0"
    summary = json.loads((out / "summary.json").read_text())
    lock = json.loads((out / "protocol_lock.json").read_text())
    script = Path(__file__).with_name("cached_core_diagnostics.py")
    if (out / "recovery.json").exists():
        recovery = json.loads((out / "recovery.json").read_text())
        assert hashlib.sha256((out / "runner_at_lock.py").read_bytes()).hexdigest() == lock["script_sha256"]
        if (out / "statistical_label_correction.json").exists():
            correction = json.loads((out / "statistical_label_correction.json").read_text())
            assert hashlib.sha256((out / "runner_after_matching_fix.py").read_bytes()).hexdigest() == recovery["recovered_script_sha256"]
            assert hashlib.sha256(script.read_bytes()).hexdigest() == correction["final_script_sha256"]
            assert correction["numeric_values_changed"] is False
        else:
            assert hashlib.sha256(script.read_bytes()).hexdigest() == recovery["recovered_script_sha256"]
        assert hashlib.sha256((out / "DF_nested_OOF_predictions.parquet").read_bytes()).hexdigest() == recovery["reused_predictions_sha256"]
        assert recovery["settings_changed"] is False and recovery["probe_refits"] == 0
    else:
        assert hashlib.sha256(script.read_bytes()).hexdigest() == lock["script_sha256"]
    assert hashlib.sha256((out / "protocol_lock.json").read_bytes()).hexdigest() == summary["protocol_sha256"]
    frame = pd.read_parquet(out / "DF_nested_OOF_predictions.parquet")
    data = load_embeddings("DF_pilot")
    rows = data.rows
    assert np.array_equal(frame.utterance_id, rows.utterance_id)
    assert np.array_equal(frame.y, rows.y)
    assert frame.groupby("speaker_group").fold.nunique().max() == 1
    assert set(frame.fold) == set(range(5))
    y = frame.y.to_numpy()
    errors = []
    for record in summary["probe_metrics"]:
        case = record["case"]
        p, score = frame[case].to_numpy(), frame[case+"_logit"].to_numpy()
        weight = np.where(y == 1, .5/(y == 1).sum(), .5/(y == 0).sum())
        ll = log_loss(y, np.clip(p, 1e-7, 1-1e-7), sample_weight=weight)
        auc = independent_auc(y, score)
        eer = independent_eer(y, score)
        errors.append({"case": case, "ll_error": abs(ll-record["balanced_logloss"]),
                       "auc_error": abs(auc-record["auroc"]), "eer_error": abs(eer-record["eer_percent"])})
    assert max(max(r[k] for k in ["ll_error", "auc_error", "eer_error"]) for r in errors) < 1e-9
    masks = json.loads((out / "channel_masks.json").read_text())["masks_removed_indices"]
    saved_scores = pd.read_parquet(out / "DF_masked_cosine_scores.parquet")
    assert np.array_equal(saved_scores.utterance_id, rows.utterance_id)
    folder, _, _ = resolve_run("DF_pilot")
    registry = json.loads((folder / "registry.json").read_text())
    threshold = registry["thresholds"]["native_cosine"]["threshold"]
    supplemental = []
    max_mask_error = 0.
    mask_metrics = pd.read_csv(out / "channel_ablation_metrics.csv")
    for dataset in ["T", "V", "DF_pilot"]:
        current = load_embeddings(dataset)
        current_y = current.rows.y.to_numpy()
        z, c = current.z.astype(float), current.centroid
        native = current.rows.cosine_s.to_numpy()
        for case, indices in masks.items():
            # Independent full zero-mask formula (runner slices retained channels).
            zero_z, zero_c = z.copy(), c.copy()
            zero_z[:, indices] = 0
            zero_c[indices] = 0
            score = np.sum(zero_z*zero_c, axis=1)/(np.sqrt(np.sum(zero_z**2, axis=1))*np.linalg.norm(zero_c))
            reference = mask_metrics[(mask_metrics.dataset == dataset) & (mask_metrics.case == case)].iloc[0]
            assert abs(independent_eer(current_y, score)-reference.eer_percent) < 1e-9
            assert abs(independent_auc(current_y, score)-reference.auroc) < 1e-9
            if dataset == "DF_pilot":
                error = float(np.max(np.abs(score-saved_scores[case].to_numpy())))
                max_mask_error = max(max_mask_error, error)
                assert error < 1e-12
            if dataset != "T":
                # Preserve the archived identity score exactly at its V threshold.
                # Equivalent summation orders can move that equality by ~1e-16.
                if case == "native":
                    score = native.copy()
                supplemental.append({"dataset": dataset, "case": case, "removed": len(indices),
                                     "fixed_native_V_threshold": threshold,
                                     "bona_false_reject_percent": float(100*np.mean(score[current_y == 1] < threshold)),
                                     "spoof_false_accept_percent": float(100*np.mean(score[current_y == 0] >= threshold)),
                                     "mean_abs_score_change": float(np.abs(score-native).mean()),
                                     "max_abs_score_change": float(np.abs(score-native).max()),
                                     "spearman_native": float(spearmanr(score, native).statistic),
                                     "within_1e12_of_threshold": int((np.abs(score-threshold) < 1e-12).sum()),
                                     "centroid_energy_removed": float(np.sum(c[indices]**2))})
    pd.DataFrame(supplemental).to_csv(out / "posthoc_mask_fixed_threshold.csv", index=False)
    # Pair constraints are independently checked against the cached metadata.
    for definition in summary["matched"]:
        if not definition["pairs"]:
            continue
        pairs = pd.read_parquet(out / f"matched_{definition['name']}_pairs.parquet")
        a, b = pairs.bona_row.to_numpy(), pairs.spoof_row.to_numpy()
        assert len(a) == definition["pairs"] and len(set(a)) == len(a) and len(set(b)) == len(b)
        assert np.all(y[a] == 1) and np.all(y[b] == 0)
        for field in definition["strata"]:
            assert np.array_equal(rows[field].to_numpy()[a], rows[field].to_numpy()[b])
        assert np.max(np.abs(rows.cosine_s.to_numpy()[a]-rows.cosine_s.to_numpy()[b])) <= definition["cosine_caliper"]
        assert np.max(np.abs(np.log(rows.raw_norm.to_numpy()[a])-np.log(rows.raw_norm.to_numpy()[b]))) <= definition["lognorm_caliper"]
    audit = {"status": "pass", "source_and_protocol_hash_match": True, "Q_opened": False,
             "OOF_row_label_alignment": True, "one_fold_per_speaker": True,
             "metrics_independently_checked": errors, "masked_metrics_checked": len(mask_metrics),
             "max_independent_mask_score_error": max_mask_error,
             "matching_constraints_checked": [r["name"] for r in summary["matched"]],
             "posthoc_scope": "all original masks at existing native V threshold; no new selection or refit",
             "threshold_equality": "native baseline uses archived cosine exactly, avoiding threshold equality changes from floating-point summation",
             "CI_levels": "95% pointwise; 99% pointwise supports Bonferroni family95% for the five listed primary comparisons only"}
    (out / "output_audit.json").write_text(json.dumps(audit, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
