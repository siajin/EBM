"""Independent tie-aware ROC and direct-vector verification of offline ablation."""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "2"
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "artifacts/geometry_cache"
OUT = CACHE / "local_analysis/low_occupancy_ablation_20261111_v1"


def eer_independent(y, score):
    order = np.argsort(-score, kind="stable")
    values, positive = score[order], y[order]
    ends = np.r_[np.flatnonzero(values[1:] != values[:-1]), len(values) - 1]
    accepted_normal = np.r_[0, np.cumsum(positive)[ends]]
    accepted_spoof = np.r_[0, np.cumsum(1 - positive)[ends]]
    fp = accepted_spoof / (len(y) - y.sum())
    fn = 1 - accepted_normal / y.sum()
    difference = fp - fn
    right = int(np.searchsorted(difference, 0))
    if difference[right] == 0:
        return float(100 * fp[right])
    left = right - 1
    weight = -difference[left] / (difference[right] - difference[left])
    return float(100 * ((1 - weight) * fp[left] + weight * fp[right]))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    results = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    assert results["status"] == "complete"
    # Explicit tied and perfectly separated boundary cases.
    assert eer_independent(np.array([1, 0]), np.array([1., 0.])) == 0
    assert eer_independent(np.array([1, 0]), np.array([1., 1.])) == 50
    assert eer_independent(np.array([1, 0]), np.array([0., 1.])) == 100
    metric = pd.read_csv(OUT / "metrics.csv", float_precision="round_trip")
    archive = np.load(OUT / "train_only_bases.npz")
    low, c = archive["low_energy_first"], archive["centroid"]
    assert np.max(abs(low.T @ low - np.eye(159))) < 1e-10
    assert np.max(abs(low.T @ c)) < 1e-10
    assert (np.diff(archive["eigenvalues_low_first"]) >= 0).all()
    catalog = json.loads((CACHE / "catalog.json").read_text())
    max_eer_error, max_vector_error = 0., 0.
    full_checks, direct_checks = [], []
    specs = [("19LA", ["frozen_19la_eval"]), ("21LA", ["21la_eval"]),
             ("DF", ["21df_pilot", "frozen_DF_remaining"])]
    for dataset, keys in specs:
        saved = pd.read_parquet(OUT / f"{dataset}_primary_scores.parquet")
        y = saved.y.to_numpy()
        rows = metric[(metric.dataset == dataset) & (metric.method == "low_energy_first")]
        baseline = saved.score_k0.to_numpy()
        for row in rows.itertuples():
            score = saved[f"score_k{row.k}"].to_numpy()
            delta = abs(eer_independent(y, score) - row.eer_percent)
            max_eer_error = max(max_eer_error, delta)
            assert delta < 1e-10
            assert np.array_equal(np.sign(score), np.sign(baseline))
            for threshold, prefix in [(row.fixed_threshold, "fixed"), (row.dev_threshold, "dev")]:
                normal_error = float(100 * np.mean(score[y == 1] < threshold))
                spoof_error = float(100 * np.mean(score[y == 0] >= threshold))
                assert abs(normal_error - getattr(row, prefix + "_normal_reject_percent")) < 1e-10
                assert abs(spoof_error - getattr(row, prefix + "_spoof_accept_percent")) < 1e-10
        assert np.array_equal(saved.score_k159, np.sign(baseline))
        full_checks.append({"dataset": dataset, "n": len(saved), "normal_n": int(y.sum()), "k_conditions": len(rows)})
        rng = np.random.default_rng(20261111)
        sample = saved.iloc[rng.choice(len(saved), 128, replace=False)]
        observed = set()
        for key in keys:
            folder = CACHE / catalog["runs"][key]["local_folder"] / "extract"
            for rank in (0, 1):
                meta = pd.read_parquet(folder / f"rank{rank}_rows.parquet")
                index = pd.Index(meta.utterance_id).get_indexer(sample.utterance_id)
                keep = index >= 0
                if not keep.any():
                    continue
                z = np.load(folder / f"rank{rank}_z.npy", mmap_mode="r")[index[keep]].astype(float)
                x = z / np.linalg.norm(z, axis=1, keepdims=True)
                observed.update(sample.loc[keep, "utterance_id"])
                for k in (0, 1, 16, 64, 128, 158):
                    changed = x - (x @ low[:, :k]) @ low[:, :k].T
                    direct = changed @ c / np.linalg.norm(changed, axis=1)
                    expected = sample.loc[keep, f"score_k{k}"].to_numpy()
                    error = float(np.max(abs(direct - expected)))
                    max_vector_error = max(max_vector_error, error)
                    assert error < 1e-8
        assert len(observed) == len(sample)
        direct_checks.append({"dataset": dataset, "sample_n": len(sample), "k": [0, 1, 16, 64, 128, 158]})
    # All methods must collapse to the same sign-only endpoint and baseline.
    for dataset, group in metric.groupby("dataset"):
        assert group[group.k == 0].eer_percent.max() - group[group.k == 0].eer_percent.min() < 1e-10
        assert group[group.k == 159].eer_percent.max() - group[group.k == 159].eer_percent.min() < 1e-10
    for path, expected in json.loads((OUT / "input_hashes.json").read_text()).items():
        assert digest(path) == expected
    for path, expected in json.loads((OUT / "existing_sources_before.json").read_text()).items():
        assert digest(path) == expected
    checks = {"status": "passed", "independent_tie_aware_EER_max_error_percentage_points": max_eer_error,
              "explicit_vector_projection_max_score_error": max_vector_error,
              "all_primary_rows_metrics_verified": full_checks, "direct_raw_embedding_sample_checks": direct_checks,
              "boundary_cases": ["perfect", "fully tied", "reversed", "all tangent axes removed gives sign-only ties"],
              "source_and_cache_hashes_unchanged": True, "training_modified": False, "new_spend_usd": 0}
    (OUT / "independent_validation.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(checks, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
