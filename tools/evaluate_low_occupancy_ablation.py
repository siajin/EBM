"""Offline whole-axis ablation of archived EBM residuals. No model updates."""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "2"
from pathlib import Path
import json
import time
import hashlib
import numpy as np
import pandas as pd
from scipy.linalg import null_space, eigh
from sklearn.metrics import roc_curve
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "artifacts/geometry_cache"
OUT = CACHE / "local_analysis/low_occupancy_ablation_20261111_v1"
SHA = "104873703a57cad98b686d770e0860e031a2cf9abef79b72668780c1660e0b24"
KS = [0, 1, 2, 4, 8, 16, 32, 64, 96, 128, 144, 152, 158, 159]
FIXED = .5093649528766109
HASHES = {}
STARTED = None


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def check(path, expected=None):
    path = Path(path).resolve()
    actual = digest(path)
    assert expected is None or actual == expected, str(path)
    HASHES[str(path)] = actual


def clock_check():
    assert time.monotonic() - STARTED < 1800, "Local CPU time ceiling reached"


def load_sources(catalog, keys):
    frames, shards = [], []
    for key in keys:
        entry = catalog["runs"][key]
        assert entry["status"] == "ready" and entry["identity"]["checkpoint_sha256"] == SHA
        folder = CACHE / entry["local_folder"]
        wanted = ["extract/scores.parquet", "input_audit.json", "extract/summary.json"]
        wanted += [f"extract/rank{rank}_{kind}" for rank in (0, 1) for kind in ("z.npy", "rows.parquet", "centroid.npy")]
        for item in entry["files"]:
            if item["path"] in wanted:
                check(folder / item["path"], item["sha256"])
        frame = pd.read_parquet(folder / "extract/scores.parquet")
        assert len(frame) == entry["sample_count"] and frame.utterance_id.is_unique
        cursor = 0
        for rank in (0, 1):
            rows = pd.read_parquet(folder / f"extract/rank{rank}_rows.parquet")
            part = frame.iloc[cursor:cursor + len(rows)]
            assert rows.utterance_id.tolist() == part.utterance_id.tolist()
            assert np.array_equal(rows.y, part.y)
            assert np.array_equal(part.embedding_row, np.arange(len(part)))
            centroid = np.load(folder / f"extract/rank{rank}_centroid.npy")
            shards.append((len(pd.concat(frames)) + cursor if frames else cursor,
                           folder / f"extract/rank{rank}_z.npy", centroid))
            cursor += len(rows)
        assert cursor == len(frame)
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    assert frame.utterance_id.is_unique
    return frame, shards


def eer_and_dev_threshold(y, score):
    fp, tp, threshold = roc_curve(y, score, drop_intermediate=False)
    fn = 1 - tp
    d = fp - fn
    j = int(np.flatnonzero(d >= 0)[0])
    if d[j] == 0:
        eer = fp[j]
    else:
        t = -d[j - 1] / (d[j] - d[j - 1])
        eer = fp[j - 1] + t * (fp[j] - fp[j - 1])
    finite = np.isfinite(threshold)
    # Same locked V-only operating rule as the preceding analysis.
    candidates = np.flatnonzero(finite)
    order = np.lexsort((-threshold[finite], fp[finite], abs(d[finite])))
    return float(100 * eer), float(threshold[candidates[order[0]]])


def operating_rates(y, score, threshold):
    normal = y == 1
    return {"normal_reject_percent": float(100 * np.mean(score[normal] < threshold)),
            "spoof_accept_percent": float(100 * np.mean(score[~normal] >= threshold))}


def transform_scores(frame, shards, c, basis):
    n = len(frame)
    scores = np.empty((len(KS), n), dtype=float)
    removed = np.empty_like(scores)
    max_baseline_error, max_explicit_error = 0., 0.
    for offset, path, actual_c in shards:
        assert np.array_equal(c, actual_c)
        z = np.load(path, mmap_mode="r")
        assert z.shape[1] == 160
        for first in range(0, len(z), 20000):
            clock_check()
            x = np.array(z[first:first + 20000], dtype=float)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            s = x @ c
            index = slice(offset + first, offset + first + len(x))
            max_baseline_error = max(max_baseline_error, float(np.max(abs(s - frame.cosine_s.to_numpy()[index]))))
            coeff = x @ basis
            squared = coeff ** 2
            total = squared.sum(axis=1)
            assert np.max(abs(total - (1 - s * s))) < 1e-9
            cumulative = np.cumsum(squared, axis=1)
            for ki, k in enumerate(KS):
                energy = np.zeros(len(x)) if k == 0 else cumulative[:, k - 1]
                remaining = np.maximum(total - energy, 0)
                if k == 0:
                    transformed = s
                elif k == 159:
                    transformed = np.sign(s)  # Exact tied endpoint: only OC-axis sign remains.
                else:
                    transformed = s / np.sqrt(s * s + remaining)
                scores[ki, index] = transformed
                removed[ki, index] = energy
                # Independent explicit zeroing/reconstruction, including renormalization.
                if first == 0 and k in (1, 16, 64, 158):
                    sample = x[:16] - coeff[:16, :k] @ basis[:, :k].T
                    direct = sample @ c / np.linalg.norm(sample, axis=1)
                    max_explicit_error = max(max_explicit_error, float(np.max(abs(direct - transformed[:16]))))
            assert np.max(abs((x[:32] - (x[:32] @ basis[:, :16]) @ basis[:, :16].T) @ c - s[:32])) < 1e-10
    assert max_baseline_error < 1e-10 and max_explicit_error < 1e-8
    assert np.isfinite(scores).all() and np.max(abs(scores)) <= 1 + 1e-10
    assert np.all(np.sign(scores) == np.sign(scores[0]))
    return scores, removed, {"max_baseline_score_error": max_baseline_error,
                             "max_explicit_projection_score_error": max_explicit_error,
                             "without_renormalization_OC_numerator_unchanged": True,
                             "all_score_signs_preserved": True}


def render(metrics, spectrum):
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 10})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout="constrained")
    for ax, dataset in zip(axes, ("19LA", "21LA", "DF")):
        f = metrics[metrics.dataset == dataset]
        for method, label, color, style in [("low_energy_first", "적은 점유 축부터", "#356AA0", "-"),
                                             ("high_energy_first", "많은 점유 축부터 (대조)", "#DB8D32", "--")]:
            b = f[f.method == method].sort_values("k")
            ax.plot(b.k, b.eer_percent, style, marker="o", markersize=3, color=color, label=label)
        random = f[f.method.str.startswith("random_")].groupby("k").eer_percent.agg(["mean", "min", "max"])
        ax.plot(random.index, random["mean"], color="#7B838B", linestyle=":", label="무작위 부분공간 3회 평균")
        ax.fill_between(random.index, random["min"], random["max"], color="#7B838B", alpha=.16)
        ax.set(title=f"{dataset} 전체 평가", xlabel="제거한 잔여 차원 수 (전체 159)", ylabel="EER (%)", xlim=(0, 159))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle("Train 정상 방향 점유율로 정한 차원 제거: 보관 batch160 · epoch50", fontsize=13)
    fig.savefig(OUT / "eer_dimension_sweep.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    axes[0].plot(np.arange(1, 160), spectrum, color="#356AA0")
    axes[0].set(xlabel="적은 점유 순서의 잔여 축", ylabel="Train 정상 방향의 평균 제곱 투영", yscale="log")
    axes[1].plot(np.arange(160), np.r_[0, np.cumsum(spectrum)] * 100, color="#356AA0")
    axes[1].set(xlabel="제거한 잔여 차원 수", ylabel="제거되는 Train 정상 방향 에너지 (%)", ylim=(0, 100))
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
    fig.suptitle("원점 보존 · 평균을 빼지 않은 Train 정상 잔여 방향의 점유율")
    fig.savefig(OUT / "train_direction_spectrum.png", dpi=180)
    plt.close(fig)


def main():
    global STARTED
    STARTED = time.monotonic()
    assert (OUT / "plan.json").exists() and not (OUT / "results.json").exists()
    protected = {str(p): digest(p) for p in (ROOT / "EBM").rglob("*.py") if "__pycache__" not in p.parts}
    save("existing_sources_before.json", protected)
    check(CACHE / "catalog.json")
    catalog = json.loads((CACHE / "catalog.json").read_text())
    tv, tvshards = load_sources(catalog, ["19la_tv"])
    c = tvshards[0][2]
    assert abs(np.linalg.norm(c) - 1) < 1e-10
    train_vectors = []
    train_ids = []
    for offset, path, actual_c in tvshards:
        assert np.array_equal(c, actual_c)
        z = np.load(path, mmap_mode="r")
        part = tv.iloc[offset:offset + len(z)]
        select = (part.split.to_numpy() == "T") & (part.y.to_numpy() == 1)
        x = np.array(z[select], dtype=float)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        r = x - (x @ c)[:, None] * c
        length = np.linalg.norm(r, axis=1)
        assert (length >= 1e-4).all()
        train_vectors.append(r / length[:, None])
        train_ids.extend(part.loc[select, "utterance_id"].tolist())
    train_u = np.concatenate(train_vectors)
    assert len(train_u) == len(set(train_ids)) == 2580
    tangent = null_space(c[None, :])
    train_coeff = train_u @ tangent
    values, vec = eigh(train_coeff.T @ train_coeff / len(train_coeff))
    assert values.min() > 0 and abs(values.sum() - 1) < 1e-10
    low = tangent @ vec
    bases = {"low_energy_first": low, "high_energy_first": low[:, ::-1]}
    for seed in (20261111, 20261112, 20261113):
        q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(159, 159)))
        bases[f"random_{seed}"] = tangent @ q
    for b in bases.values():
        assert np.max(abs(b.T @ b - np.eye(159))) < 1e-10
        assert np.max(abs(b.T @ c)) < 1e-10
    np.savez(OUT / "train_only_bases.npz", eigenvalues_low_first=values, centroid=c, **bases)
    pd.DataFrame({"utterance_id": train_ids}).to_parquet(OUT / "fit_train_UIDs.parquet", index=False)
    pd.DataFrame({"axis_low_rank": np.arange(1, 160), "directional_second_moment": values,
                  "cumulative_removed": np.cumsum(values)}).to_csv(OUT / "train_axis_occupancy.csv", index=False)
    print("Train-only 159-axis ranking fixed. Evaluating V thresholds first.", flush=True)
    metrics, groups, validation, thresholds = [], [], [], {}
    cached_tv = {}
    for method, basis in bases.items():
        scores, removed, checks = transform_scores(tv, tvshards, c, basis)
        validation.append({"dataset": "TV", "method": method, **checks})
        normal_train = (tv.split.to_numpy() == "T") & (tv.y.to_numpy() == 1)
        select_v = tv.split.to_numpy() == "V"
        for ki, k in enumerate(KS):
            eer, threshold = eer_and_dev_threshold(tv.y.to_numpy()[select_v], scores[ki, select_v])
            thresholds[(method, k)] = threshold
            metrics.append({"dataset": "V", "method": method, "k": k, "n": int(select_v.sum()),
                            "normal_n": int(tv.y.to_numpy()[select_v].sum()), "eer_percent": eer,
                            "dev_threshold": threshold, "fixed_threshold": FIXED,
                            **{"fixed_" + key: value for key, value in operating_rates(tv.y.to_numpy()[select_v], scores[ki, select_v], FIXED).items()},
                            **{"dev_" + key: value for key, value in operating_rates(tv.y.to_numpy()[select_v], scores[ki, select_v], threshold).items()},
                            "train_normal_direction_energy_removed": float(np.sum((train_u @ basis[:, :k]) ** 2) / len(train_u))})
        cached_tv[method] = scores[:, select_v].copy()
    np.savez(OUT / "V_scores.npz", y=tv.y.to_numpy()[tv.split.to_numpy() == "V"], **cached_tv)
    save("dev_thresholds.json", [{"method": m, "k": k, "threshold": v} for (m, k), v in thresholds.items()])
    del scores, removed, cached_tv
    targets = [("19LA", ["frozen_19la_eval"], 71237), ("21LA", ["21la_eval"], 148176),
               ("DF", ["21df_pilot", "frozen_DF_remaining"], 533928)]
    for dataset, keys, expected_n in targets:
        frame, shards = load_sources(catalog, keys)
        assert len(frame) == expected_n
        y = frame.y.to_numpy()
        baseline_eer = None
        exported = {}
        for method, basis in bases.items():
            clock_check()
            print(f"{dataset}: {method}, {len(frame):,} rows", flush=True)
            scores, removed, checks = transform_scores(frame, shards, c, basis)
            validation.append({"dataset": dataset, "method": method, **checks})
            if method == "low_energy_first":
                exported = {"utterance_id": frame.utterance_id.to_numpy(), "y": y}
                for ki, k in enumerate(KS):
                    exported[f"score_k{k}"] = scores[ki]
                pd.DataFrame(exported).to_parquet(OUT / f"{dataset}_primary_scores.parquet", index=False)
                del exported
            for ki, k in enumerate(KS):
                eer, _ = eer_and_dev_threshold(y, scores[ki])
                if k == 0:
                    if baseline_eer is None:
                        baseline_eer = eer
                    assert abs(eer - baseline_eer) < 1e-12
                threshold = thresholds[(method, k)]
                metrics.append({"dataset": dataset, "method": method, "k": k, "n": len(frame),
                                "normal_n": int(y.sum()), "eer_percent": eer,
                                "dev_threshold": threshold, "fixed_threshold": FIXED,
                                **{"fixed_" + key: value for key, value in operating_rates(y, scores[ki], FIXED).items()},
                                **{"dev_" + key: value for key, value in operating_rates(y, scores[ki], threshold).items()},
                                "normal_embedding_energy_removed": float(removed[ki, y == 1].mean()),
                                "spoof_embedding_energy_removed": float(removed[ki, y == 0].mean()),
                                "mean_abs_score_change": float(abs(scores[ki] - scores[0]).mean())})
                # Keep every attack/codec/source; settings are never chosen by these rows.
                if method == "low_energy_first":
                    for column in ("attack_id", "codec", "source_corpus"):
                        if column not in frame:
                            continue
                        for value, idx in frame.groupby(column, dropna=False).indices.items():
                            isspoof = y[idx] == 0
                            if not isspoof.any():
                                continue
                            ss = scores[ki, idx][isspoof]
                            base = scores[0, idx][isspoof]
                            groups.append({"dataset": dataset, "k": k, "grouping": column, "group": str(value),
                                           "spoof_n": len(ss), "fixed_spoof_accept_percent": float(100 * np.mean(ss >= FIXED)),
                                           "dev_spoof_accept_percent": float(100 * np.mean(ss >= threshold)),
                                           "mean_abs_score_change": float(np.mean(abs(ss - base)))})
            del scores, removed
        pd.DataFrame(metrics).to_csv(OUT / "metrics_progress.csv", index=False)
        print(f"{dataset}: all methods finished; baseline EER={baseline_eer:.6f}%", flush=True)
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(OUT / "metrics.csv", index=False)
    pd.DataFrame(groups).to_csv(OUT / "attack_source_codec_diagnostics.csv", index=False)
    render(metrics, values)
    for path, expected in HASHES.items():
        assert digest(path) == expected
    for path, expected in protected.items():
        assert digest(path) == expected
    save("input_hashes.json", HASHES)
    result = {"status": "complete", "checkpoint_sha256": SHA, "model": "archived batch160 epoch50",
              "basis_fit_normal_T_n": len(train_u), "basis_centered": False,
              "basis_criterion": "ascending E[(unit tangent direction dot axis)^2]",
              "k_grid": KS, "methods": list(bases), "metrics_rows": len(metrics),
              "checks": validation, "input_hashes_unchanged": len(HASHES),
              "existing_python_sources_unchanged": len(protected),
              "safety": {"training_changes": 0, "model_updates": 0, "new_inferences": 0,
                         "network_calls": 0, "remote_operations": 0, "new_spend_usd": 0, "Q_opened": False},
              "elapsed_seconds": time.monotonic() - STARTED,
              "interpretation_limits": ["All eval sweeps are exploratory, not unseen confirmation.",
                                        "Train low occupancy is not attack-independent irrelevance.",
                                        "Post-hoc projection is not equivalent to DTRS training.",
                                        "Without renormalization an exactly tangent projection leaves OC dot scores unchanged.",
                                        "Deleting all tangent axes leaves sign-only tied scores.",
                                        "V is used for threshold calibration, not independent validation."]}
    save("results.json", result)
    print(json.dumps({"status": "complete", "seconds": result["elapsed_seconds"], "metrics_rows": len(metrics)}), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
