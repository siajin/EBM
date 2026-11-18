"""Local, read-only diagnosis of archived embeddings. No torch/network/SSH imports.

This script never constructs an EBM or optimizer. The primary basis is fitted to
Train bona fide residuals only; all target use is descriptive. See locked plan.
"""
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "2"

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.metrics import roc_curve
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SHA = "104873703a57cad98b686d770e0860e031a2cf9abef79b72668780c1660e0b24"
SEED = 20261111
DURATION_EDGES = [0, 1, 2, 3, 4, 6, 10, np.inf]
K_VALUES = [1, 2, 4, 8, 16]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                                     allow_nan=False) + "\n", encoding="utf-8")


def geometry(z, centroid):
    z = np.asarray(z, dtype=np.float64)
    norm = np.linalg.norm(z, axis=1)
    assert np.all(norm > 1e-12) and np.isfinite(z).all()
    x = z / norm[:, None]
    s = x @ centroid
    r = x - s[:, None] * centroid
    a2 = np.einsum("ij,ij->i", r, r)
    valid = a2 >= 1e-8
    u = np.zeros_like(r)
    u[valid] = r[valid] / np.sqrt(a2[valid, None])
    assert np.max(np.abs(r @ centroid), initial=0) < 1e-10
    assert np.max(np.abs(a2 - (1 - s * s)), initial=0) < 1e-10
    return s, r, u, a2, valid


def top_basis(r):
    centered = r - r.mean(axis=0)
    cov = centered.T @ centered / (len(r) - 1)
    values, vectors = eigh(cov)
    values, vectors = np.maximum(values[::-1], 0), vectors[:, ::-1]
    for i in range(vectors.shape[1]):
        if vectors[np.argmax(np.abs(vectors[:, i])), i] < 0:
            vectors[:, i] *= -1
    return values, vectors


def math_checks():
    c = np.array([0., 0., 1.])
    z = np.array([[.6, 0., .8], [0., .6, .8]])
    s, r, u, a2, valid = geometry(z, c)
    assert np.allclose(s, .8) and np.allclose(a2, .36) and valid.all()
    q = (u @ np.array([[1.], [0.], [0.]]))[:, 0] ** 2
    assert np.allclose(q, [1., 0.])
    # Direction ratio is invariant to residual magnitude at fixed direction.
    r2 = r * 2
    assert np.allclose((r2[:, 0] ** 2) / (r2 * r2).sum(1), q)
    # Tangent norm alone cannot distinguish +c from -c.
    s2, _, _, a22, _ = geometry(np.array([[0., 0., 1.], [0., 0., -1.]]), c)
    assert np.array_equal(s2, [1., -1.]) and np.array_equal(a22, [0., 0.])
    return {"same_score_different_direction": "pass", "scale_invariant_ratio": "pass",
            "negative_score_residual_caveat": "pass"}


class Inputs:
    def __init__(self, root):
        self.root = root
        self.catalog = json.loads((root / "catalog.json").read_text(encoding="utf-8"))
        self.hashes = {}
        self.audit = []
        self.centroid = None

    def check_file(self, path, expected=None):
        path = Path(path).resolve()
        if str(path) not in self.hashes:
            actual = digest(path)
            assert expected is None or actual == expected, f"Hash mismatch: {path}"
            self.hashes[str(path)] = actual

    def read_run(self, key):
        entry = self.catalog["runs"][key]
        assert entry["status"] == "ready" and entry["identity"]["checkpoint_sha256"] == SHA
        folder = self.root / entry["local_folder"]
        for item in entry["files"]:
            rel = item["path"]
            if rel in ("input_audit.json", "extract/summary.json", "extract/scores.parquet") or (
                rel.startswith("extract/rank") and any(rel.endswith(suffix) for suffix in
                ("_z.npy", "_centroid.npy", "_rows.parquet", "_model.json"))):
                p = folder / rel
                assert p.stat().st_size == item["bytes"]
                self.check_file(p, item["sha256"])
        audit = json.loads((folder / "input_audit.json").read_text(encoding="utf-8"))
        assert audit["checkpoint_sha256"] == SHA
        summary = json.loads((folder / "extract/summary.json").read_text(encoding="utf-8"))
        assert summary["status"] == "pass"
        rows = pd.read_parquet(folder / "extract/scores.parquet")
        assert len(rows) == entry["sample_count"] == summary["sample_count"]
        assert rows.utterance_id.is_unique and set(rows.y.unique()) == {0, 1}
        normal_parts, z_parts = [], []
        cursor, max_cos_error, max_raw_error = 0, 0., 0.
        for rank in (0, 1):
            shard = pd.read_parquet(folder / f"extract/rank{rank}_rows.parquet")
            selected = rows.iloc[cursor:cursor + len(shard)]
            assert shard.utterance_id.tolist() == selected.utterance_id.tolist()
            assert np.array_equal(shard.y, selected.y)
            assert np.array_equal(selected.embedding_row, np.arange(len(shard)))
            assert (selected["rank"] == rank).all()
            c = np.load(folder / f"extract/rank{rank}_centroid.npy", allow_pickle=False)
            assert c.shape == (160,) and abs(np.linalg.norm(c) - 1) < 1e-10
            if self.centroid is None:
                self.centroid = c
            assert np.array_equal(c, self.centroid)
            z = np.load(folder / f"extract/rank{rank}_z.npy", mmap_mode="r", allow_pickle=False)
            assert z.shape == (len(shard), 160) and z.dtype == np.dtype("float32")
            for first in range(0, len(z), 10000):
                block = np.array(z[first:first + 10000], dtype=np.float64)
                assert np.isfinite(block).all()
                score = block / np.linalg.norm(block, axis=1, keepdims=True) @ c
                record = selected.iloc[first:first + len(block)]
                max_cos_error = max(max_cos_error, np.max(np.abs(score - record.cosine_s)))
                max_raw_error = max(max_raw_error, np.max(np.abs(score - record.raw_score)))
            mask = selected.y.to_numpy() == 1
            nrows = selected.loc[mask].copy()
            nrows["cache_row"] = np.flatnonzero(mask) + cursor
            nrows["cache_key"] = key
            normal_parts.append(nrows)
            z_parts.append(np.array(z[mask], dtype=np.float64))
            cursor += len(shard)
        assert cursor == len(rows) and max_cos_error < 1e-10 and max_raw_error < 1e-5
        normals = pd.concat(normal_parts, ignore_index=True)
        s, r, u, a2, valid = geometry(np.concatenate(z_parts), self.centroid)
        assert np.max(np.abs(s - normals.cosine_s)) < 1e-10
        normals["score"] = s
        normals["residual_energy"] = a2
        normals["valid_direction"] = valid
        for name, default in (("source_corpus", "asvspoof19"), ("codec", "not_recorded")):
            if name not in normals:
                normals[name] = default
            normals[name] = normals[name].fillna(default)
        normals["duration_band"] = pd.cut(normals.duration, DURATION_EDGES, right=False).astype(str)
        assert normals.duration.notna().all() and np.isfinite(normals.duration).all()
        self.audit.append({"run": key, "rows": len(rows), "bona": len(normals),
                           "checkpoint_sha256": SHA, "cache_fingerprint": entry["cache_fingerprint"],
                           "max_cosine_error": float(max_cos_error), "max_raw_score_error": float(max_raw_error)})
        return rows, normals, r, u


def subset(data, mask):
    f, r, u = data
    mask = np.asarray(mask)
    return f.loc[mask].reset_index(drop=True), r[mask], u[mask]


def attach_projections(data, basis, threshold):
    f, r, u = data
    f = f.copy()
    f["error"] = f.score < threshold
    coeff = r @ basis[:, :16]
    directional = u @ basis[:, :16]
    for k in K_VALUES:
        f[f"parallel_energy_{k}"] = np.sum(coeff[:, :k] ** 2, axis=1)
        f[f"rho_{k}"] = np.where(f.valid_direction, np.sum(directional[:, :k] ** 2, axis=1), np.nan)
    for j in range(16):
        f[f"pc{j + 1}_fraction"] = np.where(f.valid_direction, directional[:, j] ** 2, np.nan)
    # Independent full-projector computation, never just the same contraction.
    p = basis[:, :8] @ basis[:, :8].T
    sample = np.arange(min(64, len(r)))
    direct = np.linalg.norm((p @ r[sample].T).T, axis=1) ** 2
    assert np.max(np.abs(direct - f.parallel_energy_8.iloc[sample])) < 1e-10
    good = f.valid_direction.to_numpy()
    assert np.all((f.rho_8.to_numpy()[good] >= -1e-10) & (f.rho_8.to_numpy()[good] <= 1 + 1e-10))
    return f, r, u


def concentration(u):
    n = len(u)
    if n == 0:
        return {"n": 0}
    resultant = np.linalg.norm(u.mean(axis=0))
    moment = u.T @ u / n
    spectrum = np.maximum(eigh(moment, eigvals_only=True), 0)
    return {"n": n, "R": float(resultant),
            "unbiased_pairwise_cosine": float((n * resultant ** 2 - 1) / (n - 1)) if n > 1 else None,
            "second_moment_PR": float(np.trace(moment) ** 2 / np.sum(moment * moment)),
            "second_moment_top1_share": float(spectrum[-1] / spectrum.sum())}


def cluster_difference(frame, strata_columns, seed):
    """Error-weighted within-cell rho difference; resample speakers jointly.

    Covariate support is fixed before bootstrap. Both error and correct means
    are recomputed on each cluster resample, preserving their dependence.
    """
    f = frame.loc[frame.valid_direction].copy()
    f["stratum"] = f[strata_columns].astype(str).agg("|".join, axis=1) if strata_columns else "all"
    cells = f.groupby(["stratum", "error"], observed=True).rho_8.agg(["count", "sum"]).unstack(fill_value=0)
    if True not in cells["count"] or False not in cells["count"]:
        return {"status": "no_error_or_accepted", "error_coverage": 0}
    keep = cells.index[(cells["count"][True] >= 1) & (cells["count"][False] >= 2)]
    f = f[f.stratum.isin(keep)]
    if len(f) == 0:
        return {"status": "no_common_support", "error_coverage": 0}
    groups = sorted(f.speaker_group.unique())
    gi = pd.Categorical(f.speaker_group, categories=groups).codes
    ci = pd.Categorical(f.stratum, categories=keep).codes
    e = f.error.to_numpy()
    sums = []
    for select, val in ((e, np.ones(len(f))), (~e, np.ones(len(f))),
                        (e, f.rho_8.to_numpy()), (~e, f.rho_8.to_numpy())):
        mat = np.zeros((len(groups), len(keep)))
        np.add.at(mat, (gi[select], ci[select]), val[select])
        sums.append(mat)
    ne, nc, se, sc = [x.sum(axis=0) for x in sums]
    estimate = np.sum(se - ne * sc / nc) / ne.sum()
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=2000)
    ne_b, nc_b, se_b, sc_b = [weights @ x for x in sums]
    supported = (ne_b > 0) & (nc_b > 0)
    denominator = (ne_b * supported).sum(axis=1)
    numerator = np.sum(np.where(supported, se_b - ne_b * sc_b / np.maximum(nc_b, 1), 0), axis=1)
    draws = numerator[denominator > 0] / denominator[denominator > 0]
    error_all = int(frame.error.sum())
    return {"status": "ok", "delta_rho": float(estimate), "CI95": np.quantile(draws, [.025, .975]).tolist(),
            "error_n_supported": int(ne.sum()), "accepted_n_supported": int(nc.sum()),
            "error_coverage": float(ne.sum() / error_all) if error_all else 0.,
            "strata": len(keep), "speaker_groups": len(groups),
            "error_speaker_groups": int(f.loc[f.error, "speaker_group"].nunique()),
            "bootstrap_draws_valid": len(draws),
            "sparse_warning": bool(ne.sum() < 20 or f.loc[f.error, "speaker_group"].nunique() < 5)}


def group_summary(f, selection, u):
    selected = f.loc[selection]
    valid = selection & f.valid_direction.to_numpy()
    return {"n": int(selection.sum()), "valid_n": int(valid.sum()),
            "score_mean": float(selected.score.mean()) if len(selected) else None,
            "negative_score_n": int((selected.score < 0).sum()),
            "duration_mean": float(selected.duration.mean()) if len(selected) else None,
            "rho8_mean": float(selected.rho_8.mean()) if valid.any() else None,
            "parallel_energy8_mean": float(selected.parallel_energy_8.mean()) if len(selected) else None,
            "residual_energy_mean": float(selected.residual_energy.mean()) if len(selected) else None,
            "concentration": concentration(u[valid])}


def analyze(name, data, out, seed):
    f, r, u = data
    err = f.error.to_numpy()
    result = {"dataset": name, "bona_n": len(f), "errors": int(err.sum()),
              "error_percent": float(100 * err.mean()),
              "error": group_summary(f, err, u), "accepted": group_summary(f, ~err, u),
              "all": group_summary(f, np.ones(len(f), dtype=bool), u)}
    result["unadjusted"] = cluster_difference(f, [], seed)
    result["duration_source_codec_adjusted"] = cluster_difference(f, ["duration_band", "source_corpus", "codec"], seed + 1)
    result["speaker_duration_source_codec_adjusted"] = cluster_difference(f, ["speaker_group", "duration_band", "source_corpus", "codec"], seed + 2)
    # Equal-size direction concentration comparison, explicitly descriptive:
    # no utterance independence or inferential significance is claimed here.
    ei = np.flatnonzero(err & f.valid_direction.to_numpy())
    ai = np.flatnonzero(~err & f.valid_direction.to_numpy())
    m = min(256, len(ei), len(ai))
    if m >= 5:
        rng = np.random.default_rng(seed + 3)
        rr, aa = [], []
        for _ in range(100):
            rr.append(concentration(u[rng.choice(ei, m, replace=False)]))
            aa.append(concentration(u[rng.choice(ai, m, replace=False)]))
        result["equal_size_direction_diagnostic"] = {"n_per_group": m, "repeats": 100,
            "scope": "row subsamples; descriptive distribution only, not a confidence interval",
            "error_R_median": float(np.median([x["R"] for x in rr])),
            "accepted_R_median": float(np.median([x["R"] for x in aa])),
            "error_PR_median": float(np.median([x["second_moment_PR"] for x in rr])),
            "accepted_PR_median": float(np.median([x["second_moment_PR"] for x in aa]))}
    sensitivity = []
    for k in K_VALUES:
        sensitivity.append({"dataset": name, "k": k, "errors": int(err.sum()),
                            "rho_error": float(f.loc[err, f"rho_{k}"].mean()) if len(ei) else None,
                            "rho_accepted": float(f.loc[~err, f"rho_{k}"].mean()) if len(ai) else None})
    f.to_parquet(out / f"{name}_bona_derived.parquet", index=False)
    dec = f.copy()
    dec["score_decile"] = pd.qcut(dec.score, 10, labels=False, duplicates="drop")
    deciles = dec.groupby("score_decile", observed=True).agg(n=("score", "size"), score_mean=("score", "mean"),
        error_rate=("error", "mean"), rho8=("rho_8", "mean"), residual_energy=("residual_energy", "mean"),
        parallel_energy8=("parallel_energy_8", "mean"), duration=("duration", "mean")).reset_index()
    deciles.insert(0, "dataset", name)
    duration = f.groupby(["duration_band", "error"], observed=True).agg(n=("score", "size"),
        rho8=("rho_8", "mean"), score_mean=("score", "mean")).reset_index()
    duration.insert(0, "dataset", name)
    axes = []
    for j in range(16):
        col = f"pc{j + 1}_fraction"
        axes.append({"dataset": name, "pc": j + 1,
                     "error_fraction": float(f.loc[err, col].mean()) if len(ei) else None,
                     "accepted_fraction": float(f.loc[~err, col].mean()) if len(ai) else None})
    print(json.dumps({"dataset": name, "n": len(f), "errors": int(err.sum()),
                      "rho_error": result["error"]["rho8_mean"], "rho_correct": result["accepted"]["rho8_mean"],
                      "adjusted": result["duration_source_codec_adjusted"]}, ensure_ascii=False), flush=True)
    return result, sensitivity, deciles, duration, axes


def plot(results, deciles, out):
    names = ["V", "DF_pilot", "21LA_pilot", "19LA_eval", "DF_full", "21LA_eval"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    yy = np.arange(len(names))
    for delta, label, color in [(-.17, "Accepted bona fide", "#4776a3"), (.17, "Rejected bona fide", "#cc7546")]:
        key = "accepted" if delta < 0 else "error"
        values = [100 * results[n][key]["rho8_mean"] if results[n][key]["rho8_mean"] is not None
                  else np.nan for n in names]
        ax[0].barh(yy + delta, values, height=.3, label=label, color=color)
        for j, value in enumerate(values):
            if np.isnan(value):
                ax[0].text(1, yy[j] + delta, "No rejected bona fide", va="center", fontsize=8)
    ax[0].set(yticks=yy, yticklabels=names, xlim=(0, 100), xlabel="Energy share in Train top-8 directions (%)",
              title="Direction share, independent of residual length")
    ax[0].invert_yaxis()
    ax[0].legend(fontsize=9, loc="lower right")
    for name, color in [("DF_pilot", "#4776a3"), ("21LA_pilot", "#cc7546"), ("V", "#548f68")]:
        d = deciles[deciles.dataset == name]
        ax[1].plot(d.score_decile + 1, 100 * d.rho8, "o-", color=color, label=name)
    ax[1].set(xlabel="Bona fide score decile (1 = lowest score)", ylabel="Train top-8 direction share (%)",
              ylim=(0, 100), xticks=np.arange(1, 11), title="Does the low-score tail use Train directions?")
    ax[1].legend()
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
        a.grid(axis="x" if a is ax[0] else "y", alpha=.15)
        a.set_axisbelow(True)
    fig.suptitle("Archived EBM batch160 / epoch50: fixed dev threshold; no retraining", fontsize=13)
    fig.savefig(out / "train_direction_share.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
    labels, estimates, lower, upper = [], [], [], []
    for name in names:
        d = results[name]["duration_source_codec_adjusted"]
        if d.get("status") != "ok":
            continue
        labels.append(f"{name} (error coverage {100 * d['error_coverage']:.0f}%)")
        estimates.append(100 * d["delta_rho"])
        lower.append(100 * (d["delta_rho"] - d["CI95"][0]))
        upper.append(100 * (d["CI95"][1] - d["delta_rho"]))
    ax.errorbar(estimates, np.arange(len(labels)), xerr=[np.maximum(lower, 0), np.maximum(upper, 0)],
                fmt="o", capsize=4, color="#4776a3")
    ax.axvline(0, color="#777777", linewidth=1)
    ax.set(yticks=np.arange(len(labels)), yticklabels=labels,
           xlabel="Rejected minus accepted: top-8 direction share (percentage points)",
           title="Duration / source / codec adjusted difference\n95% speaker-cluster bootstrap intervals")
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=.15)
    fig.savefig(out / "adjusted_direction_share.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    cache = workspace / "artifacts/geometry_cache"
    out = cache / "local_analysis/bona_tangent_errors_20261111_v1"
    assert (out / "plan.json").is_file() and not (out / "results.json").exists()
    started = time.monotonic()
    source_snapshot = {str(p.resolve()): digest(p) for p in (workspace / "EBM").rglob("*.py")
                       if ".git" not in p.parts and "__pycache__" not in p.parts}
    save_json(out / "source_hashes_before.json", source_snapshot)
    inputs = Inputs(cache)
    inputs.check_file(cache / "catalog.json")
    checks = math_checks()
    rows_tv, normal_tv, r_tv, u_tv = inputs.read_run("19la_tv")
    train = normal_tv.split.to_numpy() == "T"
    assert train.sum() == 2580 and (~train).sum() == 1351
    values, basis = top_basis(r_tv[train])
    assert np.max(np.abs(basis[:, :16].T @ inputs.centroid)) < 1e-8
    assert np.max(np.abs(basis.T @ basis - np.eye(160))) < 1e-10
    # Independent SVD principal projector check (basis signs are arbitrary).
    centered = r_tv[train] - r_tv[train].mean(axis=0)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    assert np.max(np.abs(values - singular ** 2 / (len(centered) - 1))) < 1e-10
    assert np.max(np.abs(basis[:, :8] @ basis[:, :8].T - vt[:8].T @ vt[:8])) < 1e-8
    np.savez(out / "train_only_basis.npz", eigenvalues=values, basis=basis, centroid=inputs.centroid,
             residual_mean=r_tv[train].mean(axis=0))
    v = rows_tv[rows_tv.split == "V"]
    fpr, tpr, thresholds = roc_curve(v.y, v.cosine_s, drop_intermediate=False)
    fnr = 1 - tpr
    finite = np.isfinite(thresholds)
    order = np.lexsort((-thresholds[finite], fpr[finite], np.abs(fpr[finite] - fnr[finite])))
    index = np.flatnonzero(finite)[order[0]]
    threshold = float(thresholds[index])
    threshold_record = {"threshold": threshold, "V_false_reject_percent": float(100 * fnr[index]),
                        "V_spoof_accept_percent": float(100 * fpr[index]), "V_rows": len(v),
                        "V_bona": int(v.y.sum()), "source": "V only; development resubstitution"}
    save_json(out / "threshold.json", threshold_record)
    data = {"T": subset((normal_tv, r_tv, u_tv), train),
            "V": subset((normal_tv, r_tv, u_tv), ~train)}
    for key, name in [("21df_pilot", "DF_pilot"), ("21la_eval", "21LA_eval"),
                      ("frozen_19la_eval", "19LA_eval"), ("frozen_DF_remaining", "DF_remaining")]:
        _, f, r, u = inputs.read_run(key)
        data[name] = (f, r, u)
    lockfile = cache / "analysis_manifests/frozen_data_sizes_v1/lock.json"
    inputs.check_file(lockfile)
    lock = json.loads(lockfile.read_text(encoding="utf-8"))["views"]["21LA_pilot"]
    manifest = lockfile.parent / lock["manifest_file"]
    inputs.check_file(manifest, lock["manifest_sha256"])
    assert lock["source_cache_fingerprint"] == inputs.catalog["runs"]["21la_eval"]["cache_fingerprint"]
    selected = pd.read_parquet(manifest).query("y == 1")
    pilot_ids = set(selected.utterance_id)
    data["21LA_pilot"] = subset(data["21LA_eval"], data["21LA_eval"][0].utterance_id.isin(pilot_ids))
    assert data["21LA_pilot"][0].utterance_id.nunique() == len(pilot_ids) == 2000
    assert set(data["DF_pilot"][0].utterance_id).isdisjoint(data["DF_remaining"][0].utterance_id)
    data["DF_full"] = (pd.concat([data["DF_pilot"][0], data["DF_remaining"][0]], ignore_index=True),
                       np.concatenate([data["DF_pilot"][1], data["DF_remaining"][1]]),
                       np.concatenate([data["DF_pilot"][2], data["DF_remaining"][2]]))
    assert len(data["DF_full"][0]) == 14869
    results, sensitivities, deciles, durations, axes = {}, [], [], [], []
    for i, name in enumerate(["T", "V", "DF_pilot", "21LA_pilot", "19LA_eval", "DF_full", "21LA_eval"]):
        assert time.monotonic() - started < 1800
        transformed = attach_projections(data[name], basis, threshold)
        result, sens, dec, dur, pcs = analyze(name, transformed, out, SEED + i * 100)
        results[name] = result
        sensitivities.extend(sens)
        deciles.append(dec)
        durations.append(dur)
        axes.extend(pcs)
    deciles = pd.concat(deciles, ignore_index=True)
    deciles.to_csv(out / "score_deciles.csv", index=False)
    pd.concat(durations, ignore_index=True).to_csv(out / "duration_groups.csv", index=False)
    pd.DataFrame(sensitivities).to_csv(out / "k_sensitivity.csv", index=False)
    pd.DataFrame(axes).to_csv(out / "individual_train_axes.csv", index=False)
    save_json(out / "computed_results_before_render.json", results)
    plot(results, deciles, out)
    for p, expected in inputs.hashes.items():
        assert digest(p) == expected, f"Input changed during analysis: {p}"
    for p, expected in source_snapshot.items():
        assert digest(p) == expected, f"Existing Python source changed: {p}"
    payload = {"status": "complete", "completed_utc": datetime.now(timezone.utc).isoformat(),
               "checkpoint_sha256": SHA, "model": "archived batch160 fixed epoch50 EBM",
               "threshold": threshold_record, "primary_k": 8,
               "train_basis": {"bona_n": int(train.sum()), "top8_centered_variance_share": float(values[:8].sum() / values.sum()),
                               "eigenvalues_first16": values[:16].tolist()},
               "datasets": results,
               "limitations": ["Fixed-model descriptive association, not DTRS efficacy or causation",
                   "V used to calibrate threshold and inspect development behavior; no fresh holdout claim",
                   "Pilots overlap full eval sets and are not independent replications",
                   "Speaker grouping is partial proxy for source-family dependency",
                   "Train-only global covariance approximates, but does not reproduce, online DTRS queue",
                   "This batch160 checkpoint is not legacy240/B/current short32; no generalization to those weights assumed",
                   "Original frontend weight identity was not independently verified in archived cache"],
               "safety": {"local_only": True, "GPU_calls": 0, "network_calls": 0, "model_updates": 0,
                          "Q_embeddings_opened": False, "remote_operations": 0, "new_spend_usd": 0},
               "validation": {"math": checks, "source_runs": inputs.audit,
                              "consumed_file_hashes_unchanged": len(inputs.hashes),
                              "existing_python_source_hashes_unchanged": len(source_snapshot),
                              "independent_svd_basis_and_projection": "pass"},
               "elapsed_seconds": time.monotonic() - started}
    save_json(out / "results.json", payload)
    save_json(out / "input_hashes.json", inputs.hashes)
    print(json.dumps({"status": "complete", "seconds": payload["elapsed_seconds"], "output": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
