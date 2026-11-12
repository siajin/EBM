"""First diagnostic pass on server-only cached T/V embeddings.

The complete specification remains unchanged. This deliberately smaller profile
uses linear probes and V-internal grouped selection; it never opens Q features.
It does not train the EBM, tune on Q, or constitute final hypothesis confirmation.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import SplineTransformer, StandardScaler


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def balanced_ll(y, p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-0.5 * (np.log(p[y == 1]).mean() + np.log1p(-p[y == 0]).mean()))


def eer(y, score):
    fpr, tpr, _ = roc_curve(y, score, pos_label=1, drop_intermediate=False)
    fnr = 1 - tpr
    difference = fpr - fnr
    upper = int(np.flatnonzero(difference >= 0)[0])
    if upper == 0:
        return float(100 * (fpr[0] + fnr[0]) / 2)
    lower = upper - 1
    weight = -difference[lower] / (difference[upper] - difference[lower])
    return float(100 * (fpr[lower] + weight * (fpr[upper] - fpr[lower])))


def metrics(y, score, probability=None):
    result = {"eer_percent": eer(y, score), "auroc": float(roc_auc_score(y, score))}
    if probability is not None:
        result["balanced_logloss"] = balanced_ll(y, probability)
    return result


def geometry(z, c):
    norms = np.linalg.norm(z, axis=1)
    assert np.all(norms > 1e-12)
    z = z / norms[:, None]
    s = z @ c
    r = z - s[:, None] * c
    t = np.linalg.norm(r, axis=1)
    valid = t >= 1e-4
    u = np.zeros_like(r)
    u[valid] = r[valid] / t[valid, None]
    return norms, z, s, valid, u


def fit_predict_candidates(xt, xv, yt):
    scaler = StandardScaler().fit(xt)
    xt, xv = scaler.transform(xt), scaler.transform(xv)
    predictions = []
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        for c in [0.01, 0.1, 1, 10]:
            model = LogisticRegression(C=c, class_weight="balanced", solver="lbfgs", max_iter=2000, tol=1e-6, random_state=101)
            model.fit(xt, yt)
            assert model.n_iter_.max() < 2000
            predictions.append(model.predict_proba(xv)[:, 1])
    return np.asarray(predictions)


def cross_selected_predictions(predictions, y, groups):
    # Candidates are trained on T. Each V speaker's score uses C selected on
    # other V speakers, never that speaker's outcomes.
    speakers = np.random.default_rng(11).permutation(np.unique(groups))
    folds = np.array_split(speakers, 5)
    output = np.full(len(y), np.nan)
    selected = []
    for i, held_speakers in enumerate(folds):
        held = np.isin(groups, held_speakers)
        selection = ~held
        losses = [balanced_ll(y[selection], p[selection]) for p in predictions]
        best = int(np.argmin(losses))
        output[held] = predictions[best, held]
        selected.append({"fold": i, "held_speaker_count": len(held_speakers), "selected_C": [0.01, 0.1, 1, 10][best]})
    assert np.isfinite(output).all()
    return output, selected


def delta_ll_bootstrap(y, baseline, candidate, groups):
    speakers = np.unique(groups)
    b, a = np.clip(baseline, 1e-7, 1 - 1e-7), np.clip(candidate, 1e-7, 1 - 1e-7)
    values = []
    for speaker in speakers:
        chosen = groups == speaker
        positive, negative = chosen & (y == 1), chosen & (y == 0)
        values.append([positive.sum(), negative.sum(),
                       (-np.log(b[positive]) + np.log(a[positive])).sum(),
                       (-np.log1p(-b[negative]) + np.log1p(-a[negative])).sum()])
    values = np.asarray(values)
    rng = np.random.default_rng(7301)
    draws = rng.integers(len(speakers), size=(2000, len(speakers)))
    totals = values[draws].sum(axis=1)
    good = (totals[:, 0] > 0) & (totals[:, 1] > 0)
    delta = 0.5 * (totals[good, 2] / totals[good, 0] + totals[good, 3] / totals[good, 1])
    return {"delta_balanced_logloss": balanced_ll(y, baseline) - balanced_ll(y, candidate),
            "CI95": np.quantile(delta, [0.025, 0.975]).tolist(),
            "speaker_groups": len(speakers), "status": "exploratory_V_internal_cross_selection"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    opts = parser.parse_args()
    root = Path(opts.root)
    assert Path("/.dockerenv").exists()
    phase = root / "extract"
    assert (phase / "complete").read_text().strip() == "0"
    out = root / "fast_analysis"
    out.mkdir(exist_ok=False)
    start = time.monotonic()
    rows = pd.read_parquet(phase / "scores.parquet")
    assert set(rows.split) == {"T", "V"}
    z = np.concatenate([np.load(phase / f"rank{rank}_z.npy") for rank in [0, 1]]).astype("float64")
    assert len(z) == len(rows)
    c = np.load(phase / "rank0_centroid.npy")
    norms, unit_z, s, valid, u = geometry(z, c)
    assert np.max(np.abs(s - rows.cosine_s.to_numpy())) < 1e-10
    train = rows.split.to_numpy() == "T"
    val = ~train
    yt = rows.loc[train, "y"].to_numpy()
    yv = rows.loc[val, "y"].to_numpy()
    groups = rows.loc[val, "speaker_group"].to_numpy()
    profile = {"profile": "fast_screen_v1", "full_specification_completed": False,
               "features_used": ["T", "V"], "Q_features_opened": False,
               "EBM_retrained": False, "probe": "L2_logistic_only",
               "selection": "V_internal_5_fold_speaker_cross_selection",
               "scope": "source-domain diagnostic; not evidence of unseen-attack generalization",
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "deferred": ["MLP probes", "three probe seeds", "remaining mask fractions", "conditional MMD", "direction shuffles", "Q and external tests", "F perturbations"]}
    save(out / "profile_lock.json", profile)
    summary = {**profile, "sample_counts": {"T": int(train.sum()), "V": int(val.sum())},
               "baseline_native_V": metrics(yv, s[val]),
               "direction_valid_rate": {"T": float(valid[train].mean()), "V": float(valid[val].mean())},
               "norm_summary": {"mean": float(norms.mean()), "std": float(norms.std()), "min": float(norms.min()), "max": float(norms.max())}}
    predictions = {}
    spline = SplineTransformer(n_knots=5, degree=3, include_bias=False).fit(s[train, None])
    scalar = spline.transform(s[:, None])
    base = np.column_stack([scalar, np.log(norms), valid.astype(float)])
    noise = np.random.default_rng(601).normal(size=u.shape)
    noise /= np.linalg.norm(noise, axis=1, keepdims=True)
    cases = {"C0": scalar, "C1": np.column_stack([scalar, np.log(norms)]), "C1v": base,
             "C2": np.column_stack([base, u]), "Cnoise": np.column_stack([base, noise])}
    c_results = []
    all_scores = rows.loc[val, ["utterance_id", "speaker_group", "y", "attack_id"]].copy()
    for name, x in cases.items():
        candidates = fit_predict_candidates(x[train], x[val], yt)
        p, folds = cross_selected_predictions(candidates, yv, groups)
        predictions[name] = p
        all_scores[name] = p
        c_results.append({"case": name, **metrics(yv, p, p), "fold_choices": folds})
        save(out / "C_partial.json", c_results)
        print("C", name, metrics(yv, p, p), flush=True)
    summary["C"] = c_results
    summary["C_direction_delta"] = delta_ll_bootstrap(yv, predictions["C1v"], predictions["C2"], groups)
    all_scores.to_parquet(out / "C_V_predictions.parquet", index=False)

    # A: top 25%, full vector, and 20 random 25% subsets. No new EBM pass.
    bona = z[train][yt == 1]
    mean_b, var_b = bona.mean(0), bona.var(0, ddof=1)
    attacks = rows.loc[train, "attack_id"].to_numpy()
    fisher = []
    for attack in sorted(set(attacks) - {"bonafide"}):
        fake = z[train][attacks == attack]
        fisher.append((mean_b-fake.mean(0))**2/(var_b+fake.var(0, ddof=1)+1e-8))
    rank = np.argsort(-np.mean(fisher, axis=0))
    np.save(out / "A_fisher_by_attack.npy", np.asarray(fisher))
    a_results = []
    masks = [("all", np.arange(160)), ("top25", rank[:40])]
    masks.extend((f"random25_{seed}", np.random.default_rng(seed).choice(160, 40, replace=False)) for seed in range(200, 220))
    for name, indices in masks:
        candidates = fit_predict_candidates(z[train][:, indices], z[val][:, indices], yt)
        p, folds = cross_selected_predictions(candidates, yv, groups)
        a_results.append({"case": name, "dimensions": len(indices), **metrics(yv, p, p), "fold_choices": folds})
        save(out / "A_partial.json", a_results)
        print("A", name, metrics(yv, p, p), flush=True)
    summary["A"] = a_results

    # E: same centroid; train-bona covariance only. Selection stays inside V.
    covariance = np.cov(unit_z[train][yt == 1].T)
    residual = unit_z[val] - c
    scale = float(np.trace(covariance) / 160)
    e_results = []
    for mode in ["diagonal", "full"]:
        for gamma in [0.01, 0.1, 0.5, 1.0]:
            shrunk = (1-gamma)*covariance + gamma*scale*np.eye(160)
            if mode == "diagonal":
                a = np.diag(1/np.maximum(np.diag(shrunk), 1e-6*max(scale, 1e-12)))
            else:
                eigenvalues, basis = np.linalg.eigh(shrunk)
                a = (basis / np.maximum(eigenvalues, 1e-6*max(scale, 1e-12))) @ basis.T
            a *= 160 / np.trace(a)
            score = 1-0.5*np.einsum("ni,ij,nj->n", residual, a, residual)
            e_results.append({"mode": mode, "gamma": gamma, **metrics(yv, score)})
    summary["E_V_selection_only"] = e_results
    d_rows = []
    for attack in sorted(set(attacks)):
        selected = train & (rows.attack_id.to_numpy() == attack) & valid
        direction = u[selected]
        moment = direction.T @ direction / len(direction)
        eig = np.linalg.eigvalsh(moment)
        d_rows.append({"attack": attack, "n_valid": len(direction), "mean_direction_norm": float(np.linalg.norm(direction.mean(0))), "participation_ratio": float(eig.sum()**2/np.sum(eig**2))})
    summary["D_train_descriptive"] = d_rows
    summary["elapsed_seconds"] = time.monotonic() - start
    summary["status"] = "fast_screen_complete_not_confirmatory"
    save(out / "summary.json", summary)
    (out / "complete").write_text("0\n")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
