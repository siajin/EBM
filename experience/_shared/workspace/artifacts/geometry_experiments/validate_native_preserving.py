"""Offline corrected controls, bounded directional residuals and metric checks.

All training uses T and all parameter/threshold selection uses V. DF was already
observed in earlier work: this is a new exploratory version, not a fresh test.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from fast_analyze import balanced_ll, eer, metrics
from geometry_cache import DEFAULT_ROOT, load_embeddings, load_predictions, read_json, resolve_run, validate_run


CS = [0.01, 0.1, 1, 10]
EPSILONS = [0.0, 0.01, 0.03, 0.1]
GAMMAS = [0.01, 0.1, 0.5, 1.0]


def save(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    temporary.replace(path)


def import_screen():
    folder = Path(__file__).parent
    for alias, name in [("fast_analyze_v1", "fast_analyze.py"), ("df_extract_v1", "df_extract.py"), ("df_screen_v1", "df_screen.py")]:
        if alias in sys.modules:
            continue
        spec = importlib.util.spec_from_file_location(alias, folder / name)
        module = importlib.util.module_from_spec(spec)
        sys.modules[alias] = module
        spec.loader.exec_module(module)
    return sys.modules["df_screen_v1"]


def bounded_score(s, g, epsilon):
    assert np.max(np.abs(g)) <= 1 + 1e-12
    score = s + epsilon*g
    assert np.max(np.abs(score-s)) <= epsilon + 1e-12
    return score


def rotate_azimuth(unit, c, theta, seed=71):
    s = unit @ c
    r = unit-s[:, None]*c
    t = np.linalg.norm(r, axis=1)
    assert np.all(t > 1e-8)
    u = r/t[:, None]
    v = np.random.default_rng(seed).normal(size=u.shape)
    v -= (v@c)[:, None]*c
    v -= np.sum(v*u, axis=1)[:, None]*u
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    moved = s[:, None]*c + t[:, None]*(np.cos(theta)*u+np.sin(theta)*v)
    return moved, v


def oc_loss(unit, c, y):
    s = unit@c
    argument = np.where(y == 1, .9-s, s-.2)
    return np.logaddexp(0, 20*argument)


def bpl_demo(x, y):
    x = x/np.linalg.norm(x, axis=1, keepdims=True)
    y = y/np.linalg.norm(y, axis=1, keepdims=True)
    return float((np.logaddexp(0, 30*(1-x@y.T))-.3).mean())


def loss_geometry_audit(data):
    unit = data.geometry()["unit_z"]
    y = data.rows.y.to_numpy()
    # Embedding-level interventions, not waveform/encoder counterfactuals.
    indices = np.r_[np.flatnonzero(y == 1)[:64], np.flatnonzero(y == 0)[:64]]
    x, labels, c = unit[indices], y[indices], data.centroid
    original = oc_loss(x, c, labels)
    errors = []
    for angle in [.3, 1.0, 2.0]:
        moved, v = rotate_azimuth(x, c, angle)
        errors.append({"angle_radians": angle,
                       "cosine_error": float(np.max(np.abs(moved@c-x@c))),
                       "unit_norm_error": float(np.max(np.abs(np.linalg.norm(moved, axis=1)-1))),
                       "OC_loss_error": float(np.max(np.abs(oc_loss(moved, c, labels)-original)))})
    step = 1e-5
    plus, _ = rotate_azimuth(x, c, step)
    minus, _ = rotate_azimuth(x, c, -step)
    derivative = (oc_loss(plus, c, labels)-oc_loss(minus, c, labels))/(2*step)
    bona = x[labels == 1]
    rotated_bona, _ = rotate_azimuth(bona, c, 1.0)
    baseline_bpl = bpl_demo(bona, bona)
    permutation = np.random.default_rng(73).permutation(len(c))
    permutation_error = float(np.max(np.abs(x[:, permutation]@c[permutation]-x@c)))
    assert max(e["cosine_error"] for e in errors) < 1e-10
    assert max(e["OC_loss_error"] for e in errors) < 1e-10
    assert permutation_error < 1e-10
    return {
        "status": "embedding_level_invariance_verified", "samples": len(indices), "rotations": errors,
        "max_finite_difference_OC_azimuth_derivative": float(np.max(np.abs(derivative))),
        "normal_pair_BPL_demo_before": baseline_bpl,
        "normal_pair_BPL_demo_after_independent_normal_rotations": bpl_demo(rotated_bona, rotated_bona),
        "spoof_rotation_BPL_change": 0.0,
        "joint_feature_centroid_channel_permutation_score_error": permutation_error,
        "implementation_evidence": {
            "archived_loss_sha256": "164be4161b57a7899ef7b289e005ab80d12da373922d57e9e021844ea828b1ac",
            "archived_classifier_sha256": "c2cbcf898f1905c9c6ee8b20ef4514c4af7cb343abbf35e2dbd1fd526ad3459f",
            "architecture_has_layer_attention_temporal_attention_and_channel_mixing": True,
            "OC_reads_only_normalized_centroid_dot_product": True,
            "BPL_uses_bonafide_long_short_pairs_only": True,
        },
        "limits": ["Rotated vectors need not lie on the LayerNorm/encoder attainable manifold",
                   "BPL numeric example uses bona long features as both views; not a measured training-batch loss",
                   "Invariance does not prove useful direction information was lost or caused performance degradation",
                   "Model already learns channels; unconstrained spoof azimuth is a different statement"]}


def make_model(c):
    return LogisticRegression(C=c, class_weight="balanced", solver="lbfgs", max_iter=2000, tol=1e-8, random_state=101)


def fit_monotonic(s, y):
    scaler = StandardScaler().fit(s[:, None])
    model = make_model(1)
    model.fit(scaler.transform(s[:, None]), y)
    slope = float(model.coef_[0, 0]/scaler.scale_[0])
    intercept = float(model.intercept_[0]-slope*scaler.mean_[0])
    assert slope > 0, "Control must preserve native score ranking"
    return {"slope": slope, "intercept": intercept}


def probability(s, calibration):
    return expit(calibration["slope"]*s+calibration["intercept"])


def fit_residual(xt, xv, yt, yv, sv, calibration):
    scaler = StandardScaler().fit(xt)
    xt, xv = scaler.transform(xt), scaler.transform(xv)
    candidates, choices = {}, []
    for c in CS:
        model = make_model(c)
        model.fit(xt, yt)
        assert model.n_iter_.max() < 2000
        candidates[c] = model
        g = np.tanh(model.decision_function(xv)/2)
        for epsilon in EPSILONS:
            score = bounded_score(sv, g, epsilon)
            choices.append({"C": c, "epsilon": epsilon, "V_balanced_logloss": balanced_ll(yv, probability(score, calibration)),
                            "V_EER_percent": eer(yv, score)})
    smallest = min(r["V_balanced_logloss"] for r in choices)
    tied = [r for r in choices if r["V_balanced_logloss"] <= smallest+1e-12]
    selected = min(tied, key=lambda r: (r["epsilon"], r["C"]))
    return {"scaler": scaler, "model": candidates[selected["C"]]}, selected, choices


def paired_intervals(y, scores, groups, contrasts, screen, draws=10000):
    speakers, codes = np.unique(groups, return_inverse=True)
    family = len(contrasts)
    rng = np.random.default_rng(7301)
    counts = rng.multinomial(len(speakers), np.full(len(speakers), 1/len(speakers)), size=draws)
    values = {}
    for name in sorted({n for pair in contrasts for n in pair}):
        prepared = screen.weighted_eer_preparation(y, scores[name], codes)
        assert abs(screen.weighted_eer(prepared, np.ones(len(speakers)))-eer(y, scores[name])) < 1e-9
        values[name] = np.asarray([screen.weighted_eer(prepared, weights) for weights in counts])
        print("BOOTSTRAP", name, flush=True)
    result = []
    for baseline, candidate in contrasts:
        delta = values[baseline]-values[candidate]
        result.append({"baseline": baseline, "candidate": candidate,
                       "delta_EER_pp": eer(y, scores[baseline])-eer(y, scores[candidate]),
                       "CI95": np.nanquantile(delta, [.025,.975]).tolist(),
                       "CI_family95_Bonferroni": np.nanquantile(delta, [.025/family,1-.025/family]).tolist(),
                       "individual_adjusted_CI_level": 1-.05/family,
                       "family_comparisons": family, "bootstrap_draws": draws,
                       "speaker_groups": len(speakers),
                       "status": "exploratory_reused_DF_unresolved_source_families_prior_search_not_controlled"})
    return result


def run(output):
    screen = import_screen()
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    metric_names = [f"E_{mode}_{gamma}" for mode in ["diagonal","full"] for gamma in [.01,.1,.5]]
    contrasts = [("native_cosine", n) for n in metric_names]
    contrasts += [("native_cosine", "native_plus_direction"), ("native_cosine", "native_plus_noise"),
                  ("native_plus_noise", "native_plus_direction")]
    protocol = {"version": "native_preserving_v1", "profile_time_utc": datetime.now(timezone.utc).isoformat(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "DF_was_already_observed": True, "independent_confirmatory_test": False,
                "EBM_retraining": False, "Q_opened": False,
                "fit_split": "T", "selection_and_threshold_split": "V", "evaluation_split": "DF_pilot",
                "native_control": "positive affine logistic calibration on T; use unsaturated logits for ranking metrics",
                "metric_logloss_note": "Metrics share the native T-fitted calibration; E probability calibration was not separately fitted",
                "direction_score": "s + epsilon*tanh(logistic_direction_decision(u)/2), native coefficient fixed to 1",
                "C_grid": CS, "epsilon_grid": EPSILONS, "metric_gamma_grid": GAMMAS,
                "residual_selection": "minimum V balanced logloss, 1e-12 ties prefer epsilon=0 then smaller C",
                "metric_selection": "minimum V EER, ties prefer gamma=1",
                "contrasts": contrasts, "family_comparisons": len(contrasts),
                "bootstrap_draws": 10000, "random_seed": 7301,
                "bootstrap_scope": "paired source-speaker clusters; source-family dependence unresolved",
                "multiplicity_scope": "current 9 comparisons only; prior DF-driven research choices not controlled"}
    save(output / "protocol.json", protocol)
    t, v = load_embeddings("T"), load_embeddings("V")
    gt, gv = t.geometry(), v.geometry()
    yt, yv = t.rows.y.to_numpy(), v.rows.y.to_numpy()
    st, sv, centroid = gt["cosine"], gv["cosine"], t.centroid
    assert t.provenance["identity"]["checkpoint_sha256"] == v.provenance["identity"]["checkpoint_sha256"]
    for dataset in ["T","V","DF_pilot"]:
        validate_run(dataset)
    save(output / "loss_geometry_audit.json", loss_geometry_audit(t))
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        calibration = fit_monotonic(st, yt)
        assert abs(eer(yt, st)-eer(yt, calibration["slope"]*st+calibration["intercept"])) < 1e-10
        # Independent per-ID features, unrelated to labels and evaluation order.
        noise_t = screen.id_noise("T::"+t.rows.utterance_id)
        noise_v = screen.id_noise("V::"+v.rows.utterance_id)
        direction, d_choice, d_panel = fit_residual(gt["direction"], gv["direction"], yt, yv, sv, calibration)
        noise, n_choice, n_panel = fit_residual(noise_t, noise_v, yt, yv, sv, calibration)
    models = {"direction": direction, "noise": noise}
    choices = {"direction": d_choice, "noise": n_choice}
    covariance = np.cov(gt["unit_z"][yt == 1].T)
    matrices = {f"E_{mode}_{gamma}": screen.covariance_metric(covariance, mode, gamma)
                for mode in ["diagonal", "full"] for gamma in GAMMAS}
    v_scores = {"native_cosine": sv}
    v_scores["monotonic_control"] = calibration["slope"]*sv+calibration["intercept"]
    for name, x in [("direction", gv["direction"]), ("noise", noise_v)]:
        model = models[name]
        g = np.tanh(model["model"].decision_function(model["scaler"].transform(x))/2)
        v_scores["native_plus_"+name] = bounded_score(sv, g, choices[name]["epsilon"])
    for name, matrix in matrices.items():
        v_scores[name] = screen.metric_score(v.z.astype("float64"), centroid, matrix)
    e_choice = {}
    for mode in ["diagonal", "full"]:
        e_choice[mode] = min(GAMMAS, key=lambda gamma: (eer(yv, v_scores[f"E_{mode}_{gamma}"]), -gamma))
    thresholds = {name: screen.threshold_record(yv, score) for name, score in v_scores.items()}
    fit_lock = {"DF_new_scores_seen": False, "calibration": calibration, "selected_residuals": choices,
                "direction_V_candidates": d_panel, "noise_V_candidates": n_panel,
                "metric_V_choices": e_choice, "V_thresholds": thresholds,
                "T_cache": t.provenance["cache_fingerprint"], "V_cache": v.provenance["cache_fingerprint"],
                "checkpoint_sha256": t.provenance["identity"]["checkpoint_sha256"]}
    save(output / "fit_lock.json", fit_lock)
    joblib.dump({"calibration": calibration, "residuals": models, "choices": choices, "matrices": matrices,
                 "centroid": centroid}, output / "posthoc_models.joblib")
    print("FIT_LOCK", json.dumps({"calibration": calibration, "residuals": choices, "metric_choices": e_choice}), flush=True)
    df = load_embeddings("DF_pilot", expected_checkpoint_sha256=fit_lock["checkpoint_sha256"])
    gd, y = df.geometry(), df.rows.y.to_numpy()
    s = gd["cosine"]
    scores = {"native_cosine": s, "monotonic_control": calibration["slope"]*s+calibration["intercept"]}
    assert abs(eer(y, scores["monotonic_control"])-eer(y, s)) < 1e-10
    residual_panel = []
    for name, x in [("direction", gd["direction"]), ("noise", screen.id_noise("DF::"+df.rows.utterance_id))]:
        model = models[name]
        g = np.tanh(model["model"].decision_function(model["scaler"].transform(x))/2)
        scores["native_plus_"+name] = bounded_score(s, g, choices[name]["epsilon"])
        for epsilon in EPSILONS:
            candidate = bounded_score(s, g, epsilon)
            residual_panel.append({"kind": name, "epsilon": epsilon, "C": choices[name]["C"],
                                   **metrics(y, candidate, probability(candidate, calibration)),
                                   "posthoc_sensitivity_only": True})
    old_pred = load_predictions("DF_pilot")
    assert old_pred.utterance_id.tolist() == df.rows.utterance_id.tolist()
    for name, matrix in matrices.items():
        scores[name] = screen.metric_score(df.z.astype("float64"), centroid, matrix)
        assert np.max(np.abs(scores[name]-old_pred[name].to_numpy())) < 1e-10
    records = []
    for name, score in scores.items():
        assert np.isfinite(score).all()
        p = expit(score) if name == "monotonic_control" else probability(score, calibration)
        threshold = thresholds[name]["threshold"]
        records.append({"case": name, **metrics(y, score, p),
                        "V_fixed_DF_bona_false_reject": float(np.mean(score[y == 1] < threshold)),
                        "V_fixed_DF_spoof_false_accept": float(np.mean(score[y == 0] >= threshold))})
    saved = df.rows[["utterance_id","dependency_group","y","attack_id","codec","source_corpus"]].copy()
    for name, score in scores.items():
        saved[name] = score
    saved.to_parquet(output / "DF_scores.parquet", index=False)
    save(output / "metrics.json", records)
    save(output / "residual_sensitivity.json", residual_panel)
    print("POINT_RESULTS", json.dumps(records), flush=True)
    intervals = paired_intervals(y, scores, df.rows.dependency_group.to_numpy(), contrasts, screen)
    save(output / "paired_intervals.json", intervals)
    segments = []
    for field in ["source_corpus", "codec"]:
        for value in sorted(set(df.rows[field])):
            mask = df.rows[field].to_numpy() == value
            if len(np.unique(y[mask])) != 2:
                continue
            segments.append({"field": field, "value": value, "bona": int(np.sum(y[mask] == 1)),
                             "spoof": int(np.sum(y[mask] == 0)),
                             "EER": {name: eer(y[mask], score[mask]) for name, score in scores.items()}})
    save(output / "segments.json", segments)
    summary = {"status": "completed_exploratory_corrected_controls", "DF_n": len(y), "no_EBM_retraining": True,
               "DF_fit_or_hyperparameter_selection": False, "prior_DF_observations": True,
               "selected_residuals": choices, "V_metric_choices": e_choice,
               "point_results": records, "contrasts": intervals,
               "loss_geometry_audit": read_json(output / "loss_geometry_audit.json"),
               "elapsed_seconds": time.monotonic()-start,
               "conclusion_scope": "Fixed current representation only. Does not test whether new joint attention/loss training learns better representations.",
               "limitations": ["DF pilot already observed; entire research history not multiplicity-controlled",
                               "source families unresolved; speaker bootstrap may miss residual dependence",
                               "single baseline checkpoint and linear bounded residual", "V known attacks already saturated",
                               "rotations are embedding-level mathematical tests, not realizable audio interventions"]}
    save(output / "summary.json", summary)
    (output / "complete").write_text("0\n")
    print("COMPLETE", output, "seconds", summary["elapsed_seconds"], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT / "local_analysis/native_preserving_v1")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.output)


if __name__ == "__main__":
    main()
