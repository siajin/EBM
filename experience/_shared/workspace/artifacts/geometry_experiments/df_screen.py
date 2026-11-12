"""Freeze T/V-fitted diagnostics, then evaluate a fixed DF pilot on server.

DF labels determine a score-independent within-class sample and metrics only.
They never select a checkpoint, feature mask, hyperparameter or threshold.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer, StandardScaler

from fast_analyze_v1 import balanced_ll, eer, geometry, metrics, save
from df_extract_v1 import digest, settings


CS = [0.01, 0.1, 1, 10]
GAMMAS = [0.01, 0.1, 0.5, 1.0]


def stable_priority(uid):
    return hashlib.sha256(("DF-pilot-20261105|" + uid).encode()).hexdigest()


def sample_pool(pool):
    pool = pool.assign(priority=pool.utterance_id.map(stable_priority))
    chosen = []
    for label, target in [(1, 2000), (0, 18000)]:
        frame = pool.loc[pool.y == label].sort_values(["priority", "utterance_id"])
        assert len(frame) >= target
        chosen.append(frame.head(target).assign(inclusion_probability=target / len(frame)))
    return pd.concat(chosen).drop(columns="priority").sort_values("utterance_id").reset_index(drop=True)


def prepare(root, source, reference):
    assert json.loads((source / "fast_analysis/summary.json").read_text())["status"] == "fast_screen_complete_not_confirmatory"
    _, args = settings(reference)
    protocol = Path(args["path_21DF"]) / "keys/DF/CM/trial_metadata.txt"
    records = []
    for line in protocol.read_text().splitlines():
        a = line.split()
        assert len(a) == 13 and a[5] in {"bonafide", "spoof"}
        if a[7] != "eval":
            continue
        speaker = a[0] if a[0] not in {"-", "unknown", ""} else "unknown_shared"
        records.append({
            "dataset": "ASVspoof2021_DF", "split": "R21DF_pilot",
            "utterance_id": a[1], "family_id": a[1],
            "family_resolution": "source_family_unresolved_speaker_group_only",
            "speaker_group": a[3] + "::" + speaker,
            "dependency_group": a[3] + "::" + speaker,
            "speaker_id_known": speaker != "unknown_shared",
            "y": int(a[5] == "bonafide"),
            "attack_id": "bonafide" if a[5] == "bonafide" else a[4],
            "codec": a[2], "source_corpus": a[3], "trim": a[6], "vocoder": a[8],
            "path": str(Path(args["path_21DF"]) / "ASVspoof2021_DF_eval/flac" / (a[1] + ".wav")),
            "condition_id": "native_DF_" + a[2], "crop_id": "native_first",
        })
    pool = pd.DataFrame(records)
    assert not pool.utterance_id.duplicated().any()
    frame = sample_pool(pool)
    assert all(Path(p).is_file() for p in frame.path)
    assert len(frame) == 20000 and not frame.utterance_id.duplicated().any()
    manifest = root / "manifest.parquet"
    assert not manifest.exists()
    frame.to_parquet(manifest, index=False)
    prior = json.loads((source / "input_audit.json").read_text())
    audit = {
        "status": "pass", "profile": "DF_fixed_pilot_v1",
        "source_run": str(source), "reference": str(reference),
        "extraction_splits": ["R21DF_pilot"], "Q_features_not_extracted": True,
        "manifest_sha256": digest(manifest), "DF_protocol_sha256": digest(protocol),
        "checkpoint_sha256": digest(reference / "model/BestCLS_50_21DF.pt"),
        "expected_checkpoint_sha256": prior["checkpoint_sha256"],
        "sample_rule": "within-class uniform SHA256 priority; 2000 bona and 18000 spoof; no scores used",
        "pool_count": len(pool), "pool_class_counts": pool.y.value_counts().to_dict(),
        "pilot_class_counts": frame.y.value_counts().to_dict(),
        "pilot_attack_count": int(frame.loc[frame.y == 0, "attack_id"].nunique()),
        "pool_attack_count": int(pool.loc[pool.y == 0, "attack_id"].nunique()),
        "pilot_speaker_groups": int(frame.speaker_group.nunique()),
        "speaker_groups_per_class": frame.groupby("y").speaker_group.nunique().to_dict(),
        "unknown_speaker_rows": int((~frame.speaker_id_known).sum()),
        "pilot_codec_counts": frame.codec.value_counts().to_dict(),
        "pilot_source_counts": frame.source_corpus.value_counts().to_dict(),
        "unresolved_source_families": True,
        "native_partition": "eval only, matching archived evaluator; progress and hidden excluded",
        "field_definition": "https://github.com/asvspoof-challenge/2021/blob/main/eval-package/README.md#df",
        "full_DF_evaluated": False, "historically_observed_benchmark": True,
        "script_sha256": digest(__file__), "start_epoch": time.time(),
    }
    assert audit["checkpoint_sha256"] == audit["expected_checkpoint_sha256"]
    save(root / "input_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False), flush=True)


def candidate_fit(xt, xv, yt, yv):
    scaler = StandardScaler().fit(xt)
    train, val = scaler.transform(xt), scaler.transform(xv)
    candidates, losses = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        for c in CS:
            model = LogisticRegression(C=c, class_weight="balanced", solver="lbfgs", max_iter=2000, tol=1e-6, random_state=101)
            model.fit(train, yt)
            assert model.n_iter_.max() < 2000
            losses.append(balanced_ll(yv, model.predict_proba(val)[:, 1]))
            candidates.append(model)
    best = int(np.argmin(losses))
    return {"scaler": scaler, "model": candidates[best]}, {"C": CS[best], "V_candidate_balanced_logloss": losses}


def feature_cases(z, c, spline, noise):
    norms, unit, s, valid, u = geometry(z, c)
    scalar = spline.transform(s[:, None])
    base = np.column_stack([scalar, np.log(norms), valid.astype(float)])
    return {"C0": scalar, "C1": np.column_stack([scalar, np.log(norms)]),
            "C1v": base, "C2": np.column_stack([base, u]), "Cnoise": np.column_stack([base, noise])}


def threshold_record(y, score):
    bona = np.sort(score[y == 1])
    threshold = float(bona[int(np.floor(0.01 * len(bona)))])
    return {"threshold": threshold, "V_bona_false_reject": float(np.mean(bona < threshold)),
            "V_spoof_false_accept": float(np.mean(score[y == 0] >= threshold))}


def covariance_metric(covariance, mode, gamma):
    scale = float(np.trace(covariance) / covariance.shape[0])
    assert scale > 1e-12
    matrix = (1 - gamma) * covariance + gamma * scale * np.eye(covariance.shape[0])
    if mode == "diagonal":
        a = np.diag(1 / np.maximum(np.diag(matrix), 1e-6 * scale))
    else:
        eigenvalues, basis = np.linalg.eigh(matrix)
        a = (basis / np.maximum(eigenvalues, 1e-6 * scale)) @ basis.T
    return a * covariance.shape[0] / np.trace(a)


def metric_score(z, c, a):
    unit = z / np.linalg.norm(z, axis=1, keepdims=True)
    residual = unit - c
    return 1 - 0.5 * np.einsum("ni,ij,nj->n", residual, a, residual)


def freeze(root, source):
    assert not (root / "registry.json").exists()
    audit = json.loads((root / "input_audit.json").read_text())
    phase = source / "extract"
    rows = pd.read_parquet(phase / "scores.parquet")
    assert set(rows.split) == {"T", "V"}
    z = np.concatenate([np.load(phase / f"rank{i}_z.npy") for i in [0, 1]]).astype("float64")
    c = np.load(phase / "rank0_centroid.npy")
    norms, unit, s, valid, u = geometry(z, c)
    assert np.max(np.abs(s - rows.cosine_s.to_numpy())) < 1e-10
    train = rows.split.to_numpy() == "T"
    val = ~train
    yt, yv = rows.y.to_numpy()[train], rows.y.to_numpy()[val]
    spline = SplineTransformer(n_knots=5, degree=3, include_bias=False).fit(s[train, None])
    noise = np.random.default_rng(601).normal(size=u.shape)
    noise /= np.linalg.norm(noise, axis=1, keepdims=True)
    cases = feature_cases(z, c, spline, noise)
    objects = {"c": c, "spline": spline, "probes": {}, "metrics": {}}
    registry = {"profile": "DF_fixed_pilot_v1", "DF_fit_or_selection": False,
                "DF_scores_opened_at_lock": False, "Q_opened": False,
                "source_run": str(source), "pilot_manifest_sha256": audit["manifest_sha256"],
                "source_manifest_sha256": json.loads((source / "input_audit.json").read_text())["manifest_sha256"],
                "checkpoint_sha256": audit["checkpoint_sha256"],
                "selection": "T fit; all V labels select one C per probe; differs from prior V-internal cross-selected scores",
                "E_selection": "V EER; exact ties prefer larger shrinkage gamma",
                "E_all_candidates": "prespecified sensitivity panel; do not select a new metric on DF",
                "noise": "T/V legacy RNG601 stream; DF independent per-ID SHA256 RNG unit noise",
                "comparisons": [["C1v", "C2"], ["Cnoise", "C2"], ["native_cosine", "C2"], ["A_all", "A_top25"]],
                "confidence": "exploratory paired speaker bootstrap 2000, seed 7301, 95%; source families unresolved",
                "probe_choices": {}, "E_choices": {}, "V_metrics": {}, "thresholds": {},
                "script_sha256": digest(__file__), "lock_epoch": time.time()}
    registry["V_metrics"]["native_cosine"] = metrics(yv, s[val])
    registry["thresholds"]["native_cosine"] = threshold_record(yv, s[val])
    bona = z[train][yt == 1]
    attacks = rows.loc[train, "attack_id"].to_numpy()
    fisher = [(bona.mean(0) - z[train][attacks == attack].mean(0))**2 /
              (bona.var(0, ddof=1) + z[train][attacks == attack].var(0, ddof=1) + 1e-8)
              for attack in sorted(set(attacks) - {"bonafide"})]
    rank = np.argsort(-np.mean(fisher, axis=0))
    masks = [("A_all", np.arange(160)), ("A_top25", rank[:40])]
    masks.extend((f"A_random25_{seed}", np.random.default_rng(seed).choice(160, 40, replace=False)) for seed in range(200, 220))
    for name, indices in masks:
        cases[name] = z[:, indices]
    for name, x in cases.items():
        probe, choice = candidate_fit(x[train], x[val], yt, yv)
        p = probe["model"].predict_proba(probe["scaler"].transform(x[val]))[:, 1]
        objects["probes"][name] = probe
        registry["probe_choices"][name] = choice
        registry["V_metrics"][name] = metrics(yv, p, p)
        registry["thresholds"][name] = threshold_record(yv, p)
        print("LOCK", name, choice["C"], flush=True)
    objects["A_masks"] = dict(masks)
    registry["A_masks"] = {k: v.tolist() for k, v in masks}
    covariance = np.cov(unit[train][yt == 1].T)
    for mode in ["diagonal", "full"]:
        choices = []
        for gamma in GAMMAS:
            name = f"E_{mode}_{gamma}"
            a = covariance_metric(covariance, mode, gamma)
            score = metric_score(z[val], c, a)
            if gamma == 1.0:
                assert np.max(np.abs(score - s[val])) < 1e-10
            objects["metrics"][name] = a
            registry["V_metrics"][name] = metrics(yv, score)
            registry["thresholds"][name] = threshold_record(yv, score)
            choices.append((registry["V_metrics"][name]["eer_percent"], -gamma, name))
        registry["E_choices"][mode] = min(choices)[2]
    bundle_path = root / "frozen_models.joblib"
    assert not bundle_path.exists()
    joblib.dump(objects, bundle_path)
    registry["bundle_sha256"] = digest(bundle_path)
    save(root / "registry.json", registry)
    print("REGISTRY_LOCKED", json.dumps({"E_choices": registry["E_choices"], "bundle_sha256": registry["bundle_sha256"]}), flush=True)


def id_noise(ids):
    output = []
    for uid in ids:
        seed = int.from_bytes(hashlib.sha256(("DF-Cnoise-601|" + uid).encode()).digest()[:8], "big")
        v = np.random.default_rng(seed).normal(size=160)
        output.append(v / np.linalg.norm(v))
    return np.asarray(output)


def weighted_eer_preparation(y, score, codes):
    order = np.argsort(-score, kind="stable")
    ys, scores = y[order], score[order]
    ends = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(scores)-1]
    return ys, codes[order], ends


def weighted_eer(prepared, counts):
    y, codes, ends = prepared
    weights = counts[codes]
    positive, negative = weights*y, weights*(1-y)
    if positive.sum() == 0 or negative.sum() == 0:
        return float("nan")
    fnr = 1 - np.r_[0, np.cumsum(positive)[ends] / positive.sum()]
    fpr = np.r_[0, np.cumsum(negative)[ends] / negative.sum()]
    difference = fpr - fnr
    upper = int(np.flatnonzero(difference >= 0)[0])
    if upper == 0:
        return float(100 * (fpr[0]+fnr[0])/2)
    lower = upper-1
    fraction = -difference[lower]/(difference[upper]-difference[lower])
    return float(100*(fpr[lower]+fraction*(fpr[upper]-fpr[lower])))


def bootstrap_contrasts(y, scores, groups, contrasts):
    speakers, codes = np.unique(groups, return_inverse=True)
    prepared = {name: weighted_eer_preparation(y, np.asarray(score), codes) for name, score in scores.items()
                if any(name in pair for pair in contrasts)}
    for name, values in prepared.items():
        assert abs(weighted_eer(values, np.ones(len(speakers))) - eer(y, scores[name])) < 1e-9
    rng = np.random.default_rng(7301)
    draws = rng.multinomial(len(speakers), np.full(len(speakers), 1/len(speakers)), size=2000)
    samples = {name: np.asarray([weighted_eer(value, count) for count in draws]) for name, value in prepared.items()}
    results = []
    contributors = [len(np.unique(groups[y == label])) for label in [0, 1]]
    for baseline, candidate in contrasts:
        delta = samples[baseline]-samples[candidate]
        results.append({"baseline": baseline, "candidate": candidate,
                        "delta_EER_pp": eer(y, scores[baseline])-eer(y, scores[candidate]),
                        "CI95": np.nanquantile(delta, [0.025, 0.975]).tolist(),
                        "dependency_groups": len(speakers), "class_contributing_groups": contributors,
                        "adequate_group_count": len(speakers) >= 10 and min(contributors) >= 5,
                        "status": "exploratory_pilot_speaker_only_families_unresolved"})
    return results


def evaluate(root):
    assert (root / "extract.exit").read_text().strip() == "0"
    assert (root / "extract/complete").read_text().strip() == "0"
    out = root / "DF_evaluation"
    out.mkdir(exist_ok=False)
    start = time.monotonic()
    registry = json.loads((root / "registry.json").read_text())
    assert digest(root / "frozen_models.joblib") == registry["bundle_sha256"]
    assert digest(root / "manifest.parquet") == registry["pilot_manifest_sha256"]
    bundle = joblib.load(root / "frozen_models.joblib")
    rows = pd.read_parquet(root / "extract/scores.parquet")
    assert set(rows.split) == {"R21DF_pilot"} and len(rows) == 20000
    z = np.concatenate([np.load(root / "extract" / f"rank{i}_z.npy") for i in [0, 1]]).astype("float64")
    assert len(rows) == len(z) and not rows.utterance_id.duplicated().any()
    noise = id_noise(rows.utterance_id)
    cases = feature_cases(z, bundle["c"], bundle["spline"], noise)
    for name, indices in bundle["A_masks"].items():
        cases[name] = z[:, indices]
    y = rows.y.to_numpy()
    native = z / np.linalg.norm(z, axis=1, keepdims=True) @ bundle["c"]
    assert np.max(np.abs(native-rows.cosine_s.to_numpy())) < 1e-10
    scores = {"native_cosine": native}
    for name, x in cases.items():
        probe = bundle["probes"][name]
        scores[name] = probe["model"].predict_proba(probe["scaler"].transform(x))[:, 1]
    for name, a in bundle["metrics"].items():
        scores[name] = metric_score(z, bundle["c"], a)
        if name.endswith("_1.0"):
            assert np.max(np.abs(scores[name]-native)) < 1e-10
    scores_frame = rows[["utterance_id", "speaker_group", "y", "attack_id", "codec", "source_corpus"]].copy()
    results = []
    for name, score in scores.items():
        assert np.isfinite(score).all()
        scores_frame[name] = score
        threshold = registry["thresholds"][name]["threshold"]
        record = {"case": name, **metrics(y, score, score if name in bundle["probes"] else None),
                  "V_fixed_bona_false_reject": float(np.mean(score[y == 1] < threshold)),
                  "V_fixed_spoof_false_accept": float(np.mean(score[y == 0] >= threshold))}
        results.append(record)
        print("DF", json.dumps(record), flush=True)
    scores_frame.to_parquet(out / "predictions.parquet", index=False)
    save(out / "metrics.json", results)
    contrasts = bootstrap_contrasts(y, scores, rows.dependency_group.to_numpy(), registry["comparisons"])
    selected_names = ["native_cosine", "C1v", "C2", "Cnoise", "A_all", "A_top25"] + list(registry["E_choices"].values())
    segments = []
    for field in ["source_corpus", "codec", "attack_id"]:
        for value in sorted(set(rows[field])):
            chosen = rows[field].to_numpy() == value
            if field == "attack_id":
                if value == "bonafide":
                    continue
                chosen |= y == 1  # Shared bona pool; these attacks are not independent replicates.
            if len(np.unique(y[chosen])) < 2:
                continue
            segments.append({"field": field, "value": value,
                             "bona": int(np.sum(chosen & (y == 1))), "spoof": int(np.sum(chosen & (y == 0))),
                             "eer_percent": {n: eer(y[chosen], scores[n][chosen]) for n in selected_names}})
    save(out / "segments.json", segments)
    random = [r for r in results if r["case"].startswith("A_random")]
    summary = {"status": "DF_pilot_complete_exploratory", "full_DF_evaluated": False,
               "n": len(y), "bona": int(y.sum()), "spoof": int((1-y).sum()),
               "DF_fit_or_selection": False, "EBM_retrained": False, "Q_evaluated": False,
               "registry_sha256": digest(root / "registry.json"),
               "selected_results": [r for r in results if r["case"] in selected_names],
               "all_E_candidates": [r for r in results if r["case"].startswith("E_")],
               "random40_EER_range_and_median": np.quantile([r["eer_percent"] for r in random], [0, 0.5, 1]).tolist(),
               "contrasts": contrasts,
               "limits": ["fixed 20000-row pilot, not full DF", "historically observed benchmark",
                          "speaker-only dependence grouping; source-family mapping unresolved",
                          "single baseline checkpoint and linear probes", "DF results cannot choose new settings for this evaluation"],
               "elapsed_seconds": time.monotonic()-start}
    save(out / "summary.json", summary)
    (out / "complete").write_text("0\n")
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "freeze", "evaluate"])
    parser.add_argument("--root", required=True)
    parser.add_argument("--source", default="/home/mentee/analysis/embedding_geometry/20261105_fast_p0_01")
    parser.add_argument("--reference", default="/home/mentee/results/CM/LA/EBM_e50_bs160_lr1e5")
    args = parser.parse_args()
    assert Path("/.dockerenv").exists()
    root, source = Path(args.root), Path(args.source)
    if args.stage == "prepare":
        prepare(root, source, Path(args.reference))
    elif args.stage == "freeze":
        freeze(root, source)
    else:
        evaluate(root)


if __name__ == "__main__":
    main()
