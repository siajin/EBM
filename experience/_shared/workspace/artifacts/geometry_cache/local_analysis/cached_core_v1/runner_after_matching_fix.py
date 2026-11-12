"""Four CPU-only diagnostics of the existing, already observed embedding cache.

This is a new exploratory protocol. DF-internal probes are explicitly separate
from frozen T->DF transfer. No torch, audio, checkpoint or SSH access is used.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from threadpoolctl import threadpool_limits

from fast_analyze import balanced_ll, eer, metrics
from geometry_cache import DEFAULT_ROOT, load_embeddings, validate_run


CS = [.01, .1, 1., 10.]
CASES = ["scalar", "direction", "noise", "metadata", "metadata_direction",
         "preLN_h", "postLN_z", "preLN_stats"]
CHECKPOINT = "104873703a57cad98b686d770e0860e031a2cf9abef79b72668780c1660e0b24"


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")


def protocol():
    return {
        "version": "cached_core_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "exploratory_followup_of_already_observed_DF_pilot",
        "datasets": ["T", "V", "DF_pilot"], "checkpoint_sha256": CHECKPOINT,
        "GPU_inference_runs": 0, "EBM_retrained": False, "Q_opened": False,
        "direction": {
            "main": "DF-internal speaker-disjoint nested 5 outer x 3 inner CV; not T->DF transfer",
            "cases": CASES, "C_grid": CS, "selection": "inner OOF balanced logloss, smallest C on ties",
            "outer_seed": 11, "inner_seed": 101,
            "scalar": "train-fold quantile cubic spline(s), log(norm_z), direction_valid",
            "metadata": "scalar + train-fold onehot(source_corpus,codec) + log(duration) + padding_fraction",
            "noise": "ID-stable seed601 Gaussian unit vectors projected perpendicular to c; dimension160",
            "fit_scope": "all preprocessing, C selection and classifier fitting exclude each outer held speaker",
            "matching": {"main_strata": ["source_corpus", "codec"],
                         "calipers_abs_s_abs_lognorm": [[.005, .001], [.01, .002], [.02, .004]],
                         "primary_caliper": [.01, .002], "replacement": False,
                         "order": "SHA256 bona ID then nearest caliper-scaled distance, stable spoof ID ties",
                         "extra": "same source x codec x speaker using primary calipers",
                         "use": "descriptive direction comparison; evaluate already-held-out CV predictions on matched rows",
                         "not_claimed": "exact conditional independence test or population performance"}},
        "channel": {"ranking": "T-only attack-macro Fisher(z), same as earlier ranking",
                    "drop_counts": [8, 16, 40, 80], "random_masks": "20 nested permutations, seeds200..219",
                    "score": "cosine(mask*z, mask*c), re-normalize both; no probe refit",
                    "main": "drop40 vs native and random drop40 distribution",
                    "no_test_selection": True},
        "LayerNorm": {"main": "scalar+h vs scalar+z, identical 160 appended features and nested L2 grid",
                      "extra": "scalar+mean(h)+log(std(h))", "centroid_on_h": False,
                      "meaning": "readability under these probes, not information-theoretic loss"},
        "conditions": {"clusters": "spherical kmeans K4; T balanced2500/class; 5 starts501..505, choose T objective",
                       "comparisons": ["class", "source_corpus", "codec", "T-duration-quintile", "T-padding-bins"],
                       "within_class": True, "descriptive_only": True},
        "uncertainty": {"paired_cluster_bootstrap": 2000, "seed": 7301,
                        "group": "source-prefixed speaker; original utterance families unresolved",
                        "CI": "individual95%; family99% for five primary comparisons",
                        "CV_limit": "bootstrap conditions on saved OOF predictions, not full nested model refits",
                        "random_mask_spread": "design variability; not a sampling confidence interval"},
        "primary_comparisons": ["scalar vs direction logloss", "noise vs direction logloss",
                                "metadata vs metadata_direction logloss", "postLN_z vs preLN_h logloss",
                                "native vs drop40 EER impairment"],
        "limits": ["fixed 20000-row pilot, not full DF", "historically observed benchmark",
                   "source-family dependencies unresolved", "linear probes do not exhaust information",
                   "channel intervention is at embedding level, not an acoustic causal intervention"]}


def stable_noise(ids, c):
    result = []
    for uid in ids:
        seed = int.from_bytes(hashlib.sha256(("core-noise601|"+str(uid)).encode()).digest()[:8], "big")
        v = np.random.default_rng(seed).normal(size=len(c))
        v -= (v@c)*c
        result.append(v/np.linalg.norm(v))
    return np.asarray(result)


def make_cases(data, geom, noise, train, test):
    """Fit every transform on train only. Returns paired arrays for all cases."""
    spline = SplineTransformer(n_knots=5, degree=3, knots="quantile", include_bias=False)
    spline.fit(geom["cosine"][train, None])
    cat = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    cat.fit(data.rows.iloc[train][["source_corpus", "codec"]])
    sides = []
    for ix in [train, test]:
        base = np.column_stack([spline.transform(geom["cosine"][ix, None]),
                                np.log(geom["norm"][ix]), geom["valid"][ix]])
        condition = np.column_stack([cat.transform(data.rows.iloc[ix][["source_corpus", "codec"]]),
                                     np.log(data.rows.duration.to_numpy()[ix]),
                                     data.rows.padding_fraction.to_numpy()[ix]])
        h, z = data.h[ix].astype(float), data.z[ix].astype(float)
        sides.append({"scalar": base, "direction": np.column_stack([base, geom["direction"][ix]]),
                      "noise": np.column_stack([base, noise[ix]]),
                      "metadata": np.column_stack([base, condition]),
                      "metadata_direction": np.column_stack([base, condition, geom["direction"][ix]]),
                      "preLN_h": np.column_stack([base, h]), "postLN_z": np.column_stack([base, z]),
                      "preLN_stats": np.column_stack([base, h.mean(1), np.log(h.std(1))])})
    output = {}
    for case in CASES:
        scaler = StandardScaler().fit(sides[0][case])
        output[case] = tuple(scaler.transform(side[case]) for side in sides)
        assert all(np.isfinite(x).all() for x in output[case])
    return output


def fit_logit(x, y, c):
    model = LogisticRegression(C=c, class_weight="balanced", max_iter=2000,
                               tol=1e-7, solver="lbfgs", random_state=101)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(x, y)
    assert model.n_iter_.max() < 2000
    return model


def nested_probes(data, geom, out):
    rows = data.rows
    y, groups = rows.y.to_numpy(), rows.speaker_group.to_numpy()
    n = len(rows)
    predictions = {case: np.full(n, np.nan) for case in CASES}
    scores = {case: np.full(n, np.nan) for case in CASES}
    fold_id = np.full(n, -1)
    noise = stable_noise(rows.family_id, data.centroid)
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=11)
    records, choices = [], []
    for fold, (tr, te) in enumerate(folds.split(np.zeros(n), y, groups)):
        assert set(groups[tr]).isdisjoint(groups[te])
        assert len(np.unique(y[tr])) == len(np.unique(y[te])) == 2
        inner_p = {case: np.full((len(CS), len(tr)), np.nan) for case in CASES}
        inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=101+fold)
        for inner_fold, (it, iv) in enumerate(inner.split(np.zeros(len(tr)), y[tr], groups[tr])):
            assert set(groups[tr[it]]).isdisjoint(groups[tr[iv]])
            assert set(groups[tr[it]]).isdisjoint(groups[te])
            cases = make_cases(data, geom, noise, tr[it], tr[iv])
            for case, (xt, xv) in cases.items():
                for j, c in enumerate(CS):
                    model = fit_logit(xt, y[tr[it]], c)
                    inner_p[case][j, iv] = expit(model.decision_function(xv))
            print(f"nested fold{fold} inner{inner_fold} complete", flush=True)
        cases = make_cases(data, geom, noise, tr, te)
        for case, (xt, xv) in cases.items():
            assert np.isfinite(inner_p[case]).all()
            losses = [balanced_ll(y[tr], p) for p in inner_p[case]]
            best = int(np.argmin(losses))
            model = fit_logit(xt, y[tr], CS[best])
            score = model.decision_function(xv)
            p = expit(score)
            predictions[case][te] = p
            scores[case][te] = score
            choices.append({"fold": fold, "case": case, "C": CS[best], "inner_losses": losses,
                            "outer_iterations": int(model.n_iter_[0]), "train_rows": len(tr), "test_rows": len(te)})
            records.append({"fold": fold, "case": case, **metrics(y[te], score, p)})
        fold_id[te] = fold
        save(out / "probe_progress.json", {"outer_folds_complete": fold+1, "total": 5})
        print("outer complete", fold, flush=True)
    assert np.all(fold_id >= 0)
    assert all(np.isfinite(p).all() for p in predictions.values())
    assert rows.assign(fold=fold_id).groupby("speaker_group").fold.nunique().max() == 1
    frame = rows[["utterance_id", "speaker_group", "source_corpus", "codec", "y"]].copy()
    frame["fold"] = fold_id
    for case, p in predictions.items():
        frame[case] = p
        frame[case+"_logit"] = scores[case]
    frame.to_parquet(out / "DF_nested_OOF_predictions.parquet", index=False)
    pd.DataFrame(records).to_csv(out / "probe_by_fold.csv", index=False)
    save(out / "probe_choices.json", choices)
    result = [{"case": case, **metrics(y, scores[case], p)} for case, p in predictions.items()]
    pd.DataFrame(result).to_csv(out / "probe_metrics.csv", index=False)
    return predictions, scores, result, fold_id


def ll_intervals(y, predictions, groups, pairs, draws=2000):
    unique, codes = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(7301)
    counts = rng.multinomial(len(unique), np.full(len(unique), 1/len(unique)), size=draws)
    class_counts = np.stack([np.bincount(codes, weights=(y == label), minlength=len(unique)) for label in [0, 1]])
    denominators = counts@class_counts.T
    good = np.all(denominators > 0, axis=1)
    output = []
    for baseline, candidate in pairs:
        b, a = [np.clip(predictions[key], 1e-7, 1-1e-7) for key in [baseline, candidate]]
        losses = {}
        for label in [0, 1]:
            contribution = np.where(y == label, (-np.log(b)+np.log(a)) if label else (-np.log1p(-b)+np.log1p(-a)), 0.)
            losses[label] = np.bincount(codes, weights=contribution, minlength=len(unique))
        delta = .5*((counts@losses[0])/np.maximum(denominators[:, 0], 1)+(counts@losses[1])/np.maximum(denominators[:, 1], 1))
        output.append({"baseline": baseline, "candidate": candidate,
                       "delta_balanced_logloss": balanced_ll(y, b)-balanced_ll(y, a),
                       "CI95": np.quantile(delta[good], [.025, .975]).tolist(),
                       "family99": np.quantile(delta[good], [.005, .995]).tolist(),
                       "groups": len(unique), "dropped_replicates": int((~good).sum()),
                       "conditional_on_OOF_predictions": True})
    return output


def match_pairs(data, geom, fields, ds, dn):
    rows = data.rows
    y = rows.y.to_numpy()
    s, norm = geom["cosine"], np.log(geom["norm"])
    result = []
    for _, index in rows.groupby(fields, sort=True, observed=True).groups.items():
        index = np.asarray(index)
        bona = index[(y[index] == 1) & geom["valid"][index]]
        spoof = index[(y[index] == 0) & geom["valid"][index]]
        spoof = np.asarray(sorted(spoof, key=lambda i: rows.utterance_id.iloc[i]), dtype=int)
        available = np.ones(len(spoof), dtype=bool)
        bona = sorted(bona, key=lambda i: hashlib.sha256(str(rows.utterance_id.iloc[i]).encode()).digest())
        for bi in bona:
            valid = available & (np.abs(s[spoof]-s[bi]) <= ds) & (np.abs(norm[spoof]-norm[bi]) <= dn)
            candidates = np.flatnonzero(valid)
            if not len(candidates):
                continue
            distance = ((s[spoof[candidates]]-s[bi])/ds)**2+((norm[spoof[candidates]]-norm[bi])/dn)**2
            j = candidates[int(np.argmin(distance))]
            result.append((int(bi), int(spoof[j])))
            available[j] = False
    return np.asarray(result, dtype=int).reshape(-1, 2)


def matching_diagnostics(data, geom, predictions, scores, out):
    rows, u = data.rows, geom["direction"]
    results = []
    definitions = [("tight", ["source_corpus", "codec"], .005, .001),
                   ("primary", ["source_corpus", "codec"], .01, .002),
                   ("wide", ["source_corpus", "codec"], .02, .004),
                   ("same_speaker", ["source_corpus", "codec", "speaker_group"], .01, .002)]
    primary = None
    for name, fields, ds, dn in definitions:
        pairs = match_pairs(data, geom, fields, ds, dn)
        record = {"name": name, "strata": fields, "cosine_caliper": ds, "lognorm_caliper": dn,
                  "pairs": len(pairs), "bona_coverage": len(pairs)/int(rows.y.sum()),
                  "spoof_coverage": len(pairs)/int((1-rows.y).sum())}
        if len(pairs):
            a, b = pairs.T
            assert len(np.unique(a)) == len(a) and len(np.unique(b)) == len(b)
            assert np.all(rows.y.to_numpy()[a] == 1) and np.all(rows.y.to_numpy()[b] == 0)
            for field in fields:
                assert np.all(rows[field].to_numpy()[a] == rows[field].to_numpy()[b])
            sd = np.abs(geom["cosine"][a]-geom["cosine"][b])
            nd = np.abs(np.log(geom["norm"][a])-np.log(geom["norm"][b]))
            assert sd.max() <= ds and nd.max() <= dn
            selected = pairs.ravel()
            labels = rows.y.to_numpy()[selected]
            pair_frame = pd.DataFrame({"bona_row": a, "spoof_row": b, "abs_cosine_diff": sd,
                                       "abs_lognorm_diff": nd, "direction_cosine": np.einsum("ij,ij->i", u[a], u[b])})
            pair_frame.to_parquet(out / f"matched_{name}_pairs.parquet", index=False)
            record.update({"mean_abs_cosine_diff": float(sd.mean()), "max_abs_cosine_diff": float(sd.max()),
                           "mean_abs_lognorm_diff": float(nd.mean()), "max_abs_lognorm_diff": float(nd.max()),
                           "direction_mean_difference_norm": float(np.linalg.norm(u[a].mean(0)-u[b].mean(0))),
                           "within_pair_direction_cosine_mean": float(pair_frame.direction_cosine.mean()),
                           "speaker_groups": int(rows.speaker_group.iloc[selected].nunique()),
                           "source_codec_cells": int(rows.iloc[a].groupby(["source_corpus", "codec"]).ngroups),
                           "probe_metrics": [{"case": case, **metrics(labels, scores[case][selected], p[selected])}
                                             for case, p in predictions.items()]})
            if name == "primary":
                primary = pairs
                contrasts = [("scalar", "direction"), ("noise", "direction"), ("metadata", "metadata_direction")]
                record["paired_intervals"] = ll_intervals(labels, {k: p[selected] for k, p in predictions.items()},
                                                         rows.speaker_group.to_numpy()[selected], contrasts)
                segments = []
                sub = rows.iloc[a].copy()
                for fields2 in [["source_corpus"], ["codec"]]:
                    for value, indices in sub.groupby(fields2, observed=True).indices.items():
                        ix = pairs[np.asarray(indices)].ravel()
                        for case in ["scalar", "direction", "noise", "metadata", "metadata_direction"]:
                            p = predictions[case][ix]
                            segments.append({"field": fields2[0], "value": str(value), "pairs": len(indices),
                                             "case": case, **metrics(rows.y.to_numpy()[ix], scores[case][ix], p)})
                pd.DataFrame(segments).to_csv(out / "matched_primary_segments.csv", index=False)
        results.append(record)
    save(out / "matched_summary.json", results)
    return results, primary


def masked_cosine(z, c, removed):
    keep = np.ones(len(c), dtype=bool)
    keep[removed] = False
    cm, zm = c[keep], z[:, keep].astype(float)
    if np.linalg.norm(cm) <= 1e-12 or np.any(np.linalg.norm(zm, axis=1) <= 1e-12):
        raise ValueError("Masked cosine is undefined")
    return (zm@cm)/(np.linalg.norm(zm, axis=1)*np.linalg.norm(cm))


def weighted_eer_setup(y, score, codes):
    order = np.argsort(-score, kind="stable")
    ys, sc = y[order], score[order]
    ends = np.r_[np.flatnonzero(sc[:-1] != sc[1:]), len(sc)-1]
    return ys, codes[order], ends


def weighted_eer(prepared, counts):
    y, codes, ends = prepared
    w = counts[codes]
    pos, neg = w*y, w*(1-y)
    if pos.sum() == 0 or neg.sum() == 0:
        return np.nan
    fpr = np.r_[0., np.cumsum(neg)[ends]/neg.sum()]
    fnr = 1-np.r_[0., np.cumsum(pos)[ends]/pos.sum()]
    diff = fpr-fnr
    hi = int(np.flatnonzero(diff >= 0)[0])
    if hi == 0:
        return float(50*(fpr[0]+fnr[0]))
    lo = hi-1
    t = -diff[lo]/(diff[hi]-diff[lo])
    return float(100*(fpr[lo]+t*(fpr[hi]-fpr[lo])))


def channel_diagnostics(datasets, geoms, out):
    train = datasets["T"]
    y, attacks = train.rows.y.to_numpy(), train.rows.attack_id.to_numpy()
    z = train.z.astype(float)
    bona = z[y == 1]
    fishers = []
    for attack in sorted(set(attacks)-{"bonafide"}):
        fake = z[attacks == attack]
        fishers.append((bona.mean(0)-fake.mean(0))**2/(bona.var(0, ddof=1)+fake.var(0, ddof=1)+1e-8))
    fisher = np.mean(fishers, axis=0)
    ranking = np.argsort(-fisher, kind="stable")
    masks = {"native": []}
    for k in [8, 16, 40, 80]:
        masks[f"drop_top{k}"] = ranking[:k].tolist()
        for seed in range(200, 220):
            masks[f"drop_random{k}_s{seed}"] = np.random.default_rng(seed).permutation(160)[:k].tolist()
    save(out / "channel_masks.json", {"ranking_T_macroFisher": ranking.tolist(), "masks_removed_indices": masks})
    pd.DataFrame({"channel": np.arange(160), "T_macroFisher": fisher}).to_csv(out / "channel_statistics.csv", index=False)
    records, df_scores = [], None
    for name, data in datasets.items():
        labels = data.rows.y.to_numpy()
        scores = {case: masked_cosine(data.z, data.centroid, removed) for case, removed in masks.items()}
        assert np.max(np.abs(scores["native"]-geoms[name]["cosine"])) < 1e-12
        for case, score in scores.items():
            records.append({"dataset": name, "case": case, "removed": len(masks[case]), **metrics(labels, score)})
        if name == "DF_pilot":
            df_scores = scores
            frame = data.rows[["utterance_id", "speaker_group", "y"]].copy()
            frame = pd.concat([frame, pd.DataFrame(scores)], axis=1)
            frame.to_parquet(out / "DF_masked_cosine_scores.parquet", index=False)
        print("channels complete", name, flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(out / "channel_ablation_metrics.csv", index=False)
    curves = []
    for name in datasets:
        native = frame[(frame.dataset == name) & (frame.case == "native")].iloc[0].eer_percent
        for k in [8, 16, 40, 80]:
            subset = frame[(frame.dataset == name) & (frame.removed == k)]
            top = float(subset[subset.case == f"drop_top{k}"].iloc[0].eer_percent)
            random = subset[subset.case.str.startswith("drop_random")].eer_percent.to_numpy()
            curves.append({"dataset": name, "removed": k, "native_EER": native, "top_EER": top,
                           "top_impairment_pp": top-native, "random_min": float(random.min()),
                           "random_q025": float(np.quantile(random, .025)), "random_median": float(np.median(random)),
                           "random_q975": float(np.quantile(random, .975)), "random_max": float(random.max()),
                           "random_masks_worse_than_top": int((random > top).sum()), "random_count": len(random)})
    pd.DataFrame(curves).to_csv(out / "channel_ablation_curves.csv", index=False)
    data = datasets["DF_pilot"]
    labels, groups = data.rows.y.to_numpy(), data.rows.speaker_group.to_numpy()
    speakers, codes = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(7301)
    draws = rng.multinomial(len(speakers), np.full(len(speakers), 1/len(speakers)), size=2000)
    estimates = {}
    for case in ["native", "drop_top8", "drop_top16", "drop_top40", "drop_top80"]:
        prepared = weighted_eer_setup(labels, df_scores[case], codes)
        assert abs(weighted_eer(prepared, np.ones(len(speakers)))-eer(labels, df_scores[case])) < 1e-9
        estimates[case] = np.asarray([weighted_eer(prepared, draw) for draw in draws])
    intervals = []
    for case in list(estimates)[1:]:
        delta = estimates[case]-estimates["native"]
        good = np.isfinite(delta)
        intervals.append({"case": case, "EER_impairment_pp": eer(labels, df_scores[case])-eer(labels, df_scores["native"]),
                          "CI95": np.quantile(delta[good], [.025, .975]).tolist(),
                          "family99": np.quantile(delta[good], [.005, .995]).tolist(),
                          "dropped_replicates": int((~good).sum())})
    save(out / "channel_paired_intervals.json", intervals)
    return curves, intervals


def spherical_kmeans(u, k=4):
    best = None
    for seed in range(501, 506):
        rng = np.random.default_rng(seed)
        centers = [u[rng.integers(len(u))]]
        for _ in range(1, k):
            distance = np.maximum(0., 1-np.max(u@np.asarray(centers).T, axis=1))
            centers.append(u[rng.choice(len(u), p=distance/distance.sum())])
        centers = np.asarray(centers)
        empty = 0
        for iteration in range(100):
            labels = np.argmax(u@centers.T, axis=1)
            updated = []
            for j in range(k):
                mean = u[labels == j].mean(0) if np.any(labels == j) else np.zeros(u.shape[1])
                if np.linalg.norm(mean) < 1e-12:
                    mean = u[rng.integers(len(u))]
                    empty += 1
                updated.append(mean/np.linalg.norm(mean))
            updated = np.asarray(updated)
            change = np.max(np.abs(centers-updated))
            centers = updated
            if change < 1e-7:
                break
        objective = float(np.max(u@centers.T, axis=1).sum())
        candidate = {"seed": seed, "objective": objective, "iterations": iteration+1, "empty_reinitializations": empty}
        if best is None or objective > best[0]:
            best = (objective, centers.copy(), candidate)
    return best[1], best[2]


def condition_diagnostics(datasets, geoms, out):
    yt = datasets["T"].rows.y.to_numpy()
    rng = np.random.default_rng(17)
    ix = np.concatenate([rng.choice(np.flatnonzero((yt == y) & geoms["T"]["valid"]), 2500, replace=False) for y in [0, 1]])
    centers, fit = spherical_kmeans(geoms["T"]["direction"][ix])
    assert np.max(np.abs(centers@datasets["T"].centroid)) < 1e-10
    np.savez(out / "direction_clusters_T.npz", centers=centers, fit_indices=ix)
    duration_edges = np.unique(np.quantile(datasets["T"].rows.duration, [.2, .4, .6, .8]))
    pads = datasets["T"].rows.padding_fraction.to_numpy()
    padding_edges = np.unique(np.r_[0., np.quantile(pads[pads > 0], [.25, .5, .75])])
    save(out / "cluster_registry.json", {"fit_dataset": "T", "fit": fit, "duration_edges": duration_edges.tolist(),
                                         "padding_edges": padding_edges.tolist(), "K": 4})
    associations, descriptions = [], []
    for name, data in datasets.items():
        valid = geoms[name]["valid"]
        clusters = np.full(len(data.rows), -1)
        clusters[valid] = np.argmax(geoms[name]["direction"][valid]@centers.T, axis=1)
        rows = data.rows.copy()
        rows["cluster"] = clusters
        rows["duration_bin"] = np.searchsorted(duration_edges, rows.duration.to_numpy(), side="right")
        rows["padding_bin"] = np.searchsorted(padding_edges, rows.padding_fraction.to_numpy(), side="left")
        rows[["utterance_id", "cluster", "y", "duration_bin", "padding_bin"]].to_parquet(out / f"{name}_clusters.parquet", index=False)
        fields = ["y", "duration_bin", "padding_bin"]
        fields += [field for field in ["source_corpus", "codec"] if field in rows]
        for field in fields:
            for scope, subset in [("all", valid), ("bona", valid & (rows.y.to_numpy() == 1)),
                                  ("spoof", valid & (rows.y.to_numpy() == 0))]:
                if field == "y" and scope != "all":
                    continue
                selected = rows.loc[subset]
                informative = selected.cluster.nunique() > 1 and selected[field].nunique() > 1
                associations.append({"dataset": name, "scope": scope, "field": field, "rows": len(selected),
                                     "n_clusters": selected.cluster.nunique(), "categories": selected[field].nunique(),
                                     "NMI": float(normalized_mutual_info_score(selected[field], selected.cluster)) if informative else 0.,
                                     "ARI": float(adjusted_rand_score(selected[field], selected.cluster)) if informative else 0.,
                                     "informative": informative})
                ct = pd.crosstab(selected.cluster, selected[field], normalize="index")
                ct.to_csv(out / f"cluster_{name}_{scope}_{field}.csv")
        for cluster, subset in rows.loc[valid].groupby("cluster"):
            descriptions.append({"dataset": name, "cluster": int(cluster), "n": len(subset),
                                 "bona_fraction": float(subset.y.mean()), "duration_median": float(subset.duration.median()),
                                 "padding_median": float(subset.padding_fraction.median())})
    pd.DataFrame(associations).to_csv(out / "cluster_associations.csv", index=False)
    pd.DataFrame(descriptions).to_csv(out / "cluster_descriptions.csv", index=False)
    return associations, descriptions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_ROOT / "local_analysis/cached_core_v1")
    parser.add_argument("--resume-tail", action="store_true", help="Reuse saved completed CV; recover matching/cluster tail only")
    args = parser.parse_args()
    if args.resume_tail:
        assert not (args.out / "complete").exists()
        lock = json.loads((args.out / "protocol_lock.json").read_text())
        assert hashlib.sha256((args.out / "runner_at_lock.py").read_bytes()).hexdigest() == lock["script_sha256"]
        assert json.loads((args.out / "probe_progress.json").read_text())["outer_folds_complete"] == 5
        save(args.out / "recovery.json", {
            "reason": "empty spoof stratum had float indices; explicit integer dtype fixes matching only",
            "original_script_sha256": lock["script_sha256"],
            "recovered_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "settings_changed": False, "probe_refits": 0,
            "reused_predictions_sha256": hashlib.sha256((args.out / "DF_nested_OOF_predictions.parquet").read_bytes()).hexdigest()})
    else:
        args.out.mkdir(parents=True, exist_ok=False)
        lock = protocol()
        lock["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        save(args.out / "protocol_lock.json", lock)
    started = time.monotonic()
    datasets, geoms, checks = {}, {}, []
    with threadpool_limits(limits=2):
        for name in ["T", "V", "DF_pilot"]:
            checks.append(validate_run(name))
            data = load_embeddings(name, expected_checkpoint_sha256=CHECKPOINT)
            geom = data.geometry()
            assert np.max(np.abs(geom["cosine"]-data.rows.cosine_s.to_numpy())) < 1e-10
            assert np.max(np.abs(geom["direction"]@data.centroid)) < 1e-10
            assert geom["valid"].all()
            datasets[name], geoms[name] = data, geom
        c = datasets["T"].centroid
        assert all(np.allclose(data.centroid, c, rtol=0, atol=1e-12) for data in datasets.values())
        print("cache checks complete", flush=True)
        # Prioritize the direct fixed-score intervention before expensive probes.
        if args.resume_tail:
            curves = pd.read_csv(args.out / "channel_ablation_curves.csv").to_dict("records")
            channel_ci = json.loads((args.out / "channel_paired_intervals.json").read_text())
            frame = pd.read_parquet(args.out / "DF_nested_OOF_predictions.parquet")
            assert np.array_equal(frame.utterance_id, datasets["DF_pilot"].rows.utterance_id)
            assert np.array_equal(frame.y, datasets["DF_pilot"].rows.y)
            predictions = {case: frame[case].to_numpy() for case in CASES}
            scores = {case: frame[case+"_logit"].to_numpy() for case in CASES}
            probe_metrics = pd.read_csv(args.out / "probe_metrics.csv").to_dict("records")
            fold = frame.fold.to_numpy()
        else:
            curves, channel_ci = channel_diagnostics(datasets, geoms, args.out)
            predictions, scores, probe_metrics, fold = nested_probes(datasets["DF_pilot"], geoms["DF_pilot"], args.out)
        df = datasets["DF_pilot"]
        pairs = [("scalar", "direction"), ("noise", "direction"),
                 ("metadata", "metadata_direction"), ("postLN_z", "preLN_h")]
        probe_ci = ll_intervals(df.rows.y.to_numpy(), predictions, df.rows.speaker_group.to_numpy(), pairs)
        save(args.out / "probe_paired_intervals.json", probe_ci)
        matches, primary = matching_diagnostics(df, geoms["DF_pilot"], predictions, scores, args.out)
        associations, descriptions = condition_diagnostics(datasets, geoms, args.out)
        summary = {"status": "complete_exploratory", "protocol_sha256": hashlib.sha256((args.out / "protocol_lock.json").read_bytes()).hexdigest(),
                   "cache_validation": checks, "GPU_inference_runs": 0, "EBM_retrained": False, "Q_opened": False,
                   "scope": "DF-internal nested speaker CV and T-fixed channel/cluster diagnostics; not new transfer improvement",
                   "probe_metrics": probe_metrics, "probe_intervals": probe_ci,
                   "matched": matches, "channel_curves": curves, "channel_intervals": channel_ci,
                   "cluster_associations": associations, "cluster_descriptions": descriptions,
                   "elapsed_seconds": (datetime.now(timezone.utc)-datetime.fromisoformat(lock["created_utc"])).total_seconds(),
                   "this_invocation_seconds": time.monotonic()-started,
                   "tail_recovered_without_probe_refit": args.resume_tail, "limits": lock["limits"]}
        save(args.out / "summary.json", summary)
        (args.out / "complete").write_text("0\n", encoding="utf-8")
        print(json.dumps({"status": summary["status"], "out": str(args.out), "elapsed_seconds": summary["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
