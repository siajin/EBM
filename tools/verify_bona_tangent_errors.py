"""Independent aggregation checks and disclosed sensitivity of offline diagnosis."""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "2"
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analyze_bona_tangent_errors import cluster_difference

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/geometry_cache/local_analysis/bona_tangent_errors_20261111_v1"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    main_result = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    historical = 0.9883067
    checks, sensitivities, subgroups, bins, axes = [], [], [], [], []
    for index, name in enumerate(["V", "DF_pilot", "21LA_pilot", "19LA_eval", "DF_full", "21LA_eval"]):
        f = pd.read_parquet(OUT / f"{name}_bona_derived.parquet")
        original = main_result["datasets"][name]
        assert int(f.error.sum()) == original["errors"]
        e, a = f[f.error], f[~f.error]
        # Independent scalar, cell-by-cell aggregation (not matrix bootstrap).
        numerator, denominator = 0., 0
        for _, cell in f.groupby(["duration_band", "source_corpus", "codec"], observed=True):
            error = cell[cell.error]
            accepted = cell[~cell.error]
            if len(error) and len(accepted) >= 2:
                numerator += len(error) * (error.rho_8.mean() - accepted.rho_8.mean())
                denominator += len(error)
        reference = original["duration_source_codec_adjusted"]
        if denominator:
            difference = abs(numerator / denominator - reference["delta_rho"])
            assert difference < 1e-12 and denominator == reference["error_n_supported"]
        else:
            difference = None
        assert np.allclose(f.rho_8 * f.residual_energy, f.parallel_energy_8, atol=1e-10)
        assert np.allclose(f[[f"pc{i}_fraction" for i in range(1, 9)]].sum(1), f.rho_8, atol=1e-10)
        h = f.copy()
        h["error"] = h.score < historical
        hresult = {"dataset": name, "threshold": historical, "bona_n": len(h), "errors": int(h.error.sum()),
                   "error_percent": float(100 * h.error.mean()),
                   "rho8_error": float(h.loc[h.error, "rho_8"].mean()) if h.error.any() else None,
                   "rho8_accepted": float(h.loc[~h.error, "rho_8"].mean()) if (~h.error).any() else None,
                   "duration_source_codec_adjusted": cluster_difference(h, ["duration_band", "source_corpus", "codec"], 7123 + index),
                   "speaker_duration_source_codec_adjusted": cluster_difference(h, ["speaker_group", "duration_band", "source_corpus", "codec"], 8123 + index)}
        sensitivities.append(hresult)
        # Score-sign check: sin^2(theta) is not monotonic for negative scores.
        for tag, select in [("score_nonnegative", f.score >= 0), ("score_negative", f.score < 0)]:
            b = f.loc[select]
            subgroups.append({"dataset": name, "subgroup": tag, "bona_n": len(b), "errors": int(b.error.sum()),
                              "rho8_error": float(b.loc[b.error, "rho_8"].mean()) if b.error.any() else None,
                              "rho8_accepted": float(b.loc[~b.error, "rho_8"].mean()) if (~b.error).any() else None,
                              "adjusted": cluster_difference(b, ["duration_band", "source_corpus", "codec"], 9123 + index) if len(b) else {"status": "empty"}})
        # PC1 is pre-specified in k sensitivity; quantify which Train direction
        # carries the DF association, with the same covariate support rules.
        pc1 = f.copy()
        pc1["rho_8"] = pc1.pc1_fraction
        axes.append({"dataset": name,
                     "PC1_error": float(e.pc1_fraction.mean()) if len(e) else None,
                     "PC1_accepted": float(a.pc1_fraction.mean()) if len(a) else None,
                     "adjusted": cluster_difference(pc1, ["duration_band", "source_corpus", "codec"], 10123 + index)})
        for key, frame in f.groupby(["source_corpus", "codec", "duration_band", "error"], observed=True):
            bins.append({"dataset": name, "source": key[0], "codec": key[1], "duration": key[2],
                         "error": bool(key[3]), "n": len(frame), "rho8": float(frame.rho_8.mean()),
                         "PC1_fraction": float(frame.pc1_fraction.mean())})
        checks.append({"dataset": name, "unique_UIDs": bool(f.utterance_id.is_unique),
                       "independent_adjusted_delta_error": difference,
                       "energy_ratio_identity": "pass", "sum_PC_fraction_identity": "pass"})
    # Validate all consumed caches again after both analyses.
    hashes = json.loads((OUT / "input_hashes.json").read_text(encoding="utf-8"))
    for path, value in hashes.items():
        assert sha(path) == value
    payload = {"status": "pass", "independent_scalar_checks": checks,
               "historical_threshold_sensitivity": sensitivities,
               "score_sign_sensitivity": subgroups, "PC1_adjusted": axes,
               "consumed_input_hashes_unchanged": len(hashes),
               "sensitivity_status": "Historical threshold check added after primary result; no target tuning",
               "resource_use": "local CPU only; no GPU, network, model updates or remote operations"}
    (OUT / "independent_validation.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame(bins).to_csv(OUT / "source_codec_duration_groups.csv", index=False)
    # Focused scientific chart using full normal populations; primary k=8 kept.
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    names = ["DF_full", "21LA_eval"]
    for axis, k, title in [(ax[0], 1, "Train PC1"), (ax[1], 8, "Train top-8 directions (primary)")]:
        for pos, key, label, color in [(-.18, "accepted", "Accepted bona fide", "#4776a3"),
                                       (.18, "error", "Rejected bona fide", "#cc7546")]:
            vals = []
            for name in names:
                f = pd.read_parquet(OUT / f"{name}_bona_derived.parquet")
                s = f.error if key == "error" else ~f.error
                vals.append(100 * f.loc[s, f"rho_{k}"].mean())
            bars = axis.bar(np.arange(2) + pos, vals, .32, label=label, color=color)
            axis.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=10)
        axis.set(xticks=[0, 1], xticklabels=["DF (14,869 bona fide)", "21LA (14,816 bona fide)"],
                 ylim=(0, 100), ylabel="Residual energy share (%)", title=title)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=.15)
        axis.set_axisbelow(True)
    ax[1].legend(fontsize=9, loc="upper right")
    fig.suptitle("Archived batch160 EBM: errors concentrate differently across datasets", fontsize=12)
    fig.savefig(OUT / "main_comparison.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"status": "pass", "historical_threshold_sensitivity": sensitivities,
                      "PC1_adjusted": axes, "score_sign_sensitivity": subgroups}, ensure_ascii=False))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
