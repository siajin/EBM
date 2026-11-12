"""Refine the existing offline DF ablation without changing its saved basis."""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "2"
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import evaluate_low_occupancy_ablation as base
from verify_low_occupancy_ablation import eer_independent

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "artifacts/geometry_cache"
PREVIOUS = CACHE / "local_analysis/low_occupancy_ablation_20261111_v1"
OUT = CACHE / "local_analysis/low_occupancy_ablation_152_158_20261111_v1"
KS = [0, 152, 153, 154, 155, 156, 157, 158]


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def main():
    started = time.monotonic()
    assert (OUT / "plan.json").exists() and not (OUT / "results.json").exists()
    protected = {str(p): base.digest(p) for p in (ROOT / "EBM").rglob("*.py") if "__pycache__" not in p.parts}
    save("source_hashes_before.json", protected)
    base.KS, base.STARTED = KS, started
    for name in ("train_only_bases.npz", "metrics.csv", "plan.json", "results.json"):
        base.check(PREVIOUS / name)
    base.check(CACHE / "catalog.json")
    archive = np.load(PREVIOUS / "train_only_bases.npz")
    basis, c, eigenvalues = archive["low_energy_first"], archive["centroid"], archive["eigenvalues_low_first"]
    assert np.max(abs(basis.T @ basis - np.eye(159))) < 1e-10
    assert np.max(abs(basis.T @ c)) < 1e-10
    catalog = json.loads((CACHE / "catalog.json").read_text())
    tv, tvshards = base.load_sources(catalog, ["19la_tv"])
    vscore, _, vcheck = base.transform_scores(tv, tvshards, c, basis)
    vmask = tv.split.to_numpy() == "V"
    vy = tv.y.to_numpy()[vmask]
    thresholds = {k: base.eer_and_dev_threshold(vy, vscore[i, vmask])[1] for i, k in enumerate(KS)}
    del vscore
    print("Saved basis unchanged; V thresholds fixed. Evaluating all 533,928 DF rows.", flush=True)
    frame, shards = base.load_sources(catalog, ["21df_pilot", "frozen_DF_remaining"])
    assert len(frame) == 533928 and int(frame.y.sum()) == 14869
    y = frame.y.to_numpy()
    scores, removed, geometry_checks = base.transform_scores(frame, shards, c, basis)
    previous = pd.read_csv(PREVIOUS / "metrics.csv", float_precision="round_trip")
    previous = previous[(previous.dataset == "DF") & (previous.method == "low_energy_first")].set_index("k")
    rows, eer_errors = [], []
    stored = frame[["utterance_id", "y"]].copy()
    for i, k in enumerate(KS):
        score = scores[i]
        eer, _ = base.eer_and_dev_threshold(y, score)
        independent = eer_independent(y, score)
        error = abs(eer - independent)
        assert error < 1e-10
        eer_errors.append(error)
        row = {"removed_dimensions": k, "remaining_residual_dimensions": 159 - k,
               "eer_percent": eer, "V_threshold": thresholds[k],
               **{"fixed_" + key: value for key, value in base.operating_rates(y, score, base.FIXED).items()},
               **{"dev_" + key: value for key, value in base.operating_rates(y, score, thresholds[k]).items()},
               "normal_embedding_energy_removed": float(removed[i, y == 1].mean()),
               "spoof_embedding_energy_removed": float(removed[i, y == 0].mean())}
        if k in previous.index:
            old = previous.loc[k]
            assert abs(eer - old.eer_percent) < 1e-10
            assert abs(row["fixed_normal_reject_percent"] - old.fixed_normal_reject_percent) < 1e-10
            assert abs(row["fixed_spoof_accept_percent"] - old.fixed_spoof_accept_percent) < 1e-10
        rows.append(row)
        stored[f"score_k{k}"] = score
    stored.to_parquet(OUT / "DF_scores.parquet", index=False)
    del stored, removed
    axis_groups, direct_errors = [], []
    # Six newly removed axes, in the exact frozen ascending-occupancy order.
    sums = {label: np.zeros((6, 2)) for label in (0, 1)}
    totals = {label: int((y == label).sum()) for label in (0, 1)}
    draw = np.random.default_rng(20261111).choice(len(frame), 128, replace=False)
    drawn_n = 0
    for offset, path, actual_c in shards:
        assert np.array_equal(actual_c, c)
        z = np.load(path, mmap_mode="r")
        for first in range(0, len(z), 20000):
            x = np.array(z[first:first + 20000], dtype=float)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            s = x @ c
            a2 = np.maximum(1 - s * s, 0)
            energy = (x @ basis[:, 152:158]) ** 2
            labels = y[offset + first:offset + first + len(x)]
            for label in (0, 1):
                mask = labels == label
                sums[label][:, 0] += energy[mask].sum(axis=0)
                sums[label][:, 1] += (energy[mask] / a2[mask, None]).sum(axis=0)
        chosen = draw[(draw >= offset) & (draw < offset + len(z))]
        if len(chosen):
            x = np.array(z[chosen - offset], dtype=float)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            for i, k in enumerate(KS):
                transformed = x - (x @ basis[:, :k]) @ basis[:, :k].T
                direct = transformed @ c / np.linalg.norm(transformed, axis=1)
                err = float(np.max(abs(direct - scores[i, chosen])))
                assert err < 1e-9
                direct_errors.append(err)
            drawn_n += len(chosen)
    assert drawn_n == 128
    for j, rank in enumerate(range(153, 159)):
        axis_groups.append({"removed_axis_rank_low_to_high": rank,
                            "train_normal_direction_energy_percent": float(100 * eigenvalues[rank - 1]),
                            "DF_normal_direction_energy_percent": float(100 * sums[1][j, 1] / totals[1]),
                            "DF_spoof_direction_energy_percent": float(100 * sums[0][j, 1] / totals[0]),
                            "DF_normal_embedding_energy_percent": float(100 * sums[1][j, 0] / totals[1]),
                            "DF_spoof_embedding_energy_percent": float(100 * sums[0][j, 0] / totals[0])})
    pd.DataFrame(axis_groups).to_csv(OUT / "newly_removed_axis_energy.csv", index=False)
    table = pd.DataFrame(rows)
    detail = table[table.removed_dimensions >= 152].copy()
    detail["eer_adjacent_delta_percentage_points"] = detail.eer_percent.diff()
    table.to_csv(OUT / "metrics.csv", index=False)
    detail.to_csv(OUT / "DF_152_to_158.csv", index=False)
    # Largest observed adjacent deterioration, not a fitted change-point test.
    worst = detail.loc[detail.eer_adjacent_delta_percentage_points.idxmax()]
    previous_k = int(worst.removed_dimensions) - 1
    previous_eer = float(detail.loc[detail.removed_dimensions == previous_k, "eer_percent"].iloc[0])
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 11})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    ax = axes[0]
    ax.plot(detail.removed_dimensions, detail.eer_percent, "o-", color="#356AA0")
    ax.axhline(float(table.iloc[0].eer_percent), color="#7B838B", linestyle="--", linewidth=1, label="제거 전 EER")
    for k, value in zip(detail.removed_dimensions, detail.eer_percent):
        ax.annotate(f"{value:.3f}", (k, value), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9)
    ax.set(xlabel="제거한 잔여 차원 수", ylabel="DF EER (%)", xticks=range(152, 159), ylim=(0, float(detail.eer_percent.max()) * 1.15))
    ax.legend(frameon=False, fontsize=9)
    ax = axes[1]
    increments = detail.iloc[1:]
    ax.bar(increments.removed_dimensions, increments.eer_adjacent_delta_percentage_points, color="#DB8D32")
    ax.axhline(0, color="#7B838B", linewidth=1)
    ax.set(xlabel="이번에 추가 제거한 축의 순번", ylabel="바로 이전 단계 대비 EER 변화 (%p)", xticks=range(153, 159))
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle("동일한 Train 정상 기준 축 순서 · DF 전체 533,928개\n보관 batch160 · epoch50 / 현재 학습 변경 없음", fontsize=12)
    fig.savefig(OUT / "DF_152_to_158.png", dpi=180)
    plt.close(fig)
    for path, expected in base.HASHES.items():
        assert base.digest(path) == expected
    for path, expected in protected.items():
        assert base.digest(path) == expected
    assert time.monotonic() - started < 900
    save("input_hashes.json", base.HASHES)
    validation = {"status": "passed", "DF_total": len(frame), "DF_normal": totals[1], "DF_spoof": totals[0],
                  "all_UIDs_unique": True, "saved_basis_refitted": False,
                  "old_k0_k152_k158_reproduced": True,
                  "independent_full_EER_max_error_percentage_points": max(eer_errors),
                  "explicit_projection_samples": drawn_n, "explicit_projection_max_score_error": max(direct_errors),
                  "cache_and_basis_hashes_unchanged": len(base.HASHES),
                  "existing_python_sources_unchanged": len(protected), "V_checks": vcheck, "DF_checks": geometry_checks}
    save("validation.json", validation)
    result = {"status": "complete", "model": "archived batch160 epoch50", "checkpoint_sha256": base.SHA,
              "largest_adjacent_deterioration": {"from_k": previous_k, "to_k": int(worst.removed_dimensions),
                                                 "from_EER_percent": previous_eer, "to_EER_percent": float(worst.eer_percent),
                                                 "delta_percentage_points": float(worst.eer_adjacent_delta_percentage_points)},
              "rows": detail.replace({np.nan: None}).to_dict(orient="records"), "axis_energy": axis_groups,
              "validation": validation, "elapsed_seconds": time.monotonic() - started,
              "scope": "same archived embeddings/basis; exploratory sequential removal, no retraining causality claim",
              "safety": {"training_modified": False, "model_updates": 0, "new_inferences": 0,
                         "remote_operations": 0, "new_spend_usd": 0, "paid_resources_created": 0,
                         "automatic_retries": 0, "analysis_process_finished": True}}
    save("results.json", result)
    print(detail[["removed_dimensions", "remaining_residual_dimensions", "eer_percent", "eer_adjacent_delta_percentage_points"]].to_string(index=False), flush=True)
    print(json.dumps(result["largest_adjacent_deterioration"]), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
