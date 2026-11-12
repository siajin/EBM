"""Offline exploratory interpretation; never changes the frozen DF registry."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from geometry_cache import DEFAULT_ROOT, load_embeddings, load_predictions, read_json, resolve_run


def main():
    folder = Path(__file__).parent
    # Import the already synthetic-checked weighted EER implementation.
    for alias, name in [("fast_analyze_v1", "fast_analyze.py"), ("df_extract_v1", "df_extract.py"), ("df_screen_v1", "df_screen.py")]:
        spec = importlib.util.spec_from_file_location(alias, folder / name)
        module = importlib.util.module_from_spec(spec)
        sys.modules[alias] = module
        spec.loader.exec_module(module)
    screen = sys.modules["df_screen_v1"]
    data = load_embeddings("DF_pilot")
    train = load_embeddings("T")
    pred = load_predictions("DF_pilot")
    assert pred.utterance_id.tolist() == data.rows.utterance_id.tolist()
    root, entry, _ = resolve_run("DF_pilot")
    metrics = {r["case"]: r for r in read_json(root / "DF_evaluation/metrics.json")}
    names = [f"E_{mode}_{gamma}" for mode in ["diagonal", "full"] for gamma in [0.01, 0.1, 0.5]]
    contrasts = screen.bootstrap_contrasts(pred.y.to_numpy(),
                {n: pred[n].to_numpy() for n in ["native_cosine"]+names},
                data.rows.dependency_group.to_numpy(), [("native_cosine", n) for n in names])
    registry = read_json(root / "registry.json")
    native = data.rows.cosine_s.to_numpy()
    order = np.argsort(native, kind="stable")
    probability = pred.C0.to_numpy()[order]
    diagnostic = {
        "C0_adjacent_probability_decreases_tol1e8": int((np.diff(probability) < -1e-8).sum()),
        "C0_max_accumulated_probability_reversal": float(np.max(np.maximum.accumulate(probability)-probability)),
        "C0_uses_spline_not_monotonic_calibration": True,
        "V_fixed_native_threshold": registry["thresholds"]["native_cosine"],
        "T_bona_cosine_quantiles": train.rows.loc[train.rows.y == 1, "cosine_s"].quantile([.05,.5,.95]).to_dict(),
        "DF_bona_cosine_quantiles": data.rows.loc[data.rows.y == 1, "cosine_s"].quantile([.05,.5,.95]).to_dict(),
    }
    report = {"status": "exploratory_posthoc_interpretation", "DF_n": len(pred),
              "cache_fingerprint": entry["cache_fingerprint"],
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "uncertainty_note": "Paired source-speaker bootstrap 2000, 95%; source families unresolved. Posthoc six-candidate panel, no multiplicity correction. Not an independent confirmatory test.",
              "metric_contrasts": contrasts, "probe_diagnostic": diagnostic}
    out = DEFAULT_ROOT / "local_analysis/df_interpretation_v1"
    out.mkdir(parents=True, exist_ok=True)
    (out / "interpretation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    pd.DataFrame(contrasts).to_csv(out / "metric_contrasts.csv", index=False)
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 10})
    fig, (ax, right) = plt.subplots(1, 2, figsize=(13, 5.8), gridspec_kw={"width_ratios": [1.1, 1]})
    main_cases = ["native_cosine", "C1v", "C2", "A_all", "A_top25"]
    labels = ["기존 코사인", "코사인·norm 후처리", "방향 추가 후처리", "전체 160채널 분류기", "상위 40채널 분류기"]
    values = [metrics[n]["eer_percent"] for n in main_cases]
    bars = ax.barh(np.arange(5), values, color=["#35495e"]+["#4679a8"]*4, height=.62)
    ax.set_yticks(np.arange(5), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 17)
    ax.set_xlabel("EER (%) · 낮을수록 좋음")
    ax.set_title("기존 코사인과 후처리 분류기", loc="left", fontweight="bold", pad=15)
    for bar, value in zip(bars, values):
        ax.text(value+.18, bar.get_y()+bar.get_height()/2, f"{value:.2f}%", va="center")
    delta_labels = ["대각 γ=0.01", "대각 γ=0.1", "대각 γ=0.5", "전체 γ=0.01", "전체 γ=0.1", "전체 γ=0.5"]
    for i, record in enumerate(contrasts):
        point = record["delta_EER_pp"]
        low, high = record["CI95"]
        right.errorbar(point, i, xerr=[[point-low],[high-point]], fmt="o", color="#4679a8", capsize=4, markersize=6)
    right.axvline(0, color="#35495e", linewidth=1, linestyle="--")
    right.set_yticks(np.arange(6), delta_labels)
    right.invert_yaxis()
    right.set_xlabel("기존 코사인 EER - 거리 방식 EER (%p)\n양수일수록 개선 · 점=관측 차이, 선=95% 구간")
    right.set_title("분산을 반영한 거리 · 탐색 비교", loc="left", fontweight="bold", pad=15)
    for axis in [ax, right]:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="x", alpha=.16)
        axis.set_axisbelow(True)
    fig.suptitle("ASVspoof2021 DF 예비 표본 20,000개: 정상 2,000 · 스푸프 18,000", fontsize=14, fontweight="bold", y=.98)
    fig.text(.02, .025, "모델 재학습 없음 · T에서 fit, V에서 설정 선택 · 오른쪽은 사후 탐색 6개 비교(다중비교 보정 없음)\nsource·화자 93그룹 paired bootstrap 2,000회 · 원발화 파생관계 미확인\n기존에 관찰한 DF의 고정 표본이며 전체 DF 성능 또는 독립적인 확증 결과가 아님", fontsize=9, color="#41464d")
    fig.subplots_adjust(left=.20, right=.98, bottom=.24, top=.84, wspace=.72)
    fig.savefig(out / "df_results.png", dpi=180, facecolor="white")
    plt.close(fig)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("FIGURE", out / "df_results.png")


if __name__ == "__main__":
    main()
