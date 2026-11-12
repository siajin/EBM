"""Export Korean figures for the EBM author from previously reviewed results.

Presentation only: no inference, fitting, mask search, or remote mutation.
"""

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, patches
from matplotlib.ticker import FixedLocator, FixedFormatter
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score

BASE = Path(__file__).resolve().parents[2]
CACHE = BASE / "artifacts/geometry_cache/local_analysis"
OUT = BASE / "docs/experiments/cached_core_20261105/figures_author"
OUT.mkdir(parents=True, exist_ok=True)

for name in ["malgun.ttf", "malgunbd.ttf"]:
    font_manager.fontManager.addfont(str(Path("C:/Windows/Fonts") / name))
plt.rcParams.update({
    "font.family": ["Malgun Gothic", "DejaVu Sans"], "font.size": 15,
    "axes.titlesize": 17, "axes.labelsize": 16,
    "xtick.labelsize": 14, "ytick.labelsize": 14,
    "legend.fontsize": 13, "axes.unicode_minus": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#75808b", "axes.labelcolor": "#25313e",
    "text.color": "#25313e", "xtick.color": "#25313e",
    "ytick.color": "#25313e", "figure.facecolor": "white",
    "axes.facecolor": "white", "savefig.facecolor": "white",
    "svg.fonttype": "path",
})
BLUE, ORANGE, GRAY, INK = "#24649a", "#bc6128", "#7d838b", "#25313e"
PALE, LINE = "#eff5fa", "#d8e0e7"
inputs, output_data = {}, {}


def source(relative):
    p = CACHE / relative
    inputs[str(p.relative_to(BASE)).replace("\\", "/")] = hashlib.sha256(p.read_bytes()).hexdigest()
    return p


def js(relative):
    return json.loads(source(relative).read_text(encoding="utf-8"))


def save(fig, stem):
    fig.savefig(OUT / (stem + ".png"), dpi=240)
    plt.close(fig)


def heading(fig, title, sub):
    fig.text(.065, .957, title, fontsize=22, weight="bold", va="top")
    fig.text(.065, .91, sub, fontsize=13.5, color="#586572", va="top")


def box(ax, x, y, w, h, title, body, accent=BLUE, fontsize=15):
    ax.add_patch(patches.FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.012,rounding_size=0.012",
                 facecolor=PALE, edgecolor=LINE, linewidth=1.2))
    ax.text(x+w/2, y+h-.055, title, ha="center", va="top", fontsize=17,
            weight="bold", color=accent)
    ax.text(x+w/2, y+.04, body, ha="center", va="bottom", fontsize=fontsize, linespacing=1.65)


def arrow(ax, a, b, color=GRAY):
    ax.annotate("", xy=b, xytext=a,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8, mutation_scale=20))


def workflow():
    rank = js("cached_core_v1/channel_masks.json")["ranking_T_macroFisher"]
    assert rank[-2:] == [113, 95]
    fig = plt.figure(figsize=(12.5, 8.6))
    heading(fig, "실험 방식: 같은 임베딩과 센트로이드에서 좌표 선택",
            "채널 = 최종 임베딩의 좌표  |  추가 학습 없이 기존 캐시의 코사인 점수 재계산")
    ax = fig.add_axes([.04, .16, .92, .68])
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    box(ax, .025, .37, .23, .23, "기존 EBM 전체 추론",
        "음성 입력 → 전체 신경망\n→ 최종 LayerNorm", fontsize=14)
    box(ax, .34, .62, .29, .25, "최종 임베딩 z",
        "기존: 160개 좌표\n선택: z[113], z[95]", fontsize=16)
    box(ax, .34, .13, .29, .25, "학습된 센트로이드 c",
        "기존: 160개 좌표\n선택: c[113], c[95]", fontsize=16)
    box(ax, .72, .34, .255, .29, "두 벡터의 코사인",
        "선택한 두 벡터를\n각각 다시 정규화\n→ 코사인 유사도", fontsize=15)
    arrow(ax, (.26, .49), (.325, .735))
    arrow(ax, (.64, .74), (.705, .55))
    arrow(ax, (.64, .255), (.705, .42))
    ax.text(.485, .505, "같은 위치의 좌표 선택", ha="center", fontsize=16, weight="bold")
    ax.text(.485, .44, "두 좌표 예시: 113번·95번 (0부터 센 번호)", ha="center", fontsize=12.5)
    fig.text(.065, .153, "선택 기준: 2019 LA 학습 데이터의 공격별 Fisher score를 평균한 순위", fontsize=15)
    fig.text(.065, .108, "해당 두 좌표는 Fisher 하위 2개이며, 학습된 가중치가 가장 큰 좌표를 고른 것이 아님.", fontsize=14, color=BLUE)
    fig.text(.065, .055, "160차원 임베딩 생성 과정은 그대로 수행. 모델 축소 또는 추론 속도 향상을 검증한 실험이 아님.", fontsize=13, color="#586572")
    save(fig, "01_method")


def load_sweep():
    frames = []
    names = ["kept", "native_EER", "top_EER", "bottom_EER", "random_EER_median", "random_EER_min", "random_EER_max"]
    for folder in ["channel_keep_sweep_v1", "channel_keep_tiny_v1", "channel_keep_one_v3"]:
        f = pd.read_csv(source(folder + "/curves.csv"))
        frames.append(f.loc[f.dataset.eq("DF_pilot"), names])
    # Higher-count experiments removed the highest-ranked coordinates.
    old = pd.read_csv(source("cached_core_v1/channel_ablation_curves.csv"))
    old = old.loc[old.dataset.eq("DF_pilot") & old.removed.isin([8, 16, 40])]
    for _, r in old.iterrows():
        frames.append(pd.DataFrame([dict(kept=160-int(r.removed), native_EER=r.native_EER,
            top_EER=np.nan, bottom_EER=r.top_EER, random_EER_median=r.random_median,
            random_EER_min=r.random_min, random_EER_max=r.random_max)]))
    frame = pd.concat(frames, ignore_index=True)
    for _, g in frame.groupby("kept"):
        for n in names[1:]:
            v = g[n].dropna().to_numpy()
            assert not len(v) or np.max(v)-np.min(v) < 1e-10
    frame = frame.drop_duplicates("kept", keep="last")
    native = float(frame.native_EER.iloc[0])
    frame = pd.concat([frame, pd.DataFrame([dict(kept=160, **{n:native for n in names[1:]})])])
    frame = frame.sort_values("kept", ascending=False).reset_index(drop=True)
    assert len(frame) == 16
    assert abs(frame.loc[frame.kept.eq(2), "bottom_EER"].iloc[0]-2.922222222222222) < 1e-12
    frame.to_csv(OUT / "eer_values.csv", index=False)
    output_data["eer"] = frame.to_dict(orient="records")
    return frame


def sweep():
    f = load_sweep()
    interval = next(r for r in js("channel_keep_tiny_v1/paired_intervals.json") if r["case"] == "keep_bottom2")
    delta, (lo, hi) = interval["EER_impairment_pp"], interval["CI95_pointwise"]
    assert abs(delta-(f.loc[f.kept.eq(2), "bottom_EER"].iloc[0]-3.35)) < 1e-12
    output_data["two_coordinate_EER_delta"] = interval
    fig = plt.figure(figsize=(12.5, 9.8))
    heading(fig, "남기는 좌표 수에 따른 EER 변화",
            "ASVspoof 2021 DF 고정 표본: 정상 2,000개 · 스푸프 18,000개  |  EER는 낮을수록 좋음")
    gs = fig.add_gridspec(2, 1, left=.10, right=.96, top=.82, bottom=.19, hspace=.34, height_ratios=[1.5, 1])
    ax, small = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    x = f.kept.to_numpy()
    ax.fill_between(x, f.random_EER_min, f.random_EER_max, color=GRAY, alpha=.16,
                    label="무작위 20회 최소–최대")
    ax.plot(x, f.random_EER_median, "s--", color=GRAY, ms=5, lw=1.8, label="무작위 20회 중앙값")
    valid = f.top_EER.notna()
    ax.plot(x[valid], f.loc[valid, "top_EER"], "^--", color=ORANGE, ms=6, lw=2, label="Fisher 상위 유지")
    ax.plot(x, f.bottom_EER, "o-", color=BLUE, ms=6, lw=2.3, label="Fisher 하위 유지")
    ax.axhline(3.35, color=INK, ls=":", lw=1.5, label="전체 160개: 3.3500%")
    ax.set(ylim=(0, 22), ylabel="EER (%)", title="A. 세 가지 좌표 선택 조건")
    ax.legend(loc="upper left", frameon=False, ncol=2, fontsize=12)
    small.plot(x, f.bottom_EER, "o-", color=BLUE, ms=6, lw=2.3)
    small.axhline(3.35, color=INK, ls=":", lw=1.5)
    small.set(ylim=(2.70, 4.72), ylabel="EER (%)", xlabel="남기는 좌표 수 k (로그 눈금; 오른쪽으로 갈수록 감소)",
              title="B. Fisher 하위 유지 조건 확대 (세로축 범위 다름)")
    labels = [(160, 3.35, "160개: 3.3500%", (8, 12), "left"),
              (2, 2.922222222222222, "2개: 2.9222%", (-10, 19), "right"),
              (1, 4.345938907275787, "1개: 4.3459%", (-10, -18), "right")]
    for k, value, text, offset, align in labels:
        small.annotate(text, (k, value), xytext=offset, textcoords="offset points",
                       fontsize=12.5, color=BLUE, ha=align,
                       bbox=dict(facecolor="white", edgecolor="none", alpha=.9, pad=1.5))
    ticks = [160, 80, 40, 20, 10, 5, 3, 2, 1]
    for a in [ax, small]:
        a.set_xscale("log", base=2); a.set_xlim(180, .9)
        a.xaxis.set_major_locator(FixedLocator(ticks)); a.xaxis.set_major_formatter(FixedFormatter([str(k) for k in ticks]))
        a.grid(axis="y", color=LINE, lw=.8); a.set_axisbelow(True)
    ax.tick_params(axis="x", labelbottom=False)
    fig.text(.10, .112, f"2개 - 전체 EER 차이: {delta:+.4f}pp  |  개별 95% 신뢰구간: {lo:+.4f} ~ {hi:+.4f}pp", fontsize=13.5)
    fig.text(.10, .072, "신뢰구간이 0을 포함함. 무작위 최소–최대는 신뢰구간이 아님. 모든 점은 실제 계산한 조건.", fontsize=12.5, color="#586572")
    fig.text(.10, .037, "이전 결과를 관찰하며 유지 수를 확장한 탐색적 비교이며, 독립 평가에서의 개선을 입증하지 않음.", fontsize=12.5, color="#586572")
    save(fig, "02_eer_by_coordinate_count")


def roc_comparison():
    score_path = source("channel_keep_one_v3/DF_scores.parquet")
    d = pd.read_parquet(score_path, columns=["y", "native", "keep_bottom2"])
    assert len(d) == 20000 and d.y.sum() == 2000
    reference = js("channel_roc_explanation_v1/summary.json")
    metrics = {r["case"]: r for r in reference["model_metrics"]}
    fig = plt.figure(figsize=(12.5, 9.2))
    heading(fig, "EER와 운용 구간에 따른 오류율 비교",
            "ASVspoof 2021 DF 고정 표본 20,000개  |  두 오류율이 모두 낮을수록 좋음")
    gs = fig.add_gridspec(1, 2, left=.09, right=.96, top=.79, bottom=.39, wspace=.24)
    axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]
    series = [("native", INK, "전체 160개", "-"), ("keep_bottom2", BLUE, "Fisher 하위 2개", "--")]
    curve_values = []
    for key, color, label, style in series:
        fp, tp, _ = roc_curve(d.y, d[key], drop_intermediate=False)
        fn = 1-tp
        assert abs(roc_auc_score(d.y, d[key])-metrics[key]["auroc"]) < 1e-12
        for ax in axes:
            ax.plot(fn[::-1]*100, fp[::-1]*100, ls=style, color=color, lw=2.2, label=label)
        for target in [1, 3]:
            observed = fp[fn <= target/100+1e-12].min()*100
            expected = next(r["minimum_spoof_false_accept_percent"] for r in reference["matched_ROC_points"]
                            if r["case"] == key and r["bona_false_reject_cap_percent"] == target)
            assert abs(observed-expected) < 1e-10
            curve_values.append(dict(case=key, bona_false_reject_cap_percent=target, spoof_false_accept_percent=float(observed)))
        eer = metrics[key]["eer_percent"]
        axes[1].plot(eer, eer, "o" if key=="native" else "s", ms=8, color=color)
    axes[0].set(xlim=(0, 10), ylim=(0, 100), title="A. 정상 오거부율 0–10% 구간", ylabel="스푸프 통과율 (%)")
    axes[1].set(xlim=(.4, 5), ylim=(0, 19), title="B. EER 부근 확대 (축 범위 다름)")
    axes[1].plot([.4, 5], [.4, 5], ":", color=GRAY, lw=1.3, label="두 오류율이 같은 선")
    axes[1].annotate("전체 EER 3.3500%", (3.35, 3.35), xytext=(2.9, 7.3), fontsize=11.5,
                     arrowprops=dict(arrowstyle="-", color=INK), color=INK,
                     bbox=dict(facecolor="white", edgecolor="none", alpha=.95, pad=2))
    axes[1].annotate("2개 EER 2.9222%", (2.922222222, 2.922222222), xytext=(.65, 1.0), fontsize=11.5,
                     arrowprops=dict(arrowstyle="-", color=BLUE), color=BLUE,
                     bbox=dict(facecolor="white", edgecolor="none", alpha=.95, pad=2))
    axes[0].legend(frameon=False, loc="upper right", fontsize=12.5)
    axes[1].legend(frameon=False, loc="upper right", fontsize=10.5)
    for ax in axes:
        ax.set_xlabel("정상 오거부율 (%)")
        ax.grid(color=LINE, lw=.8); ax.set_axisbelow(True)
    t_ax = fig.add_axes([.09, .115, .87, .19]); t_ax.axis("off")
    rows = [["전체 160개", "3.3500%", "0.995943", "6.1056%", "3.5111%"],
            ["Fisher 하위 2개", "2.9222%", "0.987659", "15.8111%", "2.8889%"]]
    for row, (key, _, _, _) in zip(rows, series):
        assert row[1] == f"{metrics[key]['eer_percent']:.4f}%"
        assert row[2] == f"{metrics[key]['auroc']:.6f}"
        for column, target in [(3, 1), (4, 3)]:
            v = next(r['spoof_false_accept_percent'] for r in curve_values
                     if r['case'] == key and r['bona_false_reject_cap_percent'] == target)
            assert row[column] == f"{v:.4f}%"
    table = t_ax.table(cellText=rows,
        colLabels=["사용 좌표", "EER ↓", "AUROC ↑", "정상 오거부 ≤1%\n스푸프 통과율 ↓", "정상 오거부 ≤3%\n스푸프 통과율 ↓"],
        cellLoc="center", colLoc="center", colWidths=[.23, .14, .16, .235, .235], bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False); table.set_fontsize(12)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(LINE)
        if r==0:
            cell.set_facecolor(PALE); cell.set_text_props(weight="bold", fontsize=11.5)
        elif c==0:
            cell.set_text_props(color=INK if r==1 else BLUE)
    fig.text(.09, .07, "곡선이 교차하므로 낮은 EER가 모든 운용 구간의 우위를 뜻하지 않음.", fontsize=13.5)
    fig.text(.09, .03, "평가 표본에서 관찰한 곡선의 기술적 비교. 배포 문턱값을 이 표본에서 선정한 결과가 아님.", fontsize=12.5, color="#586572")
    output_data["roc_operating_points"] = curve_values
    output_data["model_metrics"] = reference["model_metrics"]
    save(fig, "03_roc_and_operating_points")


def captions():
    text = """# EBM 좌표 부분집합 실험: 참고 그림 및 캡션

아래 PNG는 흰 배경의 고해상도 이미지다. 워드에서 본문 너비에 맞춰 삽입할 수 있다.

## 그림 1. 좌표 부분집합 실험의 수행 과정

![좌표 부분집합 실험의 수행 과정](01_method.png)

기존 EBM 전체 추론으로 얻은 160차원 임베딩과 학습된 센트로이드에서 같은 좌표를 선택한 뒤, 선택 벡터를 각각 다시 정규화하여 코사인 유사도를 계산했다. 예시에 사용한 113번과 95번은 ASVspoof 2019 LA train에서 공격별 Fisher score를 평균한 순위의 하위 두 좌표다. 모델과 센트로이드의 재학습은 수행하지 않았다. 좌표 번호는 0부터 시작한다.

보고서 삽입 위치: 2. 실험 또는 3.2 좌표 선택 기준.

## 그림 2. 유지 좌표 수에 따른 EER

![유지 좌표 수에 따른 EER](02_eer_by_coordinate_count.png)

ASVspoof 2021 DF eval의 고정 표본 20,000개(정상 2,000개, 스푸프 18,000개)에서 유지 좌표 수와 선택 조건별 EER를 비교했다. 가로축은 로그 눈금이며 오른쪽으로 갈수록 좌표 수가 감소한다. 음영은 고정된 무작위 마스크 20개의 최소–최대 범위로, 신뢰구간이 아니다. 아래 패널은 Fisher 하위 유지 조건을 다른 세로축 범위로 확대했다. 하위 두 좌표의 EER는 2.9222%, 전체 160개는 3.3500%였다. 두 방법의 EER 차이에 대한 개별 95% 신뢰구간은 0을 포함했다. 유지 수는 이전 결과를 관찰하며 확장했으므로 탐색적 비교다.

보고서 삽입 위치: 4.1 유지 좌표 수에 따른 EER.

## 그림 3. 전체 좌표와 하위 두 좌표의 오류율 곡선 및 운용 구간 비교

![오류율 곡선과 운용 구간 비교](03_roc_and_operating_points.png)

같은 20,000개 표본에서 전체 160개와 Fisher 하위 두 좌표의 점수를 비교했다. 가로축은 정상 오거부율, 세로축은 스푸프 통과율이며 두 값 모두 낮을수록 좋다. 오른쪽 패널은 EER 부근을 확대했다. 두 좌표의 EER는 낮았지만 AUROC는 전체 160개가 높았고, 정상 오거부율을 1% 이하로 제한하는 구간에서도 전체 160개가 유리했다. 표의 스푸프 통과율은 각 정상 오거부율 상한을 만족하는 경험적 ROC 지점 중 최솟값이다. 이는 평가 곡선의 기술적 비교로, 배포 문턱값을 평가 표본에서 선정한 결과가 아니다.

보고서 삽입 위치: 4.2 EER와 AUROC 비교 또는 4.3 정상 오거부율 조건에 따른 비교.

## 그림 사용 범위

본 수치는 한 모델의 탐색적 분석이며 전체 2021 DF 평가셋 성능이나 논문 성능의 재현을 의미하지 않는다. 모든 좌표 조건에서 기존 160차원 임베딩 생성 과정은 유지했다.
"""
    (OUT / "captions.md").write_text(text, encoding="utf-8")


def main():
    workflow(); sweep(); roc_comparison(); captions()
    # Serialised data and hashes keep presentation traceable to the fixed results.
    def clean(value):
        if isinstance(value, dict): return {k:clean(v) for k,v in value.items()}
        if isinstance(value, list): return [clean(v) for v in value]
        if isinstance(value, float) and not np.isfinite(value): return None
        return value
    provenance = dict(scope="Presentation of existing, reviewed results only", new_inference=False,
        new_training=False, inputs=inputs, displayed_data=clean(output_data),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        output_pngs={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob("*.png")})
    (OUT / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(dict(output=str(OUT), pngs=[p.name for p in OUT.glob("*.png")],
                          input_files=len(inputs), status="data_checks_passed"), ensure_ascii=False))


if __name__ == "__main__":
    main()
