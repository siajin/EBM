"""Class-coloured point clouds for the channel-subset report, using cached z.

The actual selected two coordinates are distinguished from PCA/t-SNE views.
No EBM inference, training, score search, or external data transfer.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from threadpoolctl import threadpool_limits

from geometry_cache import load_embeddings, validate_run, resolve_run
from author_reference_figures import heading, BLUE, ORANGE, INK, LINE

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "artifacts/geometry_cache/local_analysis"
OUT = ROOT / "docs/experiments/cached_core_20261105/figures_clusters"
WORK = CACHE / "author_distribution_v1"
CHECKPOINT = "104873703a57cad98b686d770e0860e031a2cf9abef79b72668780c1660e0b24"
SEED = 20261105
STYLE = {0: (ORANGE, "스푸프"), 1: (BLUE, "정상")}


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def export(fig, name):
    fig.savefig(OUT / name, dpi=240)
    plt.close(fig)


def common(ax):
    ax.grid(color=LINE, linewidth=.7)
    ax.set_axisbelow(True)


def scatter(ax, coords, y, size=6, alpha=.24):
    for value in [0, 1]:
        color, label = STYLE[value]
        ix = np.flatnonzero(y == value)
        ax.scatter(coords[ix, 0], coords[ix, 1], s=size, c=color, alpha=alpha,
                   edgecolors="none", rasterized=True, label=f"{label} ({len(ix):,}개)", zorder=2+value)


def legend(ax, y, loc="upper left"):
    handles = [Line2D([], [], ls="", marker="o", ms=7, color=STYLE[v][0],
              label=f"{STYLE[v][1]} ({np.sum(y==v):,}개)") for v in [1, 0]]
    ax.figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(.52, .86),
                     ncol=2, frameon=False, fontsize=13)


def selected_pair(data, y):
    pair = data.z[:, [113, 95]].astype(float)
    c = data.centroid[[113, 95]]
    unit = pair/np.linalg.norm(pair, axis=1, keepdims=True)
    c_unit = c/np.linalg.norm(c)
    score = unit @ c_unit
    saved = pd.read_parquet(CACHE / "channel_keep_one_v3/DF_scores.parquet", columns=["utterance_id", "y", "keep_bottom2"])
    assert np.array_equal(saved.utterance_id, data.rows.utterance_id)
    assert np.array_equal(saved.y, y)
    error = float(np.max(np.abs(score-saved.keep_bottom2)))
    assert error < 1e-12

    fig = plt.figure(figsize=(12.5, 8.6))
    heading(fig, "선택한 두 좌표에서 정상·스푸프가 분포하는 모습",
            "ASVspoof 2021 DF 고정 표본 전체 20,000개  |  점 1개 = 발화 1개  |  PCA·t-SNE를 적용하지 않은 실제 좌표")
    gs = fig.add_gridspec(1, 2, left=.085, right=.96, top=.78, bottom=.30, wspace=.29)
    axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]
    scatter(axes[0], pair, y, size=7, alpha=.24)
    # Equal unit lengths and a common numeric range make raw directions readable.
    lo, hi = float(pair.min())-.18, float(pair.max())+.18
    axes[0].set(xlim=(lo, hi), ylim=(lo, hi), xlabel="임베딩 좌표 z[113]", ylabel="임베딩 좌표 z[95]",
                title="A. 두 좌표의 실제 값")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].axhline(0, color="#aab2bb", lw=.8)
    axes[0].axvline(0, color="#aab2bb", lw=.8)
    legend(axes[0], y)
    scatter(axes[1], unit, y, size=8, alpha=.16)
    theta = np.linspace(0, 2*np.pi, 1000)
    axes[1].plot(np.cos(theta), np.sin(theta), color="#aab2bb", lw=.9, zorder=1)
    axes[1].annotate("", xy=c_unit, xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.9, mutation_scale=20), zorder=5)
    axes[1].annotate("센트로이드 방향", xy=c_unit*.62, xytext=(-.97, .55), fontsize=12,
        arrowprops=dict(arrowstyle="-", color=INK),
        bbox=dict(facecolor="white", edgecolor="none", alpha=.95, pad=2), zorder=6)
    axes[1].set(xlim=(-1.15, 1.15), ylim=(-1.15, 1.15),
                xlabel="정규화한 z[113]", ylabel="정규화한 z[95]", title="B. 두 좌표를 L2 정규화한 방향")
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].set_xticks([-1, -.5, 0, .5, 1]); axes[1].set_yticks([-1, -.5, 0, .5, 1])
    for ax in axes:
        common(ax)
    fig.text(.085, .166, "113번·95번은 학습 데이터 Fisher score 하위 두 좌표이며, 좌표 번호는 0부터 시작함.", fontsize=13.5)
    fig.text(.085, .12, "오른쪽에서는 크기가 사라지고 방향만 남음. 코사인 점수는 검은 화살표와 이루는 각도로 결정됨.", fontsize=13)
    fig.text(.085, .073, "두 클래스가 주로 모이는 영역이 다르지만 일부 겹침. 겹친 점이 많아 보이는 정도는 밀도·투명도의 영향도 받음.", fontsize=12.5, color="#586572")
    fig.text(.085, .036, "같은 최종 160차원 임베딩에서 좌표만 선택한 결과이며, 모델 재학습이나 새로운 차원 축소를 수행하지 않음.", fontsize=12.5, color="#586572")
    export(fig, "01_selected_two_coordinates.png")
    return dict(coordinates=[113, 95], all_samples=int(len(y)), score_reconstruction_max_error=error,
                class_means={str(v):pair[y==v].mean(0).tolist() for v in [0, 1]})


def full_projection(data, y):
    # Fixed balanced display sample. Selection does not use embedding values or scores.
    rng = np.random.default_rng(SEED)
    ix = np.r_[np.flatnonzero(y==1), rng.choice(np.flatnonzero(y==0), size=2000, replace=False)]
    ix = np.sort(ix)
    label = y[ix]
    assert len(ix)==4000 and len(np.unique(ix))==4000 and label.sum()==2000
    x = data.z[ix].astype(np.float64)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    pca = PCA(n_components=50, svd_solver="full")
    pcs = pca.fit_transform(x)
    print("PCA ready", pca.explained_variance_ratio_[:2].sum(), flush=True)
    tsne = TSNE(n_components=2, perplexity=40, init="pca", random_state=SEED,
                learning_rate="auto", max_iter=1000, method="barnes_hut", n_jobs=2,
                verbose=1)
    embedding = tsne.fit_transform(pcs)
    assert np.isfinite(embedding).all()
    table = data.rows.iloc[ix][["utterance_id", "y", "source_corpus", "codec"]].copy()
    table["cache_row"] = ix
    table["PC1"], table["PC2"] = pcs[:, 0], pcs[:, 1]
    table["tSNE1"], table["tSNE2"] = embedding[:, 0], embedding[:, 1]
    table.to_parquet(WORK / "projection_points.parquet", index=False)
    np.savez_compressed(WORK / "pca_parameters.npz", mean=pca.mean_, components=pca.components_,
                        explained_variance_ratio=pca.explained_variance_ratio_)
    render_full_projection(pcs[:, :2], embedding, label, pca.explained_variance_ratio_)
    return dict(display_rows=4000, class_counts={"normal":2000, "spoof":2000}, sample_seed=SEED,
                PCA_fit_population="same 4000 balanced display rows, after L2 normalisation",
                PCA_dimensions=50, PCA_variance_ratios=pca.explained_variance_ratio_.tolist(),
                TSNE=dict(perplexity=40, seed=SEED, max_iter=1000, init="pca", learning_rate="auto",
                          input="50 PCA axes from unit-normalised 160D z", KL_divergence=float(tsne.kl_divergence_)),
                labels_used_for_projection=False, labels_used_for_balanced_display_sampling=True)


def render_full_projection(pcs, embedding, label, variance_ratios):
    fig = plt.figure(figsize=(12.5, 8.5))
    heading(fig, "전체 160차원 표현을 2차원에 투영한 분포",
            "같은 4,000개 발화: 정상 2,000개 · 스푸프 2,000개  |  점 1개 = 발화 1개  |  색상 = 실제 클래스")
    gs = fig.add_gridspec(1, 2, left=.09, right=.96, top=.78, bottom=.30, wspace=.30)
    axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]
    for ax, coords in zip(axes, [pcs[:, :2], embedding]):
        scatter(ax, coords, label, size=7, alpha=.35)
        ax.set_aspect("equal", adjustable="box")
        common(ax)
    ratio = np.asarray(variance_ratios)*100
    axes[0].set(xlabel=f"주성분 1 ({ratio[0]:.1f}%)", ylabel=f"주성분 2 ({ratio[1]:.1f}%)", title="A. PCA: 선형 투영")
    axes[1].set(xlabel="t-SNE 축 1", ylabel="t-SNE 축 2", title="B. t-SNE: 국소 이웃 관계의 시각화")
    legend(axes[0], label)
    fig.text(.09, .173, "160차원 벡터를 L2 정규화한 뒤 투영. 두 패널은 같은 발화를 표시하며 클래스 정답은 투영 계산에 넣지 않음.", fontsize=12.8)
    fig.text(.09, .126, "정상 2,000개는 모두 포함; 스푸프 18,000개 중 고정 난수로 2,000개 선택. 두 좌표 유지 실험과는 다른 시각화.", fontsize=12.5)
    fig.text(.09, .079, f"PCA 2축의 분산 설명률: {ratio[:2].sum():.1f}%. t-SNE: PCA 50축 입력, perplexity 40, seed {SEED}.", fontsize=12.5, color="#586572")
    fig.text(.09, .037, "특히 t-SNE의 군집 간 거리·크기는 원공간과 같지 않으며, 시각적 분리만으로 EER나 일반화 성능을 판단할 수 없음.", fontsize=12.5, color="#586572")
    export(fig, "02_full160_pca_tsne.png")


def main():
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    validation = validate_run("DF_pilot")
    print("Cache validation passed", flush=True)
    data = load_embeddings("DF_pilot", expected_checkpoint_sha256=CHECKPOINT)
    y = data.rows.y.to_numpy()
    assert len(y)==20000 and y.sum()==2000
    rank_path = CACHE / "cached_core_v1/channel_masks.json"
    rank = json.loads(rank_path.read_text(encoding="utf-8"))["ranking_T_macroFisher"]
    assert rank[-2:]==[113, 95]
    with threadpool_limits(limits=2):
        pair_record = selected_pair(data, y)
        projection_record = full_projection(data, y)
    captions = """# 정상·스푸프 임베딩 분포 그림

## 그림 1. 선택한 두 좌표의 실제 분포
"""
    captions += """
![선택한 두 좌표의 실제 분포](01_selected_two_coordinates.png)

ASVspoof 2021 DF eval의 고정 표본 20,000개(정상 2,000개, 스푸프 18,000개)를 모두 표시했다. 왼쪽은 최종 160차원 임베딩에서 113번·95번 좌표를 직접 선택한 실제 값이다. 좌표 번호는 0부터 시작하며, 해당 두 좌표는 ASVspoof 2019 LA train에서 공격별 Fisher score를 평균한 순위의 하위 두 개다. 오른쪽은 이 두 좌표만으로 이루어진 벡터를 L2 정규화한 분포다. 검은 화살표는 동일한 좌표를 선택한 센트로이드의 정규화 방향이다. 오른쪽의 두 클래스는 같은 단위원 위에 있으며, 코사인 점수는 센트로이드와 이루는 각도로 결정된다. 이 그림에는 PCA 또는 t-SNE를 적용하지 않았다. 일부 클래스 겹침이 있으며, 점의 중첩으로부터 정확한 오류율을 읽을 수는 없다.

삽입 위치: 보고서 ‘4.5 선택된 좌표의 실제 값’.

## 그림 2. 전체 160차원 표현의 PCA·t-SNE 투영

![전체 160차원 표현의 PCA·t-SNE 투영](02_full160_pca_tsne.png)

같은 DF 고정 표본에서 정상 2,000개 전체와 스푸프 18,000개 중 고정 난수(seed 20261105)로 선택한 2,000개를 표시했다. 두 패널은 동일한 4,000개 발화를 사용한다. 전체 160차원 임베딩을 L2 정규화하고, 같은 4,000개 발화에서 학습한 PCA로 투영했다. t-SNE에는 PCA 50차원 결과를 입력했으며 perplexity 40, seed 20261105, 초기화 PCA, 학습률 auto, 최대 반복 1,000회로 계산했다. 클래스 정답은 표시 표본의 균형을 맞추고 색상을 정하는 데만 사용했고 투영 알고리즘의 입력에는 포함하지 않았다. PCA 축의 괄호는 각 축의 분산 설명률이다.

이 그림은 두 좌표만 남겨 점수를 계산한 실험과 다르며, 전체 160차원 표현을 관찰하기 위한 별도의 시각화다. 특히 t-SNE의 군집 간 거리·크기·빈 공간은 원공간의 전역 구조를 그대로 보존하지 않는다. 그림에서 보이는 분리 정도를 EER, 독립 데이터의 일반화 성능 또는 공식적인 군집 수의 근거로 해석하지 않는다. 보고서의 EER는 기존 20,000개 표본에서 계산한 값이며 이 4,000개 시각화 표본에서 새로 평가한 값이 아니다.
"""
    (OUT / "captions.md").write_text(captions, encoding="utf-8")
    _, entry, _ = resolve_run("DF_pilot")
    record = dict(cache_validation=validation, checkpoint=CHECKPOINT, cache_fingerprint=entry["cache_fingerprint"],
                  script_sha256=digest(__file__), mask_file_sha256=digest(rank_path),
                  score_file_sha256=digest(CACHE / "channel_keep_one_v3/DF_scores.parquet"),
                  pair=pair_record, projection=projection_record, new_EBM_inference=False,
                  new_classifier_training=False, elapsed_seconds=time.monotonic()-started,
                  output_png_sha256={p.name:digest(p) for p in OUT.glob("*.png")})
    (WORK / "provenance.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(status="complete", output=str(OUT), elapsed=record["elapsed_seconds"],
                         PCA_2D_variance=sum(projection_record["PCA_variance_ratios"][:2])), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
