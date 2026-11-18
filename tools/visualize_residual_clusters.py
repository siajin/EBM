"""Scientific visualization of archived unit tangent residual directions.

Offline CPU only. Clusters/PCA/t-SNE are descriptive geometry, not EBM training.
No model, optimizer, torch, network or remote process is accessed.
"""
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "2"
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.cluster import kmeans_plusplus
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "artifacts/geometry_cache"
OUT = CACHE / "visualizations/residual_clusters_20261111_v1"
SHA = "104873703a57cad98b686d770e0860e031a2cf9abef79b72668780c1660e0b24"
DATASETS = ["T", "V", "19LA", "21LA", "DF"]
NAMES = {"T": "Train", "V": "Dev", "19LA": "19LA eval", "21LA": "21LA pilot", "DF": "DF pilot"}
COLORS = {1: "#3673A5", 0: "#D8863B", "error": "#BE3445"}
CLUSTER_COLORS = ["#356AA0", "#DB8D32", "#3D936E", "#B75E87", "#7E6AAD", "#8E785D"]
SEED = 20261111
THRESHOLD = 0.5093649528766109
READ_HASHES = {}
VALIDATION = []


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_file(path, expected=None):
    path = Path(path).resolve()
    value = digest(path)
    assert expected is None or value == expected, f"Input hash mismatch: {path}"
    READ_HASHES[str(path)] = value


def save_json(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_run(catalog, key, select_ids=None):
    entry = catalog["runs"][key]
    assert entry["status"] == "ready" and entry["identity"]["checkpoint_sha256"] == SHA
    folder = CACHE / entry["local_folder"]
    wanted = ["extract/scores.parquet", "input_audit.json", "extract/summary.json"]
    wanted += [f"extract/rank{rank}_{kind}" for rank in (0, 1) for kind in ("z.npy", "rows.parquet", "centroid.npy")]
    for item in entry["files"]:
        if item["path"] in wanted:
            assert (folder / item["path"]).stat().st_size == item["bytes"]
            check_file(folder / item["path"], item["sha256"])
    rows = pd.read_parquet(folder / "extract/scores.parquet")
    assert rows.utterance_id.is_unique and len(rows) == entry["sample_count"]
    centroid = np.load(folder / "extract/rank0_centroid.npy", allow_pickle=False)
    assert abs(np.linalg.norm(centroid) - 1) < 1e-10
    chunks, selected_rows = [], []
    cursor = 0
    for rank in (0, 1):
        shard = pd.read_parquet(folder / f"extract/rank{rank}_rows.parquet")
        part = rows.iloc[cursor:cursor + len(shard)]
        assert shard.utterance_id.tolist() == part.utterance_id.tolist()
        assert np.array_equal(shard.y, part.y)
        assert np.array_equal(part.embedding_row, np.arange(len(part)))
        c = np.load(folder / f"extract/rank{rank}_centroid.npy", allow_pickle=False)
        assert np.array_equal(c, centroid)
        z = np.load(folder / f"extract/rank{rank}_z.npy", mmap_mode="r", allow_pickle=False)
        assert z.shape == (len(part), 160)
        take = np.ones(len(part), dtype=bool) if select_ids is None else part.utterance_id.isin(select_ids).to_numpy()
        chunks.append(np.asarray(z[take], dtype=np.float64))
        selected_rows.append(part.loc[take].copy())
        cursor += len(shard)
    f = pd.concat(selected_rows, ignore_index=True)
    z = np.concatenate(chunks)
    x = z / np.linalg.norm(z, axis=1, keepdims=True)
    s = x @ centroid
    r = x - s[:, None] * centroid
    norm = np.linalg.norm(r, axis=1)
    valid = norm >= 1e-4
    assert np.isfinite(x).all() and np.max(np.abs(s - f.cosine_s)) < 1e-10
    assert np.max(np.abs(norm ** 2 - (1 - s * s))) < 1e-10
    assert np.max(np.abs(r @ centroid)) < 1e-10
    u = r[valid] / norm[valid, None]
    f = f.loc[valid].reset_index(drop=True)
    f["score"] = s[valid]
    f["rejected_bona"] = (f.y == 1) & (f.score < THRESHOLD)
    assert np.max(np.abs(np.linalg.norm(u, axis=1) - 1)) < 1e-10
    assert np.max(np.abs(u @ centroid)) < 1e-9
    VALIDATION.append({"run": key, "checkpoint_sha256": SHA, "selected": len(valid),
                       "valid_directions": int(valid.sum()), "invalid_excluded": int((~valid).sum()),
                       "fingerprint": entry["cache_fingerprint"]})
    return f, u, centroid


def hash_sample(f, limit=1000):
    out = []
    for y in (1, 0):
        indices = np.flatnonzero(f.y.to_numpy() == y)
        priority = [hashlib.sha256(("residual-cluster-20261111|" + str(f.utterance_id.iloc[i])).encode()).hexdigest() for i in indices]
        out.extend(indices[np.argsort(priority)[:limit]])
    return np.asarray(out, dtype=int)


def spherical_kmeans(x, k, seed, starts=3):
    best = None
    for start in range(starts):
        centers, _ = kmeans_plusplus(x, n_clusters=k, random_state=seed + start)
        labels = np.full(len(x), -1)
        for iteration in range(100):
            next_labels = np.argmax(x @ centers.T, axis=1)
            if np.array_equal(labels, next_labels):
                break
            labels = next_labels
            sums = np.zeros((k, x.shape[1]))
            np.add.at(sums, labels, x)
            for j in range(k):
                length = np.linalg.norm(sums[j])
                if length > 1e-12:
                    centers[j] = sums[j] / length
                else:
                    centers[j] = x[np.argmin(np.max(x @ centers.T, axis=1))]
        similarity = np.sum(x * centers[labels], axis=1).mean()
        if best is None or similarity > best[0]:
            best = (float(similarity), centers.copy(), labels.copy(), iteration + 1)
    return best


def direction_stats(u, plane, seed):
    n, dim = u.shape
    mean = u.mean(axis=0)
    R = np.linalg.norm(mean)
    moment = u.T @ u / n
    eigenvalues = np.maximum(eigh(moment, eigvals_only=True), 0)[::-1]
    bearing_xy = u @ plane
    angle = np.arctan2(bearing_xy[:, 1], bearing_xy[:, 0])
    bins = np.linspace(-np.pi, np.pi, 25)
    counts, _ = np.histogram(angle, bins)
    mass = counts / n
    double = np.r_[mass, mass]
    sector60 = max(double[i:i + 4].sum() for i in range(24))
    rng = np.random.default_rng(seed)
    left = rng.integers(n, size=50000)
    right = rng.integers(n - 1, size=50000)
    right += right >= left
    pair_angles = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", u[left], u[right]), -1, 1)))
    own_angles = np.degrees(np.arccos(np.clip(u @ (mean / R), -1, 1)))
    return {"n": n, "R": float(R), "pairwise_mean_cosine_exact": float((n * R * R - 1) / (n - 1)),
            "second_moment_PR": float(1 / np.sum(moment * moment)),
            "second_moment_top1_share": float(eigenvalues[0]),
            "mean_axis_angle_p50": float(np.quantile(own_angles, .5)),
            "mean_axis_angle_p90": float(np.quantile(own_angles, .9)),
            "within_30deg_of_mean_fraction": float(np.mean(own_angles <= 30)),
            "pair_angle_p10_p50_p90": np.quantile(pair_angles, [.1, .5, .9]).tolist(),
            "plane_energy_mean": float(np.mean(np.sum(bearing_xy ** 2, axis=1))),
            "bearing_occupied_bins_of24": int((counts > 0).sum()),
            "bearing_max_60deg_mass": float(sector60), "bearing_bin_counts": counts.tolist(),
            "uniform_reference": {"tangent_dimension": dim - 1, "population_R": 0,
                                  "population_second_moment_PR": dim - 1, "population_top1_share": 1 / (dim - 1),
                                  "expected_sample_R_squared": 1 / n,
                                  "projected_uniform_60deg_mass": 1 / 6}}, pair_angles


def nn_overlap(x, labels, seed):
    idx = NearestNeighbors(n_neighbors=16, metric="euclidean", n_jobs=2).fit(x).kneighbors(return_distance=False)[:, :15]
    # kneighbors(X=None) already excludes each query's own row.
    observed = float(np.mean(labels[idx] != labels[:, None]))
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(100):
        shuffled = rng.permutation(labels)
        null.append(np.mean(shuffled[idx] != shuffled[:, None]))
    return {"n": len(x), "k": 15, "opposite_class_neighbor_fraction": observed,
            "permuted_label_reference_mean": float(np.mean(null)),
            "reference_p025_p975": np.quantile(null, [.025, .975]).tolist(),
            "scope": "equal class samples, original residual-direction space; not classifier accuracy or a population p-value"}


def savefig(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=175, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def facet_projection(points, statistics, method, title, filename):
    fig, axes = plt.subplots(1, 5, figsize=(17.5, 4.2), sharex=True, sharey=True, layout="constrained")
    xcol, ycol = ("pca1", "pca2") if method == "PCA" else ("tsne1", "tsne2")
    for ax, dataset in zip(axes, DATASETS):
        f = points[points.dataset == dataset]
        for y in (0, 1):
            b = f[f.y == y]
            ax.scatter(b[xcol], b[ycol], s=7, alpha=.35, color=COLORS[y], linewidths=0, rasterized=True)
        e = f[f.rejected_bona]
        ax.scatter(e[xcol], e[ycol], s=19, marker="x", color=COLORS["error"], linewidths=.65, rasterized=True)
        ax.set_title(f"{NAMES[dataset]}\n표시 정상 {int((f.y == 1).sum()):,} / 스푸프 {int((f.y == 0).sum()):,}", fontsize=10)
        ax.set_aspect("equal", adjustable="box")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("공통 PCA 1" if method == "PCA" else "공동 t-SNE 1")
    axes[0].set_ylabel("공통 PCA 2" if method == "PCA" else "공동 t-SNE 2")
    handles = [Line2D([], [], marker="o", linestyle="", color=COLORS[1], label="정상"),
               Line2D([], [], marker="o", linestyle="", color=COLORS[0], label="스푸프"),
               Line2D([], [], marker="x", linestyle="", color=COLORS["error"], label="오류 정상 (고정 dev 문턱)")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle(title + "\n보관 batch160 · epoch50 / 잔차 길이를 제거한 단위 방향", fontsize=13)
    savefig(fig, filename)


def plot_clusters(points, cluster_summary):
    fig, axes = plt.subplots(2, 5, figsize=(17.5, 7.5), sharex="row", sharey="row", layout="constrained")
    for row, method in enumerate(("pca", "tsne")):
        for col, dataset in enumerate(DATASETS):
            ax = axes[row, col]
            f = points[points.dataset == dataset]
            for j in range(6):
                b = f[f.cluster == j]
                ax.scatter(b[f"{method}1"], b[f"{method}2"], s=7, alpha=.5,
                           color=CLUSTER_COLORS[j], linewidths=0, rasterized=True)
            ax.set_title(NAMES[dataset], fontsize=11)
            ax.set_aspect("equal", adjustable="box")
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xlabel("공통 PCA 1" if row == 0 else "공동 t-SNE 1")
        axes[row, 0].set_ylabel("공통 PCA 2" if row == 0 else "공동 t-SNE 2")
    handles = [Line2D([], [], marker="o", linestyle="", color=CLUSTER_COLORS[j], label=f"C{j + 1}") for j in range(6)]
    fig.legend(handles=handles, loc="outside lower center", ncol=6, frameon=False)
    fig.suptitle("원래 잔여 방향 공간의 공통 6개 군집\n같은 색은 모든 데이터셋에서 같은 군집 · 군집 수 6은 분석용 설정", fontsize=13)
    savefig(fig, "03_shared_clusters")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True, layout="constrained")
    for ax, y, label in zip(axes, (1, 0), ("정상", "스푸프")):
        bottom = np.zeros(5)
        for j in range(6):
            values = [cluster_summary[(d, y)][j] for d in DATASETS]
            ax.bar(np.arange(5), values, bottom=bottom, color=CLUSTER_COLORS[j], label=f"C{j + 1}")
            bottom += values
        ax.set(xticks=np.arange(5), xticklabels=[NAMES[d] for d in DATASETS], ylim=(0, 1), title=label)
        ax.tick_params(axis="x", rotation=20)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("해당 클래스 전체 중 군집 비율")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    fig.suptitle("데이터셋별 군집 구성 (표시 표본이 아닌 전체 보관 표본)", fontsize=12)
    savefig(fig, "04_cluster_composition")


def plot_coverage(statistics, pair_angles):
    fig, axes = plt.subplots(1, 5, figsize=(17.5, 4.5), subplot_kw={"projection": "polar"}, layout="constrained")
    theta = (np.linspace(-np.pi, np.pi, 25)[:-1] + np.pi / 24)
    maxrad = max(max(statistics[f"{d}:{y}"]["bearing_bin_counts"]) / statistics[f"{d}:{y}"]["n"] for d in DATASETS for y in (0, 1))
    for ax, d in zip(axes, DATASETS):
        for y in (1, 0):
            st = statistics[f"{d}:{y}"]
            mass = np.array(st["bearing_bin_counts"]) / st["n"]
            ax.plot(np.r_[theta, theta[0]], np.r_[mass, mass[0]], color=COLORS[y], linewidth=1.7)
            ax.fill(np.r_[theta, theta[0]], np.r_[mass, mass[0]], color=COLORS[y], alpha=.1)
        ax.plot(np.linspace(0, 2 * np.pi, 100), np.full(100, 1 / 24), color="#777777", linestyle="--", linewidth=1)
        ax.set_ylim(0, maxrad * 1.12)
        ax.set_xticks(np.radians([0, 90, 180, 270]))
        ax.set_xticklabels(["0°", "90°", "180°", "270°"])
        ax.set_yticklabels([])
        ax.set_title(NAMES[d], y=1.1)
    handles = [Line2D([], [], color=COLORS[1], label="정상"), Line2D([], [], color=COLORS[0], label="스푸프"),
               Line2D([], [], color="#777777", linestyle="--", label="투영 방위가 균등할 때")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("같은 두 축에서 본 360° 방위 분포 (15° 구간별 비율)\n반지름은 해당 구간 표본 비율 · 고차원 전체의 각도는 아님", fontsize=13)
    savefig(fig, "05_projected_360_bearings")
    fig, axes = plt.subplots(1, 5, figsize=(17.5, 4.2), sharex=True, sharey=True, layout="constrained")
    rng = np.random.default_rng(SEED)
    null_x = rng.normal(size=(20000, 159))
    null_x /= np.linalg.norm(null_x, axis=1, keepdims=True)
    null_angles = np.degrees(np.arccos(np.clip(np.sum(null_x[:10000] * null_x[10000:], axis=1), -1, 1)))
    edges = np.arange(0, 181, 5)
    for ax, d in zip(axes, DATASETS):
        for y in (1, 0):
            ax.hist(pair_angles[f"{d}:{y}"], bins=edges, weights=np.full(50000, 100 / 50000),
                    histtype="step", color=COLORS[y], linewidth=1.6)
        ax.hist(null_angles, bins=edges, weights=np.full(len(null_angles), 100 / len(null_angles)),
                histtype="step", color="#777777", linestyle="--", linewidth=1.1)
        ax.set(title=NAMES[d], xlim=(0, 180), xticks=[0, 45, 90, 135, 180], xlabel="발화 쌍의 잔여 방향 각도 (°)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("해당 각도 구간의 쌍 비율 (%)")
    fig.legend(handles=handles[:2] + [Line2D([], [], color="#777777", linestyle="--", label="159차원 균등 방향 참조")],
               loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("원래 고차원 공간의 각도: 작은 각도일수록 같은 방향에 집중\n클래스별 무작위 50,000쌍 · 독립 시행 수나 신뢰구간이 아님", fontsize=13)
    savefig(fig, "06_original_space_angles")


def interactive(points):
    payload = points[["dataset", "y", "rejected_bona", "cluster", "score", "duration", "pca1", "pca2", "tsne1", "tsne2"]].copy()
    for c in ("score", "duration", "pca1", "pca2", "tsne1", "tsne2"):
        payload[c] = payload[c].round(5)
    records = payload.to_dict(orient="records")
    html = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>EBM 잔여 방향 탐색</title><style>body{font-family:system-ui,sans-serif;margin:24px;max-width:1250px;color:#253345}h1{font-size:23px}label{margin-right:20px}select{padding:7px;margin:8px}#plot{width:100%;height:70vh;min-height:420px}p{line-height:1.6}.note{color:#566274}</style>
<h1>데이터셋별 잔여 방향: 겹침과 군집</h1><p>보관 batch160 · epoch50 모델. 잔차 길이를 제거한 단위 방향입니다. 모든 데이터셋은 같은 PCA 축 또는 한 번에 계산한 공동 t-SNE 좌표를 사용합니다.</p>
<label>투영<select id="method"><option value="pca">공통 PCA</option><option value="tsne">공동 t-SNE</option></select></label>
<label>색상<select id="color"><option value="class">정상 / 스푸프 / 오류 정상</option><option value="dataset">데이터셋</option><option value="cluster">공통 6개 군집</option></select></label>
<label>표시 데이터<select id="dataset"><option>전체</option><option>T</option><option>V</option><option>19LA</option><option>21LA</option><option>DF</option></select></label>
<div id="plot"></div><p id="counts"></p><p class="note">마우스로 확대하고 범례를 클릭해 그룹을 켜고 끌 수 있습니다. 군집은 원래 160좌표의 단위 잔여 방향에서 계산했습니다. t-SNE의 빈 공간·방위·군집 간 거리는 원래 공간의 각도가 아닙니다. 이 화면은 모델 학습이나 판정 규칙을 바꾸지 않습니다.</p>
<script>__PLOTLY__</script><script>
const points=__DATA__;
const palette=['#356AA0','#DB8D32','#3D936E','#B75E87','#7E6AAD','#8E785D'];
function render(){const method=document.getElementById('method').value, coloring=document.getElementById('color').value, dataset=document.getElementById('dataset').value;let selected=points.filter(p=>dataset==='전체'||p.dataset===dataset);const groups={};for(const p of selected){let group=coloring==='dataset'?p.dataset:coloring==='cluster'?'C'+(p.cluster+1):p.rejected_bona?'오류 정상':p.y===1?'정상':'스푸프';(groups[group]??=[]).push(p);}const traces=Object.entries(groups).map(([name,rows],i)=>({type:'scattergl',mode:'markers',name:name,x:rows.map(p=>p[method+'1']),y:rows.map(p=>p[method+'2']),customdata:rows.map(p=>[p.dataset,p.y===1?'정상':'스푸프','C'+(p.cluster+1),p.score,p.duration]),hovertemplate:'%{customdata[0]} / %{customdata[1]} / %{customdata[2]}<br>cosine=%{customdata[3]}<br>길이=%{customdata[4]}초<extra></extra>',marker:{size:name==='오류 정상'?7:4,opacity:name==='오류 정상'?.9:.45,symbol:name==='오류 정상'?'x':'circle',color:coloring==='class'?({'정상':'#3673A5','스푸프':'#D8863B','오류 정상':'#BE3445'}[name]):coloring==='cluster'?palette[Number(name.slice(1))-1]:palette[['T','V','19LA','21LA','DF'].indexOf(name)]}}));Plotly.react('plot',traces,{margin:{l:55,r:20,b:55,t:15},xaxis:{title:method==='pca'?'공통 PCA 1':'공동 t-SNE 1'},yaxis:{title:method==='pca'?'공통 PCA 2':'공동 t-SNE 2',scaleanchor:'x',scaleratio:1},legend:{orientation:'h'},hovermode:'closest',uirevision:method+dataset},{responsive:true,displaylogo:false});document.getElementById('counts').textContent='표시 '+selected.length.toLocaleString()+'개 · 데이터셋×클래스당 고정 1,000개 표본';}
for(const id of ['method','color','dataset'])document.getElementById(id).addEventListener('change',render);render();
</script></html>'''
    html = html.replace("__PLOTLY__", get_plotlyjs()).replace("__DATA__", json.dumps(records, ensure_ascii=False, separators=(",", ":")))
    (OUT / "residual_direction_viewer.html").write_text(html, encoding="utf-8")


def main():
    started = time.monotonic()
    assert (OUT / "plan.json").is_file() and not (OUT / "statistics.json").exists()
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 10})
    protected = {str(p): digest(p) for p in (ROOT / "EBM").rglob("*.py") if "__pycache__" not in p.parts}
    check_file(CACHE / "catalog.json")
    catalog = json.loads((CACHE / "catalog.json").read_text(encoding="utf-8"))
    frame_tv, u_tv, c = load_run(catalog, "19la_tv")
    data = {d: (frame_tv.loc[frame_tv.split == d].reset_index(drop=True), u_tv[frame_tv.split.to_numpy() == d]) for d in ("T", "V")}
    f, u, other = load_run(catalog, "frozen_19la_eval")
    assert np.array_equal(c, other)
    data["19LA"] = (f, u)
    lockpath = CACHE / "analysis_manifests/frozen_data_sizes_v1/lock.json"
    check_file(lockpath)
    lock = json.loads(lockpath.read_text())["views"]["21LA_pilot"]
    manifest = lockpath.parent / lock["manifest_file"]
    check_file(manifest, lock["manifest_sha256"])
    selected = pd.read_parquet(manifest)
    f, u, other = load_run(catalog, "21la_eval", set(selected.utterance_id))
    assert np.array_equal(c, other) and set(f.utterance_id) == set(selected.utterance_id)
    assert lock["source_cache_fingerprint"] == catalog["runs"]["21la_eval"]["cache_fingerprint"]
    data["21LA"] = (f, u)
    f, u, other = load_run(catalog, "21df_pilot")
    assert np.array_equal(c, other)
    data["DF"] = (f, u)
    parts, vectors = [], []
    for d in DATASETS:
        f, u = data[d]
        idx = hash_sample(f)
        selected_f = f.iloc[idx].copy()
        selected_f["dataset"] = d
        parts.append(selected_f)
        vectors.append(u[idx])
    points = pd.concat(parts, ignore_index=True)
    x = np.concatenate(vectors)
    assert len(x) == 10000 and points.groupby(["dataset", "y"]).size().eq(1000).all()
    points[["dataset", "utterance_id", "y"]].to_parquet(OUT / "sample_lock.parquet", index=False)
    train = points.dataset.to_numpy() == "T"
    pca = PCA(n_components=50, svd_solver="full").fit(x[train])
    xy = pca.transform(x)[:, :2]
    points[["pca1", "pca2"]] = xy
    _, singular, vt = np.linalg.svd(x[train], full_matrices=False)
    plane = vt[:2].T.copy()
    for j in range(2):
        if plane[np.argmax(np.abs(plane[:, j])), j] < 0:
            plane[:, j] *= -1
    assert np.max(np.abs(plane.T @ c)) < 1e-8
    np.savez(OUT / "projection_parameters.npz", pca_components=pca.components_, pca_mean=pca.mean_,
             pca_variance_ratio=pca.explained_variance_ratio_, bearing_plane=plane, centroid=c)
    print("Common PCA fitted. Original-space spherical clustering starts.", flush=True)
    similarity, centers, labels, iterations = spherical_kmeans(x, 6, SEED)
    order = np.argsort((centers - pca.mean_) @ pca.components_[0])
    centers = centers[order]
    labels = np.argmax(x @ centers.T, axis=1)
    points["cluster"] = labels
    np.save(OUT / "cluster_centers.npy", centers)
    cluster_settings = {"K6": {"mean_assigned_cosine": similarity, "iterations": iterations}}
    for k in (3, 10):
        quality, alternative_centers, alternative_labels, it = spherical_kmeans(x, k, SEED, starts=2)
        cluster_settings[f"K{k}"] = {"mean_assigned_cosine": quality, "iterations": it,
                                     "ARI_against_K6": float(adjusted_rand_score(labels, alternative_labels))}
        points[f"cluster_k{k}"] = alternative_labels
    # Stability of the pre-specified K=6 partition under a different random seed.
    q2, _, lab2, it2 = spherical_kmeans(x, 6, SEED + 100, starts=2)
    cluster_settings["K6_other_seed"] = {"mean_assigned_cosine": q2, "iterations": it2,
                                          "ARI_against_main": float(adjusted_rand_score(labels, lab2))}
    save_json("cluster_settings.json", cluster_settings)
    print("Joint t-SNE main starts (10,000 fixed points).", flush=True)
    tsne = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", max_iter=750,
                metric="euclidean", random_state=SEED, n_jobs=2, verbose=1)
    points[["tsne1", "tsne2"]] = tsne.fit_transform(x.astype(np.float32))
    points.to_parquet(OUT / "plotted_points_main.parquet", index=False)
    print("Joint t-SNE sensitivity starts (same points, perplexity50/other seed).", flush=True)
    tsne2 = TSNE(n_components=2, perplexity=50, init="pca", learning_rate="auto", max_iter=750,
                 metric="euclidean", random_state=SEED + 1, n_jobs=2, verbose=1)
    points[["tsne_alt1", "tsne_alt2"]] = tsne2.fit_transform(x.astype(np.float32))
    points.to_parquet(OUT / "plotted_points.parquet", index=False)
    statistics, angles, cluster_summary, overlaps, per_group = {}, {}, {}, {}, []
    for di, d in enumerate(DATASETS):
        f, u = data[d]
        assigned = np.argmax(u @ centers.T, axis=1)
        assigned_rows = f[["utterance_id", "y", "score", "rejected_bona"]].copy()
        assigned_rows["dataset"] = d
        assigned_rows["cluster"] = assigned
        assigned_rows.to_parquet(OUT / f"{d}_cluster_assignments.parquet", index=False)
        for y in (1, 0):
            mask = f.y.to_numpy() == y
            key = f"{d}:{y}"
            statistics[key], angles[key] = direction_stats(u[mask], plane, SEED + di * 10 + y)
            freq = np.bincount(assigned[mask], minlength=6) / mask.sum()
            cluster_summary[(d, y)] = freq
            statistics[key]["cluster_fractions"] = freq.tolist()
            per_group.append({"dataset": d, "class": "bona" if y == 1 else "spoof",
                              **{k: v for k, v in statistics[key].items() if isinstance(v, (int, float))}})
        view = points.dataset.to_numpy() == d
        overlaps[d] = nn_overlap(x[view], points.y.to_numpy()[view], SEED + di)
        print(json.dumps({"dataset": d, "R_bona": statistics[f"{d}:1"]["R"],
                          "R_spoof": statistics[f"{d}:0"]["R"], "NN_opposite_fraction": overlaps[d]["opposite_class_neighbor_fraction"]}), flush=True)
    pd.DataFrame(per_group).to_csv(OUT / "directional_concentration.csv", index=False)
    hist_overlap = []
    for y in (1, 0):
        for a in DATASETS:
            for b in DATASETS:
                hist_overlap.append({"class": "bona" if y == 1 else "spoof", "dataset_a": a, "dataset_b": b,
                                     "shared_cluster_histogram_overlap": float(np.minimum(cluster_summary[(a, y)], cluster_summary[(b, y)]).sum())})
    pd.DataFrame(hist_overlap).to_csv(OUT / "between_dataset_cluster_overlap.csv", index=False)
    # A common 2D embedding means coordinate equality can be directly checked.
    assert points.utterance_id.notna().all() and np.isfinite(points[["pca1", "pca2", "tsne1", "tsne2"]]).all().all()
    sensitivity = {"main_KL": float(tsne.kl_divergence_), "alternative_KL": float(tsne2.kl_divergence_)}
    # Trustworthiness on a fixed balanced row sample (same original distances).
    sample = np.random.default_rng(SEED).choice(len(x), 2000, replace=False)
    sensitivity["trustworthiness_main_2000_rows"] = float(trustworthiness(x[sample], points[["tsne1", "tsne2"]].to_numpy()[sample], n_neighbors=15))
    sensitivity["trustworthiness_alt_2000_rows"] = float(trustworthiness(x[sample], points[["tsne_alt1", "tsne_alt2"]].to_numpy()[sample], n_neighbors=15))
    pca_title = f"공통 PCA: 데이터셋별 정상·스푸프 겹침 (Train 설명분산 {100 * pca.explained_variance_ratio_[:2].sum():.1f}%)"
    facet_projection(points, statistics, "PCA", pca_title, "01_common_pca")
    facet_projection(points, statistics, "tSNE", "공동 t-SNE: 모든 데이터셋을 한 좌표계에서 계산", "02_joint_tsne")
    alt_points = points.copy()
    alt_points[["tsne1", "tsne2"]] = alt_points[["tsne_alt1", "tsne_alt2"]].to_numpy()
    facet_projection(alt_points, statistics, "tSNE", "t-SNE 설정 민감도: 같은 표본 / 다른 seed / perplexity 50", "07_tsne_sensitivity")
    plot_clusters(points, cluster_summary)
    plot_coverage(statistics, angles)
    interactive(points)
    for path, before in READ_HASHES.items():
        assert digest(path) == before
    for path, before in protected.items():
        assert digest(path) == before
    output = {"status": "complete", "checkpoint_sha256": SHA, "model": "archived batch160 epoch50, not current training",
              "sample_n": len(points), "per_dataset_class_plot_n": 1000,
              "full_group_statistics": statistics, "nearest_neighbor_class_overlap": overlaps,
              "PCA_train_variance_ratio_first2": pca.explained_variance_ratio_[:2].tolist(),
              "cluster_settings": cluster_settings, "tSNE_sensitivity": sensitivity,
              "validation": VALIDATION, "input_hashes_verified_unchanged": len(READ_HASHES),
              "existing_python_sources_unchanged": len(protected),
              "safety": {"network_calls": 0, "remote_operations": 0, "GPU_calls": 0, "EBM_updates": 0, "new_spend_usd": 0,
                         "Q_embeddings_opened": False},
              "limitations": ["PCA/tSNE distances and apparent overlaps are not original-space decision errors",
                              "tSNE direction and island spacing have no physical angular meaning",
                              "360-degree bearings cover one fixed 2D plane, not the 159D tangent sphere",
                              "K6 partitions are descriptive, not a natural cluster-count discovery",
                              "Cached extraction/input policy and this checkpoint only; full eval sets and pilots are distinguished",
                              "Unit residual directions remove cosine and magnitude, so class overlap is not native detection failure",
                              "Uniform references are geometric descriptions, not population p-values under dependent utterances"],
              "elapsed_seconds": time.monotonic() - started}
    save_json("statistics.json", output)
    save_json("input_hashes.json", READ_HASHES)
    print(json.dumps({"status": "complete", "seconds": output["elapsed_seconds"], "PCA_variance": output["PCA_train_variance_ratio_first2"],
                      "output": str(OUT)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
