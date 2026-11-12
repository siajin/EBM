"""User-requested 3D views; offline descriptive geometry, no EBM updates."""
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "2"
from pathlib import Path
import json
import hashlib
import time
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE, trustworthiness
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from plotly.offline import get_plotlyjs
import visualize_residual_clusters as shared

OUT = shared.OUT
DATASETS = shared.DATASETS


def interactive_3d(points, metadata):
    columns = ["dataset", "y", "rejected_bona", "cluster", "score", "duration"]
    columns += [p + str(i) for p in ("pca", "tsne", "direction") for i in (1, 2, 3)]
    f = points[columns].copy()
    for c in columns:
        if f[c].dtype.kind == "f":
            f[c] = f[c].round(5)
    records = json.dumps(f.to_dict(orient="records"), ensure_ascii=False, separators=(",", ":"))
    html = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>EBM 잔여 방향 3D</title><style>
body{font-family:system-ui,'Malgun Gothic',sans-serif;margin:20px auto;padding:0 18px;max-width:1350px;color:#253345;background:white}h1{font-size:24px;margin:12px 0}p{line-height:1.55;margin:8px 0}.controls{display:flex;flex-wrap:wrap;gap:12px 20px;align-items:center}select{padding:7px;margin-left:5px}label{white-space:nowrap}fieldset{border:0;padding:0;margin:0;display:flex;flex-wrap:wrap;gap:12px}input{accent-color:#3673a5}#plot{width:100%;height:72vh;min-height:480px}.note{color:#586576;font-size:14px}#meta{font-variant-numeric:tabular-nums}button{padding:6px 12px;background:#f2f4f7;border:1px solid #a8b3be;border-radius:4px;cursor:pointer}
</style><h1>잔여 방향 3D: 데이터셋과 군집 비교</h1>
<p>마우스 드래그로 회전 · 휠로 확대 · 범례를 클릭해 그룹 표시/숨김. <strong>학습 중인 모델이 아닌 보관 batch160 · epoch50 EBM</strong>입니다.</p>
<div class="controls"><label>좌표<select id="method"><option value="pca">3D PCA — Train 공통축</option><option value="tsne">3D t-SNE — 전체 공동배치</option><option value="direction">3D 방향투영 — 원점 보존</option></select></label>
<label>색상<select id="color"><option value="class">정상 / 스푸프 / 오류 정상</option><option value="dataset">데이터셋</option><option value="cluster">공통 잔여 방향 군집</option></select></label><button id="reset" type="button">시점 초기화</button></div>
<fieldset id="datasets" aria-label="표시 데이터셋"><label><input type="checkbox" value="T" checked>Train</label><label><input type="checkbox" value="V" checked>Dev</label><label><input type="checkbox" value="19LA" checked>19LA eval</label><label><input type="checkbox" value="21LA" checked>21LA pilot</label><label><input type="checkbox" value="DF" checked>DF pilot</label></fieldset>
<fieldset id="classes" aria-label="표시 클래스"><label><input type="checkbox" value="1" checked>정상</label><label><input type="checkbox" value="0" checked>스푸프</label></fieldset>
<div id="plot"></div><p id="meta"></p><p id="explanation" class="note"></p>
<p class="note">군집은 이 그림의 3차원 좌표가 아니라 원래 단위 잔여 방향 공간에서 계산한 공통 6개 구역입니다. 색이 같아도 원인·화자·공격이 같다는 뜻은 아닙니다. PCA/t-SNE에서 겹쳐 보이는 것과 원래 모델의 판정 오류는 다릅니다. 모든 점은 데이터셋×클래스별 UID 해시로 고정한 500개 표본이며 오류를 보고 고르지 않았습니다.</p>
<script>__PLOTLY__</script><script>
const points=__DATA__;
const metadata=__META__;
const palette=['#356AA0','#DB8D32','#3D936E','#B75E87','#7E6AAD','#8E785D'];
let lastMethod='';let resetVersion=0;
function render(){
const method=document.getElementById('method').value,coloring=document.getElementById('color').value;
const datasets=[...document.querySelectorAll('#datasets input:checked')].map(x=>x.value),classes=[...document.querySelectorAll('#classes input:checked')].map(x=>Number(x.value));
const selected=points.filter(p=>datasets.includes(p.dataset)&&classes.includes(p.y));const groups={};
for(const p of selected){const group=coloring==='dataset'?p.dataset:coloring==='cluster'?'C'+(p.cluster+1):p.rejected_bona?'오류 정상':p.y===1?'정상':'스푸프';(groups[group]??=[]).push(p);}
const traces=Object.entries(groups).map(([name,rows],index)=>({type:'scatter3d',mode:'markers',name:name,x:rows.map(p=>p[method+'1']),y:rows.map(p=>p[method+'2']),z:rows.map(p=>p[method+'3']),customdata:rows.map(p=>[p.dataset,p.y===1?'정상':'스푸프','C'+(p.cluster+1),p.score,p.duration]),hovertemplate:'%{customdata[0]} / %{customdata[1]} / %{customdata[2]}<br>cosine=%{customdata[3]}<br>원음 길이=%{customdata[4]}초<extra></extra>',marker:{size:name==='오류 정상'?4:2.4,opacity:name==='오류 정상'?.95:.55,symbol:name==='오류 정상'?'x':'circle',color:coloring==='class'?({'정상':'#3673A5','스푸프':'#D8863B','오류 정상':'#BE3445'}[name]):coloring==='cluster'?palette[Number(name.slice(1))-1]:palette[['T','V','19LA','21LA','DF'].indexOf(name)],line:{width:0}}}));
if(method==='direction'){const xx=[],yy=[],zz=[];for(let latitude=-60;latitude<=60;latitude+=30){const a=latitude*Math.PI/180;for(let i=0;i<=72;i++){const b=2*Math.PI*i/72;xx.push(Math.cos(a)*Math.cos(b));yy.push(Math.cos(a)*Math.sin(b));zz.push(Math.sin(a));}xx.push(null);yy.push(null);zz.push(null);}traces.push({type:'scatter3d',mode:'lines',x:xx,y:yy,z:zz,line:{color:'#c9cfd6',width:1},hoverinfo:'skip',showlegend:false});}
const axisTitle=method==='pca'?'PC':method==='tsne'?'t-SNE ':'방향 축 ';
const bounds=metadata.bounds[method];const axis=(j)=>({title:axisTitle+j,range:bounds[j-1],zeroline:true,showbackground:false,gridcolor:'#e5e9ee'});
Plotly.react('plot',traces,{margin:{l:0,r:0,b:0,t:5},scene:{xaxis:axis(1),yaxis:axis(2),zaxis:axis(3),aspectmode:'cube',camera:{eye:{x:1.5,y:1.5,z:1.1}},uirevision:method+resetVersion},legend:{orientation:'h',y:1},uirevision:method+resetVersion},{responsive:true,displaylogo:false,toImageButtonOptions:{format:'png',filename:'ebm-residual-3d',width:1400,height:1000}});
document.getElementById('meta').textContent='표시 '+selected.length.toLocaleString()+'개 · 정상 '+selected.filter(p=>p.y===1).length.toLocaleString()+' · 스푸프 '+selected.filter(p=>p.y===0).length.toLocaleString()+' · 오류 정상 '+selected.filter(p=>p.rejected_bona).length;
document.getElementById('explanation').textContent=method==='pca'?'Train에서만 정한 동일한 세 축입니다. Train 방향 분산의 '+metadata.pcaVariance.toFixed(1)+'%를 설명하며 보이지 않는 나머지 방향도 있습니다.':method==='tsne'?'모든 데이터셋을 함께 계산했습니다. 가까운 이웃을 보기 위한 그림이며 섬 사이 거리·방위·빈 공간을 실제 각도나 전방향 분포로 해석할 수 없습니다.':'원점을 보존한 실제 선형투영입니다. 옅은 구는 단위 구면 기준이며, 점을 구 표면으로 재정규화하지 않았습니다. 안쪽에 있다는 것은 원래 방향 에너지 일부가 다른 차원에 남는다는 뜻입니다.';
lastMethod=method;
}
for(const id of ['method','color'])document.getElementById(id).addEventListener('change',render);
document.querySelectorAll('input').forEach(x=>x.addEventListener('change',render));document.getElementById('reset').addEventListener('click',()=>{resetVersion++;render();});render();
</script></html>'''
    html = html.replace("__PLOTLY__", get_plotlyjs()).replace("__DATA__", records).replace("__META__", json.dumps(metadata))
    (OUT / "residual_direction_3d.html").write_text(html, encoding="utf-8")


def static_3d(points, method, name, variance=None):
    fig = plt.figure(figsize=(16.5, 9.5), layout="constrained")
    bounds = []
    for j in (1, 2, 3):
        values = points[f"{method}{j}"]
        pad = (values.max() - values.min()) * .04
        bounds.append((values.min() - pad, values.max() + pad))
    for i, d in enumerate(DATASETS):
        ax = fig.add_subplot(2, 3, i + 1, projection="3d")
        f = points[points.dataset == d]
        for y in (0, 1):
            b = f[f.y == y]
            ax.scatter(b[f"{method}1"], b[f"{method}2"], b[f"{method}3"], s=5, alpha=.35,
                       color=shared.COLORS[y], linewidths=0, rasterized=True)
        b = f[f.rejected_bona]
        ax.scatter(b[f"{method}1"], b[f"{method}2"], b[f"{method}3"], s=20, alpha=.95,
                   marker="x", color=shared.COLORS["error"], linewidths=.7, rasterized=True)
        for j, setter in enumerate((ax.set_xlim, ax.set_ylim, ax.set_zlim)):
            setter(bounds[j])
        ax.set_title(shared.NAMES[d], pad=12)
        ax.set(xlabel="軸1".replace("軸", "축"), ylabel="축2", zlabel="축3")
        ax.view_init(elev=22, azim=-58)
        ax.set_box_aspect((1, 1, 1))
        ax.tick_params(labelsize=7, pad=0)
    ax = fig.add_subplot(2, 3, 6)
    ax.axis("off")
    handles = [Line2D([], [], marker="o", linestyle="", color=shared.COLORS[1], label="정상"),
               Line2D([], [], marker="o", linestyle="", color=shared.COLORS[0], label="스푸프"),
               Line2D([], [], marker="x", linestyle="", color=shared.COLORS["error"], label="오류 정상")]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=12)
    note = "데이터셋·클래스당 고정 500개\n모든 패널은 같은 축과 시점\n군집과 겹침은 원래 공간에서도 확인\n\n보관 batch160 · epoch50 모델\n현재 학습은 변경하지 않음"
    if variance is not None:
        note += f"\n\n공통 3D PCA의 Train 설명분산 {variance:.1f}%"
    else:
        note += "\n\nt-SNE의 전역 거리·방위는 실제 각도가 아님"
    ax.text(.02, .65, note, transform=ax.transAxes, va="top", fontsize=11, linespacing=1.8)
    fig.suptitle("데이터셋별 잔여 방향: " + ("공통 3D PCA" if method == "pca" else "공동 3D t-SNE"), fontsize=17)
    shared.savefig(fig, name)


def main():
    started = time.monotonic()
    assert not (OUT / "statistics_3d.json").exists()
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 10})
    protected = {str(p): shared.digest(p) for p in (shared.ROOT / "EBM").rglob("*.py") if "__pycache__" not in p.parts}
    catalog = json.loads((shared.CACHE / "catalog.json").read_text())
    shared.check_file(shared.CACHE / "catalog.json")
    f, u, c = shared.load_run(catalog, "19la_tv")
    data = {d: (f.loc[f.split == d].reset_index(drop=True), u[f.split.to_numpy() == d]) for d in ("T", "V")}
    f, u, c2 = shared.load_run(catalog, "frozen_19la_eval")
    assert np.array_equal(c, c2)
    data["19LA"] = (f, u)
    lockpath = shared.CACHE / "analysis_manifests/frozen_data_sizes_v1/lock.json"
    lock = json.loads(lockpath.read_text())["views"]["21LA_pilot"]
    manifest = lockpath.parent / lock["manifest_file"]
    shared.check_file(lockpath)
    shared.check_file(manifest, lock["manifest_sha256"])
    selected = pd.read_parquet(manifest)
    f, u, c2 = shared.load_run(catalog, "21la_eval", set(selected.utterance_id))
    assert np.array_equal(c, c2) and set(f.utterance_id) == set(selected.utterance_id)
    data["21LA"] = (f, u)
    f, u, c2 = shared.load_run(catalog, "21df_pilot")
    assert np.array_equal(c, c2)
    data["DF"] = (f, u)
    sample_lock = pd.read_parquet(OUT / "sample_lock.parquet")
    lock3 = sample_lock.groupby(["dataset", "y"], sort=False).head(500).reset_index(drop=True)
    frames, vectors = [], []
    for d in DATASETS:
        f, u = data[d]
        requested = lock3.loc[lock3.dataset == d, "utterance_id"]
        index = pd.Index(f.utterance_id).get_indexer(requested)
        assert np.all(index >= 0)
        part = f.iloc[index].copy()
        part["dataset"] = d
        frames.append(part)
        vectors.append(u[index])
    points = pd.concat(frames, ignore_index=True)
    x = np.concatenate(vectors)
    assert len(x) == 5000 and points.groupby(["dataset", "y"]).size().eq(500).all()
    params = np.load(OUT / "projection_parameters.npz", allow_pickle=False)
    assert np.array_equal(c, params["centroid"])
    pca_xyz = (x - params["pca_mean"]) @ params["pca_components"][:3].T
    points[["pca1", "pca2", "pca3"]] = pca_xyz
    train_x = data["T"][1]
    train_f = data["T"][0]
    idx = shared.hash_sample(train_f, 1000)
    _, _, vt = np.linalg.svd(train_x[idx], full_matrices=False)
    plane3 = vt[:3].T
    for j in range(3):
        if plane3[np.argmax(np.abs(plane3[:, j])), j] < 0:
            plane3[:, j] *= -1
    direction_xyz = x @ plane3
    assert np.max(np.linalg.norm(direction_xyz, axis=1)) <= 1 + 1e-10
    assert np.max(np.abs(plane3.T @ c)) < 1e-8
    points[["direction1", "direction2", "direction3"]] = direction_xyz
    centers = np.load(OUT / "cluster_centers.npy", allow_pickle=False)
    points["cluster"] = np.argmax(x @ centers.T, axis=1)
    np.save(OUT / "bearing_basis3.npy", plane3)
    np.save(OUT / "plot_unit_directions_3d.npy", x.astype(np.float32))
    points.to_parquet(OUT / "plotted_points_3d_before_tsne.parquet", index=False)
    statistics, angles, cluster_summary, overlap, compact = {}, {}, {}, {}, []
    for di, d in enumerate(DATASETS):
        f, u = data[d]
        assigned = np.argmax(u @ centers.T, axis=1)
        save = f[["utterance_id", "y", "score", "rejected_bona"]].copy()
        save["cluster"] = assigned
        save.to_parquet(OUT / f"{d}_cluster_assignments.parquet", index=False)
        for y in (1, 0):
            mask = f.y.to_numpy() == y
            key = f"{d}:{y}"
            statistics[key], angles[key] = shared.direction_stats(u[mask], params["bearing_plane"], shared.SEED + di * 10 + y)
            cluster_summary[(d, y)] = np.bincount(assigned[mask], minlength=6) / mask.sum()
            statistics[key]["cluster_fractions"] = cluster_summary[(d, y)].tolist()
            compact.append({"dataset": d, "class": "bona" if y == 1 else "spoof",
                            **{k: v for k, v in statistics[key].items() if isinstance(v, (int, float))}})
        view = points.dataset.to_numpy() == d
        overlap[d] = shared.nn_overlap(x[view], points.y.to_numpy()[view], shared.SEED + di)
    pd.DataFrame(compact).to_csv(OUT / "directional_concentration.csv", index=False)
    shared.plot_coverage(statistics, angles)
    pca_var = float(params["pca_variance_ratio"][:3].sum() * 100)
    static_3d(points, "pca", "08_common_pca_3d", pca_var)
    shared.save_json("statistics_before_tsne_3d.json", {"full_group_statistics": statistics, "nearest_neighbor_class_overlap": overlap})
    print("3D PCA and concentration figures complete. Joint 3D t-SNE: 5,000 points starts.", flush=True)
    tsne = TSNE(n_components=3, perplexity=30, init="pca", learning_rate="auto", max_iter=600,
                metric="euclidean", random_state=shared.SEED, n_jobs=2, angle=.65, verbose=2)
    points[["tsne1", "tsne2", "tsne3"]] = tsne.fit_transform(x.astype(np.float32))
    points.to_parquet(OUT / "plotted_points_3d.parquet", index=False)
    static_3d(points, "tsne", "09_joint_tsne_3d")
    bounds = {}
    for method in ("pca", "tsne", "direction"):
        bounds[method] = []
        for j in (1, 2, 3):
            a = points[f"{method}{j}"].min()
            b = points[f"{method}{j}"].max()
            pad = (b - a) * .05
            bounds[method].append([float(a - pad), float(b + pad)])
    bounds["direction"] = [[-1.05, 1.05]] * 3
    interactive_3d(points, {"pcaVariance": pca_var, "bounds": bounds})
    hist = []
    for y in (1, 0):
        for a in DATASETS:
            for b in DATASETS:
                hist.append({"class": "bona" if y == 1 else "spoof", "dataset_a": a, "dataset_b": b,
                             "shared_cluster_histogram_overlap": float(np.minimum(cluster_summary[(a, y)], cluster_summary[(b, y)]).sum())})
    pd.DataFrame(hist).to_csv(OUT / "between_dataset_cluster_overlap.csv", index=False)
    # Same source data; visual clustering partitions only. No new EBM training.
    sensitivity = json.loads((OUT / "cluster_settings.json").read_text())
    checkidx = np.random.default_rng(shared.SEED).choice(len(x), 1200, replace=False)
    trust = float(trustworthiness(x[checkidx], points[["tsne1", "tsne2", "tsne3"]].to_numpy()[checkidx], n_neighbors=15))
    for path, value in shared.READ_HASHES.items():
        assert shared.digest(path) == value
    for path, value in protected.items():
        assert shared.digest(path) == value
    result = {"status": "complete", "dimension": 3, "model": "archived batch160 epoch50", "checkpoint_sha256": shared.SHA,
              "plot_n": len(points), "plot_per_dataset_class": 500, "pca_train_explained_variance_percent": pca_var,
              "full_group_statistics": statistics, "nearest_neighbor_class_overlap": overlap,
              "cluster_settings": sensitivity, "tSNE_KL": float(tsne.kl_divergence_), "tSNE_trustworthiness_1200_rows": trust,
              "tSNE_method": {"joint": True, "perplexity": 30, "seed": shared.SEED, "iterations": 600, "angle": .65},
              "validation": shared.VALIDATION, "input_hashes_unchanged": len(shared.READ_HASHES),
              "existing_python_sources_unchanged": len(protected),
              "limitations": ["3D is a projection of a 159D tangent sphere", "tSNE global gaps and bearings are not angular coverage",
                              "Unit residual directions omit native cosine and length", "K6 is an exploratory partition, not six proven natural modes",
                              "19LA is full eval; DF and 21LA are existing locked pilots; plots use 500 per class", "Uniform geometric references are not population p-values"],
              "safety": {"network_calls": 0, "GPU_inferences": 0, "remote_operations": 0, "EBM_updates": 0, "Q_opened": False, "new_spend_usd": 0},
              "elapsed_seconds": time.monotonic() - started}
    shared.save_json("statistics_3d.json", result)
    shared.save_json("input_hashes_3d.json", shared.READ_HASHES)
    print(json.dumps({"status": "complete", "seconds": result["elapsed_seconds"], "PCA_3D_variance_percent": pca_var,
                      "viewer": str(OUT / "residual_direction_3d.html")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
