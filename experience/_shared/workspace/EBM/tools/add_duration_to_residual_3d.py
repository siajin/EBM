"""Offline duration coloring and a Train-only descriptive length-axis probe."""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "2"
from pathlib import Path
import json
import time
import shutil
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import visualize_residual_clusters as shared

ROOT, CACHE = shared.ROOT, shared.CACHE
ORIGINAL = shared.OUT
OUT = CACHE / "visualizations/residual_duration_20261111_v1"
ALPHAS = [0.0001, 0.001, 0.01, 0.1, 1., 10.]
CAP = 64600 / 16000


def save(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def choose_alpha(x, y, groups, folds=5):
    split = list(GroupKFold(folds).split(x, y, groups))
    losses = []
    for alpha in ALPHAS:
        squared, n = 0., 0
        for train, test in split:
            assert set(groups[train]).isdisjoint(groups[test])
            m = Ridge(alpha=alpha, solver="cholesky").fit(x[train], y[train])
            squared += float(np.sum((m.predict(x[test]) - y[test]) ** 2))
            n += len(test)
        losses.append(squared / n)
    return ALPHAS[int(np.argmin(losses))], dict(zip(map(str, ALPHAS), losses))


def rho(a, b):
    if len(a) < 20 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    return float(spearmanr(a, b).statistic)


def adjusted_rank_rho(f, value):
    # Descriptive sensitivity; controls are not asserted to be causal confounders.
    a, b = rankdata(f.duration.to_numpy()), rankdata(value)
    columns = ["speaker_group", "source_corpus", "codec", "attack_id"]
    control = pd.DataFrame(index=f.index)
    for col in columns:
        if col in f:
            control[col] = f[col].fillna("not_recorded").astype(str)
    dummy = pd.get_dummies(control, drop_first=True, dtype=float).to_numpy()
    design = np.c_[np.ones(len(f)), rankdata(f.score.to_numpy()) / len(f), dummy]
    target = np.c_[a, b]
    fitted = design @ np.linalg.lstsq(design, target, rcond=None)[0]
    residual = target - fitted
    if min(np.std(residual, axis=0)) < 1e-10:
        return None
    return float(np.corrcoef(residual.T)[0, 1])


def patch_viewer(points, statistics, basis3):
    source = ORIGINAL / "residual_direction_3d.html"
    html = source.read_text(encoding="utf-8")
    shutil.copy2(source, OUT / "original_residual_direction_3d.html")
    assert html.count("const points=") == 1
    old_data = html.split("const points=", 1)[1].split(";\nconst metadata=", 1)[0]
    old_points = json.loads(old_data)
    assert len(old_points) == len(points)
    for record, (_, row) in zip(old_points, points.iterrows()):
        assert record["dataset"] == row.dataset and record["y"] == row.y
        assert abs(record["duration"] - row.duration) < 0.000006
        for prefix in ("pca", "tsne", "direction"):
            for j in (1, 2, 3):
                # Existing t-SNE columns are float32; vectorized rounding retains
                # float32 representation noise. Keep the original JSON literal.
                assert abs(record[prefix + str(j)] - float(row[prefix + str(j)])) < 0.000012
        for j in (1, 2, 3):
            record[f"length{j}"] = round(float(row[f"length{j}"]), 5)
    new_data = json.dumps(old_points, ensure_ascii=False, separators=(",", ":"))
    html = html.replace("const points=" + old_data, "const points=" + new_data, 1)
    meta_text = html.split("const metadata=", 1)[1].split(";\nconst palette=", 1)[0]
    metadata = json.loads(meta_text)
    metadata["bounds"]["length"] = []
    for j in (1, 2, 3):
        low, high = points[f"length{j}"].min(), points[f"length{j}"].max()
        pad = (high - low) * .05
        metadata["bounds"]["length"].append([float(low - pad), float(high + pad)])
    metadata["durationProbe"] = {"nestedCVrho": statistics["nested_CV_spearman"], "fit": "Train 정상만 사용"}
    html = html.replace("const metadata=" + meta_text, "const metadata=" + json.dumps(metadata), 1)
    html = html.replace('<option value="direction">3D 방향투영 — 원점 보존</option>', '<option value="direction">3D 방향투영 — 원점 보존</option><option value="length">Train 길이 예측축 + 직교 2축</option>')
    html = html.replace('<select id="color">', '<select id="color"><option value="duration" selected>원음 길이 — 연속 색상</option>', 1)
    html = html.replace('<button id="reset"', '<label>길이 색상 눈금<select id="durationScale"><option value="log" selected>로그 간격</option><option value="linear">선형 간격</option></select></label><button id="reset"', 1)
    html = html.replace('<div id="plot"></div>', '<p class="note" id="durationNote">길이 색상: 짧음(보라) → 김(노랑), 모든 데이터셋에 같은 0.5–16초 눈금. 색은 원음 길이입니다. 보관 모델 입력은 모두 약 4.04초이며, 짧은 원음은 반복하고 긴 원음은 앞부분을 사용했습니다.</p><div id="plot"></div>', 1)
    start = html.index("function render(){", html.index("const points="))
    end = html.index("for(const id of ['method','color'])", start)
    render = r'''function render(){
const method=document.getElementById('method').value,coloring=document.getElementById('color').value;
const isDuration=coloring==='duration',logColor=document.getElementById('durationScale').value==='log';
document.getElementById('durationScale').disabled=!isDuration;
document.getElementById('durationNote').style.display=isDuration?'block':'none';
const datasets=[...document.querySelectorAll('#datasets input:checked')].map(x=>x.value),classes=[...document.querySelectorAll('#classes input:checked')].map(x=>Number(x.value));
const selected=points.filter(p=>datasets.includes(p.dataset)&&classes.includes(p.y));const groups={};
for(const p of selected){const group=coloring==='dataset'?p.dataset:coloring==='cluster'?'C'+(p.cluster+1):p.rejected_bona?'오류 정상':p.y===1?'정상':'스푸프';(groups[group]??=[]).push(p);}
const traces=Object.entries(groups).map(([name,rows],index)=>({type:'scatter3d',mode:'markers',name:name,x:rows.map(p=>p[method+'1']),y:rows.map(p=>p[method+'2']),z:rows.map(p=>p[method+'3']),customdata:rows.map(p=>[p.dataset,p.y===1?'정상':'스푸프','C'+(p.cluster+1),p.score,p.duration]),hovertemplate:'%{customdata[0]} / %{customdata[1]} / %{customdata[2]}<br>cosine=%{customdata[3]}<br>원음 길이=%{customdata[4]:.3f}초<extra></extra>',marker:{size:name==='오류 정상'?4:2.4,opacity:isDuration?.8:name==='오류 정상'?.95:.55,symbol:name==='오류 정상'?'x':'circle',color:isDuration?rows.map(p=>logColor?Math.log2(p.duration):p.duration):coloring==='class'?({'정상':'#3673A5','스푸프':'#D8863B','오류 정상':'#BE3445'}[name]):coloring==='cluster'?palette[Number(name.slice(1))-1]:palette[['T','V','19LA','21LA','DF'].indexOf(name)],coloraxis:isDuration?'coloraxis':undefined,line:{width:0}}}));
if(method==='direction'){const xx=[],yy=[],zz=[];for(let latitude=-60;latitude<=60;latitude+=30){const a=latitude*Math.PI/180;for(let i=0;i<=72;i++){const b=2*Math.PI*i/72;xx.push(Math.cos(a)*Math.cos(b));yy.push(Math.cos(a)*Math.sin(b));zz.push(Math.sin(a));}xx.push(null);yy.push(null);zz.push(null);}traces.push({type:'scatter3d',mode:'lines',x:xx,y:yy,z:zz,line:{color:'#c9cfd6',width:1},hoverinfo:'skip',showlegend:false});}
const axisTitle=method==='pca'?'PC':method==='tsne'?'t-SNE ':'방향 축 ';
const bounds=metadata.bounds[method];const axis=(j)=>({title:method==='length'?(j===1?'Train 길이 예측축':'직교 축 '+j):axisTitle+j,range:bounds[j-1],zeroline:true,showbackground:false,gridcolor:'#e5e9ee'});
const layout={margin:{l:0,r:isDuration?85:0,b:0,t:5},scene:{xaxis:axis(1),yaxis:axis(2),zaxis:axis(3),aspectmode:'cube',camera:{eye:{x:1.5,y:1.5,z:1.1}},uirevision:method+resetVersion},legend:{orientation:'h',y:1},uirevision:method+resetVersion};
if(isDuration)layout.coloraxis={colorscale:'Viridis',cmin:logColor?-1:.5,cmax:logColor?4:16,colorbar:{title:{text:'원음 길이<br>(초)'},tickvals:logColor?[-1,0,1,2,3,4]:[.5,2,4,8,12,16],ticktext:logColor?['0.5','1','2','4','8','16']:['0.5','2','4','8','12','16'],thickness:17,len:.65,x:1.02}};
Plotly.react('plot',traces,layout,{responsive:true,displaylogo:false,toImageButtonOptions:{format:'png',filename:'ebm-residual-duration-3d',width:1400,height:1000}});
document.getElementById('meta').textContent='표시 '+selected.length.toLocaleString()+'개 · 정상 '+selected.filter(p=>p.y===1).length.toLocaleString()+' · 스푸프 '+selected.filter(p=>p.y===0).length.toLocaleString()+' · 오류 정상 '+selected.filter(p=>p.rejected_bona).length;
document.getElementById('explanation').textContent=method==='pca'?'Train에서만 정한 동일한 세 축입니다. Train 방향 분산의 '+metadata.pcaVariance.toFixed(1)+'%를 설명하며 보이지 않는 나머지 방향도 있습니다.':method==='tsne'?'모든 데이터셋을 함께 계산했습니다. 가까운 이웃을 보기 위한 그림이며 섬 사이 거리·방위·빈 공간을 실제 각도나 전방향 분포로 해석할 수 없습니다.':method==='length'?'첫 축은 Train 정상만으로 원음 log₂ 길이를 예측하도록 정한 방향이며, 다른 두 축은 여기에 직교합니다. 작은 변화를 볼 수 있게 각 축 범위를 따로 확대했습니다. 화면상의 축 길이·각도는 원래 공간의 크기·각도가 아닙니다. 훈련 색상 정렬은 축을 만든 목적의 영향을 받으므로 별도 데이터셋 상관과 함께 해석해야 합니다.':'원점을 보존한 실제 선형투영입니다. 옅은 구는 단위 구면 기준이며 점을 구 표면으로 늘리지 않았습니다. 원음 길이와의 관계에는 반복·잘림 처리도 관련될 수 있습니다.';
lastMethod=method;
}
'''
    html = html[:start] + render + html[end:]
    html = html.replace("for(const id of ['method','color'])", "for(const id of ['method','color','durationScale'])", 1)
    (OUT / "residual_direction_duration_3d.html").write_text(html, encoding="utf-8")
    return source, html


def main():
    started = time.monotonic()
    assert not (OUT / "results.json").exists()
    protected = {str(p): shared.digest(p) for p in (ROOT / "EBM").rglob("*.py") if "__pycache__" not in p.parts}
    save("existing_sources_before.json", protected)
    for path in (ORIGINAL / "plotted_points_3d.parquet", ORIGINAL / "plot_unit_directions_3d.npy", ORIGINAL / "projection_parameters.npz"):
        shared.check_file(path)
    points = pd.read_parquet(ORIGINAL / "plotted_points_3d.parquet")
    plot_u = np.load(ORIGINAL / "plot_unit_directions_3d.npy").astype(float)
    catalog = json.loads((CACHE / "catalog.json").read_text())
    shared.check_file(CACHE / "catalog.json")
    f, u, c = shared.load_run(catalog, "19la_tv")
    data = {d: (f.loc[f.split == d].reset_index(drop=True), u[f.split.to_numpy() == d]) for d in ("T", "V")}
    f, u, c2 = shared.load_run(catalog, "frozen_19la_eval")
    assert np.array_equal(c, c2)
    data["19LA"] = (f, u)
    lockpath = CACHE / "analysis_manifests/frozen_data_sizes_v1/lock.json"
    shared.check_file(lockpath)
    lock = json.loads(lockpath.read_text())["views"]["21LA_pilot"]
    manifest = lockpath.parent / lock["manifest_file"]
    shared.check_file(manifest, lock["manifest_sha256"])
    selected = pd.read_parquet(manifest)
    f, u, c2 = shared.load_run(catalog, "21la_eval", set(selected.utterance_id))
    assert np.array_equal(c, c2) and set(f.utterance_id) == set(selected.utterance_id)
    data["21LA"] = (f, u)
    f, u, c2 = shared.load_run(catalog, "21df_pilot")
    assert np.array_equal(c, c2)
    data["DF"] = (f, u)
    train_f, train_all = data["T"]
    normal = train_f.y.to_numpy() == 1
    x = train_all[normal]
    y = np.log2(train_f.duration.to_numpy()[normal])
    groups = train_f.speaker_group.to_numpy()[normal]
    assert len(x) == 2580 and len(set(groups)) == 20
    pred, baseline = np.zeros(len(y)), np.zeros(len(y))
    outer_coefs, fold_info = [], []
    for fold, (itr, ite) in enumerate(GroupKFold(5).split(x, y, groups)):
        alpha, loss = choose_alpha(x[itr], y[itr], groups[itr], 4)
        m = Ridge(alpha=alpha, solver="cholesky").fit(x[itr], y[itr])
        pred[ite], baseline[ite] = m.predict(x[ite]), y[itr].mean()
        outer_coefs.append(m.coef_ / np.linalg.norm(m.coef_))
        fold_info.append({"fold": fold, "alpha": alpha, "train_n": len(itr), "test_n": len(ite),
                          "train_speakers": sorted(set(groups[itr])), "test_speakers": sorted(set(groups[ite]))})
    alpha, loss = choose_alpha(x, y, groups)
    model = Ridge(alpha=alpha, solver="cholesky").fit(x, y)
    coef = model.coef_.copy()
    axis = coef - (coef @ c) * c
    axis /= np.linalg.norm(axis)
    params = np.load(ORIGINAL / "projection_parameters.npz")
    vectors = [axis]
    for v in params["pca_components"]:
        v = v.copy()
        for q in vectors:
            v -= (q @ v) * q
        if np.linalg.norm(v) > 1e-8:
            vectors.append(v / np.linalg.norm(v))
        if len(vectors) == 3:
            break
    basis3 = np.array(vectors).T
    assert np.max(abs(basis3.T @ basis3 - np.eye(3))) < 1e-10
    assert np.max(abs(basis3.T @ c)) < 1e-8
    points[["length1", "length2", "length3"]] = plot_u @ basis3
    points.to_parquet(OUT / "plotted_points_duration.parquet", index=False)
    np.savez(OUT / "duration_probe.npz", coefficient=coef, intercept=model.intercept_, unit_axis=axis, basis3=basis3,
             centroid=c, alpha=alpha, train_oof_prediction=pred, train_target_log2duration=y, train_fold_baseline=baseline)
    train_stats = {"fit": "Train bona fide only", "n": len(x), "speakers": len(set(groups)), "alpha": alpha,
                   "alpha_CV_MSE": loss, "nested_CV_spearman": rho(y, pred), "nested_CV_R2": float(r2_score(y, pred)),
                   "nested_CV_MSE_reduction_vs_fold_mean": float(1 - mean_squared_error(y, pred) / mean_squared_error(y, baseline)),
                   "fold_axis_cosine_to_full_fit": [float(a @ axis) for a in outer_coefs], "folds": fold_info,
                   "axis_projection_energy_in_original_PCA3": float(np.sum((params["pca_components"][:3] @ axis) ** 2))}
    save("train_only_probe.json", train_stats)
    rows, correlations = [], []
    for d in shared.DATASETS:
        frame, directions = data[d]
        assert frame.crop_samples.eq(64600).all() and frame.crop_start.eq(0).all()
        assert np.max(abs(frame.padding_fraction - np.maximum(0, 1 - frame.duration / CAP))) < 1e-6
        score = directions @ axis
        predict = model.predict(directions)
        pca = (directions - params["pca_mean"]) @ params["pca_components"][:3].T
        bearing = directions @ np.load(ORIGINAL / "bearing_basis3.npy")
        for label in (1, 0):
            keep = frame.y.to_numpy() == label
            part = frame.loc[keep].reset_index(drop=True)
            length, value = part.duration.to_numpy(), score[keep]
            short = length < CAP
            rows.append({"dataset": d, "class": "normal" if label else "spoof", "n": len(part),
                         "length_axis_spearman": rho(length, value),
                         "partial_rank_rho_score_speaker_source_codec_attack": adjusted_rank_rho(part, value),
                         "short_n": int(short.sum()), "short_spearman": rho(length[short], value[short]),
                         "long_n": int((~short).sum()), "long_spearman": rho(length[~short], value[~short]),
                         "log2_duration_prediction_R2": float(r2_score(np.log2(length), predict[keep])),
                         "scope": "Train resubstitution" if d == "T" else "No target fitting"})
            for method, values in [("pca", pca), ("direction", bearing)]:
                for j in range(3):
                    correlations.append({"dataset": d, "class": "normal" if label else "spoof", "axis": f"{method}{j+1}",
                                         "n": len(part), "spearman_original_duration": rho(length, values[keep, j])})
        print(f"{d}: length associations complete", flush=True)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "duration_associations.csv", index=False)
    pd.DataFrame(correlations).to_csv(OUT / "original_axes_duration_correlations.csv", index=False)
    # Preserve original coordinates and show smooth common duration colors by class.
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 9})
    fig = plt.figure(figsize=(18, 8.8), layout="constrained")
    bounds = [(points[f"pca{j}"].min() - .04, points[f"pca{j}"].max() + .04) for j in (1, 2, 3)]
    axes = []
    for row, label in enumerate((1, 0)):
        for col, dataset in enumerate(shared.DATASETS):
            ax = fig.add_subplot(2, 5, row * 5 + col + 1, projection="3d")
            b = points[(points.dataset == dataset) & (points.y == label)]
            scatter = ax.scatter(b.pca1, b.pca2, b.pca3, c=b.duration, cmap="viridis", norm=LogNorm(.5, 16), s=6, alpha=.8, linewidths=0, rasterized=True)
            ax.set_title(shared.NAMES[dataset] + (" · 정상" if label else " · 스푸프"))
            ax.set(xlim=bounds[0], ylim=bounds[1], zlim=bounds[2], xlabel="PC1", ylabel="PC2", zlabel="PC3")
            ax.view_init(elev=22, azim=-58)
            ax.tick_params(labelsize=6, pad=0)
            axes.append(ax)
    bar = fig.colorbar(scatter, ax=axes, orientation="horizontal", shrink=.6, aspect=45, pad=.03)
    bar.set_ticks([.5, 1, 2, 4, 8, 16], labels=["0.5", "1", "2", "4", "8", "16"])
    bar.set_label("원음 길이 (초) · 공통 로그 색상 눈금")
    fig.suptitle("기존 공통 3D PCA 좌표 유지 · 길이에 따른 연속 색상\n보관 batch160·epoch50 / 입력은 모두 4.04초로 반복·자름 / 데이터셋×클래스당 동일한 500개", fontsize=13)
    fig.savefig(OUT / "duration_common_pca_3d.png", dpi=175)
    plt.close(fig)
    source, html = patch_viewer(points, train_stats, basis3)
    for path, value in shared.READ_HASHES.items():
        assert shared.digest(path) == value
    for path, value in protected.items():
        assert shared.digest(path) == value
    assert time.monotonic() - started < 1200
    save("input_hashes.json", shared.READ_HASHES)
    result = {"status": "analysis_complete_pending_browser_check", "model": "archived batch160 epoch50", "checkpoint_sha256": shared.SHA,
              "plot_n": len(points), "original_coordinates_unchanged": True, "duration_source": "original waveform seconds",
              "input_duration_seconds": CAP, "train_probe": train_stats, "associations": rows,
              "input_hashes_unchanged": len(shared.READ_HASHES), "existing_python_sources_unchanged": len(protected),
              "elapsed_seconds": time.monotonic() - started,
              "safety": {"EBM_training_modified": False, "EBM_updates": 0, "new_GPU_inferences": 0,
                         "remote_operations": 0, "new_spend_usd": 0, "Q_opened": False}}
    save("results.json", result)
    print(metrics.to_string(index=False), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
