"""Centroid-anchored scientific figures from existing, authorized local caches.

No model training, new inference, cache sync, GPU, or remote mutation.
The polar view preserves theta to c, but compresses 159 tangent dimensions
into a two-axis bearing. Pairwise angles are measured in the full space.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import datetime as dt
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from geometry_cache import DEFAULT_ROOT, load_embeddings, validate_run

OUT = DEFAULT_ROOT / "visualizations/centroid_centered_20261105_v2"
SEED = 20261105
BLUE, ORANGE, GRAY = "#286b9c", "#c65f25", "#465260"
NAMES = ["T", "DF_pilot", "21LA_pilot"]
TITLES = {"T": "2019 LA · 학습 표본", "DF_pilot": "2021 DF · 고정 20,000개", "21LA_pilot": "2021 LA · 고정 20,000개"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def pick(rows, per_class, key):
    # UID-only selection; never select points by score, angle, or appearance.
    priority = np.array([hashlib.sha256(f"{key}|{u}".encode()).hexdigest() for u in rows.utterance_id])
    ix = []
    for label in [0, 1]:
        pool = np.flatnonzero(rows.y.to_numpy() == label)
        ix.extend(pool[np.argsort(priority[pool])[:per_class]].tolist())
    return np.asarray(sorted(ix), dtype=int)


def load_data():
    data = {}
    checks = {}
    for alias in ["T", "DF_pilot", "21LA_eval"]:
        checks[alias] = validate_run(alias, verify_hashes=True)
        d = load_embeddings(alias)
        if alias == "21LA_eval":
            root = DEFAULT_ROOT / "analysis_manifests/frozen_data_sizes_v1"
            lock = json.loads((root / "lock.json").read_text(encoding="utf-8"))["views"]["21LA_pilot"]
            assert digest(root / lock["manifest_file"]) == lock["manifest_sha256"]
            assert d.provenance["cache_fingerprint"] == lock["source_cache_fingerprint"]
            m = pd.read_parquet(root / lock["manifest_file"])
            ix = m.cache_row.to_numpy(dtype=int)
            assert d.rows.iloc[ix].utterance_id.tolist() == m.utterance_id.tolist()
            assert np.array_equal(d.rows.iloc[ix].y, m.y)
            d.z, d.h, d.rows = d.z[ix], d.h[ix], d.rows.iloc[ix].reset_index(drop=True)
            alias = "21LA_pilot"
        data[alias] = d
    c = data["T"].centroid
    for d in data.values():
        assert np.max(np.abs(d.centroid - c)) < 1e-12
        assert not d.rows.utterance_id.duplicated().any()
        assert set(d.rows.y.unique()) == {0, 1}
    return data, checks


def geometry(d, c):
    x = d.z.astype(np.float64)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    s = np.clip(x @ c, -1, 1)
    theta = np.degrees(np.arccos(s))
    tangent = x - s[:, None] * c
    tn = np.linalg.norm(tangent, axis=1)
    valid = tn >= 1e-4
    u = np.zeros_like(tangent)
    u[valid] = tangent[valid] / tn[valid, None]
    assert np.max(np.abs(s - d.rows.cosine_s.to_numpy())) < 1e-10
    assert np.max(np.abs(np.sum((x-c)**2, axis=1) - (2-2*s))) < 1e-12
    assert np.max(np.abs(tangent @ c)) < 1e-12
    return dict(x=x, s=s, theta=theta, tangent=tangent, u=u, valid=valid)


def pairs(x, n, seed):
    rng = np.random.default_rng(seed)
    a = rng.integers(len(x), size=n)
    b = rng.integers(len(x)-1, size=n)
    b += b >= a
    angles = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", x[a], x[b]), -1, 1)))
    return angles


def concentration(x):
    n = len(x)
    r = np.linalg.norm(x.mean(axis=0))
    # Exact mean over all ordered distinct pairs, without an O(n^2) matrix.
    mean_cosine = (n*r*r-1)/(n-1)
    return dict(mean_resultant_length=float(r), all_distinct_pairs_mean_cosine=float(mean_cosine))


def summary(data, geo, basis, pca):
    stats, groups, pair_arrays = {}, [], {}
    for name, d in data.items():
        g, y = geo[name], d.rows.y.to_numpy()
        coords = g["u"] @ basis.T
        phi = np.arctan2(coords[:, 1], coords[:, 0])
        radial = g["theta"][:, None] * np.column_stack([np.cos(phi), np.sin(phi)])
        radial[~g["valid"]] = np.nan
        linear = (g["x"]-d.centroid) @ pca.components_.T
        g.update(phi=phi, polar=radial, linear=linear, tangent_energy=np.sum(coords**2, axis=1))
        assert np.max(np.abs(np.linalg.norm(radial[g["valid"]], axis=1)-g["theta"][g["valid"]])) < 1e-10
        assert np.max(np.linalg.norm(linear, axis=1) - np.linalg.norm(g["x"]-d.centroid, axis=1)) < 1e-10
        stats[name] = {"N":len(y), "classes":{}, "fingerprint":d.provenance["cache_fingerprint"]}
        for label, key in [(1,"bona"),(0,"spoof")]:
            m = y == label
            pp = pairs(g["x"][m], 50000, SEED+label)
            pair_arrays[(name, key)] = pp
            a = g["theta"][m]
            valid = m & g["valid"]
            unitstats = concentration(g["x"][m])
            ustats = concentration(g["u"][valid])
            stats[name]["classes"][key] = {
                "N":int(m.sum()), "centroid_angle_deg_q05_q50_q95":np.quantile(a,[.05,.5,.95]).tolist(),
                "centroid_angle_deg_mean":float(a.mean()), "centroid_angle_over_90_percent":float(100*np.mean(a>90)),
                "full_space":unitstats, "tangent_directions":ustats,
                "within_class_pair_angle_deg_q05_q50_q95":np.quantile(pp,[.05,.5,.95]).tolist(),
                "pair_draws":len(pp), "pairs_are_dependent_descriptive_only":True,
                "bearing_axes_tangent_energy_mean":float(np.mean(g["tangent_energy"][valid])),
                "bearing_axes_tangent_energy_median":float(np.median(g["tangent_energy"][valid])),
                "invalid_tangent_N":int(np.sum(m & ~g["valid"])),
            }
            # Independently verify exact pairwise-mean identity on a small block.
            block = g["x"][m][:80]
            gram = block @ block.T
            empirical = (gram.sum()-np.trace(gram))/(len(block)*(len(block)-1))
            assert abs(empirical-concentration(block)["all_distinct_pairs_mean_cosine"]) < 1e-12
        for field in ["attack_id", "source_corpus", "vocoder", "codec"]:
            if field not in d.rows:
                continue
            for key, sub in d.rows[y==0].groupby(field, dropna=False):
                ix = sub.index.to_numpy()
                groups.append(dict(dataset=name, field=field, group=str(key), n=len(ix),
                    angle_q05=float(np.quantile(g["theta"][ix],.05)), angle_median=float(np.median(g["theta"][ix])),
                    angle_q95=float(np.quantile(g["theta"][ix],.95)),
                    resultant_length=float(np.linalg.norm(g["x"][ix].mean(axis=0))),
                    tangent_resultant_length=float(np.linalg.norm(g["u"][ix].mean(axis=0)))))
    return stats, pd.DataFrame(groups), pair_arrays


def legend():
    return [Line2D([],[],marker="o",markerfacecolor="none",markeredgecolor=BLUE,ls="none",label="정상"),
            Line2D([],[],marker="x",color=ORANGE,ls="none",label="스푸프"),
            Line2D([],[],marker="*",color="#222222",ls="none",markersize=10,label="학습된 정상 센트로이드 c")]


def save(fig, name):
    fig.savefig(OUT/f"{name}.png", dpi=180, facecolor="white")
    fig.savefig(OUT/f"{name}.svg", facecolor="white")
    plt.close(fig)


def plot_maps(data, geo, stats, samples, mode):
    fig, axes = plt.subplots(1,3,figsize=(17,7))
    if mode == "polar":
        title = "센트로이드를 중심에 놓고 펼친 각도 지도"
        footer = ("반지름 = 원래 160차원에서 센트로이드와 이루는 각도(°). 원은 30° 간격의 눈금이며 결정경계가 아닙니다.\n"
                  "지도 방위는 잔여 방향 2축으로 압축했습니다. 점 사이의 거리·각도·면적은 보존하지 않습니다.\n"
                  "180° 테두리는 구면의 반대점에 해당합니다. 테두리 부근의 호 길이를 실제 스푸프 간 거리로 읽으면 안 됩니다.")
    else:
        title = "센트로이드를 원점에 놓은 실제 선형 투영"
        footer = ("정규화한 160차원 벡터와 c를 같은 PCA에 투영한 뒤, c의 투영 좌표를 빼서 (0,0)에 놓았습니다.\n"
                  "모든 패널에 같은 학습 표본 PCA·같은 축척을 사용합니다. 투영 거리는 원래 거리보다 작거나 같습니다.")
    fig.suptitle(title, fontsize=20, fontweight="bold", y=.98)
    fig.text(.5,.91,"같은 고정 EBM · 그림은 각 클래스 최대 2,000개 · 통계는 각 데이터의 전체 표본",ha="center",fontsize=11,color=GRAY)
    for ax, name in zip(axes, NAMES):
        g, d, ix = geo[name], data[name], samples[name]
        xy = g[mode if mode=="polar" else "linear"]
        y = d.rows.y.to_numpy()
        if mode == "polar":
            for r in [30,60,90,120,150,180]:
                ax.add_patch(Circle((0,0),r,fill=False,color="#b0b7c0",lw=.6,alpha=.7))
                ax.text(r/np.sqrt(2),r/np.sqrt(2),f"{r}°",fontsize=8,color=GRAY,
                        bbox=dict(facecolor="white",edgecolor="none",alpha=.85,pad=.5))
            lim = 190
            ax.set_xticks([]); ax.set_yticks([])
        else:
            lim=2.05
            ax.set_xticks([-2,-1,0,1,2]); ax.set_yticks([-2,-1,0,1,2])
            ax.set_xlabel("PCA 1 · c 기준 좌표")
            ax.set_ylabel("PCA 2 · c 기준 좌표")
            ax.grid(alpha=.12)
        ax.axhline(0,color="#a0a6ae",lw=.6,zorder=0)
        ax.axvline(0,color="#a0a6ae",lw=.6,zorder=0)
        for label,col,marker in [(0,ORANGE,"x"),(1,BLUE,"o")]:
            j=ix[y[ix]==label]
            if label==0:
                ax.scatter(xy[j,0],xy[j,1],s=9,color=col,marker=marker,alpha=.23,lw=.5,rasterized=True)
            else:
                ax.scatter(xy[j,0],xy[j,1],s=11,facecolors="none",edgecolors=col,alpha=.32,lw=.55,rasterized=True)
        ax.scatter(0,0,marker="*",s=170,color="#111111",edgecolors="white",lw=.7,zorder=8)
        ax.set_xlim(-lim,lim);ax.set_ylim(-lim,lim);ax.set_aspect("equal")
        ax.set_title(TITLES[name],fontsize=13,fontweight="bold",pad=18)
        b=stats[name]["classes"]["bona"]; s=stats[name]["classes"]["spoof"]
        if mode=="polar":
            ax.text(.5,-.10,f"중심 각도 중앙값  정상 {b['centroid_angle_deg_q05_q50_q95'][1]:.1f}° / 스푸프 {s['centroid_angle_deg_q05_q50_q95'][1]:.1f}°",transform=ax.transAxes,ha="center",fontsize=10)
            ax.text(.5,-.16,f"방위 2축의 스푸프 잔여 방향 에너지: 평균 {100*s['bearing_axes_tangent_energy_mean']:.1f}%",transform=ax.transAxes,ha="center",fontsize=9,color=GRAY)
    fig.legend(handles=legend(),loc="lower center",bbox_to_anchor=(.5,.137),ncol=3,frameon=False,fontsize=11)
    fig.text(.035,.02,footer,fontsize=10,color=GRAY,linespacing=1.6)
    fig.subplots_adjust(left=.045,right=.98,top=.82,bottom=.29,wspace=.16)
    save(fig,"01_centroid_angle_map" if mode=="polar" else "02_centroid_linear_projection")


def plot_quantitative(data, geo, stats, pair_arrays):
    fig, axes=plt.subplots(2,3,figsize=(16.2,8.3))
    bins=np.linspace(0,180,61)
    for col,name in enumerate(NAMES):
        y=data[name].rows.y.to_numpy()
        for label,key,color,ls in [(1,"bona",BLUE,"-"),(0,"spoof",ORANGE,"--")]:
            a=geo[name]["theta"][y==label]
            b=pair_arrays[(name,key)]
            for ax,arr in [(axes[0,col],a),(axes[1,col],b)]:
                counts,_=np.histogram(arr,bins)
                ax.stairs(100*counts/len(arr),bins,color=color,ls=ls,lw=1.7,label="정상" if label else "스푸프")
        for row in [0,1]:
            ax=axes[row,col]
            ax.set_xlim(0,180);ax.set_xticks([0,30,60,90,120,150,180]);ax.set_ylim(bottom=0)
            ax.grid(alpha=.15);ax.spines[["top","right"]].set_visible(False)
            ax.set_ylabel("클래스 안의 비율 (%) / 3° 구간")
            ax.set_xlabel("센트로이드와의 각도 (°)" if row==0 else "같은 클래스의 두 발화 사이 각도 (°)")
        axes[0,col].set_title(TITLES[name],fontsize=12,fontweight="bold",pad=13)
        med=stats[name]["classes"]["spoof"]["within_class_pair_angle_deg_q05_q50_q95"][1]
        axes[1,col].text(.97,.94,f"스푸프끼리 중앙값 {med:.1f}°",transform=axes[1,col].transAxes,ha="right",va="top",fontsize=10)
    # Common y scales within each row for comparable magnitudes.
    for row in [0,1]:
        high=max(ax.get_ylim()[1] for ax in axes[row])
        for ax in axes[row]:ax.set_ylim(0,high)
    fig.suptitle("중심에서 멀어진 정도와 스푸프끼리 퍼진 정도를 구분",fontsize=19,fontweight="bold",y=.99)
    fig.legend(handles=legend()[:2],loc="lower center",bbox_to_anchor=(.5,.075),ncol=2,frameon=False)
    fig.text(.035,.025,"원래 160차원에서 계산 · 위: 전체 발화 · 아래: 클래스별 무작위 50,000쌍(자기 자신 제외, 독립 표본 수가 아님)",fontsize=10,color=GRAY)
    fig.subplots_adjust(left=.07,right=.98,top=.88,bottom=.18,hspace=.5,wspace=.32)
    save(fig,"03_full_space_angles")


def plot_attack_map(data, geo, samples):
    # Expose 21LA attack differences without inventing more than 13 categories.
    name="21LA_pilot";d=data[name];g=geo[name];ix=samples[name];y=d.rows.y.to_numpy()
    attacks=sorted(d.rows.loc[y==0,"attack_id"].astype(str).unique())
    fig,axes=plt.subplots(3,5,figsize=(15.5,10.7))
    fig.suptitle("2021 LA: 공격별로 나누어 본 센트로이드 중심 각도 지도",fontsize=19,fontweight="bold",y=.99)
    bona=ix[y[ix]==1]
    for ax,attack in zip(axes.flat,attacks):
        for r in [60,120,180]:ax.add_patch(Circle((0,0),r,fill=False,color="#bcc1c8",lw=.6))
        attackmask=(y==0)&(d.rows.attack_id.astype(str).to_numpy()==attack)
        selected=ix[attackmask[ix]]
        ax.scatter(g["polar"][bona,0],g["polar"][bona,1],s=3,color=BLUE,alpha=.08,rasterized=True)
        ax.scatter(g["polar"][selected,0],g["polar"][selected,1],s=9,color=ORANGE,marker="x",lw=.5,alpha=.5,rasterized=True)
        ax.scatter(0,0,s=75,marker="*",color="#111111",zorder=5)
        ax.set_title(f"{attack} · n={attackmask.sum():,}\n中心각 중앙값 {np.median(g['theta'][attackmask]):.1f}°".replace("中心각","중심 각도"),fontsize=10)
        ax.set_xlim(-190,190);ax.set_ylim(-190,190);ax.set_aspect("equal");ax.set_xticks([]);ax.set_yticks([])
    for ax in list(axes.flat)[len(attacks):]:ax.axis("off")
    fig.legend(handles=legend(),loc="lower center",bbox_to_anchor=(.5,.07),ncol=3,frameon=False)
    fig.text(.035,.02,"반지름 = 실제 센트로이드 각도 · 방위는 학습 표본의 동일한 2축으로 압축 · 원은 60° 간격의 눈금\n각 공격의 통계는 해당 공격 전체, 점은 앞 그림과 같은 UID 고정 표본만 사용 · 파란색은 같은 정상 표본",fontsize=10,color=GRAY)
    fig.subplots_adjust(left=.04,right=.98,top=.90,bottom=.17,hspace=.45,wspace=.20)
    save(fig,"04_21LA_attack_facets")


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({"font.family":"Malgun Gothic","font.size":10,"axes.unicode_minus":False,"svg.fonttype":"none"})
    data,checks=load_data()
    print("Validated local caches",flush=True)
    c=data["T"].centroid
    geo={name:geometry(d,c) for name,d in data.items()}
    fitix=pick(data["T"].rows,2500,"centroid-map-fit-v1")
    fitu=geo["T"]["u"][fitix]
    _,sv,vt=np.linalg.svd(fitu,full_matrices=False)
    basis=vt[:2].copy()
    # Deterministic orientation: largest-magnitude coefficient is positive.
    for row in basis:
        if row[np.argmax(np.abs(row))]<0:row*=-1
    assert np.max(np.abs(basis@c))<1e-10
    assert np.max(np.abs(basis@basis.T-np.eye(2)))<1e-10
    pca=PCA(n_components=2,svd_solver="full").fit(geo["T"]["x"][fitix])
    stats,groups,pair_arrays=summary(data,geo,basis,pca)
    samples={name:pick(d.rows,2000,"centroid-map-display-v1") for name,d in data.items()}
    stats["method"]={"UTC":dt.datetime.now(dt.timezone.utc).isoformat(),"seed":SEED,
        "centroid":"Fixed learned bona centroid, not an empirical test mean", "fit_split":"T only", "fit_N":len(fitix),
        "bearing":"atan2(u dot tangent_axis2, u dot tangent_axis1); axes from uncentered SVD of unit tangent directions",
        "radius":"theta = degrees(acos(unit_z dot c)); exact", "pairwise_geometry":"Full original 160 dimensions",
        "linear":"PCA(unit_z) - PCA(c) = (unit_z-c) dot components.T",
        "linear_T_explained_variance_ratio":pca.explained_variance_ratio_.tolist(),
        "fit_tangent_2axis_energy":float((sv[:2]**2).sum()/(sv**2).sum()),
        "display_class_cap":2000,"new_inference":False,"training":False,"Q_accessed":False,
        "limitations":["2D bearing compresses 159 dimensions; pairwise distances and areas are distorted",
                       "Radial rings indicate angles, not class boundaries or uniform spherical density",
                       "Angle is relative to a learned centroid, not physical direction",
                       "Previously observed fixed pilots, one checkpoint, no population inference or causal claim",
                       "DF and LA share speakers; not independent repetitions"],
        "script_sha256":digest(__file__)}
    dump(OUT/"statistics.json",stats)
    groups.to_csv(OUT/"group_statistics.csv",index=False,encoding="utf-8-sig")
    points=[]
    for name,ix in samples.items():
        g,d=geo[name],data[name]
        fields=[k for k in ["utterance_id","y","attack_id","source_corpus","codec"] if k in d.rows]
        frame=d.rows.iloc[ix][fields].copy()
        frame["dataset"]=name
        frame["theta_deg"]=g["theta"][ix];frame["bearing_rad"]=g["phi"][ix]
        frame["map_x"]=g["polar"][ix,0];frame["map_y"]=g["polar"][ix,1]
        frame["linear_x"]=g["linear"][ix,0];frame["linear_y"]=g["linear"][ix,1]
        points.append(frame)
    pd.concat(points,ignore_index=True).to_parquet(OUT/"plotted_points.parquet",index=False)
    np.savez(OUT/"fixed_axes.npz",tangent_basis=basis,pca_components=pca.components_,centroid=c,fit_indices=fitix)
    plot_maps(data,geo,stats,samples,"polar")
    plot_maps(data,geo,stats,samples,"linear")
    plot_quantitative(data,geo,stats,pair_arrays)
    plot_attack_map(data,geo,samples)
    dump(OUT/"verification.json",{"status":"pass","input_checks":checks,
        "checks":["SHA256 input cache checks","locked manifest UID and label alignment","same learned centroid",
                  "native cosine reproduction","normalized chord-angle identity","tangent orthogonality",
                  "fixed axes orthonormality","polar radial preservation","linear projection contraction",
                  "all-pair cosine formula independently matched to explicit Gram matrices"],
        "visual_inspection":"pending","new_model_training_or_GPU_inference":False})
    print(json.dumps({name:stats[name]["classes"] for name in NAMES},ensure_ascii=False,indent=2),flush=True)


def redraw_maps():
    """Repair chart presentation using the exact previously saved plotted rows."""
    from types import SimpleNamespace
    plt.rcParams.update({"font.family":"Malgun Gothic","font.size":10,"axes.unicode_minus":False,"svg.fonttype":"none"})
    points=pd.read_parquet(OUT/"plotted_points.parquet")
    stats=json.loads((OUT/"statistics.json").read_text(encoding="utf-8"))
    data,geo,samples={},{},{}
    for name in NAMES:
        frame=points.loc[points.dataset==name].reset_index(drop=True)
        data[name]=SimpleNamespace(rows=frame)
        geo[name]={"polar":frame[["map_x","map_y"]].to_numpy(),"linear":frame[["linear_x","linear_y"]].to_numpy()}
        samples[name]=np.arange(len(frame))
    plot_maps(data,geo,stats,samples,"polar")
    plot_maps(data,geo,stats,samples,"linear")
    stats["method"]["script_sha256"]=digest(__file__)
    dump(OUT/"statistics.json",stats)
    print("Re-rendered exact saved points; statistical results unchanged.",flush=True)


if __name__=="__main__":
    import sys
    with threadpool_limits(limits=2):
        if "--redraw-maps" in sys.argv:
            redraw_maps()
        else:
            main()
