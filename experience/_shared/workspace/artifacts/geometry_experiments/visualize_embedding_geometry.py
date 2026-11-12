"""Reproducible scientific figures from existing local T/V/DF caches only."""

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from fast_analyze import eer
from geometry_cache import DEFAULT_ROOT, load_embeddings, read_json, resolve_run, validate_run


BLUE = "#346fa2"
ORANGE = "#c77c29"
ERROR = "#9b3658"
GRAY = "#46525e"
OUT = DEFAULT_ROOT / "visualizations/embedding_geometry_v1"


def choose_rows(rows, per_class, seed):
    rng = np.random.default_rng(seed)
    indices = []
    for label in [0, 1]:
        pool = np.flatnonzero(rows.y.to_numpy() == label)
        indices.extend(rng.choice(pool, min(per_class, len(pool)), replace=False).tolist())
    return np.asarray(sorted(indices))


def padded(values):
    low, high = float(np.min(values)), float(np.max(values))
    margin = max((high-low)*.06, .01)
    return low-margin, high+margin


def style(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(alpha=.15)
    axis.set_axisbelow(True)


def export(fig, name):
    fig.savefig(OUT / name, dpi=180, facecolor="white")
    plt.close(fig)


def plot_objective():
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.4, 5.5), gridspec_kw={"width_ratios": [1, 1.35]})
    s = .6
    radius = np.sqrt(1-s*s)
    theta = np.linspace(0, 2*np.pi, 721)
    left.plot(radius*np.cos(theta), radius*np.sin(theta), color=GRAY, linewidth=1.4)
    for angle, label in zip([0, np.pi/3, np.pi/2, 4*np.pi/3], ["A", "B", "C", "D"]):
        point = radius*np.array([np.cos(angle), np.sin(angle)])
        left.plot([0,point[0]],[0,point[1]],color=ORANGE,alpha=.5,linewidth=1)
        left.scatter(*point,color=ORANGE,s=45,zorder=3)
        left.text(point[0]*1.17,point[1]*1.17,label,ha="center",va="center",fontweight="bold")
    left.axhline(0,color=GRAY,linewidth=.5,alpha=.5)
    left.axvline(0,color=GRAY,linewidth=.5,alpha=.5)
    left.set_aspect("equal")
    left.set_xlim(-1.07,1.07); left.set_ylim(-1.07,1.07)
    left.set_xlabel("센트로이드 축에 수직인 방향 1")
    left.set_ylabel("센트로이드 축에 수직인 방향 2")
    left.set_title("같은 코사인, 서로 다른 잔여 방향",loc="left",fontweight="bold",pad=14)
    left.text(0,-1.42,"A·B·C·D 모두 코사인 = 0.60\n스푸프 OC 손실도 모두 동일",ha="center",va="top",fontsize=10)
    unit = np.column_stack([np.full(len(theta),s),radius*np.cos(theta),radius*np.sin(theta)])
    c = np.array([1.,0.,0.])
    a = np.diag([1.,2.,.5])*3/3.5  # Explicit explanatory metric; not fitted to real data.
    score = 1-.5*np.sum(((unit-c)@a)*(unit-c),axis=1)
    assert np.max(np.abs(1-.5*np.sum((unit-c)**2,axis=1)-s)) < 1e-12
    degrees = theta*180/np.pi
    right.plot(degrees,np.full(len(theta),s),color=GRAY,label="기존 코사인 점수",linewidth=2,linestyle="--")
    right.plot(degrees,score,color=BLUE,label="방향별 가중 거리 점수(예시)",linewidth=2)
    right.set_xlim(0,360); right.set_ylim(0,1)
    right.set_xticks([0,90,180,270,360])
    right.set_xlabel("센트로이드 축 주위의 회전각 (도)")
    right.set_ylabel("탐지 점수 · 클수록 정상")
    right.set_title("기존 점수는 같고, 가중 점수는 달라짐",loc="left",fontweight="bold",pad=14)
    right.legend(loc="lower right",frameon=False,fontsize=9)
    style(right)
    fig.suptitle("학습 목적의 차이: 센트로이드와의 각도 vs 그 축 주위의 방향",fontsize=14,fontweight="bold",y=.99)
    fig.text(.02,.025,"설명용 가상 3차원 벡터와 임의의 대각 metric: 실제 데이터의 군집이나 학습 결과가 아님\n실제 모델 출력은 LayerNorm 제약이 있어 임의의 회전이 모두 실현 가능하다는 뜻도 아님",fontsize=9,color=GRAY)
    fig.subplots_adjust(left=.08,right=.98,top=.83,bottom=.27,wspace=.35)
    export(fig,"01_objective_geometry.png")


def plot_scores(data, geometry, threshold, statistics):
    fig, axes = plt.subplots(1,3,figsize=(14,4.8),sharex=True,sharey=True)
    edges = np.linspace(-1,1,81)
    for axis, name in zip(axes,data):
        y = data[name].rows.y.to_numpy(); s=geometry[name]["cosine"]
        for label,color,linestyle,text in [(1,BLUE,"-","정상"),(0,ORANGE,"--","스푸프")]:
            counts,_ = np.histogram(s[y==label],bins=edges)
            pct=100*counts/counts.sum()
            axis.stairs(np.where(pct>0,pct,np.nan),edges,color=color,linewidth=1.5,linestyle=linestyle,label=text)
            assert abs(pct.sum()-100)<1e-10
        axis.axvline(threshold,color=ERROR,linestyle=":",linewidth=1.5)
        axis.set_yscale("log")
        axis.set_ylim(.003,150);axis.set_xlim(-1.02,1.02)
        axis.set_yticks([.01,.1,1,10,100],["0.01","0.1","1","10","100"])
        axis.minorticks_off()
        axis.set_title(f"{name}  ·  EER {statistics[name]['EER_percent']:.2f}%\n정상 {int(y.sum()):,} / 스푸프 {int((1-y).sum()):,}",fontsize=11,fontweight="bold",pad=12)
        axis.set_xlabel("센트로이드 기준 코사인 점수 s")
        style(axis)
    axes[0].set_ylabel("클래스 안의 발화 비율 (%) · 로그축")
    handles,labels=axes[0].get_legend_handles_labels()
    handles.append(Line2D([],[],color=ERROR,linestyle=":",label="V에서 고정한 문턱값 0.9883"))
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,.11),ncol=3,frameon=False)
    fig.suptitle("원래 코사인 점수의 분포: DF 정상 일부가 낮은 점수로 이동",fontsize=14,fontweight="bold",y=.99)
    fig.text(.025,.02,"전체 캐시 사용 · 정상/스푸프를 각각 100%로 정규화 · 0인 bin은 로그축에서 생략\nDF 정상의 V 문턱값 오거부율 39.0%는 EER 3.35%와 다른 운용 지표",fontsize=9,color=GRAY)
    fig.subplots_adjust(left=.07,right=.98,top=.76,bottom=.28,wspace=.15)
    export(fig,"02_score_distributions.png")


def plot_geometry(data, geometry, pc, threshold, model, statistics):
    first = np.concatenate([pc[n][:,0] for n in data])
    second = np.concatenate([pc[n][:,1] for n in data])
    lim1,lim2=padded(first),padded(second)
    fig,axes=plt.subplots(2,3,figsize=(14.5,8.8))
    records=[]
    for col,name in enumerate(data):
        rows=data[name].rows
        sample=choose_rows(rows,1000,117)
        y=rows.y.to_numpy()[sample]
        s=geometry[name]["cosine"][sample]
        projection=pc[name][sample]
        rejected=(y==1)&(s<threshold)
        positions=[(s,projection[:,0]),(projection[:,0],projection[:,1])]
        for row,(x,z) in enumerate(positions):
            axis=axes[row,col]
            for label,color,marker in [(0,ORANGE,"x"),(1,BLUE,"o")]:
                selected=y==label
                if label==1:
                    axis.scatter(x[selected],z[selected],s=9,facecolors="none",edgecolors=color,linewidths=.4,alpha=.45)
                else:
                    axis.scatter(x[selected],z[selected],s=7,color=color,marker=marker,linewidths=.4,alpha=.25)
            if name=="DF_pilot":
                axis.scatter(x[rejected],z[rejected],s=18,facecolors="none",edgecolors=ERROR,marker="^",linewidths=.65,alpha=.7)
            if row==0:
                axis.set_xlim(-1.05,1.05);axis.set_ylim(*lim1)
                axis.axvline(threshold,color=ERROR,linestyle=":",linewidth=1)
                axis.set_xlabel("원래 코사인 s")
                axis.set_ylabel("잔여 단위방향 u의 PCA 1")
                axis.set_title(f"{name} · 코사인과 잔여 방향",fontweight="bold",fontsize=11,pad=12)
            else:
                axis.set_xlim(*lim1);axis.set_ylim(*lim2)
                axis.set_xlabel("잔여 단위방향 u의 PCA 1")
                axis.set_ylabel("잔여 단위방향 u의 PCA 2")
                axis.set_title(f"{name} · 잔여 방향만 표시",fontweight="bold",fontsize=11,pad=12)
            style(axis)
        records.extend({"dataset":name,"utterance_id":rows.iloc[i].utterance_id,"y":int(rows.iloc[i].y),
                        "cosine":float(geometry[name]["cosine"][i]),"u_pc1":float(pc[name][i,0]),"u_pc2":float(pc[name][i,1]),
                        "V_threshold_rejected_bona":bool(rows.iloc[i].y==1 and geometry[name]["cosine"][i]<threshold)} for i in sample)
        statistics[name]["plotted_bona"]=int((y==1).sum());statistics[name]["plotted_spoof"]=int((y==0).sum())
        statistics[name]["cosine_uPC1_spearman"]=float(spearmanr(geometry[name]["cosine"],pc[name][:,0]).statistic)
    pd.DataFrame(records).to_parquet(OUT/"scatter_points.parquet",index=False)
    legend=[Line2D([],[],marker="o",color=BLUE,markerfacecolor="none",linestyle="none",label="정상"),
            Line2D([],[],marker="x",color=ORANGE,linestyle="none",label="스푸프"),
            Line2D([],[],marker="^",color=ERROR,markerfacecolor="none",linestyle="none",label="V 문턱값에서 거부된 DF 정상")]
    fig.legend(handles=legend,loc="lower center",bbox_to_anchor=(.5,.085),ncol=3,frameon=False)
    explained=100*model.explained_variance_ratio_
    fig.suptitle(f"같은 축으로 본 임베딩: T에서 fit한 잔여 방향 PCA (설명 분산 {explained[0]:.1f}% + {explained[1]:.1f}%)",fontsize=14,fontweight="bold",y=.99)
    fig.text(.025,.02,"PCA는 T의 클래스 균형 5,000개에만 fit · 패널마다 그림용으로 정상/스푸프 각 1,000개 표시\n위 행은 s와 u의 한 축, 아래 행은 u의 두 축: 같은 2차원 그림에서 겹친다고 160차원의 정보 부재가 증명되는 것은 아님",fontsize=9,color=GRAY)
    fig.subplots_adjust(left=.065,right=.985,top=.90,bottom=.17,hspace=.40,wspace=.30)
    export(fig,"03_embedding_geometry.png")


def plot_channels(data, geometry, statistics):
    train=data["T"];yt=train.rows.y.to_numpy()
    unit=geometry["T"]["unit_z"]
    covariance=np.cov(unit[yt==1].T)
    scale=np.trace(covariance)/160
    diagonal=(1-.01)*np.diag(covariance)+.01*scale
    weights=1/np.maximum(diagonal,1e-6*scale)
    weights*=160/weights.sum()
    top=np.argsort(-weights)[:24]
    source,_,_=resolve_run("T")
    fisher=np.load(source/"fast_analysis/A_fisher_by_attack.npy",allow_pickle=False).mean(axis=0)
    rank=np.argsort(-fisher)
    denominator=np.sqrt((np.var(unit[yt==1],axis=0,ddof=1)+np.var(unit[yt==0],axis=0,ddof=1))/2+1e-8)
    effects=[]
    for name in data:
        y=data[name].rows.y.to_numpy();z=geometry[name]["unit_z"]
        effects.append((z[y==1].mean(axis=0)-z[y==0].mean(axis=0))/denominator)
    effects=np.asarray(effects)
    fig,(upper,lower)=plt.subplots(2,1,figsize=(13.8,7.7),gridspec_kw={"height_ratios":[1.1,1]})
    upper.bar(np.arange(24),weights[top],color=BLUE,width=.72)
    upper.axhline(1,color=GRAY,linestyle="--",linewidth=1,label="정규화 Euclidean의 거리계수 = 1")
    upper.set_xticks(np.arange(24),[str(i) for i in top])
    upper.set_ylim(0,max(weights[top])*1.17)
    upper.set_xlabel("최종 임베딩 채널 번호 · 가중치가 큰 24개만 표시")
    upper.set_ylabel("대각 거리계수 A[j,j] · 평균 1")
    upper.set_title("T 정상의 분산으로 계산한 거리계수 (γ=0.01)",loc="left",fontweight="bold",pad=12)
    upper.legend(frameon=False,fontsize=9,loc="upper right")
    style(upper)
    extent=float(np.max(np.abs(effects)))
    cmap=LinearSegmentedColormap.from_list("signed_effect",[BLUE,"#f4f4f4",ORANGE])
    image=lower.imshow(effects[:,rank],aspect="auto",cmap=cmap,vmin=-extent,vmax=extent,interpolation="nearest")
    lower.axvline(39.5,color=GRAY,linestyle="--",linewidth=1)
    lower.set_yticks([0,1,2],["T","V","DF pilot"])
    lower.set_xticks([0,39,79,119,159],["1","40","80","120","160"])
    lower.set_xlabel("T Fisher 중요도 순위로 고정한 채널 순서 (왼쪽 40개 = 기존 상위 25%)")
    lower.set_title("정상 평균 - 스푸프 평균 · T의 클래스 내 분산으로 표준화",loc="left",fontweight="bold",pad=12)
    bar=fig.colorbar(image,ax=lower,pad=.015,fraction=.025)
    bar.set_label("평균 차이 / T 표준편차",fontsize=9)
    fig.suptitle("채널을 다르게 다루는 두 방법: 분산 기반 거리계수와 판별력 순위",fontsize=14,fontweight="bold",y=.99)
    fig.text(.025,.025,"모든 계수·채널 순서는 T에서 결정 · V/DF는 관찰용 비교 · 위/아래 지표의 의미는 다름\n거리계수는 attention 학습 결과나 인과적 중요도가 아니며, 채널 번호는 이 checkpoint의 좌표",fontsize=9,color=GRAY)
    fig.subplots_adjust(left=.075,right=.97,top=.88,bottom=.16,hspace=.67)
    export(fig,"04_channel_structure.png")
    pd.DataFrame({"channel":np.arange(160),"distance_weight_gamma001":weights,"T_fisher_score":fisher,
                  "T_standardized_mean_difference":effects[0],"V_standardized_mean_difference":effects[1],
                  "DF_standardized_mean_difference":effects[2]}).to_csv(OUT/"channel_statistics.csv",index=False)
    statistics["channel_weight_range"]=[float(weights.min()),float(weights.max())]
    statistics["top24_weight_channels"]=top.tolist()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"Malgun Gothic","font.size":10,"axes.unicode_minus":False})
    names=["T","V","DF_pilot"]
    data={};geometry={};statistics={}
    for name in names:
        validate_run(name)
        data[name]=load_embeddings(name)
        geometry[name]=data[name].geometry()
        assert geometry[name]["valid"].all()
    assert max(np.max(np.abs(data[n].centroid-data['T'].centroid)) for n in names)<1e-12
    base,_,_=resolve_run("DF_pilot")
    registry=read_json(base/"registry.json")
    threshold=registry["thresholds"]["native_cosine"]["threshold"]
    for name in names:
        y=data[name].rows.y.to_numpy();s=geometry[name]["cosine"]
        statistics[name]={"n":len(y),"bona":int(y.sum()),"spoof":int((1-y).sum()),
                          "EER_percent":eer(y,s),"V_threshold_bona_false_reject":float(np.mean(s[y==1]<threshold)),
                          "V_threshold_spoof_false_accept":float(np.mean(s[y==0]>=threshold)),
                          "valid_direction_rate":float(geometry[name]["valid"].mean())}
    fit_indices=choose_rows(data['T'].rows,2500,17)
    model=PCA(n_components=2,svd_solver="full").fit(geometry['T']['direction'][fit_indices])
    pc={n:model.transform(geometry[n]['direction']) for n in names}
    np.savez(OUT/"T_direction_pca.npz",components=model.components_,mean=model.mean_,
             explained_variance_ratio=model.explained_variance_ratio_,fit_indices=fit_indices)
    plot_objective()
    plot_scores(data,geometry,threshold,statistics)
    plot_geometry(data,geometry,pc,threshold,model,statistics)
    plot_channels(data,geometry,statistics)
    statistics.update({"PCA_fit_split":"T only","PCA_fit_n":len(fit_indices),
                       "PCA_explained_variance_ratio":model.explained_variance_ratio_.tolist(),
                       "V_threshold":threshold,"score_histogram_bin_edges":np.linspace(-1,1,81).tolist(),
                       "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "source_cache_fingerprints":{n:data[n].provenance['cache_fingerprint'] for n in names},
                       "limits":["PCA is an exploratory 2D projection, not proof of separability/density in 160D",
                                 "DF is a fixed 20000-row previously observed pilot; not the full benchmark",
                                 "Explanation diagram uses virtual 3D vectors, not actual encoder outputs",
                                 "No new model training, inference, SSH or Q data used"]})
    (OUT/"visualization_stats.json").write_text(json.dumps(statistics,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in statistics.items() if k!='score_histogram_bin_edges'},ensure_ascii=False,indent=2))
    print('OUTPUT',OUT)


if __name__=="__main__":
    with threadpool_limits(limits=2):main()
