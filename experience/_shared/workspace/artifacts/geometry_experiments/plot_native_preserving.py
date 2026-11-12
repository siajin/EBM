"""Render the completed cached-control validation without refitting anything."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from geometry_cache import DEFAULT_ROOT, read_json


root = DEFAULT_ROOT / "local_analysis/native_preserving_v1"
summary = read_json(root / "summary.json")
assert summary["status"] == "completed_exploratory_corrected_controls"
points = {r["case"]: r for r in summary["point_results"]}
intervals = summary["contrasts"]
pd.DataFrame(summary["point_results"]).to_csv(root / "metrics.csv", index=False)
pd.DataFrame(intervals).to_csv(root / "paired_intervals.csv", index=False)
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 10})
fig, (left, right) = plt.subplots(1, 2, figsize=(13.5, 7), gridspec_kw={"width_ratios": [1, 1.2]})
cases = ["native_cosine", "monotonic_control", "native_plus_direction", "E_diagonal_0.01"]
labels = ["기존 코사인", "단조 변환 대조군", "코사인 + 방향 보정", "대각 거리 γ=0.01\n(탐색 후보)"]
values = [points[n]["eer_percent"] for n in cases]
bars = left.barh(np.arange(4), values, height=.58, color=["#35495e"]+["#4679a8"]*3)
left.set_yticks(np.arange(4), labels)
left.invert_yaxis()
left.set_xlim(0, 4)
left.set_xlabel("EER (%) · 낮을수록 좋음")
left.set_title("기존 코사인을 유지한 비교", loc="left", fontweight="bold", pad=12)
for bar, value in zip(bars, values):
    left.text(value+.06, bar.get_y()+bar.get_height()/2, f"{value:.2f}%", va="center")
names = ["대각 γ=.01", "대각 γ=.1", "대각 γ=.5", "전체 γ=.01", "전체 γ=.1", "전체 γ=.5",
         "방향 보정", "잡음 보정(선택=0)", "방향 대 잡음 보정"]
for i, record in enumerate(intervals):
    point = record["delta_EER_pp"]
    low, high = record["CI_family95_Bonferroni"]
    right.plot([low,high],[i,i], color="#a4abb3", linewidth=5, solid_capstyle="butt", label="9개 비교 보정 구간" if i==0 else None)
    low, high = record["CI95"]
    right.plot([low,high],[i,i], color="#4679a8", linewidth=2, label="개별 95% 구간" if i==0 else None)
    right.plot(point,i,"o",color="#35495e",markersize=5)
right.axvline(0, color="#35495e", linestyle="--", linewidth=1)
right.set_yticks(np.arange(9), names)
right.invert_yaxis()
right.set_xlim(-.18,.55)
right.set_xlabel("대조군 EER - 후보 EER (%p) · 양수면 개선\n마지막 행 외에는 기존 코사인이 대조군")
right.set_title("개선 폭과 다중비교 보정", loc="left", fontweight="bold", pad=12)
right.legend(loc="upper right", bbox_to_anchor=(1.03,-.15), frameon=False, fontsize=9)
for axis in [left,right]:
    axis.spines[["top","right"]].set_visible(False)
    axis.grid(axis="x",alpha=.15)
    axis.set_axisbelow(True)
fig.suptitle("DF 고정 표본 20,000개 · 기존 표현의 후처리 재검증", fontsize=15,fontweight="bold",y=.98)
fig.text(.02,.035,"T에서 학습 / V에서 설정 선택 · 같은 DF 표본을 이미 관찰한 뒤 수행한 탐색 버전\nsource·화자 93그룹 paired bootstrap 10,000회 · 원발화 파생관계 미확인\n보정은 이번 9개 비교에만 적용: 이전 탐색 전체나 새 모델 학습 효과를 검증한 결과가 아님",fontsize=9,color="#41464d")
fig.subplots_adjust(left=.16,right=.98,top=.85,bottom=.27,wspace=.80)
fig.savefig(root / "validation.png",dpi=180,facecolor="white")
plt.close(fig)
print(root / "validation.png")
