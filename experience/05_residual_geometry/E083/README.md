<a id="e083-원래-160차원-관계를-보존하거나-왜곡을-밝힌-2차원구면-시각화"></a>

# 원래 160차원 관계를 보존하거나 왜곡을 밝힌 2차원·구면 시각화

**실행 상태:** 보조 기하 분석·그림 검증

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

중심축을 유지한투영과고차원검산을 함께 두면 기하구조를 덜오해할 수 있다.

## 그렇게 설계한 이유

방향·정상 기준 각·분산 주축의 관계를 시각화하되 그림 자체를고 차원 검정으로 오해하지 않게한다.

대조와 통제의 근거: 같은 화면 방향이 두 모델에서 같은 음향 방향이라는 정렬을 하지 않았다. c 축 포함 전체 에너지와 옆 성분 내 설명 에너지를 구분한다.

## 모델·데이터·표본

저장된 배치 160 특징 및 EBM-Repeat/EBM-NoRepeat 50epoch 모델의 잠근 768개 진단 특징. 기존 그림들의 보조 분석이며 새 학습 없음.

## 실험 내용과 상세 절차

1. 정상 기준 c를 한 축으로 고정하고 c 직 교 공간의 PCA/SVD 두 축을 추가해(x의 옆 투영 2개,s)좌표를 그린다. 재정규화로 그림의 구표면에 억지로 올리지 않는다.
2. 원자료 전체 PCA/t-SNE·선택 두 좌표 그림은 서로 다른 투영임을 명시하고 주 축 fit 자료와 표본 선택을 남긴다.
3. EBM-NoRepeat/EBM-Repeat 모델은 각자 별도 기저를 추정한다. 원래 160 좌표에서 score·직교성·설명 에너지·이웃/경로 판정을 검산한다.
4. 선택 좌표 예시는 113 번·95 번 두 좌표를 사용했다. 전체 160차원 그림은 DF 진단 자료에서 정상·위조 각 2,000개, seed 20261105로 표시 표본을 고르고 그 4,000개 에 PCA 50축을 맞춘 뒤 t-SNE를 적용했다. t-SNE는 perplexity 40, seed 20261105, 1,000회, PCA 초기화, 자동 학습률이다. 투영 계산에 라벨을 넣지는 않았지만 표시 표본의 클래스 균형에는 라벨을 사용했다.

## 측정과 검증

실제 각도·투영 에너지·축 fit 범위·그림 범례/잘림·정확한 score 재계산.

## 결과

768개 진단에서 두 옆 축은 EBM-NoRepeat 잔여 에너지 66.829%, 원래 24073.776%였고 c 축 포함 전체 에너지는 97.496/97.627%. 후자를 잔여 방향 거의 전부 보존으로 해석하면 안 된다.

## 해석·실패한 부분·한계

화면상의 클러스터·한 방향 모양·거리만으로 물리적 원인이나 모델 본질 차원을 정하지 않는다.

## 배운 점

2차원에서보존한 전체에너지와잔여에너지는 다르다. 서로다른모델의 투영축도 자동으로 같은의미를 갖지 않는다.

## 실험 코드

- [소스: centroid_centered_maps.py](code/artifacts/geometry_experiments/centroid_centered_maps.py)
- [소스: plot_author_embedding_distribution.py](code/artifacts/geometry_experiments/plot_author_embedding_distribution.py)

코드의 파일별 원본 위치·SHA256은 [원본 파일 위치와 해시 목록](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [상세 설계·실험 기록](../../_shared/references/docs/experiments/centroid_centered_maps_20261105.md)
- [상세 설계·실험 기록](../../_shared/references/docs/experiments/ebm_B_final_results_20261108.md)
- [소스: centroid_centered_maps.py](../../_shared/workspace/artifacts/geometry_experiments/centroid_centered_maps.py)
- [설정·결과 근거](../../_shared/references/artifacts/geometry_cache/local_analysis/B_final_analysis_20261108_v1/analysis/projection_basis.json)
- [설정·결과 근거](../../_shared/references/artifacts/geometry_cache/local_analysis/author_distribution_v1/provenance.json)

## 연결

다른 관련 가설: 주 가설에서만 분류한다.

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
