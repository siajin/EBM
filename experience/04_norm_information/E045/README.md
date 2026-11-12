# E045. LayerNorm 전후 특징·평균·표준편차를 읽는 내부 분류기

**실행 상태:** 추가 분류기·내부 교차검증 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

LayerNorm 이전의 평균·스케일·방향 조합이 같은 읽기 분류기에 더 잘 보일 수 있다.

## 그렇게 설계한 이유

정규화 전 특징과 정규화 후 특징 중 어느 쪽이 같은 선형 분류기에 더 쉽게 읽히는지 본다.

대조와 통제의 근거: 정규화 전후는 같은 발화의 동일 forward 에서 추출한 특징이다. 원래 단일 중심 코사인 대신 새 분류기 점수를 비교한다.

## 모델·데이터·표본

[E043](../../05_residual_geometry/E043/README.md)과 같은 고정 DF 진단 20,000개·5×3 화자 분리 fold.

## 실험 내용과 상세 절차

1. 기본 점수/크기 입력에 마지막 선형 출력 h160개, LayerNorm 출력 z160개, 또는 h의 좌표 평균과 log 표준편차 2개만 추가한다.
2. 각 조건에 같은 L2 logistic·C =0.01/0.1/1/10·fold 내 표준화와 선택을 적용한다.
3. 각 발화 held-out 예측을 저장하고 차이를 동일 화자 군집 재표집으로 계산한다. h에 z의 정상 기준 벡터를 곧바로 적용하는 점수는 사용하지 않는다.

## 측정과 검증

균형로그 손실·EER·2,000회 화자 군집 구간. 로그 손실 차이는 개별 95%다.

## 결과

scalar +z EER 1.7000%·LL0.064016,scalar +h1.4944%·0.040671,scalar +h 통계 3.2444%·0.091780. h의 LL 이득 0.023345의 95%구간[0.010456,0.042250].

## 해석·실패한 부분·한계

동일한 선형 분류기에 더 잘 읽힌다는 결과다. LayerNorm이 탐지 정보를 비가역적으로 없앤다거나 현재 오류의 인과 원인이라는 증명은 아니다.

## 배운 점

h의 내부 판독 이득은 정규화가 정보를 비가역적으로 버렸다는 증명이 아니다. 같은 분류기와fold로 비교해야 한다.

## 실험 코드

- [artifacts/geometry_experiments/cached_core_diagnostics.py](code/artifacts/geometry_experiments/cached_core_diagnostics.py)

코드의 파일별 원본 위치·SHA256은 [source_manifest.json](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [docs/experiments/ebm_cached_core_results_20261105.md](../../_shared/references/docs/experiments/ebm_cached_core_results_20261105.md)
- [artifacts/geometry_experiments/cached_core_diagnostics.py](../../_shared/workspace/artifacts/geometry_experiments/cached_core_diagnostics.py)
- [artifacts/geometry_cache/local_analysis/cached_core_v1/probe_metrics.csv](../../_shared/references/artifacts/geometry_cache/local_analysis/cached_core_v1/probe_metrics.csv)

## 연결

다른 관련 가설: [길이·유효 발화량과 앞뒤 저에너지 편향](../../01_duration_bias/README.md) · [잔여 방향·좌표·거리·기하와 변형 반응](../../05_residual_geometry/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
