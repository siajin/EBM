# E049. 학습 정상 공분산으로 거리 방향을 바꾸되 정상 기준 벡터는 고정

**실행 상태:** 캐시거리 평가 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

Train 정상 공분산을 반영한 비등방 거리가 단일 코사인보다 전이에 유리할 수 있다.

## 그렇게 설계한 이유

단순 코사인과 비등방성 거리의 차이를 원래 중심을 유지한 채 비교한다.

대조와 통제의 근거: unit 공간의 단순 유클리드 거리 ² =2 −2cosine 이므로 그것만 바꾸면 순위/EER가 같다. raw 특징 거리·중심 교체와 구별한다.

## 모델·데이터·표본

배치 160 고정 특징. 학습 자료 정상 2,580개 unit 벡터의 공분산을 적합하고 개발 선택·DF/21LA 각 20,000 평가.

## 실험 내용과 상세 절차

1. 공분산 Σ와평균고유값 a =trace(Σ)/160을 추정하고 Σγ=(1 −γ)Σ+γaI를 만든다.
2. γ=0.01/0.1/0.5/1, 대각 공분산과 전체 공분산을 각각 사용한다. 작은 고유값은 1e-6a로 제한하고 역행렬 A의 trace를 160으로 정규화한다.
3. 원래 정상 중심 c를 고정하고 점수 1 −0.5(x −c)ᵀA(x −c)를 계산한다. A =I가 원래 cosine과 같은지 검산한다.
4. 개발 EER 최소를 선택하며 동률이면 더 큰 γ를 고른다. 기준 identity와 8 후보를 모두 보고한다.

## 측정과 검증

각 후보 EER·클래스/출처오류·10,000회 paired 군집 구간·8 후보 보정 구간.

## 결과

개발은 모든 후보 EER 0으로 identity가 선택됐다. 관찰 후보 대각 γ0.01의 DF 3.05%는 기준 3.35%보다 낮지만 개발 선택된 주 모델이 아니다.

## 해석·실패한 부분·한계

수축 거리 전체가 무용하다는 결론도, 가장 좋은 DF 후보가 전이 개선을 확정했다는 결론도 아니다. [E042](../E042/README.md)의 유사 후보와 재사용 관계를 명시한다.

## 배운 점

개발에서는identity가 선택됐다. DF에서만 좋은 후보를 골라 일반화 성공으로 보고하면 안 된다.

## 실험 코드

- [artifacts/geometry_experiments/frozen_n0.py](code/artifacts/geometry_experiments/frozen_n0.py)

코드의 파일별 원본 위치·SHA256은 [source_manifest.json](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [artifacts/geometry_experiments/frozen_n0.py](../../_shared/workspace/artifacts/geometry_experiments/frozen_n0.py)
- [artifacts/geometry_cache/local_analysis/frozen_all_v1/F3/metric_registry.json](../../_shared/references/artifacts/geometry_cache/local_analysis/frozen_all_v1/F3/metric_registry.json)
- [artifacts/geometry_cache/local_analysis/frozen_all_v1/F3/paired_effects.json](../../_shared/references/artifacts/geometry_cache/local_analysis/frozen_all_v1/F3/paired_effects.json)

## 연결

다른 관련 가설: [코사인 외 norm과 정규화 전 통계](../../04_norm_information/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
