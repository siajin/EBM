<a id="e046-정상-기준-밖의-방향-군집과-출처코덱길이의-연관"></a>

# 정상 기준 밖의 방향 군집과 출처·코덱·길이의 연관

**실행 상태:** 캐시 군집 분석 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

잔여 방향 군집은 라벨뿐 아니라 출처·길이 같은 조건을 반영할 수 있다.

## 그렇게 설계한 이유

표현 방향 구조가 정상/위조 라벨만 반영하는지 다른 데이터 조건도 반영하는지 본다.

대조와 통제의 근거: 점수 cosine이 같은 표현도 옆 방향 군집이 다를 수 있다. 군집 번호는 공격 ID 나 음향 환경 정답이 아니다.

## 모델·데이터·표본

배치 160 고정 특징. 학습 자료 클래스 균형 5,000개로 군집 fit, 개발·DF 진단에 고정 적용.

## 실험 내용과 상세 절차

1. 단위 옆 방향 u에 구면 k-means K4를 seed 501~505로 적합하고 Train의 cosine 목적이 가장 좋은 초기화를 선택한다.
2. 개발·DF 에는 기준을 다시 fit 하지 않고 가장 가까운 방향 군집을 할당한다.
3. 전체와 정상/위조 내부 각각에서 군집과 라벨·출처·코덱·길이구간·패딩 구간의 정규화 상호 정보량(NMI)을 계산한다. 길이/패딩 경계는 학습 자료에서 정한다.

## 측정과 검증

NMI·그룹별 표본 수·방향 평균·자료 조건 분포.

## 결과

DF 전체 라벨 NMI0.264455·출처 0.118216·코덱 0.004659. 정상 내부 출처 NMI0.343788로 데이터 출처와의 연관이 남았다.

## 해석·실패한 부분·한계

범주수·군집 수에 영향 받는 연관 지표다. 인과 중요도 순위나 모든 방향이 불필요한 단서라는 의미가 아니다.

## 배운 점

방향에는 정상 출처 구조가 남았다. NMI를 인과적 중요도나 물리적 음향 축으로 해석할 수 없다.

## 실험 코드

- [소스: cached_core_diagnostics.py](code/artifacts/geometry_experiments/cached_core_diagnostics.py)

코드의 파일별 원본 위치·SHA256은 [원본 파일 위치와 해시 목록](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [상세 설계·실험 기록](../../_shared/references/docs/experiments/ebm_cached_core_results_20261105.md)
- [소스: cached_core_diagnostics.py](../../_shared/workspace/artifacts/geometry_experiments/cached_core_diagnostics.py)
- [집계 결과 표](../../_shared/references/artifacts/geometry_cache/local_analysis/cached_core_v1/cluster_associations.csv)

## 연결

다른 관련 가설: [정상 다양성·도메인 지지와 정상쌍 학습](../../03_bonafide_diversity/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
