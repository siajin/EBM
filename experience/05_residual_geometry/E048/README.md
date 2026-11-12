# E048. 같은 화자·코덱·점수·크기의 정상/위조 방향 분포 매칭

**실행 상태:** 계산 완료·표본 부족으로 주가 설 판정 불가

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

화자·코덱·점수·norm까지 맞춘 정상/위조의 방향 분포를 비교할 수 있다.

## 그렇게 설계한 이유

같은 점수와 크기에서 정상/위조의 옆 방향 분포가 다른지 더 엄격한 조건으로 조사한다.

대조와 통제의 근거: 사전 최소 기준은 200쌍·10 화자 그룹. 방향 유효 조건·실제 점수/norm 잔차·선택 비율을 공개한다.

## 모델·데이터·표본

고정 배치 160의 DF/21LA 진단 각 20,000. 대역폭 추정은 학습 자료 정상 500 +위조 500만 사용.

## 실험 내용과 상세 절차

1. DF는 출처 × 화자 × 코덱,21LA는 화자 × 코덱 × 전송 조건을 같게 한다.
2. 주 허용 차이는 |점수차 |≤0.02·|lognorm 차 |≤0.002, 넓은 민감도는 0.05/0.005다. 중복 없는 정상/위조 매칭을 고정 seed 17 순서로 만들고 최대 2,000쌍을 남긴다.
3. 클래스 방향 평균·평균 방향 각·평균 벡터 길이·중심화 공분산/비중심 2차 모멘트의 유효 차원(PR)을 계산한다.
4. 학습 자료 균형 1,000개에서 정한 RBF 대역폭으로 biased MMD² 를 구하고, 짝의 화자 그룹을 10,000회 재표집한다.

## 측정과 검증

MMD²·방향 통계와 매칭 coverage. biased MMD의 양의 구간을 동일 분포 귀무가설 기각으로 오독하지 않는다.

## 결과

주 조건 DF 27쌍/9 그룹,21LA 10쌍/6 그룹. 넓게 해도 58/15,21/14로 쌍 수 기준 미달이었다.

## 해석·실패한 부분·한계

방향 차이가 없다는 결론이 아니라 현재 지지집합에서 판정 불가다. 초기 출처 × 코덱 106쌍과 허용 차이·화자 조건이 다르다.

## 배운 점

엄격 매칭에서 표본이 사전 기준에 미달했다. 차이가 없다는 결론 대신 판정 불가를 기록해야 한다.

## 실험 코드

- [artifacts/geometry_experiments/frozen_n0.py](code/artifacts/geometry_experiments/frozen_n0.py)

코드의 파일별 원본 위치·SHA256은 [source_manifest.json](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [artifacts/geometry_experiments/frozen_n0.py](../../_shared/workspace/artifacts/geometry_experiments/frozen_n0.py)
- [artifacts/geometry_cache/local_analysis/frozen_all_v1/F2/summary.json](../../_shared/references/artifacts/geometry_cache/local_analysis/frozen_all_v1/F2/summary.json)
- [artifacts/geometry_cache/local_analysis/frozen_all_v1/F2/matching_balance.csv](../../_shared/references/artifacts/geometry_cache/local_analysis/frozen_all_v1/F2/matching_balance.csv)
- [docs/experiments/ebm_frozen_model_protocol_20261105.json](../../_shared/references/docs/experiments/ebm_frozen_model_protocol_20261105.json)

## 연결

다른 관련 가설: [정상 다양성·도메인 지지와 정상쌍 학습](../../03_bonafide_diversity/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
