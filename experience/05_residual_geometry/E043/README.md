# E043. DF 안에서 점수·크기를 통제한 잔여 방향의 판별 정보

**실행 상태:** 추가 분류기·내부 교차검증 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

같은 점수·norm에서도 잔여 방향에 클래스 판별 신호가 남을 수 있다.

## 그렇게 설계한 이유

기존 점수와 표현 크기가 설명하지 못하는 방향 관련 신호가 고정 특징 안에 남는지 본다.

대조와 통제의 근거: 주매칭 106쌍·28 화자 그룹이며 같은 화자까지 요구하면 18쌍·7 그룹이다. 후속 엄격 매칭 [E048](../E048/README.md)과 조건이 다르다.

## 모델·데이터·표본

배치 160 고정 모델의 DF 진단 20,000개·출처 × 화자 93 그룹. Train은 이 내부 분류기를 fit 하는 자료가 아니다.

## 실험 내용과 상세 절차

1. 화자가 겹치지 않는 바깥 5fold·안쪽 3fold를 고정한다. 각 학습 fold 안에서 5-knot cubic spline·표준화·조건 one-hot을 fit 한다.
2. 기본 입력은 spline(s)+log ||z ||+방향 유효 표시다. 실제 u160개 추가, 같은 차원의 ID 고정 단위 tangent 잡음 추가, 출처/코덱/log 길이/패딩 추가, 조건 +u 추가를 비교한다.
3. L2 logistic의 C =0.01/0.1/1/10을 안쪽 균형로그 손실로 고른 뒤 바깥 학습 부분에 다시 적합 하고 각 held-out 발화의 예측을 한 번만 저장한다.
4. 같은 출처 × 코덱에서 정상/위조를 중복 없이 매칭한다. 주 허용 차이는 점수 0.01·lognorm0.002, 좁게 0.005/0.001·넓게 0.02/0.004도 분석한다. 이미 저장한 held-out 예측을 사용하고 다시 적합 하지 않는다.

## 측정과 검증

클래스 균형로그 손실·EER·AUROC, 출처 × 화자 2,000회 seed 7301paired 구간. 저장 예측에 조건부이며 매 bootstrap 마다 재학습하지 않는다.

## 결과

기본 EER 3.5056%→방향 추가 1.20%, 잡음 추가 4.20%. 조건 메타데이터 포함 2.0778%→조건 +방향 0.60%. 주 106쌍에서는 50.9434 →19.8113%.

## 해석·실패한 부분·한계

DF 정답으로 내부 분류기를 학습한 정보 접근성 진단이다. 학습 자료 →DF 전이 성능이나 조건부상호 정보량의 정확 추정이 아니다. 비교 일부 fold 에서 출처 × 클래스 공급 부족 제한이 있다.

## 배운 점

DF 내부의 신호는 확인했지만 정답을 쓴 내부CV다. 새 도메인으로의 배포 성능과 구분해야 한다.

## 실험 코드

- [artifacts/geometry_experiments/cached_core_diagnostics.py](code/artifacts/geometry_experiments/cached_core_diagnostics.py)

코드의 파일별 원본 위치·SHA256은 [source_manifest.json](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [docs/experiments/ebm_cached_core_results_20261105.md](../../_shared/references/docs/experiments/ebm_cached_core_results_20261105.md)
- [artifacts/geometry_experiments/cached_core_diagnostics.py](../../_shared/workspace/artifacts/geometry_experiments/cached_core_diagnostics.py)
- [artifacts/geometry_cache/local_analysis/cached_core_v1/protocol_lock.json](../../_shared/references/artifacts/geometry_cache/local_analysis/cached_core_v1/protocol_lock.json)
- [artifacts/geometry_cache/local_analysis/cached_core_v1/summary.json](../../_shared/references/artifacts/geometry_cache/local_analysis/cached_core_v1/summary.json)
- [artifacts/geometry_cache/local_analysis/cached_core_v1/matched_summary.json](../../_shared/references/artifacts/geometry_cache/local_analysis/cached_core_v1/matched_summary.json)

## 연결

다른 관련 가설: [정상 다양성·도메인 지지와 정상쌍 학습](../../03_bonafide_diversity/README.md) · [코사인 외 norm과 정규화 전 통계](../../04_norm_information/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
