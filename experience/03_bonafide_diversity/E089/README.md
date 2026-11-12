# E089. 정상 특징으로 여러 평균 기준 벡터를 만든 초기 실험

**실행 상태:** 기존 실제 결과 확인·문서화 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

여러정상중심이 단일중심보다 정상다양성을 잘대표할 수 있다.

## 그렇게 설계한 이유

정상 분포를 한 벡터보다 여러 벡터로 대표하면 분리가 개선되는지 본다.

대조와 통제의 근거: 특징 추출기·분류기 가중치는 고정이다. 분할은 발화 단위이며 화자·출처 독립성을 보장하지 않는다.

## 모델·데이터·표본

배치 80·80epoch의 고정 모델(체크포인트 SHA256 30043199664bc84da780654e01fd430235f647b894e7832a52f777c7fee5939a). 별도 명시가 없으면 DF eval에서 seed 1로 정상·위조 각 1,000개를 클래스 안에서 중복 없이 무작위 추출한 같은 2,000개다. 추론 배치 4, worker 2, 원래 앞 64,600샘플·부족하면 순환 반복, 증강 없음이다.

## 실험 내용과 상세 절차

1. 고정 특징에서 분할마다 정상 600개로 k-means와 구면 재정렬을 수행한다. 기준 개수는 1/2/3/4/8개다.
2. 정상 200개·위조 200개로 각 방법의 EER 근사 임계값을 정하고, 별도 정상 200개·위조 200개를 평가한다.
3. 10회 분할에서 원래 학습 기준 벡터와 새 기준 중 최대 코사인 점수를 비교한다.
4. k-means는 n_init=10이며, 얻은 중심을 단위화한 뒤 최대 100회의 구면 재정렬을 수행했다.

## 측정과 검증

10회 평균·최솟값·최댓값·원래 기준 대비 대응 차이를 기록한다. 겹치는 분할 범위를 신뢰구간으로 부르지 않는다.

## 결과

분할 평균 EER은 원래 기준 1.85%, 새 평균 하나 1.80%, 기준 2/3/4/8개에서 3.80/3.95/7.25/10.75%였다. 기준을 늘린 효과가 일관된 개선은 아니었다.

## 해석·실패한 부분·한계

하나의 고정 학습 모델과 탐색 표본의 결과다. 전체 공식 평가 EER, 여러 학습 seed의 효과, 손실을 제거한 재학습의 효과로 해석하지 않는다. 반복 관찰한 표본과 서로 겹치는 후속 분석은 독립 실험 표본으로 합산하지 않는다.

## 배운 점

중심수를늘리면 이탐색에서는 악화했다. 겹치는10분할 범위를CI나 독립seed 재현으로 부르지 않는다.

## 실험 코드

- [historical_analysis/multicenter_21df.py](code/multicenter_21df.py)

코드의 파일별 원본 위치·SHA256은 [source_manifest.json](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [artifacts/experiment_methods_audit_20261111/historical_receipts/EBM80_21DF_multicenter.json](../../_shared/references/artifacts/experiment_methods_audit_20261111/historical_receipts/EBM80_21DF_multicenter.json)
- [artifacts/experiment_methods_audit_20261111/historical_method_source_checks.json](../../_shared/references/artifacts/experiment_methods_audit_20261111/historical_method_source_checks.json)

## 연결

다른 관련 가설: [잔여 방향·좌표·거리·기하와 변형 반응](../../05_residual_geometry/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
