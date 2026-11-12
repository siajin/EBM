# 겹치지 않는 분할로 길이를 맞추고 꼬리 손실을 0.5초 미만으로 제한

**실행 상태:** 분할 계획·수학적 상한 검증 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

겹치지 않는 연속 분할은 꼬리 손실을 제한하면서 클래스 길이 분포를 맞출 수 있다.

## 그렇게 설계한 이유

속도·패딩·원음 제외 없이 연속 분할만으로 길이 분포를 맞출 때의 절충을 계산한다.

대조와 통제의 근거: 원음마다 대안 분할들을 동시에 만들어 중복 노출하지 않는다. 분모는 생성 조각 수이므로 원음 균등 확률 실험과 노출 기준이 다르다.

## 모델·데이터·표본

[학습·개발·평가 자료의 원음 길이와 공격별 클래스 구성](../E028/README.md)의 803,565개 메타데이터. 원래 0.5초 미만인 19LA 위조 1개·DF 위조 9개 도 한 번씩 보존한다.

## 실험 내용과 상세 절차

1. 원음의 정수 샘플 수를 n이라 한다. 800샘플(0.05초) 간격의 목표 샘플 수 L≥8,000과 정수 조각 수 k에 대해 `0≤n−kL<8,000`인 후보를 만든다. 한 원음의 조각은 모두 L샘플이며 남는 꼬리는 0.5초 미만이다.
2. 비슷한 길이 원음을 묶되 모든 원음에서 가능한 후보만 남기고, 원음마다 한 후보를 선택한다.
3. 클래스별 조각 수의 정규화를 고정 비율별 선형 계획으로 풀고, 정수 선택과 제한 시간 혼합 정수 최적화로 보완한다.
4. 조각 수를 줄이는 구성과 일치율 우선 구성을 보존한다. 평균 조각 2초 이상 후보 중 겹침률이 큰 것을 실용 표에 표시한다.
5. 모든 원음을 0.5초로 분할하는 별도 구성이 달성하는 정확 전역 최대도 계산한다. 좌표[소스: optimize_duration_partitions.py](code/EBM/tools/optimize_duration_partitions.py)
- [소스: validate_duration_partitions.py](code/EBM/tools/validate_duration_partitions.py)
- [소스: compact_duration_partitions.py](code/EBM/tools/compact_duration_partitions.py)

코드의 파일별 원본 위치·SHA256은 [원본 파일 위치와 해시 목록](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [상세 설계·실험 기록](../../_shared/references/artifacts/duration_partitions_20261109/README.md)
- [검산 결과](../../_shared/references/artifacts/duration_partitions_20261109/validation.json)
- [설정·결과 근거](../../_shared/references/artifacts/duration_partitions_20261109/independent_validation.json)

## 연결

다른 관련 가설: 주 가설에서만 분류한다.

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
