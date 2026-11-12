# E031. 겹치지 않는 분할로 길이를 맞추고 꼬리 손실을 0.5초 미만으로 제한

**실행 상태:** 분할 계획·수학적 상한 검증 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

겹치지 않는 연속 분할은 꼬리 손실을 제한하면서 클래스 길이 분포를 맞출 수 있다.

## 그렇게 설계한 이유

속도·패딩·원음 제외 없이 연속 분할만으로 길이 분포를 맞출 때의 절충을 계산한다.

대조와 통제의 근거: 원음마다 대안 분할들을 동시에 만들어 중복 노출하지 않는다. 분모는 생성 조각 수이므로 원음 균등 확률 실험과 노출 기준이 다르다.

## 모델·데이터·표본

[E028](../E028/README.md)의 803,565개 메타데이터. 원래 0.5초 미만인 19LA 위조 1개·DF 위조 9개 도 한 번씩 보존한다.

## 실험 내용과 상세 절차

1. 원음의 정수 샘플 수를 n이라 한다. 800샘플(0.05초) 간격의 목표 샘플 수 L≥8,000과 정수 조각 수 k에 대해 `0≤n−kL<8,000`인 후보를 만든다. 한 원음의 조각은 모두 L샘플이며 남는 꼬리는 0.5초 미만이다.
2. 비슷한 길이 원음을 묶되 모든 원음에서 가능한 후보만 남기고, 원음마다 한 후보를 선택한다.
3. 클래스별 조각 수의 정규화를 고정 비율별 선형 계획으로 풀고, 정수 선택과 제한 시간 혼합 정수 최적화로 보완한다.
4. 조각 수를 줄이는 구성과 일치율 우선 구성을 보존한다. 평균 조각 2초 이상 후보 중 겹침률이 큰 것을 실용 표에 표시한다.
5. 모든 원음을 0.5초로 분할하는 별도 구성이 달성하는 정확 전역 최대도 계산한다. 좌표[jL,(j +1)L), 전체 ID·길이·꼬리·중복을 검산한다.

## 측정과 검증

분포 겹침·조각 수배율·평균 길이·버린 총 시간·최대 꼬리. 오디오 생성·학습은 하지 않았다.

## 결과

실용 학습 자료 겹침 99.6730%, 평균 2.956초·1.117배 조각·총 시간 3.601% 손실. 전부 0.5초 면 학습 자료 100%지만 6.352배 조각이다.

## 해석·실패한 부분·한계

일치율만 극대화하면 거의 0.5초로 몰릴 수 있다. 실용 최적화는 제한 시간 최선 해일 수 있고 전역 최적성을 주장하지 않는다. 별도 0.5초 상한만 정확히 증명했다.

## 배운 점

일치율만 최적화하면0.5초 조각으로 몰린다. 내용 보존·조각 수·문맥 길이의 절충이 필요하다.

## 실험 코드

- [EBM/tools/optimize_duration_partitions.py](code/EBM/tools/optimize_duration_partitions.py)
- [EBM/tools/validate_duration_partitions.py](code/EBM/tools/validate_duration_partitions.py)
- [EBM/tools/compact_duration_partitions.py](code/EBM/tools/compact_duration_partitions.py)

코드의 파일별 원본 위치·SHA256은 [source_manifest.json](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [artifacts/duration_partitions_20261109/README.md](../../_shared/references/artifacts/duration_partitions_20261109/README.md)
- [artifacts/duration_partitions_20261109/validation.json](../../_shared/references/artifacts/duration_partitions_20261109/validation.json)
- [artifacts/duration_partitions_20261109/independent_validation.json](../../_shared/references/artifacts/duration_partitions_20261109/independent_validation.json)

## 연결

다른 관련 가설: 주 가설에서만 분류한다.

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
