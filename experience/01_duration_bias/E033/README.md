# E033. 학습 원음을 균등 선택하며 0.5~6초 길이 확률을 맞추는 초기 정책

**실행 상태:** 정책 계산·구간 검증 완료; 최종 학습 안 아님

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

원음 확률을 유지한 조건부 길이 추출로 클래스 길이 목표를 맞출 수 있다.

## 그렇게 설계한 이유

원음·화자·공격 비율을 유지하면서 클래스별 선택 길이를 같게 만들 수 있는지 본다.

대조와 통제의 근거: 앞/뒤 중복 구간은 유지한다. 클래스별 quota는 별도이며 이 정책이 정상: 위조 1:9를 바꾸지 않는다.

## 모델·데이터·표본

학습 자료 25,380개만 사용.0.5~6초를 0.5초 간격으로 12 길이.

## 실험 내용과 상세 절차

1. 목표 확률을 5,10,15,20,18,15,8,4,2,1.5,1,0.5%로 고정한다.
2. 클래스 안의 원음을 균등 선택하고 원음 길이 bucket 별 가능한 목표 길이 조건부 확률 표를 맞춘다.
3. 방향을 앞/뒤 50%로 선택한 뒤 해당 격자에서 구간을 균등 선택한다. 짧은 원음에 불가능한 길이를 요구하면 거부한다.
4. 모든 확률과 원음/화자/공격 기대 비율을 검산하고 실제 4,000개 구간 좌표를 확인한다.

## 측정과 검증

클래스 길이 기대 확률 차이·분포 겹침·화자/공격의 원래 비율 보존·구간유효성.

## 결과

원음 균 등 조건의 클래스 길이 기대 겹침은 약 100%,4,000개 구간 검사 통과. 이후 화자 × 공격까지 공동 길이 분포를 맞추려할 때 6초 목표가 불가능한 그룹이 발견됐다.

## 해석·실패한 부분·한계

이 초기 정책을 이후 0.5~4초 최종 학습과 동일시하지 않는다. 모형 성능 EER을 측정하지 않았다.

## 배운 점

클래스 전체의 가능성이 모든 화자·공격 그룹의 가능성을 보장하지 않는다. 공급 지지집합부터 확인해야 한다.

## 실험 코드

- [EBM/tools/design_duration_sampling_policy.py](code/EBM/tools/design_duration_sampling_policy.py)
- [EBM/tools/duration_sampling_policy.py](code/EBM/tools/duration_sampling_policy.py)

코드의 파일별 원본 위치·SHA256은 [source_manifest.json](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [artifacts/duration_sampling_design_20261110/policy.json](../../_shared/references/artifacts/duration_sampling_design_20261110/policy.json)
- [artifacts/duration_sampling_design_20261110/validation.json](../../_shared/references/artifacts/duration_sampling_design_20261110/validation.json)

## 연결

다른 관련 가설: 주 가설에서만 분류한다.

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
