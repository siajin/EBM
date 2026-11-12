<a id="e035-목표-길이-안에서-정상위조의-음성활동-비율까지-맞추는-정책"></a>

# 목표 길이 안에서 정상·위조의 음성활동 비율까지 맞추는 정책

**실행 상태:** 정책 생성·실제 파형 검증 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

길이를 맞춘 뒤에도 남는 음성활동 차이는 원음 내부 후보 확률 조정으로 완화할 수 있다.

## 그렇게 설계한 이유

짧은 조각 안의 음성/비음성 비율 차이가 남는 것을 완화한다.

대조와 통제의 근거: VAD는 자동 추정이며 학습 파형을 자르거나 무 음을 제거하는 필터가 아니다. RawBoost 후의 활동 량까지 같은 것은 아니다.

## 모델·데이터·표본

학습 자료 25,380개, 앞/뒤 격자 후보 중 동일 구간 중복 제거 후 734,966개. 개발·평가 자료는 사용하지 않는다.

## 실험 내용과 상세 절차

1. 증강 전 16kHz 파형을 20ms 프레임으로 나눠 WebRTC VAD mode2로 음성 프레임 수를 계산한다.
2. 음성활동 0 인 경우를 별도 구간으로, 양수는 20% 간격 다섯 구간으로 나누고 길이별 평균도 계산한다.
3. 원음 선택 확률과 140 그룹 길이 목표는 유지하면서 원음 내부의 길이·구간 선택 확률을 조정해 두 클래스의 6구간 비중과 평균을 맞춘다.
4. 처음 화자별 제약을 시도한 버전은 일부 공급 지지집합 부족으로 전체 클래스 범위로 정정했다. 선형 계획 제한 시간 종료를 제약 불가능 증명으로 처리하지 않았다.
5. epoch1 원음 ID·증강 seed 전후 일치와 실제 1,010개 구간 VAD를 검사하고 2worker 입력을 확인한다.

## 측정과 검증

기대 길이별 VAD 평균·6구간 TV, 실제 선택 분포 잔차, 원음별 유효 후보 수, 시간 구간 사용률.

## 결과

평균 활동량 차이 6.30~8.04%p가 기대 값상 0.000003%p 미만으로 감소. 유효후보수 중앙값 13.87 →12.79. 화자별·연속적 모든 활동 분포 동일성은 보장하지 않는다.

## 해석·실패한 부분·한계

이 항목 자체는 모형 학습/EER 평가가 아니다. 후보 다양성과 시간 사용률도 변하므로 활동 비율만의 독립 학습 효과로 볼 수 없다.

## 배운 점

전체 클래스 기대 활동량은 맞췄으나 화자별·연속 분포까지 같지는 않았다. 후보 다양성과 사용 시간의 비용도 남는다.

## 실험 코드

- [소스: activity_policy.py](code/EBM/activity_policy.py)
- [소스: fit_activity_policy.py](code/EBM/tools/fit_activity_policy.py)
- [소스: validate_activity_policy.py](code/EBM/tools/validate_activity_policy.py)
- [소스: build_activity_candidates.py](code/EBM/tools/build_activity_candidates.py)

코드의 파일별 원본 위치·SHA256은 [원본 파일 위치와 해시 목록](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [설정·결과 근거](../../_shared/references/artifacts/joint_activity_balance_20261110/state.json)
- [설정·결과 근거](../../_shared/references/artifacts/joint_activity_balance_20261110/final_audit.json)
- [상세 설계·실험 기록](../../../POSITIVE_BPL.md)
- [소스: activity_policy.py](../../../activity_policy.py)

## 연결

다른 관련 가설: [반복 입력과 학습·추론의 상호작용](../../02_repetition/README.md) · [정상 다양성·도메인 지지와 정상쌍 학습](../../03_bonafide_diversity/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
