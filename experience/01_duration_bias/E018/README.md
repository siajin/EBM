<a id="e018-고정-모델에서-길이-정규화와-속도-변형-비교"></a>

# 고정 모델에서 길이 정규화와 속도 변형 비교

**실행 상태:** 추론 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

고정 모델의 길이 정규화는 내용 증가보다 속도 처리 흔적에 더 민감할 수 있다.

## 그렇게 설계한 이유

길이를 일정하게 만드는 처리와 발화 속도 변화, 처리 알고리즘의 흔적을 구별한다.

대조와 통제의 근거: 전체를 목표 길이 T로 만드는 배율은 원음 길이/T 다. 짧은 음성의 극단 감속과 긴 음성의 극단 가속이 같은 변형 세기가 아니다.

## 모델·데이터·표본

EBM-Repeat/EBM-NoRepeat와 EBM-FullContext의 50epoch. 개발·19LA·21LA·DF 각각 정상 64 +위조 64 =128개, 총 512개. 각 클래스는 원음 길이 4구간에서 16개 씩 선택.

## 실험 내용과 상세 절차

1. 발화 ID 해시·화자별 번갈아 선택으로 표본을 고정하고 각 클래스 최단·최장 예시도 점수 확인 전에 포함한다.
2. 기준은 앞 64,600 상한·비반복이다. 원음 전체 입력, 같은 기준 구간 0.8배 감속,1.25배 가속, 원음 전체 →4.0375초,0.8배 감속 후 기준 길이로 복원하는 왕복 처리를 비교한다.
3. librosa0.11.0 phase-vocoder의 FFT2,048·hop512를 사용한다. 배율 1은 정확한 복사이며 추가 길이 보정 절단·채움을 하지 않는다.
4. 모델은 고정하고 pooling 잡음 0에서 6조건 ×3 모델 =9,216개 점수를 계산한다.

## 측정과 검증

층화 표본 EER·점수 변화·원래 전체 개발 임계값 오류, 변형 길이·rate 검산. 전체 benchmark EER로 사용하지 않는다.

## 결과

EBM-NoRepeat의 전체 →4.0375초 정상 거부는 각 집합 64개 중 63/64,64/64,62/64,64/64였다. 왕복 처리도 길이를 복원했지만 손상이 남았다.

## 해석·실패한 부분·한계

phase-vocoder·고정 모델의 실패를 모든 속도 증강 학습의 실패로 일반화하지 않는다. 전체 내용·서로 다른 전역 속도·정확한 동일 길이를 동시에 보장할 수 없다.

## 배운 점

phase-vocoder의 심한 실패는 모든 속도 증강 학습의 실패가 아니다. 왕복 처리도 무손실 대조가 아니다.

## 실험 코드

- [소스: diagnose_speed_views.py](code/EBM/tools/diagnose_speed_views.py)

코드의 파일별 원본 위치·SHA256은 [원본 파일 위치와 해시 목록](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [상세 설계·실험 기록](../../_shared/references/docs/experiments/ebm_B_C_all_checkpoint_evaluation_20261109.md)
- [설정·결과 근거](../../_shared/references/artifacts/speed_duration_feasibility_20261109/probe/protocol.json)
- [설정·결과 근거](../../_shared/references/artifacts/speed_duration_feasibility_20261109/probe/results.json)
- [소스: diagnose_speed_views.py](../../../tools/diagnose_speed_views.py)

## 연결

다른 관련 가설: 주 가설에서만 분류한다.

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
