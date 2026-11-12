# E019. 같은 표본에서 두 속도 변형 알고리즘을 직접 비교

**실행 상태:** 추론 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

같은 속도 배율에서도 알고리즘을 바꾸면 고정 모델의 점수 손상이 달라질 수 있다.

## 그렇게 설계한 이유

[E018](../E018/README.md)의 실패가 속도 변경 자체인지 특정 변형 알고리즘에 민감한지 본다.

대조와 통제의 근거: pooling 잡음 0. 모델 갱신 없음. atempo 전체 →약 4초는 실제 3.4331875~4.061875초로 정확 4.0375초 대조가 아니다.

## 모델·데이터·표본

[E018](../E018/README.md)에서 각 집합 × 클래스 × 길이 구간마다 메타데이터 순서 2개 씩, 총 64개(정상 32·위조 32). 같은 세 50epoch 모델.

## 실험 내용과 상세 절차

1. 같은 64개 의 기준 입력·배율·6조건을 유지하고 변형만 FFmpeg atempo로 바꾼다.
2. 기존 phase-vocoder 결과도 정확히 같은 64개로 제한한다. 두 실행의 원본 기준 점수가 오차 0으로 일치하는지 확인한다.
3. atempo 결과의 실제 길이를 기록하고 목표와 다르더라도 추가 절단·패딩으로 숨기지 않는다.

## 측정과 검증

같은 전체 개발 임계값의 정상 거부·위조 통과와 점수 변화. 작은 층화 표본이므로 전체 성능 추정으로 쓰지 않는다.

## 결과

B 정상 32개에서 원본 거부 2개,0.8배 phase-vocoder31개,0.8배 atempo2개. atempo1.25배 5개, 왕복 2개, 전체 →약 4초 7개였다.

## 해석·실패한 부분·한계

변형 알고리즘이 중요한 변수다. 한 감속 조건에서 추가 오탐이 없었다고 모든 속도·코덱·내용에서 안전하다고 결론 내리지 않는다.

## 배운 점

atempo와 phase-vocoder가 크게 달랐다. 변형 알고리즘과 실제 출력 길이를 조건으로 기록해야 한다.

## 실험 코드

- [EBM/tools/diagnose_speed_atempo.py](code/EBM/tools/diagnose_speed_atempo.py)
- [artifacts/speed_duration_feasibility_20261109/write_report.py](code/artifacts/speed_duration_feasibility_20261109/write_report.py)

코드의 파일별 원본 위치·SHA256은 [source_manifest.json](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [docs/experiments/ebm_B_C_all_checkpoint_evaluation_20261109.md](../../_shared/references/docs/experiments/ebm_B_C_all_checkpoint_evaluation_20261109.md)
- [artifacts/speed_duration_feasibility_20261109/state.json](../../_shared/references/artifacts/speed_duration_feasibility_20261109/state.json)
- [artifacts/speed_duration_feasibility_20261109/write_report.py](../../_shared/workspace/artifacts/speed_duration_feasibility_20261109/write_report.py)

## 연결

다른 관련 가설: 주 가설에서만 분류한다.

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
