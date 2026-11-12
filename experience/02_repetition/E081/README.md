<a id="e081-네-모델전체-체크포인트의-반복비반복-공통-평가-준비와-실행-전-취소"></a>

# 네 모델·전체 체크포인트의 반복/비반복 공통 평가 준비와 실행 전 취소

**실행 상태:** 검사·분석 재현만 완료; 추가 GPU 평가 취소

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

모든보존checkpoint를 같은반복/비반복 조건에서 평가하면 최소EER 비교를 더 공정하게 만들 수 있다.

## 그렇게 설계한 이유

최소 EER을 완전히 같은 두 입력 조건에서 비교하는 후속 평가를 준비했다가 사용자의 범위 정정으로 실행하지 않았다.

대조와 통제의 근거: 실제 GPU 사전 실행 검사·전체 평가·새최소표는 미완료다. 준비된 코드를 현재 실험 완료 자료로 사용하지 않는다.

## 모델·데이터·표본

예정은 배치 240 고정 길이 반복 입력 학습 보존 8개 +4.0375초 상한 비반복 학습 10epoch 모델개 +원음 전체 비반복 학습 10epoch 모델개 +현재 학습 10개 =38 체크포인트,19LA/21LA/DF 전체. 실제 새 음성 점수는 없다.

## 실험 내용과 상세 절차

1. 입력은 앞 64,600 비반복/mask와 순환 반복 64,600, 학습 OC 중심·공통 frontend·잡음 1e-5·모든 head 공통 난수로 고정한 별도 snapshot을 만들었다.
2. 전체 보존 최소와 공통 15~50epoch8개그리드최소·같은 epoch 차이·길이/IPF/31조건 진단의 자동 연결 코드를 검사했다.
3. 로컬 9 검사와 기존 완료 파생 점수 296,352행의 분석 경로 검산을 통과했다. 이것은 새 모델 38개 의 추론 성능 검사가 아니다.
4. 현재 학습 종료와 GPU 유휴를 기다리던 무자손 CPUwatcher만 사용자의 ‘지금까지의 실험만’정정으로 해제했다. launch/pipeline/stage 표식과새 GPU 평가가 없음을 확인했다.

## 측정과 검증

취소표식·기존 학습 불변·소스/원모델보존·새추론 0회.

## 결과

새 평가 점수 없음. 기존 두 모델 50epoch 전체 반복 진단 [개발 전체와 21LA 전체에서 반복/비반복 입력을 직접 교차 평가](../E014/README.md)와 모든 38개 checkpoint 평가 계획은 다르다.

## 해석·실패한 부분·한계

취소한 watcher를 재활성화하거나 추가 학습을 실행하지 않았다.

## 배운 점

준비·CPU재현만 했고 사용자요청으로 취소됐다. 과거2모델50평가를38checkpoint 전체평가로 대체해 쓰지 않는다.

## 실험 코드

- [소스: paired_evaluation.py](code/EBM/paired_evaluation.py)
- [소스: paired_queue_guard.py](code/EBM/paired_queue_guard.py)
- [소스: run_paired_evaluation.py](code/EBM/tools/run_paired_evaluation.py)

코드의 파일별 원본 위치·SHA256은 [원본 파일 위치와 해시 목록](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [상세 설계·실험 기록](../../_shared/references/docs/experiments/ebm_paired_minimum_eer_evaluation_20261111.md)
- [설정·결과 근거](../../_shared/references/artifacts/paired_min_eer_20261111/queue_cancelled.json)
- [검산 결과](../../_shared/references/artifacts/paired_min_eer_20261111/validation.json)
- [설정·결과 근거](../../_shared/references/artifacts/paired_min_eer_20261111/derived_analysis_validation.json)

## 연결

다른 관련 가설: [길이·유효 발화량과 앞뒤 저에너지 편향](../../01_duration_bias/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
