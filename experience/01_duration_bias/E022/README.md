<a id="e022-중단한-가변-길이-학습의-저장-20epoch를-21ladf-전체-평가"></a>

# 중단한 가변 길이 학습의 저장 20epoch를 21LA·DF 전체 평가

**실행 상태:** 평가·검산 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

중단한 학습도 사용자가 지정한 저장20epoch로 외부 일반화를 정직하게 평가할 수 있다.

## 그렇게 설계한 이유

학습을 재개하지 않고 사용자가 지정한 저장 모델의 외부 성능을 확인한다.

대조와 통제의 근거: 개발 평가 없음, optimizer0회. EBM-NoRepeat와 EBM-FullContext 기존 점수와 입력/점수 정책은 같지만 실제 난수 실행은 다르다.

## 모델·데이터·표본

[정상·위조 길이와 음성활동을 맞추고 독립 정상 묶음을 정렬한 학습](../E021/README.md)의 CLS20/TrainState20. 21LA 148,176 →DF 533,928 전체를 순차 평가한다.19LA 20 결과는 기존 평가를 보존한다.

## 실험 내용과 상세 절차

1. 두 저장 파일의 파라미터·epoch20·Adam2,160회 일치를 검사한다.
2. 앞 64,600 상한·비반복·마스크·GPU 별 배치 8·평가 seed 1729·원래 pooling 잡음 1e-5·학습 OC 중심 cosine을 사용한다.
3. 두 GPU 사전 실행 검사 192개 와 저장 19LA 64개 점수 차이를 검사한 뒤 전체 평가를 실행한다.
4. 전체 UID/index·클래스·길이 cap·SHA·유한 점수와 독립 동점 ROC를 검산한다. EBM-NoRepeat와 EBM-FullContext의 20epoch 모델 비교와 EBM-NoRepeat와 EBM-FullContext 최소값 참고 표를 구별한다.

## 측정과 검증

전체 EER, 독립 재계산, 동일 epoch 차이. 평가된 21LA·DF 체크포인트는 20 하나뿐이다.

## 결과

19LA 2.708118%,21LA 8.908218%, DF 2.286637%. EBM-NoRepeat/EBM-FullContext 20epoch 모델은 21LA 4.792117/5.707109%, DF 2.205932/2.326989%였다.

## 해석·실패한 부분·한계

21LA·DF의 여러 epoch 최소 EER이 아니다. 사용자 중단 후 20epoch를 지정했고 21LA/DF 결과로 이 모델을 골랐던 것은 아니다.

## 배운 점

21LA는 크게 악화했고 DF는 EBM-NoRepeat와 EBM-FullContext의 20epoch 모델 부근이었다. 중단 상태와 지정 checkpoint의 전체 평가 완료는 별개로 표시해야 한다.

## 실험 코드

- [소스: evaluate_coverage_checkpoint.py](code/EBM/tools/evaluate_coverage_checkpoint.py)
- [소스: compare_coverage_epoch20.py](code/EBM/tools/compare_coverage_epoch20.py)

코드의 파일별 원본 위치·SHA256은 [원본 파일 위치와 해시 목록](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [소스: evaluate_coverage_checkpoint.py](../../../tools/evaluate_coverage_checkpoint.py)
- [모델·조건별 비교 결과](../../_shared/references/artifacts/ebm_coverage_e20_eval_20261110/comparison.json)
- [설정·결과 근거](../../_shared/references/artifacts/ebm_coverage_e20_eval_20261110/analysis_complete.json)

## 연결

다른 관련 가설: [반복 입력과 학습·추론의 상호작용](../../02_repetition/README.md) · [정상 다양성·도메인 지지와 정상쌍 학습](../../03_bonafide_diversity/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
