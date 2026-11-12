<a id="e010-두-반복-제거-학습-모델의-모든-저장-epoch-평가와-최소-eer-비교"></a>

# 두 반복 제거 학습 모델의 모든 저장 epoch 평가와 최소 EER 비교

**실행 상태:** 평가 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

고정50 한 점보다 저장된 전체 epoch를 보면 EBM-NoRepeat와 EBM-FullContext의 과적합·집합별 최적 시점을 파악할 수 있다.

## 그렇게 설계한 이유

마지막 50epoch 한 점 대신 저장된 5~50epoch 전체를 같은 조건으로 평가한다.

대조와 통제의 근거: EBM-NoRepeat와 EBM-FullContext의 입력·난수는 같은 실행에서 통제한다. EBM-Repeat 의최소표는 과거 반복 평가 15~50epoch8개 기록으로, 이 20개 평가와 입력 조건이 다르다.

## 모델·데이터·표본

EBM-NoRepeat와 EBM-FullContext 각각 5,10,15,20,25,30,35,40,45,50epoch의 총 20개 모델. 개발·19LA·21LA·DF 전체 778,185개.

## 실험 내용과 상세 절차

1. 각 모델의 파라미터와 저장 학습 상태가 같은지, Adam 갱신 수가 epoch×108 인지 검사한다.
2. 앞 64,600 상한·비반복·마스크·GPU 별 배치 8의 입력을 만들고, 한 XLS-R 출력을 20개 분류기가 공유한다.
3. 각 입력의 모든 모델에 동일 pooling 난수를 복원한다. 원래 잡음 1e-5를 유지한다.
4. 각 집합 × 모델 ×epoch EER을 계산하고 모든 저장 결과를 남긴다. 요청에 따른 최소값은 사후 요약이며 epoch 선택용 독립 검증 성능은 아니다.

## 측정과 검증

모든 EER·최소값/epoch, 개발 임계값 고정 오류, 길이별 분모. LA의 모든 21개 점수 열은 별도 동점 ROC로 검산했다. DF 완료표와 metric도 확인했다.

## 결과

EBM-NoRepeat의 최소는 19LA 0.462271%(45),21LA 3.759449%(45), DF 2.084875%(5). EBM-FullContext 모델은 0.488400%(40),4.622076%(30),2.152549%(5). 세 집합의 최적 epoch가 다르다.

## 해석·실패한 부분·한계

중간 문서의 DF ‘평가 중’은 최신 완료 상태가 아니다. 12행짜리 합성 summary_qc 결과는 실제 전체 평가로 사용하지 않는다. 네 모델의 반복/비반복 전체 epoch 대조는 미실행이다.

## 배운 점

target별 관측 최소는 사후 요약이다. 다른 집합에서 각각 고른 epoch를 단일 배포 모델의 성능처럼 제시하면 안 된다.

## 실험 코드

- [두 비반복 학습의 전체 저장점 평가 코드](code/EBM/tools/evaluate_B_C_checkpoints.py)
- [EBM-NoRepeat의 저장점 평가 코드](code/EBM/tools/evaluate_B_checkpoints.py)

코드의 파일별 원본 위치·SHA256은 [원본 파일 위치와 해시 목록](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [두 비반복 학습의 전체 저장점 평가 코드](../../../tools/evaluate_B_C_checkpoints.py)
- [집계 평가 지표](../../_shared/references/artifacts/ebm_B_C_checkpoints_20261109/metrics.json)
- [완료 확인 기록](../../_shared/references/artifacts/ebm_B_C_checkpoints_20261109/complete.json)
- [검산 결과](../../_shared/references/artifacts/ebm_B_C_checkpoints_20261109/LA_analysis/validation.json)
- [설정·결과 근거](../../_shared/references/artifacts/ebm_B_C_checkpoints_20261109/completed_B_C_dataset_minima.json)

## 연결

다른 관련 가설: [길이·유효 발화량과 앞뒤 저에너지 편향](../../01_duration_bias/README.md) · [정상 다양성·도메인 지지와 정상쌍 학습](../../03_bonafide_diversity/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
