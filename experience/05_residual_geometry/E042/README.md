<a id="e042-원래-코사인을-유지한-방향-잔차와-공분산-거리의-전이-대조"></a>

# 원래 코사인을 유지한 방향 잔차와 공분산 거리의 전이 대조

**실행 상태:** 추가 분류기·캐시 평가 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

기존 코사인 계수를 보존하면 사후 보정의 순위 손상과 방향 이득을 분리할 수 있다.

## 그렇게 설계한 이유

새읽기 분류기의 점수 범위 변화와 방향이 득을 구별하기 위해 기존 점수 계수 1을 보존한다.

대조와 통제의 근거: 주 9개 비교, 출처 × 화자 pairedbootstrap10,000회 seed 7301. 같은 native 확률 보정을 후보에도 써 순위와 calibration을 구별한다.

## 모델·데이터·표본

배치 160 고정 모델의 학습 자료·개발 선택·DF 진단 20,000. 기존 모델 가중치·중심 고정.

## 실험 내용과 상세 절차

1. 학습 자료만으로 단조 확률 보정 p =sigmoid(8.710543812499347s −0.7151751091389169)를 고정하고 순위에는 포화되지 않은 logit을 쓴다.
2. 옆 방향 u의 L2 logistic decision에 tanh를 적용한 잔차를 s +epsilon×tanh(decision/2)로 더 한다. C =0.01/0.1/1/10,epsilon =0/0.01/0.03/0.1를 개발로그 손실로 선택한다.
3. 차원이 같은 고정 잡음 방향으로 동일 잔차를 학습한다. 동률 1e-12 에서는 epsilon0·작은 EBM-FullContext 모델을 우선한다.
4. 같은 학습 정상 공분산의 대각/전체 거리와 수축 0.01/0.1/0.5/1를 비교하고, 개발 EER 동률은 gamma1 =identity를 고른다.

## 측정과 검증

EER·AUROC·균형로그 손실·개발 고정 오류·9개 비교 보정 구간. 후속 사후 해석에서는 6개 거리 후보 각각에 대해 출처·화자 군집 2,000회 대응 bootstrap의 점별 95% 구간을 계산했다. 6개 후보의 다중 비교 보정은 하지 않았다. 이 재분석은 독립적인 새 검증 표본이 아니다.

## 결과

방향 잔차의 선택은 분류기 규제 계수 1·epsilon0.1 이었으나 DFEER 3.35%로같고로그 손실은 악화했다. 잡음 잔차는 epsilon0, 거리 선택도 identity 였다.

## 해석·실패한 부분·한계

특정 후보 대각 gamma0.01의 3.05%는 관찰 값이지 개발에서 선택한 주 모델이 아니다. DF를 이미 본 뒤의 탐색이므로 새 확증 검증이 아니다.

## 배운 점

개발 선택 방향 잔차는 DF EER을 개선하지 못했다. 사후 좋은 거리 후보를 주 결과로 바꾸면 안 된다.

## 실험 코드

- [소스: validate_native_preserving.py](code/artifacts/geometry_experiments/validate_native_preserving.py)
- [소스: plot_native_preserving.py](code/artifacts/geometry_experiments/plot_native_preserving.py)
- [소스: verify_native_preserving_math.py](code/artifacts/geometry_experiments/verify_native_preserving_math.py)

코드의 파일별 원본 위치·SHA256은 [원본 파일 위치와 해시 목록](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [설정·결과 근거](../../_shared/references/artifacts/geometry_cache/local_analysis/native_preserving_v1/protocol.json)
- [설정·결과 근거](../../_shared/references/artifacts/geometry_cache/local_analysis/native_preserving_v1/summary.json)
- [소스: validate_native_preserving.py](../../_shared/workspace/artifacts/geometry_experiments/validate_native_preserving.py)
- [설정·결과 근거](../../_shared/references/artifacts/geometry_cache/local_analysis/df_interpretation_v1/interpretation.json)

## 연결

다른 관련 가설: 주 가설에서만 분류한다.

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
