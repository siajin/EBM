<a id="e063-출처길이를-통제한-음향-특징-블록별-정상-오류-예측"></a>

# 출처·길이를 통제한 음향 특징 블록별 정상 오류 예측

**실행 상태:** 추가 진단 분류기 교차검증 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

출처·길이 이상의 음향 블록별 정보가 정상 오류를 예측할 수 있다.

## 그렇게 설계한 이유

어떤 음향 블록이 출처와 길이만 아는 것보다 오류를 더 잘 설명하는지 검사한다.

대조와 통제의 근거: 음성 위조 검출기 재학습이 아니라 기존 모델의 정상 오류를 예측하는 별도 분류기다. 목표 오류가 어느 임계값에서 정의됐는지 고정한다.

## 모델·데이터·표본

[정상 도메인 오류와 연결되는 실제 음향 특성 48개 측정](../E062/README.md)에서 DFnocodec 정상 1,156개. 배치 160 모델의 고정 19LA 임계값 오류를 목표로 한다.

## 실험 내용과 상세 절차

1. 모든 분류기에 출처와 원음 길이를 포함한다. 기본 metadata만, 음량, 스펙트럼, MFCC, 음높이/주기성, 시간 변화, 전체 48개, 차원이 같은 ID 고정 잡음을 비교한다.
2. 분류기 규제 계수 1·max_iter3000·tol1e-7 logistic을 사용하고 하이퍼 파라미터 탐색은 하지 않는다. 결측채움·표준화·출처 one-hot은 학습 fold 에서만 fit 한다.
3. 출처 × 화자 그룹으로 5fold,seed 11/22/33을 사용한다. 같은 화자가 train/test에 겹치지 않게하고 발화별 held-out 오류 확률을 저장한다.
4. 출처별 화자 군집 2,000회로 기본 대비로그 손실 감소와 7개 음향/잡음 비교의 보정 구간을 계산한다. 화자·길이를 잔차화한 순위 상관도 보조 계산한다.

## 측정과 검증

OOFAUROC·로그 손실 개선·7 비교 보정 구간·출처 내 및 화자/길이 보정 상관.

## 결과

48개 특징은 출처 +길이 이상의 신뢰할 만한 추가 개선을 보이지 않았다. 화자/길이 보정 상관 최대 절댓값은 0.160597 이었다.

## 해석·실패한 부분·한계

측정하지 않은 음향 요인·비선형 관계·작은 오류 수의 한계가 남는다. 실패가 음향 요인이 전혀 없음을 증명하는 것은 아니다.

## 배운 점

화자 분리와잡음 대조 후 안정된 추가 이득이 없었다. 작은 표본과 미측정 비선형 요인까지 부정한 것은 아니다.

## 실험 코드

- [소스: analyze_normal_cause_audit.py](code/artifacts/geometry_experiments/analyze_normal_cause_audit.py)

코드의 파일별 원본 위치·SHA256은 [원본 파일 위치와 해시 목록](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [소스: analyze_normal_cause_audit.py](../../_shared/workspace/artifacts/geometry_experiments/analyze_normal_cause_audit.py)
- [설정·결과 근거](../../_shared/references/artifacts/geometry_cache/local_analysis/normal_cause_audit_v1/diagnostic_model_lock.json)
- [설정·결과 근거](../../_shared/references/artifacts/geometry_cache/local_analysis/normal_cause_audit_v1/acoustic_error_model_effects.json)

## 연결

다른 관련 가설: 주 가설에서만 분류한다.

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
