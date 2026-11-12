<a id="e057-기존-모델의-층-가중치최종-시간-가중치를-균등하게-바꾸는-개입"></a>

# 기존 모델의 층 가중치·최종 시간 가중치를 균등하게 바꾸는 개입

**실행 상태:** 실제 추론 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

층·시간attention을 균등화하면 어떤 집계가 현재 분리에 기여하는지 드러날 수 있다.

## 그렇게 설계한 이유

기존 attention을 제거할 때 출력이 얼마나 달라지는지 본다.

대조와 통제의 근거: 원래 고정 4.0375초 입력과 ASP 잡음 1e-5를 유지 하며 발화 seed를 복원한다. 원래 모델 재학습은 하지 않는다.

## 모델·데이터·표본

배치 160 고정 모델·DF/21LA 진단 각 20,000. 각 2,000개를 먼저 실행한 뒤 같은 고정 목록 전체로 확장했다.

## 실험 내용과 상세 절차

1. 기준은 원래 층·시간 attention 둘 다 유지한다.
2. 층 가중치를 만드는 벡터를 0으로 대체해 25개 층 각 1/25로 한다. 각 층 내 pooling 자체를 삭제하지 않는다.
3. 최종 시간 attention 파라미터를 0으로 대체해 유효 시간 프레임에 균등 가중치를 준다. 평균/표준편차 pooling 자체를 없애지 않는다.
4. 기준·층만 균등·시간만 균등·둘 다 균등의 2×2 네 조건을 functional parameter override로 계산하고 저장 파라미터 해시 불변을 확인한다. 같은 원음·XLS-R 특징·난수를 공유한다.

## 측정과 검증

조건별 EER/AUROC·발화별 점수/방향 변화·개발 고정 오류와 후처리 gate 반응.

## 결과

DF 기준/층균 등/시간 균등/둘 다 EER은 3.35/7.488889/4.094444/11.133333%.21LA는 0.8/4.855556/0.805556/8.911111%.

## 해석·실패한 부분·한계

attention을 제거한 채 다시 학습한 모델의 성능이 아니다. 정상 점수만 좋아져도 위조 점수가 함께 올라 전체 EER은 악화할 수 있다.

## 배운 점

층attention 제거의 손상이 컸다. 이는 재학습 없는 개입 결과이며 균등attention으로 새로 학습한 성능이 아니다.

## 실험 코드

- [소스: frozen_r23.py](code/artifacts/geometry_experiments/frozen_r23.py)

코드의 파일별 원본 위치·SHA256은 [원본 파일 위치와 해시 목록](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [소스: frozen_r23.py](../../_shared/workspace/artifacts/geometry_experiments/frozen_r23.py)
- [상세 설계·실험 기록](../../_shared/references/docs/experiments/ebm_frozen_all_results_20261105.md)
- [상세 설계·실험 기록](../../_shared/references/docs/experiments/ebm_perturbation_candidate_audit_and_plan_20261106.md)

## 연결

다른 관련 가설: [코사인 외 norm과 정규화 전 통계](../../04_norm_information/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
