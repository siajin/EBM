<a id="e065-틀린-정상-특징이-학습-위조-또는-평가-정상의-어디에-가까운지-조사"></a>

# 틀린 정상 특징이 학습 위조 또는 평가 정상의 어디에 가까운지 조사

**실행 상태:** 최근접 참조 분석 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

틀린 새출처 정상은 Train 정상보다 target 정상 지지에 더 가까울 수 있다.

## 그렇게 설계한 이유

새출처의 정상이 현재 고정 특징에서 어떤 기존 집단과 가까운지 본다.

대조와 통제의 근거: DF 참조에는 평가 도메인 학습 fold의 라벨 정보가 있다. 타깃 정답을 전혀 사용하지 않는 검출 점수라고 부르지 않는다.

## 모델·데이터·표본

배치 160 고정 모델의 학습 자료·DF 진단. [정상 예시를 더 잘 대표하는 중심·이동·정규화 점수 7 종](../E064/README.md)와 같은 화자 분리 fold 에서 각 학습 부분의 정상/위조 참조를 사용한다.

## 실험 내용과 상세 절차

1. 160차원 전체 z의 cosine 거리를 사용한다. 화면의 2차원 거리로 최근접을 정하지 않는다.
2. 학습 자료 참조와 DF 바깥 학습 fold 참조 각각의 정상/위조 개수를 같게 ID 해시로 선택한다.
3. 기존 임계값에서 틀린 정상·맞은 정상별로 최근접 1개 와 가까운 5개 평균 거리의 정상 우세 여부를 계산한다.
4. 출처·길이·코덱 공통 조건을 분해하고 다음 [DF 자체 EER 부근 오류 꼬리와 길이·최근접·음향 특성 재검토](../E066/README.md)의 다른 오류 정의에서도 다시 검사한다.

## 측정과 검증

참조 class 균형·fold 분리·최근접 라벨 비중·거리차이·source 별 분모.

## 결과

기존 임계값에서 틀린 VCC2018 정상 334개 중 학습 자료 최근접 정상 29.94%, DFheld-out 참조 정상 90.12%;VCC2020 정상 154개는 44.16%/96.75%였다.

## 해석·실패한 부분·한계

훈련 정상 지지와 새 정상 지지의 차이를 보이는 기하 진단이다. 원천 녹음 원인이나 새 정상 추가 학습의 효과를 측정하지 않는다.

## 배운 점

최근접 정상 비율이 target참조에서 크게 높았다. 새정상 추가 학습의 실제 이득은 아직 시험하지 않았다.

## 실험 코드

- [소스: trace_normal_neighborhoods.py](code/artifacts/geometry_experiments/trace_normal_neighborhoods.py)

코드의 파일별 원본 위치·SHA256은 [원본 파일 위치와 해시 목록](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [설정·결과 근거](../../_shared/references/artifacts/geometry_cache/local_analysis/normal_cause_audit_v1/neighborhood_lock.json)
- [설정·결과 근거](../../_shared/references/artifacts/geometry_cache/local_analysis/normal_cause_audit_v1/neighborhood_complete.json)
- [소스: trace_normal_neighborhoods.py](../../_shared/workspace/artifacts/geometry_experiments/trace_normal_neighborhoods.py)

## 연결

다른 관련 가설: [잔여 방향·좌표·거리·기하와 변형 반응](../../05_residual_geometry/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
