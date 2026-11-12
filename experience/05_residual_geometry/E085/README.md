# E085. 기존 변형 결과에서 정상 안정성과 위조의 정상 방향 이동을 따로 검토

**실행 상태:** 기존 반응 점수의 탐색적 재분석 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

변형반응이 정상은안정시키고 위조만정상쪽으로 옮기는 특성이 있을 수 있다.

## 그렇게 설계한 이유

두 클래스의 평균 이동 차이만 보고 정상은 유지 되고 위조만 정상 쪽으로 이동한다고 결론 내릴 수 있는지 점검한다.

대조와 통제의 근거: 변형 전 점수가 높은 정상과 낮은 위조의 전체 평균을 직접 비교하면 점수 상한·출발점 차이가 섞인다. 정상의 평균 이동이 0에 가깝다는 것과 개별 정상의 변화가 작다는 것도 구별한다.

## 모델·데이터·표본

배치 160의 고정 모델. DF·21LA 진단 표본 각 20,000개 에 저장 한 층/시간 가중치 개입과 잡음·Opus 결과, 별도로 고정한 DF 471개 와 그중 긴 원음 144개 의 반복·절단 결과를 재사용했다. 새 음성 추론·분류기 학습은 없다.

## 실험 내용과 상세 절차

1. 발화별 변형 전 점수 s와 변형 후 점수 s_after로 d =s_after −s를 만든다. 클래스별 평균 d, 평균 |d |, 중앙값, 절대 변화의 90 백 분위, |d |≤0.02 비율과 d>0.05 비율을 함께 계산한다.
2. 층 가중치 균등, 시간 가중치 균등, 둘 다 균등, 백색잡음 20dB·10dB, Opus 32kbps의 6개 조건을 비교한다. 2,000개 확인용 표본은 20,000개 안에 있으므로 독립 반복으로 합치지 않는다.
3. 긴 원음 절단은 앞·뒤 점수를 원음별로 먼저 평균한다. 4.0375초에서 비반복 1.5초 또는 2초로 줄이는 비교와, 같은 짧은 내용의 반복/비반복 비교를 나누어 내용 감소와 반복 문맥을 구별한다.
4. 출처 안에서 화자 군집을 5,000회 복원 추출해 대응 점별 95% 구간을 계산한다. 여러 조건의 다중 비교 보정은 하지 않았으며, 이미 관찰한 결과를 다시 본 탐색 분석이다.
5. 변형 전 점수와 메타데이터가 비슷한 정상·위조의 반응 차이도 조사했다. 초기 단순 매칭의 한계를 확인한 뒤 전체 자료의 최적 매칭은 [E070](../E070/README.md)으로 확장했다. 확대 반응 추론은 실행 하지 않았다.

## 측정과 검증

클래스별 부호 있는 이동과 절대 이동, 출처·길이 별 안정성, 같은 출발점수의 반응 차이. 원행 156개 통계의 독립 산술 대조 최대 오차는 1.51e-14 였으나 bootstrap 구간을 독립 재실행한 것은 아니다.

## 결과

층 가중치 균등화에서 DF 정상 평균 이동 +0.304742, 위조 +1.314300 이고 정상 평균 절대 이동은 0.308168 이었다. DF 정상 중앙값 −0.000925만 보면 놓치는 큰 변화가 있었다. 1.5초 비반복 절단에서도 정상 +0.124023, 위조 +1.245396 이지만 출발 점수 평균은 정상 0.871198, 위조 −0.950127로 달랐다. 정상 유지·위조만 이동하는 방법을 확정하지 못했다.

## 해석·실패한 부분·한계

전체 평균 격차와 조건부판별 정보, 변형 후 점수 하나의 EER과 변형 전 점수 +반응을 읽는 새 분류기의 성능은 각각 다른 질문이다. 후자의 전이 성능은 미실행이다. 후보 실패를 모든 반응 정보의 부재로 해석하지 않는다.

## 배운 점

전체평균과출발점수 차이가 섞여 단순가설을 지지하지 못했다. 평균0도 개별안정성을 뜻하지 않는다.

## 실험 코드

- [artifacts/geometry_experiments/audit_perturbation_candidates_20261106.py](code/artifacts/geometry_experiments/audit_perturbation_candidates_20261106.py)

코드의 파일별 원본 위치·SHA256은 [source_manifest.json](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [docs/experiments/ebm_perturbation_candidate_audit_and_plan_20261106.md](../../_shared/references/docs/experiments/ebm_perturbation_candidate_audit_and_plan_20261106.md)
- [artifacts/geometry_experiments/audit_perturbation_candidates_20261106.py](../../_shared/workspace/artifacts/geometry_experiments/audit_perturbation_candidates_20261106.py)
- [artifacts/geometry_cache/local_analysis/perturbation_candidate_audit_20261106_v1/audit_scope.json](../../_shared/references/artifacts/geometry_cache/local_analysis/perturbation_candidate_audit_20261106_v1/audit_scope.json)
- [artifacts/geometry_cache/local_analysis/perturbation_candidate_audit_20261106_v1/summary.json](../../_shared/references/artifacts/geometry_cache/local_analysis/perturbation_candidate_audit_20261106_v1/summary.json)
- [artifacts/geometry_cache/local_analysis/perturbation_candidate_audit_20261106_v1/independent_verification.json](../../_shared/references/artifacts/geometry_cache/local_analysis/perturbation_candidate_audit_20261106_v1/independent_verification.json)

## 연결

다른 관련 가설: [정상 다양성·도메인 지지와 정상쌍 학습](../../03_bonafide_diversity/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
