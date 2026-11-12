# E070. 같은 변형 전 점수 표본의 전체 가능 매칭과 이미 저장된 변형 반응의 범위

**실행 상태:** 기존캐시매칭·산술 검증 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

같은 출발점수·메타데이터의 정상/위조를 맞추면 선택적 위조 반응 가설을 더 엄격히 볼 수 있다.

## 그렇게 설계한 이유

‘위조만 정상 방향으로 움직인다’는 반응 가설을 같은 출발점에서 검사할 자료가 있는지 본다.

대조와 통제의 근거: 두 표본 집합의 최대 매칭을 독립적으로 구하므로 두 쌍 수 차이를 단순 누락 수로 빼지 않는다.3 허용 차 ×2 엄격 도 ×2 자료 ×2 가용 범위 =24 매칭 파일.

## 모델·데이터·표본

고정 배치 160의 DF 전체 533,928·21LA 전체 148,176 변형 전 점수. 실제층/시간 attention·잡음/Opus 반응은 각 진단 20,000 에만 있다.

## 실험 내용과 상세 절차

1. 엄격 조건은 출처·화자·코덱·길이 구간 일치와 원음 길이 차 ≤0.25초·변형 전 점수 차 ≤0.02다.21LA는 전송 조건도 일치시킨다. 완화는 화자 일치만 해제한다.
2. 길이 구간은[0,2),[2,3),[3,4.0375),[4.0375,6),[6, 무한대), 점수 허용 차 0.01/0.005도 각각 계산한다.
3. 반응 점수를 보지 않고 먼저 중복 없는 쌍 수를 최대화하고, 그 최대 개수 안에서 변형 전 점수 차이 합을 최소화한다. 최대매칭 =최소 vertex-cover로 확인한다.
4. 전체 변형 전 점수 자료와 반응이 양쪽 모두 있는 자료에서 각각 매칭한다. 반응 차이는 D =(변형 후 −전 위조 점수)−(변형 후 −전 정상 점수)로 계산한다.
5. 기존 471개 패딩/문맥 반응을 같은 UID 라는 이유로 attention/잡음 반응으로 대체하지 않는다. 누락 endpoint 명세만 만들고 새 추론은 하지 않는다.

## 측정과 검증

쌍 수·화자 수·점수/길이 잔차·기존 반응 D·최대 개수/최소비용 검산. 작은 지지 집합에 유의성 주장을 붙이지 않는다.

## 결과

허용 차 0.02 전체 엄격 DF 515쌍·21LA 106쌍이지만 기존 반응만 쓰면 6/3쌍. 층 균등 D는 DF 엄격 −0.000093·21LA 엄격 −0.675100으로 선택적 위조 이동을 입증하지 못했다.

## 해석·실패한 부분·한계

자료가 적다는 결론을 반응 차이 0으로 바꾸지 않는다. 확대 반응 추론·새 분류기 학습은 실행 하지 않았다. 초기 greedy 하한과 후속 최적 매칭 v2를 구별한다.

## 배운 점

전체점수 매칭은 충분해도 기존 반응까지 있는쌍은 매우적었다. 결측반응을0으로 채우면 안 된다.

## 실험 코드

- [artifacts/geometry_experiments/check_full_score_overlap_20261106.py](code/artifacts/geometry_experiments/check_full_score_overlap_20261106.py)

코드의 파일별 원본 위치·SHA256은 [source_manifest.json](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [docs/experiments/ebm_full_score_overlap_results_20261106.md](../../_shared/references/docs/experiments/ebm_full_score_overlap_results_20261106.md)
- [docs/experiments/ebm_perturbation_candidate_audit_and_plan_20261106.md](../../_shared/references/docs/experiments/ebm_perturbation_candidate_audit_and_plan_20261106.md)
- [artifacts/geometry_experiments/check_full_score_overlap_20261106.py](../../_shared/workspace/artifacts/geometry_experiments/check_full_score_overlap_20261106.py)
- [artifacts/geometry_cache/local_analysis/full_score_overlap_20261106_v2/summary.json](../../_shared/references/artifacts/geometry_cache/local_analysis/full_score_overlap_20261106_v2/summary.json)
- [artifacts/geometry_cache/local_analysis/perturbation_candidate_audit_20261106_v1/summary.json](../../_shared/references/artifacts/geometry_cache/local_analysis/perturbation_candidate_audit_20261106_v1/summary.json)

## 연결

다른 관련 가설: 주 가설에서만 분류한다.

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
