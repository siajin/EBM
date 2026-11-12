# 같은 변형 전 점수 표본의 전체 가능 매칭과 이미 저장된 변형 반응의 범위

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
2. 길이 구간은[소스: check_full_score_overlap_20261106.py](code/artifacts/geometry_experiments/check_full_score_overlap_20261106.py)

코드의 파일별 원본 위치·SHA256은 [원본 파일 위치와 해시 목록](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [상세 설계·실험 기록](../../_shared/references/docs/experiments/ebm_full_score_overlap_results_20261106.md)
- [상세 설계·실험 기록](../../_shared/references/docs/experiments/ebm_perturbation_candidate_audit_and_plan_20261106.md)
- [소스: check_full_score_overlap_20261106.py](../../_shared/workspace/artifacts/geometry_experiments/check_full_score_overlap_20261106.py)
- [설정·결과 근거](../../_shared/references/artifacts/geometry_cache/local_analysis/full_score_overlap_20261106_v2/summary.json)
- [설정·결과 근거](../../_shared/references/artifacts/geometry_cache/local_analysis/perturbation_candidate_audit_20261106_v1/summary.json)

## 연결

다른 관련 가설: 주 가설에서만 분류한다.

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
