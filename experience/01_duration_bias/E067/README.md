<a id="e067-원음-길이별-정상-평균-기준과-점수-보정무작위-중심-대조"></a>

# 원음 길이별 정상 평균 기준과 점수 보정·무작위 중심 대조

**실행 상태:** 캐시 점수 평가·독립 검산 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

길이별 정상 중심과점수보정이 단일중심의 단음 취약성을 줄일 수 있다.

## 그렇게 설계한 이유

단일 정상 중심을 길이별 중심으로 바꾸면 짧은 정상 취약성을 해결하는지 본다.

대조와 통제의 근거: 주 방법은 여러 중심 중 최대 값을 취하지 않는다. 입력의 원음 길이로 중심 하나를 고른다. DF 정답으로 중심/구간/계수를 fit 하지 않는다.

## 모델·데이터·표본

배치 160 고정 모델. 학습 자료 정상 2,580개 로만 중심 fit, 개발 선택 14,983은 임계값용, DF 전체·19LA·21LA·진단·개발 나머지 평가.

## 실험 내용과 상세 절차

1. 원음 길이를<2,[2,3),[3,4.0375), ≥4.0375초로 나눈다. 학습 자료 정상 수는 109/877/1,018/576이다.
2. unit 특징 전체 평균 μ와길이별 μ_k를 구해 정규화한다. 기존 학습 중심, 전체 평균 중심, 원음 길이에 맞는 평균 중심 중 하나를 선택하는 주 방법을 비교한다.
3. 기존/전체 평균 점수에서 학습 자료 정상의 길이별 평균 차이만 빼는 두 스칼라 대조를 만든다. 각 길이 내 순위는 유지된다.
4. 기존 중심 보존 보조는 unit(c +[n_k/(n_k +100)]P_c(μ_k −μ)), 수축 평균은 unit((n_k μ_k +100 μ)/(n_k +100))다. 계수 100은 고정한다.
5. 무작위 대조 199회는 학습 자료 길이 label과 각 평가 길이 label을 별도 순열하되 각 구간 개수는 유지해 4개 중심을 만든다. seed =202611060100 +r 다.
6. 각 방법의 개발 위조 통과율 ≤1%가 되는 가장 낮은 임계값을 고정해 적용한다. DF 주 4 비교는 출처별 화자 10,000회 seed 2026110601과 98.75%보정 구간을 사용한다.

## 측정과 검증

전체·길이 내·출처 × 길이 EER/AUROC·운영 오류·오류 전환·199개 대조 분포·학습 자료 화자 500회 중심 안정성.

## 결과

DF 전체 기준 3.241828%, 전체 평균 3.335799%, 길이별 평균 3.315623%, 길이 스 칼라 보정 3.243177%, 기존 중심 보존 3.246837%. 주 방법은 기준보다 0.073795%p 악화했다.

## 해석·실패한 부분·한계

고정 정상 표현의 길이별 평균 교체가 실패했다는 결과다. 길이 조건 공동 학습이나 원음 입력 변경을 기각한 것은 아니다.199개 무작위 분포는 인과 randomization p 값이 아니다.

## 배운 점

길이별평균 중심은 DF전체에서 악화했다. 단순 사후 중심 교체와 길이조건 공동학습은 다른 가설이다.

## 실험 코드

- [소스: duration_conditioned_centroids.py](code/artifacts/geometry_experiments/duration_conditioned_centroids.py)

코드의 파일별 원본 위치·SHA256은 [원본 파일 위치와 해시 목록](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [상세 설계·실험 기록](../../_shared/references/docs/experiments/ebm_duration_conditioned_centroids_plan_20261106.md)
- [상세 설계·실험 기록](../../_shared/references/docs/experiments/ebm_duration_conditioned_centroids_results_20261106.md)
- [소스: duration_conditioned_centroids.py](../../_shared/workspace/artifacts/geometry_experiments/duration_conditioned_centroids.py)
- [집계 결과 표](../../_shared/references/artifacts/geometry_cache/local_analysis/duration_centroids_20261106_v1/metrics_overall.csv)

## 연결

다른 관련 가설: [반복 입력과 학습·추론의 상호작용](../../02_repetition/README.md) · [정상 다양성·도메인 지지와 정상쌍 학습](../../03_bonafide_diversity/README.md) · [잔여 방향·좌표·거리·기하와 변형 반응](../../05_residual_geometry/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
