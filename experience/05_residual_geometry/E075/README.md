# E075. 학습 정상이 드물게 차지하는 방향 축부터 통째로 제거

**실행 상태:** 기존캐시 점수개입·독립 검수 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

Train정상 점유가낮은 방향부터 지워도 탐지성능을 유지하거나 개선할 수 있다.

## 그렇게 설계한 이유

학습 정상 점유가 낮은 방향이 정말 불필요한지 실제 점수 변화로 확인한다.

대조와 통제의 근거: 평균을 빼지 않는 방향 점유 기저다. 기존 특징 좌표 Fisher 제거와 다르며 encoder160차원 계산은 그대로 한다.

## 모델·데이터·표본

배치 160 고정 모델. 학습 자료 정상 2,580개 만으로 c에 수직인 159차원 기저 fit. 개발 선택·19LA/21LA/DF 전체 평가.

## 실험 내용과 상세 절차

1. unit 옆 방향 u의 비중심 2차 모멘트 E[uu ᵀ]를 c 직 교 공간에서 구하고 작은 고유값부터 정렬한다. 공격 라벨로 순위 fit 하지 않는다.
2. k =0/1/2/4/8/16/32/64/96/128/144/152/158/159개 축을 지울 때 x 에서 해당 투영 성분을 빼고 다시 길이 1로 정규화한 뒤 원래 c와 cosine을 계산한다.
3. 큰고유값부터 제거하는 대조와 seed 20261111/20261112/20261113의 무작위 부분 공간 대조를 같은 k 에서 수행한다.
4. 원래 개발 임계값 0.5093649528766109 고정과 각 변형의 개발 재보정 임계값을 둘 다 적용한다. 기본 주 k16을 평가 최적 k로 바꾸지 않는다.
5. 정규 직교성·직교 중심·명시적 투영 대 에너지 계산·동점 ROC·모든 159축 제거 시 부호만 남는 끝 점을 검증한다.

## 측정과 검증

전체 EER·공격/출처/코덱 오류·제거 에너지·원래/재설정 임계값 오류·소스해시 불변.

## 결과

작은 점유 축 1~32개 제거는 세 평가 EER 변화 없음.152개 제거의 DF는 3.241828 →3.149353%,158개는 5.066091%로 악화했다.

## 해석·실패한 부분·한계

학습 자료 정상 점유가 작다고 미지 위조 판별에 불필요한 것은 아니다. 사후 DF 최적 k를 배포 설정으로 선택하지 않는다.

## 배운 점

소수축제거는 변화없었으나 과도한제거는 악화했다. 정상에서 드문축이 미지위조에도 불필요하다는 보장은 없다.

## 실험 코드

- [EBM/tools/evaluate_low_occupancy_ablation.py](code/EBM/tools/evaluate_low_occupancy_ablation.py)
- [EBM/tools/verify_low_occupancy_ablation.py](code/EBM/tools/verify_low_occupancy_ablation.py)

코드의 파일별 원본 위치·SHA256은 [source_manifest.json](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [artifacts/geometry_cache/local_analysis/low_occupancy_ablation_20261111_v1/plan.json](../../_shared/references/artifacts/geometry_cache/local_analysis/low_occupancy_ablation_20261111_v1/plan.json)
- [artifacts/geometry_cache/local_analysis/low_occupancy_ablation_20261111_v1/findings.json](../../_shared/references/artifacts/geometry_cache/local_analysis/low_occupancy_ablation_20261111_v1/findings.json)
- [artifacts/geometry_cache/local_analysis/low_occupancy_ablation_20261111_v1/independent_validation.json](../../_shared/references/artifacts/geometry_cache/local_analysis/low_occupancy_ablation_20261111_v1/independent_validation.json)

## 연결

다른 관련 가설: [길이·유효 발화량과 앞뒤 저에너지 편향](../../01_duration_bias/README.md) · [정상 다양성·도메인 지지와 정상쌍 학습](../../03_bonafide_diversity/README.md) · [코사인 외 norm과 정규화 전 통계](../../04_norm_information/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
