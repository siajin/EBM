# E060. 정상 점수와 위조 점수의 데이터셋을 따로 교체하는 교차 평가

**실행 상태:** 캐시교차 평가·재현 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

정상과위조의 도메인을 따로 교체하면 EER 악화에 민감한 클래스 분포를 찾을 수 있다.

## 그렇게 설계한 이유

도메인별 EER 차이가 어느 클래스 점수 분포의 교체에 민감한지 본다.

대조와 통제의 근거: 점수 집합 조합이며 실제 음성 생성·입력 교체 실험이 아니다. 다른 순서의 EER 변화는 비선형이므로 edge 값을 합쳐 원인 비율을 만들지 않는다.

## 모델·데이터·표본

배치 160 고정 모델의 19LA 전체·21LA 전체·DF 전체 점수. 새 특징 추출이나 분류기 학습은 없다.

## 실험 내용과 상세 절차

1. 정상 점수 집합과 위조 점수 집합을 독립적으로 선택해 2×2 EER 행렬을 만든다.19LA ↔DF,21LA ↔DF,19LA ↔21LA를 각각 계산한다.
2. DF 정상은 ASVspoof/VCC2018/VCC2020로 분리한 교차 표를 만들고, 공유 공격 비중을 재가중한 보조 비교도 한다.
3. DF nocodec 제한과 양 클래스의 길이를 각각 19LA 분포에 맞추는 민감도를 계산한다. 길이 구간은<1,1~1.5,1.5~2,2~3,3~4.0375,4.0375~6, ≥6초다.
4. 출처별 화자 군집 2,000회, 공유 ASV 화자의 가중치는 데이터셋 사이에도 공유한다. 기준 셋 별 네 교체 edge에 95%와 Bonferroni98.75%구간을 계산한다.

## 측정과 검증

교차 EER/AUROC·개발 위조 통과 1% 기준 고정 오류·출처/길이/코덱 민감도.

## 결과

19LA 정상 +19LA 위조 0.435177%, DF 정상 +19LA 위조 5.999058%,19LA 정상 +DF 위조 0.285520%, DF 정상 +DF 위조 3.241828%. 정상 교체가 악화를 주도했다.

## 해석·실패한 부분·한계

정상 학습 다양성 부족이 원인이라고 증명한 것은 아니다. 고정 개발 임계값의 균형 오류는 오히려 DF 에서 낮을 수 있으므로 모든 운용 점의 악화로 일반화하지 않는다.

## 배운 점

DF 정상 교체의 영향이 컸다. 이것은 정상 학습 다양성 부족의 인과 증명이 아니라 후속 가설의 근거다.

## 실험 코드

- [artifacts/geometry_experiments/test_bonafide_domain_shift.py](code/artifacts/geometry_experiments/test_bonafide_domain_shift.py)
- [artifacts/geometry_experiments/evaluate_19la_to_21la_cross.py](code/artifacts/geometry_experiments/evaluate_19la_to_21la_cross.py)

코드의 파일별 원본 위치·SHA256은 [source_manifest.json](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [docs/experiments/ebm_bonafide_cross_recheck_results_20261106.md](../../_shared/references/docs/experiments/ebm_bonafide_cross_recheck_results_20261106.md)
- [docs/experiments/ebm_19la_to_21la_cross_results_20261106.md](../../_shared/references/docs/experiments/ebm_19la_to_21la_cross_results_20261106.md)
- [artifacts/geometry_experiments/test_bonafide_domain_shift.py](../../_shared/workspace/artifacts/geometry_experiments/test_bonafide_domain_shift.py)
- [artifacts/geometry_experiments/evaluate_19la_to_21la_cross.py](../../_shared/workspace/artifacts/geometry_experiments/evaluate_19la_to_21la_cross.py)
- [artifacts/geometry_cache/local_analysis/bonafide_cross_recheck_20261106_v1/protocol.json](../../_shared/references/artifacts/geometry_cache/local_analysis/bonafide_cross_recheck_20261106_v1/protocol.json)

## 연결

다른 관련 가설: [길이·유효 발화량과 앞뒤 저에너지 편향](../../01_duration_bias/README.md) · [잔여 방향·좌표·거리·기하와 변형 반응](../../05_residual_geometry/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
