# E009. 두 고정 50epoch 모델을 같은 비반복 입력으로 전체 평가

**실행 상태:** 평가 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

모델마다 다른 평가 전처리를 통일하면 모델 표현의 차이를 더 공정하게 비교할 수 있다.

## 그렇게 설계한 이유

과거 반복 학습 모델과 반복 제거 학습 모델의 표현을 같은 추론 입력에서 비교한다.

대조와 통제의 근거: 공통 입력 비교와 ‘기존 모델의 과거 반복 파이프 라인 대 B의 비반복 파이프라인’ 비교를 별도 표로 둔다.

## 모델·데이터·표본

배치 240 원래 모델과 반복 제거 모델(B)의 50epoch. 개발 전체 24,844·19LA 71,237·21LA 148,176·21DF 533,928, 총 778,185개.

## 실험 내용과 상세 절차

1. 모든 모델에 원음 앞 min(원음 길이,64,600)을 사용한다. 짧은 원음은 반복하지 않는다.
2. 물리배치 8과 실제 길이 마스크, 고정 XLS-R 출력을 공유한다. 두 모델은 각자 학습한 정상 기준 벡터로 코사인 점수를 낸다.
3. 원래 pooling 잡음 1e-5를 유지하고 같은 입력·공통 난수로 점수를 계산한다. 평가 중 optimizer 갱신은 하지 않는다.
4. 발화 ID·라벨·원음 길이·평가 입력 길이의 정확한 정렬, 전체 개수, 유한 점수, 과거 B50 점수의 재현을 검사한다.

## 측정과 검증

전체 EER·AUROC와 개발 자료에서 고정한 임계값의 정상 거부·위조 통과. 3개 평가 셋 ×2개 클래스 차이는 화자 군집 5,000회, 6 비교 Bonferroni 동시 95% 구간.

## 결과

공통 비반복 EER 에서 원래 240 →B는 19LA 1.128484 →0.679810%,21LA 6.688715 →4.792117%, DF 8.877530 →2.723788%. 과거 반복 평가 대비로는 DF만 낮았다.

## 해석·실패한 부분·한계

공통 비반복 입력의 우위가 과거 전체 파이프라인 대비 우위를 뜻하지 않는다. 원래 모델과 B의 학습 구현 차이를 반복 하나의 효과로 해석하지 않는다.

## 배운 점

공통 입력 우위와 실제 역사적 파이프라인 우위는 다르다. 입력·중심·난수를 함께 잠그는 비교가 필요하다.

## 실험 코드

- [artifacts/ebm_git_private_20261106/evaluate_B_final.py](code/artifacts/ebm_git_private_20261106/evaluate_B_final.py)
- [artifacts/ebm_git_private_20261106/render_B_final_analysis.py](code/artifacts/ebm_git_private_20261106/render_B_final_analysis.py)

코드의 파일별 원본 위치·SHA256은 [source_manifest.json](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [docs/experiments/ebm_B_final_results_20261108.md](../../_shared/references/docs/experiments/ebm_B_final_results_20261108.md)
- [artifacts/geometry_cache/local_analysis/B_final_analysis_20261108_v1/analysis/comparison.json](../../_shared/references/artifacts/geometry_cache/local_analysis/B_final_analysis_20261108_v1/analysis/comparison.json)
- [artifacts/geometry_cache/local_analysis/B_final_analysis_20261108_v1/analysis/validation.json](../../_shared/references/artifacts/geometry_cache/local_analysis/B_final_analysis_20261108_v1/analysis/validation.json)

## 연결

다른 관련 가설: [길이·유효 발화량과 앞뒤 저에너지 편향](../../01_duration_bias/README.md) · [정상 다양성·도메인 지지와 정상쌍 학습](../../03_bonafide_diversity/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
