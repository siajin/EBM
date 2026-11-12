# E011. 원음 길이별 오류와 임계값 민감도 분석

**실행 상태:** 기존 점수 분석 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

전체 EER 개선이 특정 길이 정상의 운용 오류 개선까지 보장하지는 않을 수 있다.

## 그렇게 설계한 이유

전체 EER 개선이 짧은 정상의 개선을 의미하는지, 임계값 선택에 따라 결론이 달라지는지 본다.

대조와 통제의 근거: 각 길이마다 평가 정답으로 새 임계값을 정하지 않는다. <2초 와 이후 ≤2초 실험의 분모를 혼용하지 않는다.

## 모델·데이터·표본

[E009](../../02_repetition/E009/README.md)의 같은 전체 점수. 길이는 반복·절단 전 원음 기준이다.

## 실험 내용과 상세 절차

1. 원음 길이를 [0,2),[2,3.0275),[3.0275,4.0375),[4.0375, 무한대)초로 나눈다. 경계는 16kHz 정수 샘플로 판단한다.
2. 각 모델의 전체 개발 자료에서 |위조 통과율 −정상 거부율 | 최소, 동률이면 낮은 위조 통과율·높은 임계값 순으로 한 기준을 정한다.
3. 그 한 임계값을 모든 평가 길이에 그대로 적용하고 정상/위조의 분모를 따로 세어 오류율·오류 전이를 구한다.
4. 추가 탐색으로 개발 위조 통과 예산 0.1%와 1% 규칙의 임계값을 다시 정해 DF<2초 정상 결과를 비교한다.

## 측정과 검증

길이 × 클래스별 오류, 동일 발화의 맞음/틀림 전환. 길이별 구간은 화자 5,000회 탐색적 95%이며 전체 다중 비교 동시 구간과 구별한다.

## 결과

DF의 2초 미만 정상 거부는 원래 배치 240 모델에서 155/1,368개, B에서 355/1,368개였다. 같은 길이의 위조 통과는 14,432/115,447개에서 1,994/115,447개로 줄었다. 개발 위조 통과 예산을 바꾸면 정상 오류 차이의 크기도 달라졌다.

## 해석·실패한 부분·한계

정상만 덜/더 거부한 결과를 분리력 개선/악화로 대체하지 않는다. 고정 개발 임계값의 도메인 전이와 EER은 다른 질문이다.

## 배운 점

DF 단음 정상 거부 증가와 위조 누락 감소가 공존했다. 개발 고정 임계값 오류와 길이 내부 EER을 같이 보아야 한다.

## 실험 코드

- [artifacts/ebm_git_private_20261106/render_B_final_analysis.py](code/artifacts/ebm_git_private_20261106/render_B_final_analysis.py)
- [EBM/tools/evaluate_B_C_checkpoints.py](code/EBM/tools/evaluate_B_C_checkpoints.py)

코드의 파일별 원본 위치·SHA256은 [source_manifest.json](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [docs/experiments/ebm_B_final_results_20261108.md](../../_shared/references/docs/experiments/ebm_B_final_results_20261108.md)
- [artifacts/geometry_cache/local_analysis/B_final_analysis_20261108_v1/analysis/threshold_sensitivity.csv](../../_shared/references/artifacts/geometry_cache/local_analysis/B_final_analysis_20261108_v1/analysis/threshold_sensitivity.csv)
- [artifacts/ebm_B_C_checkpoints_20261109/LA_analysis/length_summary.csv](../../_shared/references/artifacts/ebm_B_C_checkpoints_20261109/LA_analysis/length_summary.csv)

## 연결

다른 관련 가설: [정상 다양성·도메인 지지와 정상쌍 학습](../../03_bonafide_diversity/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
