# 원음 길이별 오류와 임계값 민감도 분석

**실행 상태:** 기존 점수 분석 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

전체 EER 개선이 특정 길이 정상의 운용 오류 개선까지 보장하지는 않을 수 있다.

## 그렇게 설계한 이유

전체 EER 개선이 짧은 정상의 개선을 의미하는지, 임계값 선택에 따라 결론이 달라지는지 본다.

대조와 통제의 근거: 각 길이마다 평가 정답으로 새 임계값을 정하지 않는다. <2초 와 이후 ≤2초 실험의 분모를 혼용하지 않는다.

## 모델·데이터·표본

[두 고정 50epoch 모델을 같은 비반복 입력으로 전체 평가](../../02_repetition/E009/README.md)의 같은 전체 점수. 길이는 반복·절단 전 원음 기준이다.

## 실험 내용과 상세 절차

1. 원음 길이를 [고정 길이 반복 학습과 비반복 학습의 결과 시각화 코드](code/artifacts/ebm_git_private_20261106/render_B_final_analysis.py)
- [두 비반복 학습의 전체 저장점 평가 코드](code/EBM/tools/evaluate_B_C_checkpoints.py)

코드의 파일별 원본 위치·SHA256은 [원본 파일 위치와 해시 목록](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [상세 설계·실험 기록](../../_shared/references/docs/experiments/ebm_B_final_results_20261108.md)
- [집계 결과 표](../../_shared/references/artifacts/geometry_cache/local_analysis/B_final_analysis_20261108_v1/analysis/threshold_sensitivity.csv)
- [집계 결과 표](../../_shared/references/artifacts/ebm_B_C_checkpoints_20261109/LA_analysis/length_summary.csv)

## 연결

다른 관련 가설: [정상 다양성·도메인 지지와 정상쌍 학습](../../03_bonafide_diversity/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
