<a id="e012-같은-768개-원음의-반복-여부를-두-모델에-교차-적용"></a>

# 같은 768개 원음의 반복 여부를 두 모델에 교차 적용

**실행 상태:** 추론·분석 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

같은 원음의 반복 여부에 대한 반응은 학습한 모델에 따라 달라질 수 있다.

## 그렇게 설계한 이유

학습 모델과 추론 반복 정책을 교차해 고정 모델의 입력 정책 민감도를 본다.

대조와 통제의 근거: 768×2 입력 ×2 모델. 큰 특징 추출 기와 두 분류기의 가중치를 바꾸지 않는다. 주기성만의 효과가 아니라 유효 길이·문맥을 포함한 반복 입력 정책의 총 효과다.

## 모델·데이터·표본

배치 240 원래 모델과 EBM-NoRepeat의 50epoch. 19LA·21LA·DF 각각 정상/위조 ×4 길이 구간에서 32개 씩, 총 768개 원음.

## 실험 내용과 상세 절차

1. 각 집합·클래스·길이 구간에서 발화 ID 해시와 화자별 번갈아 선택만으로 32개를 고르고 추론 전에 고정한다. 점수나 오류로 고르지 않는다.
2. 같은 원음 앞부분에서 native는 min(원음 길이,64,600), 반복 조건은 부족한 경우만 64,600까지 순환 반복한다.
3. 두 고정 모델 × 두 입력을 계산한다. pooling 잡음은 0으로 끄며 긴 원음의 두 입력·특징·점수 정확 일치를 확인한다.
4. 같은 발화의 점수·정상 기준 벡터 각도·최종 특징 차이와 개발 임계값 오류를 분석한다.

## 측정과 검증

발화 내 paired 차이와 화자 bootstrap 탐색 구간. 같은 길이 구간의 32개를 전체 분포의 오류율 추정에 사용하지 않는다.

## 결과

DF<2초 정상 32개에서 EBM-Repeat 모델은 반복을 없애면 평균 기준 각 52.364 →4.086°로 가까워졌고, EBM-NoRepeat 모델은 6.966 →43.190° 로 멀어졌다. 모델에 따라 반응이 반대였다.

## 해석·실패한 부분·한계

최근 전체 21LA 진단과 표본·잡음 조건을 구별한다. 두 GPU·두 입력·같은 발화를 독립 표본으로 합산하지 않는다.

## 배운 점

EBM-Repeat와 EBM-NoRepeat의 반응이 반대였다. 반복 입력을 모든 모델에 유해하거나 유익한 것으로 일반화할 수 없다.

## 실험 코드

- [학습 모델과 추론 반복 정책의 교차 개입 코드](code/geometry_B_intervention.py)
- [고정 길이 반복 학습과 비반복 학습의 결과 시각화 코드](code/artifacts/ebm_git_private_20261106/render_B_final_analysis.py)

코드의 파일별 원본 위치·SHA256은 [원본 파일 위치와 해시 목록](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [상세 설계·실험 기록](../../_shared/references/docs/experiments/ebm_B_final_results_20261108.md)
- [설정·결과 근거](../../_shared/references/artifacts/geometry_cache/local_analysis/B_final_analysis_20261108_v1/mechanism_2x2/protocol.json)
- [설정·결과 근거](../../_shared/references/artifacts/geometry_cache/local_analysis/B_final_analysis_20261108_v1/analysis/geometry_statistics.json)
- [집계 결과 표](../../_shared/references/artifacts/repetition_shortcut_20261111/cached_paired_interventions.csv)

## 연결

다른 관련 가설: [길이·유효 발화량과 앞뒤 저에너지 편향](../../01_duration_bias/README.md) · [잔여 방향·좌표·거리·기하와 변형 반응](../../05_residual_geometry/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
