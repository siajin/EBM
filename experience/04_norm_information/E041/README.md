<a id="e041-초기-학습-자료-분류기와-선택-좌표-분류기의-df-전이-평가"></a>

# 초기 학습 자료 분류기와 선택 좌표 분류기의 DF 전이 평가

**실행 상태:** 추가 분류기·전이 평가 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

Train에서 읽은 방향·좌표 정보가 다른 DF 출처로 전이될 수 있다.

## 그렇게 설계한 이유

개발에서 잘 맞는 사후 분류기가 다른 출처의 DF 에도 일반화하는지 본다.

대조와 통제의 근거: 추가 classifier가 없는 기존 cosine, 선택 좌표에 학습한 새 분류기, 선택 좌표의 기존 cosine은 서로 다른 실험이다.

## 모델·데이터·표본

고정 배치 160 모델의 학습 자료·개발 선택 부분과 DF 진단 20,000개. DF는 정상 2,000·위조 18,000을 클래스별 ID 해시로 뽑았고 110 공격·9 코덱을 포함한다.

## 실험 내용과 상세 절차

1. [초기 개발 자료에서 점수·크기·옆 방향 특징을 읽는 분류기 비교](../E040/README.md)의 다섯 점수/크기/방향/잡음 입력을 학습 자료에서 fit 한다. 원음 전체 비반복 학습은 개발 전체에서 클래스 균형로그 손실 최소로 하나 선택한다.
2. 추가로 raw z160개 전부, 학습 자료 공격별 Fisher 상위 40개, 고정 seed 200~219의 무작위 40개 각각에 같은 L2 logistic 분류기를 학습한다.
3. Fisher는 정상과 각 공격의 평균 차이 제곱을 두 분산 합 +1e-8로 나눈 뒤 공격별 평균한다. 개발/DF로 좌표 순위를 선택하지 않는다.
4. 정상 unit 특징 공분산의 대각/전체·수축 0.01/0.1/0.5/1 거리 후보도 고정한다. 모든 설정과 threshold를 DF 평가 전에 잠근다.

## 측정과 검증

DFEER·AUROC·로그손실·개발 고정 오류, 출처 × 화자 93 그룹 bootstrap. DF 에서 fit 하거나 원음 전체 비반복 학습을 선택하지 않는다.

## 결과

DF 기준 cosine3.35%에 비해 초기 점수 spline17.10%, 방향 추가 11.238889%,raw 전체 좌표 분류기 13.05%, Fisher 상위 40 분류기 5.311111%로 대부분 전이 악화했다.

## 해석·실패한 부분·한계

DF 내부에서 학습한 후속 분류기의 낮은 EER과 비교해 같은 방법으로 부르지 않는다. 같은 자료·예측을 재계산한 해시 폴더 4개는 독립 실험 4회가 아니다.

## 배운 점

내부 판독 가능성과 전이는 달랐다. 개발에서 좋아 보인 추가 분류기 대부분이 DF에서 악화했다.

## 실험 코드

- [소스: df_screen.py](code/artifacts/geometry_experiments/df_screen.py)
- [소스: fast_analyze.py](code/artifacts/geometry_experiments/fast_analyze.py)

코드의 파일별 원본 위치·SHA256은 [원본 파일 위치와 해시 목록](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [소스: df_screen.py](../../_shared/workspace/artifacts/geometry_experiments/df_screen.py)
- [소스: fast_analyze.py](../../_shared/workspace/artifacts/geometry_experiments/fast_analyze.py)
- [상세 설계·실험 기록](../../_shared/references/artifacts/geometry_experiments/README.md)
- [집계 결과 표](../../_shared/references/artifacts/geometry_cache/local_analysis/6ac3505da0708324/metrics.csv)
- [설정·결과 근거](../../_shared/references/artifacts/geometry_cache/local_analysis/7f2722d6dad62841/verification.json)
- [설정·결과 근거](../../_shared/references/artifacts/geometry_cache/local_analysis/9e2d4fccadfa67a5/verification.json)

## 연결

다른 관련 가설: [길이·유효 발화량과 앞뒤 저에너지 편향](../../01_duration_bias/README.md) · [잔여 방향·좌표·거리·기하와 변형 반응](../../05_residual_geometry/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
