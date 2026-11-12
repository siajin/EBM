<a id="e053-정상위조-라벨로-대각-거리-가중치만-학습하는-대조"></a>

# 정상·위조 라벨로 대각 거리 가중치만 학습하는 대조

**실행 상태:** 추가 모듈 학습·전이 평가 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

분류 정답으로 대각 거리 가중치를 학습하면 비지도 공분산보다 유용할 수 있다.

## 그렇게 설계한 이유

공분산으로 계산한 거리와 달리 분류 정답을 사용해 거리 방향의 가중치를 배울 수 있는지 본다.

대조와 통제의 근거: 이것은 학습 자료 정상 공분산의 비지도 추정이 아니다. 분류 정답을 쓰는 160 파라미터 모듈이며 원래 중심과 encoder를 학습하지 않는다.

## 모델·데이터·표본

[고정 특징 위에서 정적·입력별 좌표 가중치와 같은 크기 신경망 학습](../E052/README.md)와 같은 학습 자료·개발·DF/21LA 자료와 seed 3개. 원래 160차원 특징·중심 고정.

## 실험 내용과 상세 절차

1. 각좌표 양의 가중치 a_j =160 exp(2tanh(v_j))/sum_l exp(2tanh(v_l))를 학습한다.
2. 점수 1 −0.5sum_j a_j(x_j −c_j)² 를 사용한다. trace160·조건 비율 ≤exp(4), 처음 a_j =1의 identity를 확인한다.
3. [고정 특징 위에서 정적·입력별 좌표 가중치와 같은 크기 신경망 학습](../E052/README.md)와 같은 두 클래스 OC 목적·0.01 규제·Adam·개발로그 손실·epoch0 선택·조기 중단을 쓴다.

## 측정과 검증

선택 epoch·거리 spectrum·seed 별 전이 EER/로그손실·native 차이.

## 결과

세 seed 모두 개발에서 epoch0이 선택돼 A =I, native와 점수가 같았다.

## 해석·실패한 부분·한계

선택된 함수가 같은 대수적 결과다. [0,0]차이 구간을 모든 거리 학습의 무효성에 대한 정밀한 통계 증거로 쓰지 않는다.

## 배운 점

세seed 모두epoch0이 선택돼 원래 점수와 같았다. 같은 함수의0차이를 전체 방법군의 무효성 증거로 과장하지 않는다.

## 실험 코드

- [소스: frozen_heads.py](code/artifacts/geometry_experiments/frozen_heads.py)

코드의 파일별 원본 위치·SHA256은 [원본 파일 위치와 해시 목록](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [소스: frozen_heads.py](../../_shared/workspace/artifacts/geometry_experiments/frozen_heads.py)
- [집계 결과 표](../../_shared/references/artifacts/geometry_cache/local_analysis/frozen_all_v1/head_metrics.csv)
- [설정·결과 근거](../../_shared/references/artifacts/geometry_cache/local_analysis/frozen_all_v1/head_paired_effects.json)

## 연결

다른 관련 가설: [정상 다양성·도메인 지지와 정상쌍 학습](../../03_bonafide_diversity/README.md) · [코사인 외 norm과 정규화 전 통계](../../04_norm_information/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
