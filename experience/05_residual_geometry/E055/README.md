# E055. 두 평가 자료 내부의 방향·정규화 전 특징·같은 차원 잡음 대조 9 종

**실행 상태:** 추가 분류기·내부 교차검증 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

독립 화자CV 안에서는 방향·정규화 전 특징의 추가 판독 정보를 검출할 수 있다.

## 그렇게 설계한 이유

전이 개선과 분리해 현재 특징에서 읽을 수 있는 정보와 정규화 전 통계의 증분을 검사한다.

대조와 통제의 근거: 모델은 [E054](../E054/README.md)와 같은 단조 기본 식·hidden16. 정답을 사용한 평가 자료 내부 분류이므로 배포용 학습 자료 전이 수치가 아니다. DF의 VCC2018 위조 화자 4 그룹뿐이라 5fold 모든 cell에 존재할 수 없다.

## 모델·데이터·표본

DF 진단 20,000과 21LA 진단 20,000을 따로 학습·검증한다. 두 자료를 합쳐 fit 하지 않는다.

## 실험 내용과 상세 절차

1. 각 자료에 StratifiedGroupKFold 바깥 5·안쪽 3,seed 11/12를 고정한다. 그룹은 출처:: 화자, 층화는 출처:: 클래스다.
2. 입력 9 종은 [E054](../E054/README.md)의 기본 식/선형 방향/선형 잡음/신경망 방향/신경망 잡음에, 기본 식 +방향 +h 평균·log 표준편차 2개, 방향 +잡음 2개, 방향 +h160개, 방향 +잡음 160개를 더 한 것이다.
3. 각 variant×seed 101/102/103× 바깥 fold 에서 ridge1e-4/1e-2의 안쪽 3개 개발 학습 곡선이 공통으로 존재하는 epoch 까지만 평균한다.
4. 안쪽 평균 LL로 ridge와 epoch를 고르고 바깥 학습 부분만 해당 epoch 수로 다시 적합한다. scaler·결측처리·fit 통계는 해당 학습 부분에서만 만든다.
5. 각 바깥 평가 발화 예측을 한번씩 저장하고 같은 fold·seed의 기본/잡음대조와 짝 지어 비교한다.

## 측정과 검증

입력별 seed 평균 EER/LL·같은 fold paired 효과.10,000회 화자 구간은 고정 OOF 예측에 조건부이며 매번 전체 분류기를 다시 적합 하지 않는다.

## 결과

DF 내부 기본 EER 3.344444%, 방향 선형 2.303704%, 방향 신경망 1.662963%, 방향 +h160은 1.944444%.21LA는 기본 0.8%, 방향 신경망 0.75%였다.

## 해석·실패한 부분·한계

내부 판별 신호가 있다는 진단과 새 도메인 일반화는 별개다. 출처 × 클래스 누락 제한을 표시하며, 사용자가 취소한 4×3 보완 학습은 실행 하지 않았다.

## 배운 점

내부CV 이득은 존재하지만 Train→미지 도메인 전이와 다르다. 출처×클래스의fold 공급 부족도 남는다.

## 실험 코드

- [artifacts/geometry_experiments/frozen_probes.py](code/artifacts/geometry_experiments/frozen_probes.py)

코드의 파일별 원본 위치·SHA256은 [source_manifest.json](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [artifacts/geometry_experiments/frozen_probes.py](../../_shared/workspace/artifacts/geometry_experiments/frozen_probes.py)
- [docs/experiments/ebm_condition_audit_20261105.md](../../_shared/references/docs/experiments/ebm_condition_audit_20261105.md)
- [artifacts/geometry_cache/local_analysis/frozen_all_v1/P3_P4/B/DF_pilot/complete.json](../../_shared/references/artifacts/geometry_cache/local_analysis/frozen_all_v1/P3_P4/B/DF_pilot/complete.json)
- [artifacts/geometry_cache/local_analysis/frozen_all_v1/P3_P4/B/21LA_pilot/complete.json](../../_shared/references/artifacts/geometry_cache/local_analysis/frozen_all_v1/P3_P4/B/21LA_pilot/complete.json)

## 연결

다른 관련 가설: [정상 다양성·도메인 지지와 정상쌍 학습](../../03_bonafide_diversity/README.md) · [코사인 외 norm과 정규화 전 통계](../../04_norm_information/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
