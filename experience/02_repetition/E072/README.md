<a id="e072-특징-복제의-표준편차-배율과-평균표준편차-경로-교체"></a>

# 특징 복제의 표준편차 배율과 평균·표준편차 경로 교체

**실행 상태:** 실제 내부 개입·수학 검수 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

반복의 큰효과가 통계pooling 표준편차의 길이배율만으로 생겼을 수 있다.

## 그렇게 설계한 이유

반복 시 통계 pooling의 길이 배율만으로 큰 점수 이동이 설명되는지 검증한다.

대조와 통제의 근거: 동일 특징 복제와 원음 반복 전처리는 다르다. 숨은 특징 전체는 디스크에 저장하지 않고 스칼라 통계를 저장한다.

## 모델·데이터·표본

[출처별 같은 원음의 순환 반복·반사채움·비반복과 긴 원음 절단 대조](../E071/README.md)의 같은 471개·배치 160 모델. 원래 반복/비반복 두 forward의 최종 pooling 벡터와 비반복 XLS-R 특징을 메모리에서 사용한다.

## 실험 내용과 상세 절차

1. 원래 pooling이 평균 sum(w×X), 표준편차 sample_std(w×X +noise)를 계산함을 확인한다. 일반적인 가중 분산 식과 다르다.
2. 같은 비반복 XLS-R 특징을 시간 축에서 2/4회 복제해 음성 내용·특징 값은 고정하고 프레임 수만 늘린다. 이대수대조의 pooling 잡음은 0으로 끈다.
3. T 프레임을 k 회 복제했을 때 표준편차 복원 배율 sqrt(k(kT −1)/(T −1))을층 내 pooling 만·최종 pooling 만·둘 다에 적용한다. k4는 무보정/둘 다를 계산한다.
4. 실제 반복/비반복 forward의 최종 평균 μ·표준편차 σ를 서로 교체해 원래 둘·평균만교체·표준편차만 교체·둘 다의 네 조합을 원래 linear/LayerNorm에 넣는다.
5. 두 교체 순서의 평균으로 각 분기의 Shapley 기여를 계산하고 정확히 총 점수 변화와 합이 같은지 확인한다. 이 교체 분해는 원래 잡음 조건을 유지한다.

## 측정과 검증

복제 전후 점수·분산 복원 오차·μ/σ분기 기여·모델해시 불변.

## 결과

특징 2/4회 복제의 평균 절대 점수 변화 0.000236/0.000350. 두 pooling 분산 배율을 복원하면 약 1.4e-6 이내로 기준 점수로 돌아왔다. VCC2018 반복 →원음 점수 변화 1.262285 중 평균 분기 1.261679였다.

## 해석·실패한 부분·한계

단순 표준편차 길이 배율이 큰 효과의 주원인이라는 가설은 지지되지 않았다. 비현실적인 평균/표준편차 조합의 내부 함수 분해이며 음성인과 기여율로 일반화하지 않는다.

## 배운 점

특징복제의 작은효과는 보정됐지만 실제파형 반복의 큰이동은 평균경로가 지배했다. 단순분산배율 설명은 부족했다.

## 실험 코드

- [소스: pooling_mechanism_sidecar.py](code/artifacts/geometry_experiments/pooling_mechanism_sidecar.py)
- [소스: verify_padding_mechanisms.py](code/artifacts/geometry_experiments/verify_padding_mechanisms.py)

코드의 파일별 원본 위치·SHA256은 [원본 파일 위치와 해시 목록](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [소스: pooling_mechanism_sidecar.py](../../_shared/workspace/artifacts/geometry_experiments/pooling_mechanism_sidecar.py)
- [상세 설계·실험 기록](../../_shared/references/docs/experiments/ebm_padding_context_mechanism_20261106.md)
- [소스: verify_padding_mechanisms.py](../../_shared/workspace/artifacts/geometry_experiments/verify_padding_mechanisms.py)

## 연결

다른 관련 가설: [길이·유효 발화량과 앞뒤 저에너지 편향](../../01_duration_bias/README.md) · [코사인 외 norm과 정규화 전 통계](../../04_norm_information/README.md) · [잔여 방향·좌표·거리·기하와 변형 반응](../../05_residual_geometry/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
