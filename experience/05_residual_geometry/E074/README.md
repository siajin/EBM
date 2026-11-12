# E074. 틀린 정상의 옆 방향이 학습 정상 주축에 집중되는지 분석

**실행 상태:** 기존캐시 통계·독립 검수 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

DF 정상오류의 잔여방향이 Train정상의 주요축에 과도하게 집중될 수 있다.

## 그렇게 설계한 이유

정상 오류가 학습 정상의 주요 변동 방향에 과도하게 놓이는지 표현 크기와 분리해 본다.

대조와 통제의 근거: 다음 저점유 제거는 unit 옆 방향의 비중심 2차 모멘트를 사용하므로 이 중심화 r 공분산과 다르다. 현재 batch240 학습 모델 결과가 아니다.

## 모델·데이터·표본

배치 160·50epoch 고정 모델. 학습 자료 정상 2,580개로 기저 fit, 개발 선택 부분으로 임계값 선택, DF/21LA 진단과 19LA/21LA/DF 전체의 정상만 분석.

## 실험 내용과 상세 절차

1. r =x −(x ᵀc)c를 구하고 학습 자료 정상 r의 평균을 뺀 공분산에서 주축을 구한다. 주 상위 8축과 민감도 1/2/4/16축은 평가 정답을 보지 않고 정한다.
2. 각 정상의 비중 q =||P_kr ||²/||r ||² 를 계산해 절대 잔여 크기와 구별한다. 임계값은 개발에서 EER 근사 규칙으로 정한 0.5093649528766109다.
3. 틀린 정상과 수락 정상의 q를 비교하고 원음 길이 0/1/2/3/4/6/10/무한대 초 × 출처 × 코덱 공 통 셀에서 오류 가중 차이를 계산한다. 셀당오류 ≥1·수락 ≥2를 요구한다.
4. 화자 군집 2,000회 구간과 화자까지 일치한 엄격 민감도, 점수 10 분위·음수 점수 제한·개별 축·같은 표본 수의 방향 집중도 비교를 한다.
5. 기존 임계값 0.9883067 민감도는 주 계산 후 보조로 정했다고 밝힌다. 평가 자료에서 주 기저를 fit 하지 않는다.

## 측정과 검증

주축 에너지 비중·방향 평균 길이·유효 rank·조건 보정 q 차이·coverage·bootstrap 구간.

## 결과

DF 정상 오류 2,955개 의 PC1 비중 65.24%대수락 18.05%, 상위 8축 보정 차이 +5.02%p[2.29,7.49].21LA는 −5.13%p[−6.84, −3.07]. DF 오류 중 2,506개는 코사인이 음수였다.

## 해석·실패한 부분·한계

DF 특정 심한 도메인 변화와 연관된 관측이다. 모든 평가에서 정상 오류의 같은 방향 집중이 성립하지 않고 방향 억제 학습의 효과를 시험하지 않았다.

## 배운 점

DF와21LA의 조건보정 차이 부호가 달랐다. 모든도메인에 같은방향 억제를 적용할 근거는 부족하다.

## 실험 코드

- [EBM/tools/analyze_bona_tangent_errors.py](code/EBM/tools/analyze_bona_tangent_errors.py)
- [EBM/tools/verify_bona_tangent_errors.py](code/EBM/tools/verify_bona_tangent_errors.py)

코드의 파일별 원본 위치·SHA256은 [source_manifest.json](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [artifacts/geometry_cache/local_analysis/bona_tangent_errors_20261111_v1/plan.json](../../_shared/references/artifacts/geometry_cache/local_analysis/bona_tangent_errors_20261111_v1/plan.json)
- [artifacts/geometry_cache/local_analysis/bona_tangent_errors_20261111_v1/findings.json](../../_shared/references/artifacts/geometry_cache/local_analysis/bona_tangent_errors_20261111_v1/findings.json)
- [artifacts/geometry_cache/local_analysis/bona_tangent_errors_20261111_v1/independent_validation.json](../../_shared/references/artifacts/geometry_cache/local_analysis/bona_tangent_errors_20261111_v1/independent_validation.json)

## 연결

다른 관련 가설: [정상 다양성·도메인 지지와 정상쌍 학습](../../03_bonafide_diversity/README.md) · [코사인 외 norm과 정규화 전 통계](../../04_norm_information/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
