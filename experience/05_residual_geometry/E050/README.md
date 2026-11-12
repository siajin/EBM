<a id="e050-방향-벡터-전체의-조건별-분산과-길이패딩-연관"></a>

# 방향 벡터 전체의 조건별 분산과 길이·패딩 연관

**실행 상태:** 캐시 통계 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

잔여 방향의 전체 분산은 길이·패딩·출처에 체계적으로 연결될 수 있다.

## 그렇게 설계한 이유

군집 번호로 줄이지 않고 전체 옆 방향 벡터가 자료 조건과 어떻게 연결되는지 본다.

대조와 통제의 근거: 이 항목은 k-means NMI와 다른 지표다. 길이와 패딩은 고정전 처리 규칙으로 연결돼 독립 두 원인으로 세지 않는다.

## 모델·데이터·표본

배치 160 고정 특징, DF/21LA 진단 각 20,000. PCA 축은 학습 자료 방향으로만 fit 한다.

## 실험 내용과 상세 절차

1. 정상/위조 각각 출처·코덱·전송 조건·원음 길이·패딩분율로 나눈다. 패딩 bin 경계는 0/0.25/0.5다.
2. 각 조건의 평균 방향 차이 제곱 합을 전체 방향 분산 제곱 합으로 나눈 vector eta² 를 계산한다.
3. 학습 자료에서 fit 한 PCA 상위 축과 원음 길이·패딩분율의 Spearman 상관을 계산한다.
4. 셀별 표본 수·화자 수와 N<20·화자<5 표시, 원래 점수 오류를 모두 저장한다. 한 범주만 있는 층의 분산 비 0은 비교 변이 없음으로 설명한다.

## 측정과 검증

vector eta²·순위 상관·방향 집중도·조건별 EER/오류·coverage.

## 결과

DF 정상 출처 eta²0.307754, DF 위조 패딩 0.163591,21LA 정상 코덱 0.168577, 위조 패딩 0.158263. 위조 PCA1-패딩 상관은 DF −0.600272,21LA −0.577569.

## 해석·실패한 부분·한계

조건을 조작했을 때 제거되는 인과적 분산 비율이 아니다. 축 부호는 임의이며 측정하지 않은 조건은 남는다.

## 배운 점

큰 연관은 확인했지만 길이와패딩은 규칙으로 연결된다. 관찰 분산 비율을 제거 가능한 인과 비율로 부르지 않는다.

## 실험 코드

- [소스: frozen_n0.py](code/artifacts/geometry_experiments/frozen_n0.py)

코드의 파일별 원본 위치·SHA256은 [원본 파일 위치와 해시 목록](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [소스: frozen_n0.py](../../_shared/workspace/artifacts/geometry_experiments/frozen_n0.py)
- [집계 결과 표](../../_shared/references/artifacts/geometry_cache/local_analysis/frozen_all_v1/F4/vector_variance.csv)
- [집계 결과 표](../../_shared/references/artifacts/geometry_cache/local_analysis/frozen_all_v1/F4/padding_duration.csv)
- [집계 결과 표](../../_shared/references/artifacts/geometry_cache/local_analysis/frozen_all_v1/F4/coverage.csv)

## 연결

다른 관련 가설: [길이·유효 발화량과 앞뒤 저에너지 편향](../../01_duration_bias/README.md) · [반복 입력과 학습·추론의 상호작용](../../02_repetition/README.md) · [정상 다양성·도메인 지지와 정상쌍 학습](../../03_bonafide_diversity/README.md) · [코사인 외 norm과 정규화 전 통계](../../04_norm_information/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
