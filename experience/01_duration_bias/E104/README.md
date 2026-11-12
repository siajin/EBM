# E104. 자연적인 앞 저에너지 길이를 읽는 특징 방향: −20dB와 안정성

**실행 상태:** 기존 실제 결과 확인·문서화 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

저에너지정의와화자분할을 바꿔도 길이정보가 어느정도 유지될 수 있다.

## 그렇게 설계한 이유

앞 길이 정의와 화자 분할을 바꿨을 때 정보·축 방향의 안정성을 확인한다.

대조와 통제의 근거: 축을 매번 보류 표본에 맞추지 않는다. 여러 분할의 예측을 한 축의 공간 좌표라고 그리지 않는다.

## 모델·데이터·표본

[E103](../E103/README.md)과 같은 기존 원음 전체 특징. 새 음성 추론·EBM 재학습 없이 회귀·분할·조건 조정만 수행했다.

## 실험 내용과 상세 절차

1. [E103](../E103/README.md)과 같은 1,050개·화자 분리를 쓰되 기준을 최대 RMS의 10% 미만인 −20dB로 정한다.
2. 주 조건 조정에는 코덱·뒤 저에너지 길이·활성 RMS와 제곱만 넣는다. 목표 앞 길이에서 뺀 파생 길이는 넣지 않는다. 전체 길이와 제곱은 보조로만 조정한다.
3. 같은 주 축을 유지하며 길이 정의만 −30/−40/−50dB로 바꿔 상관을 계산한다.
4. 다른 화자 다섯 분할의 축·예측 상관·R², 단순 평균 변화 방향, 학습 PCA, 양의 길이와 1.5초 이하 범위를 검토한다.

## 측정과 검증

발화별 대응 점수·원래 160차원 코사인·클래스별 분모와 저장 검산을 확인한다.

## 결과

주 보류 길이−축 상관 0.6127, 조정 후 0.5731, 길이−정상 기준 코사인 −0.1100이었다. 분할별 축과 주 축의 방향 코사인은 0.097~0.872로 달랐고 길이 정보가 유일한 물리축이라는 결론은 없었다.

## 해석·실패한 부분·한계

하나의 고정 학습 모델과 탐색 표본의 결과다. 전체 공식 평가 EER, 여러 학습 seed의 효과, 손실을 제거한 재학습의 효과로 해석하지 않는다. 반복 관찰한 표본과 서로 겹치는 후속 분석은 독립 실험 표본으로 합산하지 않는다.

## 배운 점

예측상관은남았으나 축방향은불안정했다. 유일한물리적길이축이라는 해석은 근거가부족하다.

## 실험 코드

- [historical_analysis/natural_lowenergy_axis_db20.py](code/natural_lowenergy_axis_db20.py)
- [historical_analysis/verify_natural_lowenergy_axis.py](code/verify_natural_lowenergy_axis.py)

코드의 파일별 원본 위치·SHA256은 [source_manifest.json](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [artifacts/experiment_methods_audit_20261111/historical_receipts/EBM80_ASV_bona_natural_lowenergy_axis_db20.json](../../_shared/references/artifacts/experiment_methods_audit_20261111/historical_receipts/EBM80_ASV_bona_natural_lowenergy_axis_db20.json)
- [artifacts/experiment_methods_audit_20261111/historical_method_source_checks.json](../../_shared/references/artifacts/experiment_methods_audit_20261111/historical_method_source_checks.json)

## 연결

다른 관련 가설: [정상 다양성·도메인 지지와 정상쌍 학습](../../03_bonafide_diversity/README.md) · [코사인 외 norm과 정규화 전 통계](../../04_norm_information/README.md) · [잔여 방향·좌표·거리·기하와 변형 반응](../../05_residual_geometry/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
