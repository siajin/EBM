# E105. 앞 저에너지의 존재·지속 길이 규칙과 모델 점수의 관계

**실행 상태:** 기존 실제 결과 확인·문서화 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

앞저에너지의단순존재보다 지속길이가 클래스와 더연결될 수 있다.

## 그렇게 설계한 이유

길이만으로 라벨을 예측할 수 있는지와 모델이 그 규칙으로 작동하는지를 나눈다.

대조와 통제의 근거: 단순 규칙의 균형 정확도는 EBM 정확도나 단서 사용 비율이 아니다. VCC를 새로 평가하지 않았다.

## 모델·데이터·표본

고정 80epoch 모델의 ASV 출처 원음 전체 캐시 정상 1,050·위조 248개와 학습 원음 정상·위조 각 1,000개. 새 추론 없음.

## 실험 내용과 상세 절차

1. ASVspoof 출처 정상 1,050개·위조 248개의 기존 전체 원음 점수와 −20dB 앞 길이를 연결한다.
2. 20/100/500ms 이상이면 정상이라는 세 규칙을 결과에 맞춰 최적화하지 않고 고정한다. 클래스별 존재 비율과 균형 정확도를 계산한다.
3. −40dB 민감도, 공격별 길이·점수, 0.9 위아래 점수층을 분석한다.
4. 학습 정상·위조 각 1,000개에서도 원음 전체·앞 64,600샘플의 존재 비율을 계산한다. 실제 랜덤 증강 입력의 결과와 구별한다.

## 측정과 검증

발화별 대응 점수·원래 160차원 코사인·클래스별 분모와 저장 검산을 확인한다.

## 결과

학습 원음의 균형 정확도는 20/100/500ms에서 50.25/56.90/74.45%, ASV 평가 표본은 약 53.94/77.38/86.63%였다. 존재만으로 구분되는 학습 분포는 아니었다.

## 해석·실패한 부분·한계

하나의 고정 학습 모델과 탐색 표본의 결과다. 전체 공식 평가 EER, 여러 학습 seed의 효과, 손실을 제거한 재학습의 효과로 해석하지 않는다. 반복 관찰한 표본과 서로 겹치는 후속 분석은 독립 실험 표본으로 합산하지 않는다.

## 배운 점

단순존재는약했고 긴문턱의예측력은컸다. 그규칙이 곧 EBM의실제판정규칙인 것은 아니다.

## 실험 코드

- [historical_analysis/asv_quiet_presence_hypothesis.py](code/asv_quiet_presence_hypothesis.py)
- [historical_analysis/asv_train_presence_context.py](code/asv_train_presence_context.py)

코드의 파일별 원본 위치·SHA256은 [source_manifest.json](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [artifacts/experiment_methods_audit_20261111/historical_receipts/EBM80_ASV_quiet_presence_hypothesis_20261030.json](../../_shared/references/artifacts/experiment_methods_audit_20261111/historical_receipts/EBM80_ASV_quiet_presence_hypothesis_20261030.json)

## 연결

다른 관련 가설: [반복 입력과 학습·추론의 상호작용](../../02_repetition/README.md) · [정상 다양성·도메인 지지와 정상쌍 학습](../../03_bonafide_diversity/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
