# E013. 원래 배치 240 모델의 반복 구간과 오류 분포 조사

**실행 상태:** 기존 기록 분석 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

반복이 필요한 발화에서 오류가 많이 관측되는 이유는 노출 비중과 반복 강도로 나눠 설명해야 한다.

## 그렇게 설계한 이유

틀린 음성에 반복이 많다는 관찰을 전체 반복 노출 비율과 구별한다.

대조와 통제의 근거: 경계<64,600샘플. 후속 자연 단음 분석의 ≤32,000샘플 여부와 다른 조건이다.

## 모델·데이터·표본

원래 배치 240 모델의 보존 점수·원음 길이와 원본 반복 전처리 기록. 학습을 바꾸지 않는다.

## 실험 내용과 상세 절차

1. 원음이 64,600 미만이면 반복이 필요하다는 지시자를 만들고, 실제 입력의 고유 음성 비율 min(N,64,600)/64,600도 계산한다.
2. 정상/위조별 반복 비율, 오류 집단의 반복 비율, 같은 반복 집단 안의 오류율을 따로 계산한다.
3. 학습 자료와 19LA/21LA/DF 평가 자료를 구분하며 단순 반복 유무와 원음이 매우 짧아 여러 번 반복되는 정도를 구별한다.
4. 초기 전체 길이 조사의 반복량은 목표 길이 M과 원음 길이 n에 대해 부족 비율 max(0, M −n)/M, 접합 횟수 floor((M −1)/n)로 계산했다. 긴 입력과 짧은 입력을 각각 조사했다. 발화 파일을 읽어 반복 신호를 탐지한 값이 아니라, 입력 생성 규칙에서 계산한 반복량이다.
5. 초기 서버 전수 조사도 다시 확인했다. 다섯 자료의 803,565개 파일 header와 실제 증강 추출의 반복량, 저에너지와 반복의 교차 분포를 사용했다. 반복 여부만으로 라벨을 예측한 규칙은 모델의 실제 사용 여부와 구별했다.

## 측정과 검증

클래스별 분모·평균 반복 비율·길이 구간 오류 집중. 인과적인 새 학습 실험이 아니다.

## 결과

21LA 정상의 78.3275%가 반복 필요 집단이다. 후속 전체 진단의 고정 native 임계값에서 반복 후 정상 오류의 74.8858%가 이 집단에 있었다. 다수 오류가 반복이라는 사실만으로 과대 표집 위험을 증명하지 못한다.

## 해석·실패한 부분·한계

단순 개수 비와 위험 비를 혼동하지 않는다. 반복 제거 학습의 효과는 이 관찰 분석에서 확인하지 않았다.

## 배운 점

오류 대부분이 반복 집단이라는 수치만으로 위험 증가를 주장할 수 없다. 집단 분모와 반복 정도가 필요하다.

## 실험 코드

- [EBM/tools/audit_original_repetition.py](code/EBM/tools/audit_original_repetition.py)
- [artifacts/padding_analysis_20261030/analyze_padding.py](code/artifacts/padding_analysis_20261030/analyze_padding.py)
- [artifacts/spoof_error_audit_20261108/analyze.py](code/artifacts/spoof_error_audit_20261108/analyze.py)

코드의 파일별 원본 위치·SHA256은 [source_manifest.json](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [artifacts/original_repetition_20261108/results.json](../../_shared/references/artifacts/original_repetition_20261108/results.json)
- [artifacts/repetition_shortcut_20261111/original_length_distribution.csv](../../_shared/references/artifacts/repetition_shortcut_20261111/original_length_distribution.csv)
- [artifacts/repetition_shortcut_20261111/reviewed_analysis/normal_repeat_groups.csv](../../_shared/references/artifacts/repetition_shortcut_20261111/reviewed_analysis/normal_repeat_groups.csv)
- [artifacts/padding_analysis_20261030/analyze_padding.py](../../_shared/workspace/artifacts/padding_analysis_20261030/analyze_padding.py)
- [artifacts/spoof_error_audit_20261108/analyze.py](../../_shared/workspace/artifacts/spoof_error_audit_20261108/analyze.py)
- [artifacts/experiment_methods_audit_20261111/historical_receipts/EBM_padding_length_20261030_v1.json](../../_shared/references/artifacts/experiment_methods_audit_20261111/historical_receipts/EBM_padding_length_20261030_v1.json)

## 연결

다른 관련 가설: [길이·유효 발화량과 앞뒤 저에너지 편향](../../01_duration_bias/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
