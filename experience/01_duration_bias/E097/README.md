# E097. 같은 본문·총길이·에너지에서 앞뒤 저에너지 배치만 바꾼 실험

**실행 상태:** 기존 실제 결과 확인·문서화 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

본문·총길이·에너지를고정하고 앞뒤배치만 바꿔도 점수가 달라질 수 있다.

## 그렇게 설계한 이유

앞 구간이 긴 정상·짧은 위조라는 연관이 같은 내용에서도 점수를 움직이는지 본다.

대조와 통제의 근거: 원기록은 −cosine을 위조 점수로 사용했다. 여기서는 부호를 뒤집어 높은 값이 정상인 코사인으로 설명한다. 원본 대 가공본은 뒤 내용이 달라질 수 있어 주비교가 아니다.

## 모델·데이터·표본

배치 80·80epoch의 고정 모델(체크포인트 SHA256 30043199664bc84da780654e01fd430235f647b894e7832a52f777c7fee5939a). 별도 명시가 없으면 DF eval에서 seed 1로 정상·위조 각 1,000개를 클래스 안에서 중복 없이 무작위 추출한 같은 2,000개다. 추론 배치 4, worker 2, 원래 앞 64,600샘플·부족하면 순환 반복, 증강 없음이다.

## 실험 내용과 상세 절차

1. 원래 앞 저에너지 구간을 찾은 뒤 공통 본문을 최대 3.0175초로 고정하고 부족한 본문은 0으로 채운다.
2. 앞 0.06/0.48/1.02초와 뒤 0.96/0.54/0초를 조합해 전체 4.0375초·총 filler 1.02초를 유지한다. filler는 0 또는 해당 원음의 앞 저에너지 파형을 반복한 것이다.
3. 뒤 filler를 없애 전체 길이도 변하는 조건과, 시작점을 −40dB로 잡은 조건을 추가한다. 실제 앞구간 달성·없는 구간을 0으로 대체한 예외·원래 뒤 내용 손실을 검사한다.
4. 개발 정상·위조 각 1,000개에서 정한 코사인 0.816285938을 고정하고 각 조건의 오류와 대응 이동을 계산한다. 정상만/위조만/둘 다 변형한 네 조합과 클래스별 긴·짧은 배치 반전도 비교한다.
5. 후속 표는 같은 변형의 짧은→긴 이동, 원래 점수 꼬리/중심, 출처별 반응과 실제 앞구간 변화량으로 나눴다. 4개짜리 smoke는 본 표본과 독립 성능 결과가 아니다.
6. 보존된 13개 점수 조건에는 원본·동일 입력 재계산, −20dB 0 filler의 세 길이, −20dB 자체 filler의 세 길이, 가변 총길이의 0.06/0.48초 두 조건, −40dB 시작점 대조의 세 길이가 들어 있다. 가변 총길이 조건에 1.02초가 있다고 가정하지 않는다.

## 측정과 검증

동일 본문·길이·에너지 보존, 무변형 반복, 화자 군집 95% 구간, 조건별 EER/AUROC·고정 임계값 오류의 독립 검산. 표본 2,000개의 결과다.

## 결과

자체 저에너지 filler에서 앞 0.06→1.02초는 정상 코사인을 평균 +0.152437, 위조 +0.069454만큼 옮겼다. 정상 길게·위조 짧게 배치 EER 2.0%, 반대 배치 6.1%였다.

## 해석·실패한 부분·한계

하나의 고정 학습 모델과 탐색 표본의 결과다. 전체 공식 평가 EER, 여러 학습 seed의 효과, 손실을 제거한 재학습의 효과로 해석하지 않는다. 반복 관찰한 표본과 서로 겹치는 후속 분석은 독립 실험 표본으로 합산하지 않는다.

## 배운 점

가공고정길이 안의 위치효과를 봤다. 원음전체에 앞구간을추가한 엄격대조와 동일한실험이 아니다.

## 실험 코드

- [historical_analysis/leading_quiet_paired_21df.py](code/leading_quiet_paired_21df.py)
- [historical_analysis/leading_quiet_shift_tables.py](code/leading_quiet_shift_tables.py)
- [historical_analysis/leading_quiet_tail_followup.py](code/leading_quiet_tail_followup.py)

코드의 파일별 원본 위치·SHA256은 [source_manifest.json](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [artifacts/experiment_methods_audit_20261111/historical_receipts/EBM80_21DF_leading_quiet_paired.json](../../_shared/references/artifacts/experiment_methods_audit_20261111/historical_receipts/EBM80_21DF_leading_quiet_paired.json)
- [artifacts/experiment_methods_audit_20261111/historical_receipts/EBM80_21DF_leading_quiet_smoke.json](../../_shared/references/artifacts/experiment_methods_audit_20261111/historical_receipts/EBM80_21DF_leading_quiet_smoke.json)

## 연결

다른 관련 가설: [반복 입력과 학습·추론의 상호작용](../../02_repetition/README.md) · [정상 다양성·도메인 지지와 정상쌍 학습](../../03_bonafide_diversity/README.md) · [잔여 방향·좌표·거리·기하와 변형 반응](../../05_residual_geometry/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
