# E101. 엄격 원음 추가 실험의 고정 평가 입력·반복 재현·경험적 정상 기준 대조

**실행 상태:** 기존 실제 결과 확인·문서화 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

엄격추가실험의반응은 입력범위·재현잡음·중심선택에 영향을받을 수 있다.

## 그렇게 설계한 이유

원음 전체와 기존 고정 입력의 차이, 추론 재현성, 학습 기준 방향 의존성을 따로 점검한다.

대조와 통제의 근거: 경험적 기준은 별도 사후 진단이다. 12개 반복은 새 독립 음성 60개로 세지 않는다.

## 모델·데이터·표본

[E100](../E100/README.md)의 저장 표본과 같은 고정 80epoch 모델. 추가 입력 범위 평가·12개 파일 5회 재현·저장 특징의 기준 교체만 사용한다.

## 실험 내용과 상세 절차

1. [E100](../E100/README.md)의 새 파일에서 기존 앞부분·순환 반복 64,600샘플 입력 전체를 보존한 채 앞/뒤에 1초 0 또는 자체 −40dB 파형을 추가한다. 원음 전체 조건과 같은 적합 표본에서 비교한다.
2. 새 출처·라벨 각 층의 첫 두 ID, 총 12개 파일을 5회 반복 계산한다. 주효과와 점수의 반복 범위를 기록하며 원래 epsilon 규칙은 바꾸지 않는다.
3. 이전 표본 정상 1,000개의 원음 전체 특징 평균을 길이 1로 만든 경험적 기준을 고정해 파일 ID가 겹치지 않는 새 표본에 적용한다. 변형 후 기준을 다시 계산하지 않는다. 이 기준과 학습 기준의 코사인은 0.988079766이었다.
4. 원행 코사인·정규화·PCM 중복·에너지·샘플 보존을 독립 검산하고 주비교 8개의 판단과 접합·용량 단조성 결과를 재확인한다.

## 측정과 검증

발화별 대응 점수·원래 160차원 코사인·클래스별 분모와 저장 검산을 확인한다.

## 결과

저장 검수는 all_checks_passed=true, 주량 독립 일치·방향 판정 일치를 기록했다. 이 검수 통과는 보편적인 앞 저에너지 효과나 성능 개선의 증거가 아니다.

## 해석·실패한 부분·한계

하나의 고정 학습 모델과 탐색 표본의 결과다. 전체 공식 평가 EER, 여러 학습 seed의 효과, 손실을 제거한 재학습의 효과로 해석하지 않는다. 반복 관찰한 표본과 서로 겹치는 후속 분석은 독립 실험 표본으로 합산하지 않는다.

## 배운 점

입력보존과반복검산은통과했다. 검수성공은 가설성공이나 일반화개선의 증거가 아니다.

## 실험 코드

- [historical_analysis/evalview_quiet_strict.py](code/evalview_quiet_strict.py)
- [historical_analysis/native_quiet_repeat_check.py](code/native_quiet_repeat_check.py)
- [historical_analysis/verify_native_quiet_strict.py](code/verify_native_quiet_strict.py)

코드의 파일별 원본 위치·SHA256은 [source_manifest.json](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [artifacts/experiment_methods_audit_20261111/historical_receipts/EBM80_native_quiet_strict.json](../../_shared/references/artifacts/experiment_methods_audit_20261111/historical_receipts/EBM80_native_quiet_strict.json)

## 연결

다른 관련 가설: [반복 입력과 학습·추론의 상호작용](../../02_repetition/README.md) · [잔여 방향·좌표·거리·기하와 변형 반응](../../05_residual_geometry/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
