# E016. 긴 원음의 같은 1초/2초 구간에서 반복량·접합부·반사대조

**실행 상태:** 추론·분석 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

내용을 고정한 채 반복량·접합만 바꿔도 점수가 변한다면 단순 내용 부족만으로 설명할 수 없다.

## 그렇게 설계한 이유

음성 내용은 같게 두고 반복 횟수·접합 처리·시간 역전 반사가 점수를 바꾸는지 본다.

대조와 통제의 근거: 336×31 =10,416행. 2초 구간을 2초로 반복은 실제 동일 입력 대조다. 반사는 뒤집힌 음성 구간을 포함하는 비자연적 처리다.

## 모델·데이터·표본

21LA 원음>64,600샘플 중 정상 128개 와 13개 위조 공격별 16개 =208개, 총 336개. 원래 240/B50 고정.

## 실험 내용과 상세 절차

1. 원음 별 해시 우선순위, 화자별 번갈아 선택, 코덱 분산을 사용해 점수와 무관하게 표본을 고정한다.
2. 원음 전체의 앞 또는 뒤에서 1초(16,000)·2초(32,000)를 각각 취한다. 뒤 구간은 원음의 실제 끝 부분이며 과거 471개 실험의 첫 4초 창 뒤 부분과 다르다.
3. 각 구간에서 실제 길이, 순환 반복 32,000/48,000/64,600,64,600 반복의 접합 smoothing, 같은 수의 비접합 smoothing, 반사채움 64,600을 만든다.
4. smoothing은 각 위치 ±80샘플 구간의 양 끝을 선형 보간한다. 실제 접합은 원래 구간 길이의 정수 배, 가짜 접합은 그 절반에서 시작한 동일 간격 위치다.
5. 원음 앞 64,600 자체와 그 자연 음성의 1초/2초 위치 smoothing 두 조건을 추가한다. 총 1 +2×(2×7 +1)=31조건이다.
6. pooling 잡음 0, 두 고정 모델, 같은 파형해시를 재사용한다. head/tail은 원음별 평균하고 화자 단위로 불확실성을 계산한다.

## 측정과 검증

조건별 점수 변화·고정/재설정 임계값 오류. 저장된 화자 군집 개별 95% 구간은 탐색적이며 전체 다중 비교 동시 구간이 아니다.

## 결과

원래 240 정상의 1초 구간은 2/3/4.0375초 반복에서 평균 점수 변화 −0.152219/−0.494815/−0.858379.2초 구간은 0/−0.333210/−0.503502. 접합 smoothing 회복은 하락량의 3.80%/7.61%였다.

## 해석·실패한 부분·한계

동일 내용의 반복으로 변화가 생겨 정보량 부족만의 설명은 충분하지 않다. 그러나 반복 길이·문맥·접합 효과를 모두 분리한 것은 아니다. 자연 단음의 모집단 오류율로 일반화하지 않는다.

## 배운 점

반복량 반응은 크지만 접합 smoothing의 회복은 작았다. 내용 길이·문맥·접합을 더 분리할 필요가 있다.

## 실험 코드

- [EBM/tools/probe_repetition_shortcut.py](code/EBM/tools/probe_repetition_shortcut.py)
- [EBM/tools/analyze_repetition_shortcut.py](code/EBM/tools/analyze_repetition_shortcut.py)

코드의 파일별 원본 위치·SHA256은 [source_manifest.json](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [EBM/tools/probe_repetition_shortcut.py](../../../tools/probe_repetition_shortcut.py)
- `artifacts/repetition_shortcut_20261111/probe_selected.csv` — 원자료/운영기록 경로; 제외 사유는 전체 출처 목록에 기록
- [artifacts/repetition_shortcut_20261111/protocol.json](../../_shared/references/artifacts/repetition_shortcut_20261111/protocol.json)
- [artifacts/repetition_shortcut_20261111/reviewed_analysis/dose_effects.csv](../../_shared/references/artifacts/repetition_shortcut_20261111/reviewed_analysis/dose_effects.csv)
- [artifacts/repetition_shortcut_20261111/reviewed_analysis/natural_smoothing_controls.csv](../../_shared/references/artifacts/repetition_shortcut_20261111/reviewed_analysis/natural_smoothing_controls.csv)

## 연결

다른 관련 가설: [길이·유효 발화량과 앞뒤 저에너지 편향](../../01_duration_bias/README.md) · [잔여 방향·좌표·거리·기하와 변형 반응](../../05_residual_geometry/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
