<a id="e014-개발-전체와-21la-전체에서-반복비반복-입력을-직접-교차-평가"></a>

# 개발 전체와 21LA 전체에서 반복/비반복 입력을 직접 교차 평가

**실행 상태:** 추론·독립 검산 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

소표본에서 본 모델×반복 상호작용이 개발·21LA 전체에서도 나타날 수 있다.

## 그렇게 설계한 이유

소표본 768개 결과를 넘어 같은 원음 전체에서 반복 입력 정책의 효과를 측정한다.

대조와 통제의 근거: 점수 346,040행에 모델 두 열. 모델·음성·라벨 고정, optimizer 갱신 0. [두 반복 제거 학습 모델의 모든 저장 epoch 평가와 최소 EER 비교](../E010/README.md)의 잡음 1e-5 정식 평가와 다른 진단 실행이다.

## 모델·데이터·표본

원래 배치 240과 EBM-NoRepeat의 50epoch 두 모델. 개발 24,844 +21LA 148,176 =173,020개 원음.

## 실험 내용과 상세 절차

1. 모든 원음에서 앞 min(N,64,600)의 비반복 입력과 64,600 까지의 순환 반복 입력을 만든다. 원음·처리 파형의 SHA256을 기록한다.
2. 두 고정 모델이 같은 XLS-R 특징을 공유한다. 물리배치 8·마스크를 사용하고 pooling 잡음을 0으로 고정한다.
3. 긴 원음은 두 조건 파형이 같으므로 중복 추론을 재사용한다. 전체에서 27,406개 동일 입력 대조를 확인한다.
4. 모델별·조건별 전체 EER, 비반복 입력의 개발 임계값을 고정한 오류, 각 입력 조건의 개발 임계값으로 다시 정한 오류, 발화별 오류 전이를 계산한다.

## 측정과 검증

전체 UID/index·라 벨 수·길이·유한 점수·해시·동일 입력 점수 검사. 독립 동점 ROC와 주 계산의 EER 최대 차이 5.11e-15%p.

## 결과

21LA EER은 EBM-Repeat 비반복 6.688715 →반복 1.052789%, EBM-NoRepeat 비반복 4.792117 →반복 7.736953%다.

## 해석·실패한 부분·한계

두 모델의 50epoch 고정 비교이며 모든 체크포인트 최소값 비교가 아니다. EBM-FullContext·현재 추가 손실 모델은 이 추론에 포함하지 않았다. 반복 자체를 위조의 보편적 단서로 단정하지 않는다.

## 배운 점

전체21LA에서도 EBM-Repeat 모델은 반복으로 개선되고 EBM-NoRepeat 모델은 악화했다. 이는 추론 정책의 효과이며 반복 학습의 단독 인과 효과는 아니다.

## 실험 코드

- [소스: probe_repetition_shortcut.py](code/EBM/tools/probe_repetition_shortcut.py)
- [소스: validate_repetition_results.py](code/EBM/tools/validate_repetition_results.py)

코드의 파일별 원본 위치·SHA256은 [원본 파일 위치와 해시 목록](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [소스: probe_repetition_shortcut.py](../../../tools/probe_repetition_shortcut.py)
- [설정·결과 근거](../../_shared/references/artifacts/repetition_shortcut_20261111/protocol.json)
- [집계 결과 표](../../_shared/references/artifacts/repetition_shortcut_20261111/reviewed_analysis/independent_full_eer.csv)
- [검산 결과](../../_shared/references/artifacts/repetition_shortcut_20261111/reviewed_analysis/validation.json)

## 연결

다른 관련 가설: [길이·유효 발화량과 앞뒤 저에너지 편향](../../01_duration_bias/README.md) · [정상 다양성·도메인 지지와 정상쌍 학습](../../03_bonafide_diversity/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
