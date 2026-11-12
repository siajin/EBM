<a id="e020-정상-두-묶음만-사용한-108회-양성쌍-손실의-단기-학습"></a>

# 정상 두 묶음만 사용한 108회 양성쌍 손실의 단기 학습

**실행 상태:** 단기 학습·진단 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

서로 다른 정상 원음 전체쌍만 정렬해도 단기 표현과 잡음 안정성을 개선할 가능성이 있다.

## 그렇게 설계한 이유

위조 분류 손실 없이 정상끼리 가깝게하면 잡음 민감도가 줄면서 표현이 붕괴하는지 조사한다.

대조와 통제의 근거: 실제 pilot의 RawBoost8은 원래 병렬 필터 +충격성 잡음 대신 필터 +색 잡음이었던 정정이 있다. 이후 코드가 고쳐졌어도 이 진단 표본 관측을 원본 증강 통제 실험으로 소급하지 않는다.

## 모델·데이터·표본

처음부터 초기화한 분류기 본체와 고정 XLS-R. GPU 마다 독립 정상 12개 씩 두 묶음,144쌍. 정상 원음은 정해진 학습 자료 길이 정책에서 복원 추출한다.

## 실험 내용과 상세 절차

1. seed 1·Adam 학습률 1e-5·weight decay1e-4·BPL alpha30으로 108회 갱신한다. OCSoftmax·교차 엔트로피·음성 쌍 negative·정상 기준 벡터 갱신은 없다.
2. 정상 원음과 구간은 양 묶음 안팎에서 독립 추출하고 중복 원음 발생을 기록한다. 파형 이어 붙이기는 하지 않는다.
3. 개발 전체에서 클래스 2× 원음 길이 4구간 ×64개 =512개를 점수와 무관하게 고정한다. 기준 점수는 고정 학습 자료 정상 256개 의 현재 unit 특징 평균 방향과의 cosine 이다.
4. 0/27/54/108회 에 깨끗한 512개, 동일 128개 의 20dB 백색잡음 조건, 특징 분산·유효 차원·발화 간 유사도를 평가한다. 진단 pooling 잡음은 0이다.

## 측정과 검증

진단 512개 EER/AUROC, 동일 128개 잡음 전후 EER, cosine 거리, covariance trace, 유효 rank. 매단계 기준 평균은 다시 계산 하므로 학습된 단일 OC 중심점수와 다르다.

## 결과

깨끗한 진단 EER 48.046875 →30.078125%, 잡음 128개 59.375 →42.1875%. 표현 분산 91.63% 감소, 유효 rank19.32 →36.11. 완전 상수는 아니지만 강한 수축이 있었다.

## 해석·실패한 부분·한계

108회·소표본·OC 없는 진단 표본이다. EBM-NoRepeat와 EBM-FullContext 또는 정식 OC +BPL의 우월성을 입증하지 않는다. RawBoost도 있어 잡음 개선을 서로 다른 정상 원음 정렬만의 효과로 분리하지 못한다.

## 배운 점

진단 EER은 내려갔지만 분산이 크게 수축했다. OC 없는108step·작은 표본·다른 RawBoost8을 정식 EBM 우위로 해석할 수 없다.

## 실험 코드

- [소스: positive_bpl_pilot.py](code/snapshots/ebm_positive_bpl_20261110_v1/positive_bpl_pilot.py)
- [소스: positive_bpl_pilot.py](code/EBM/tools/positive_bpl_pilot.py)
- [소스: analyze_positive_bpl_pilot.py](code/EBM/tools/analyze_positive_bpl_pilot.py)

코드의 파일별 원본 위치·SHA256은 [원본 파일 위치와 해시 목록](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [설정·결과 근거](../../_shared/references/artifacts/positive_only_bpl_20261110/protocol.json)
- [설정·결과 근거](../../_shared/references/artifacts/positive_only_bpl_20261110/analysis.json)
- [설정·결과 근거](../../_shared/references/artifacts/positive_only_bpl_20261110/state.json)

## 연결

다른 관련 가설: [반복 입력과 학습·추론의 상호작용](../../02_repetition/README.md) · [잔여 방향·좌표·거리·기하와 변형 반응](../../05_residual_geometry/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
