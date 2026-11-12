<a id="e038-가변-길이-oc-bpl과-평가-중심난수저장의-실제-연결-검사"></a>

# 가변 길이 OC +BPL과 평가 중심·난수·저장의 실제 연결 검사

**실행 상태:** 구현 검증 완료·장기 성능 실험 아님

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

OC를 유지한 가변길이 학습도 중심·난수·저장 정의를 맞추면 기존 평가와 연결할 수 있다.

## 그렇게 설계한 이유

OC를 유지한 최종 가변 길이 학습과 기존 모델의 평가 점수가 동일한 정의인지 확인한다.

대조와 통제의 근거: 이전 shared_train_mean 검증은 별도 평균 중심을 저장하는 실험이었으며 학습 OCweight를 덮어 쓰지 않았다. 최종 OC +BPL 학습 평가에 그 평균을 사용하지 않는다.

## 모델·데이터·표본

학습 자료 정책·실제 파형·두 GPU의 짧은 갱신 배치. EBM-Repeat/EBM-NoRepeat와 EBM-FullContext 보존 모델은 평가 전용으로만 읽는다.

## 실험 내용과 상세 절차

1. OC 정상 12 +위조 108 의 가변 입력과 BPL 왼쪽 동일 정상 12·오른쪽 독립 정상 12를 만든다.
2. 같은 원음 오른쪽 조건과 독립 원음 오른쪽 조건을 각각 두 논리 배치 갱신하고 본체/중심 각 2회·144쌍·유한 gradient·고정 frontend를 확인한다.
3. 두 모델 평가를 학습된 각 OC 기준 벡터 cosine 으로 정정한다. 이전‘양성 BPL만 학습/학습 자료 평균 중심’분기는 최종 요청과 다른 과거 검증으로 남긴다.
4. 앞 64,600·비반복·mask·batch8·seed 1729·원래 ASP 잡음의 공통 난수와 훈련 난수 복원을 확인한다. 모델 저장/불러오기 후 점수를 대조한다.

## 측정과 검증

11개 단위 검사·동일 checkpoint 점수·실제 update 횟수·저장 재로 딩·rank 간 일치.

## 결과

저장 모델 평가 점수 최대 차이 4.10e-8·EER 동일 등 연결 검사를 통과했다. 이후 장기 학습은 [정상·위조 길이와 음성활동을 맞추고 독립 정상 묶음을 정렬한 학습](../E021/README.md)로 별도 실행됐다.

## 해석·실패한 부분·한계

짧은 실제 갱신 검사와 단위 검사가 정식 평가 성능 개선의 증거는 아니다. 과거 검증 조건을 현재 최종 설계로 소급하지 않는다.

## 배운 점

과거 BPL-only·평균중심 검증은 최종 설계와 달랐다. 현재 실행의 손실·scorer 경로를 명시해야 한다.

## 실험 코드

- [소스: positive_bpl.py](code/EBM/positive_bpl.py)
- [소스: controlled_data_loaders.py](code/EBM/controlled_data_loaders.py)
- [소스: test_positive_bpl.py](code/EBM/tests/test_positive_bpl.py)

코드의 파일별 원본 위치·SHA256은 [원본 파일 위치와 해시 목록](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [설정·결과 근거](../../_shared/references/artifacts/oc_bpl_matched_variable_20261110/state.json)
- [설정·결과 근거](../../_shared/references/artifacts/positive_bpl_code_review_20261110/state.json)
- [설정·결과 근거](../../_shared/references/artifacts/oc_bpl_code_audit_20261110/analysis.json)
- [상세 설계·실험 기록](../../../POSITIVE_BPL.md)

## 연결

다른 관련 가설: [반복 입력과 학습·추론의 상호작용](../../02_repetition/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
