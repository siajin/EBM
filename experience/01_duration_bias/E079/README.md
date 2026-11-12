# E079. 원음 길이에 따라 5초/4초 또는 4초/3초를 정렬하는 보류 안

**실행 상태:** 코드 초안·CPU 검사만; 실행 보류

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

원음길이에 따라5/4초 또는4/3초를 정렬하면 추가문맥을 활용할 수 있다.

## 그렇게 설계한 이유

원래 긴 입력보다 긴 원음에는 추가 긴 문맥, 짧은 원음에는 3초 참조를 주는 길이 항을 제안했다.

대조와 통제의 근거: 4초/3초는 64,600/48,440 이고 5초는 80,000이다. 이후 선택한 원음 ≤2초 3/2초 양 방향 항과 다른 설계다.

## 모델·데이터·표본

계획상 학습 자료·원래 240 OC·BPL. 실제 새 학습·GPU 사전 실행 검사·원격 실행 root 없음.

## 실험 내용과 상세 절차

1. 원음>64,600이면 80,000/64,600샘플, 원음 ≤64,600이면 64,600/48,440샘플의 같은 원음 입력을 만든다. 부족분은 사용자 선택대로 반복한다.
2. OC +0.5 기존 BPL +0.5 추가 길이 항을 계획했다. 길이 정책은 각 원음별로 결정하고 긴/짧은 별도 optimizer 나 번갈아 클래스 균형 배치를 사용하지 않는 안이었다.
3. 당시 초안은 긴 참조 gradient 끊기를 포함했다. 실제 길이·nested 구간·반복·난수와 기존 분기 동일성의 CPU 검사 4개만 통과했다.

## 측정과 검증

단위 검사와 실행 보류 상태만 있다. 학습 EER을 보고하지 않는다.

## 결과

사용자의 ‘아직 수행하지 마’로 보류됐고 학습·GPU 추론 0회다.

## 해석·실패한 부분·한계

이 초안을 새 3/2초 방식의 실행 코드로 대체해 보고하지 않는다.

## 배운 점

보류안의 CPU검사는 성능실험이 아니다. 이후 선택된3/2초 양방향항과 합쳐 설명하면 안 된다.

## 실험 코드

- [EBM/conditional_bpl2.py](code/EBM/conditional_bpl2.py)
- [EBM/conditional_bpl2_steps.py](code/EBM/conditional_bpl2_steps.py)
- [EBM/tests/test_conditional_bpl2.py](code/EBM/tests/test_conditional_bpl2.py)

코드의 파일별 원본 위치·SHA256은 [source_manifest.json](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [artifacts/bpl2_conditional_54_43_20261111/plan.json](../../_shared/references/artifacts/bpl2_conditional_54_43_20261111/plan.json)
- [artifacts/bpl2_conditional_54_43_20261111/state.json](../../_shared/references/artifacts/bpl2_conditional_54_43_20261111/state.json)
- [EBM/tests/test_conditional_bpl2.py](../../../tests/test_conditional_bpl2.py)

## 연결

다른 관련 가설: [잔여 방향·좌표·거리·기하와 변형 반응](../../05_residual_geometry/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
