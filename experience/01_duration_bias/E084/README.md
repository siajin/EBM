<a id="e084-학습분산-기울기저장worker-오류의-실행-전-통합-검증"></a>

# 학습·분산 기울기·저장·worker 오류의 실행 전 통합 검증

**실행 상태:** 여러 버전의 구현 검증 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

부분배치·분산기울기·저장·worker오류까지 검증하면 실행실패를 줄일 수 있다.

## 그렇게 설계한 이유

과학적 성능 결과와 분리해 각 실행을 중단시킬 코드 오류와 학습 동등성을 검사한다.

대조와 통제의 근거: 단위 검사의 임시 optimizer 갱신과 실제 음성 본 학습 갱신을 구분한다. 불변 검사·완료 검사는 하나의 독립 성능 실험이 아니다.

## 모델·데이터·표본

합성 CPU 모델, 한정된 실제 학습 자료 파형·실제 XLS-R/GPU 배치, 저장 후각 평가 64개 등의 검증 전용 입력. 정식 전체 평가와 다르다.

## 실험 내용과 상세 절차

1. 원래 손실과 로컬/전역 BPL의 입력·파라미터 gradient, 두단계 Adam, 마지막 부분 배치·작은 loader 순환을 대조한다.
2. 실제 마스크·길이·원음 샘플러·RawBoost0~8의 배열 일치, 소스 Git/SHA,codec 비활성, 고정 frontend를 검사한다.
3. 새 3/2초 항에서는 K0가 원래 함수와 같은지, K =N가 0.5/0.5 인지, 두 rank의 K가 다른 경우·양방향 gradient·원래 난수 보존을 검사한다.
4. 저장 실패·평가 한쪽 rank 실패·불균등 평가 shard·DataLoader 다중 workerimport를 실제 프로세스에서 검사한다. 모델/Adam/epoch/난수 저장·재로딩·완료표식을 검증한다.
5. 현재 학습 snapshot은 서버 57 검사와 두 GPU3 논리 배치 120/72/114 입력·각 집합 64개 평가를 통과한 뒤 본 학습을 시작했다. 실패 v1 검사 경로 수정은 새 snapshot 에만 반영했다.

## 측정과 검증

검사 수·오류/skip·실제 update·finitegradient·동일성 허용 오차·각 stage 종료 코드.

## 결과

EBM-NoRepeat 초기 두 GPU 각 105 검사, OC +BPL 단기 연결 11 검사, 현재 짧은 정상 음성의 3.0275초·2초 정렬 학습 서버 57 검사 등 버전별 범위를 확인했다. 검사가 통과해도 최종 EER 개선을 뜻하지 않는다.

## 해석·실패한 부분·한계

과거 일부 구현은 검사를 통과한 뒤 진입부/평가 import 에서 실패했다. 통과 범위와 아직 수행하지 않은 실제 단계를 분리해 기록한다.

## 배운 점

unit통과뒤에도 실제진입·평가import가 실패했다. 검증범위와실행단계의 빈틈을 명시해야 한다.

## 실험 코드

- [소스: test_activity_policy.py](code/EBM/tests/test_activity_policy.py)
- [실험 구현·검증 소스](code/EBM/tests/test_baseline240_mild.py)
- [소스: test_conditional_bpl2.py](code/EBM/tests/test_conditional_bpl2.py)
- [소스: test_intermediate_19la.py](code/EBM/tests/test_intermediate_19la.py)
- [실험 구현·검증 소스](code/EBM/tests/test_mild_duration.py)
- [소스: test_paired_evaluation.py](code/EBM/tests/test_paired_evaluation.py)
- [소스: test_positive_bpl.py](code/EBM/tests/test_positive_bpl.py)
- [소스: test_repetition_shortcut.py](code/EBM/tests/test_repetition_shortcut.py)
- [소스: test_role_code_audit.py](code/EBM/tests/test_role_code_audit.py)
- [소스: test_role_experiment.py](code/EBM/tests/test_role_experiment.py)
- [실험 구현·검증 소스](code/EBM/tests/test_short32_bpl.py)
- [실험 구현·검증 소스](code/EBM/tests/test_short32_lifecycle.py)
- [실험 구현·검증 소스](code/EBM/tests/test_short32_runner.py)
- [소스: test_source_coverage.py](code/EBM/tests/test_source_coverage.py)
- [실험 구현·검증 소스](code/EBM/tools/validate_short32.py)

코드의 파일별 원본 위치·SHA256은 [원본 파일 위치와 해시 목록](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [검산 결과](../../_shared/references/artifacts/short32_train_20261111_v2/container_validation/validation.json)
- [설정·결과 근거](../../_shared/references/artifacts/short32_train_20261111_v2/smoke_done_rank0.json)
- [설정·결과 근거](../../_shared/references/artifacts/short32_robustness_20261111/state.json)
- [설정·결과 근거](../../_shared/references/artifacts/short32_original240_validation_20261111/change_audit.json)
- [설정·결과 근거](../../_shared/references/artifacts/positive_bpl_code_review_20261110/gpu_checks_complete.json)
- [완료 확인 기록](../../_shared/references/artifacts/geometry_cache/local_analysis/control_B_20261106_v1/complete.json)

## 연결

다른 관련 가설: [반복 입력과 학습·추론의 상호작용](../../02_repetition/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
