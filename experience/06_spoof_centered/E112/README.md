<a id="e112-역할-평가의-점수-부호smoke-범위checkpoint-용도-오류-감사"></a>

# 역할 평가의 점수 부호·smoke 범위·checkpoint 용도 오류 감사

**실행 상태:** 결함 재현·로컬 수정·86검사 통과; 실제 사용자 실행의 원인 미확정

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

역할을모르는평가기와smoke범위오표기가 일부높은EER 또는잘못된완료판정을 만들수있다.

## 그렇게 설계한 이유

높은EER 보고를 성능 실패로 단정하기 전에 역할정보가 평가경로에서 사라지는 결함을 검사한다.

대조와 통제의 근거: 기존모델·manifest·score를 덮어쓰지 않았다. 이결함이 실제사용자 실행에 쓰였는지는 확인되지 않았다.

## 모델·데이터·표본

[정상 중심과 스푸프 중심을 명시적으로 전환하는 원본240 기반 구현](../E111/README.md) 로컬구현·합성CPU·두Gloo프로세스. 실제 사용자run root는 확보되지 않았으며 원격실행·실제음성 추론0회.

## 실험 내용과 상세 절차

1. 초기tensor-only spoof중심 CLS를 기존bona 평가기가 strict=True로도 받아+cos를 출력하는 경우를 재현한다. 부호만분리한 합성예제의 올바른EER0%·legacy100%를 확인한다.
2. role_state_dict와역할·용도를 담은v2 CLS/TrainState를 도입해 unaware tensor-only fallback이 조용히 잘못읽지 못하도록 한다. 기존유효v1은 역할대응로더로 유지한다.
3. smoke64개를 모두 처리한경우 full로 표시되던 범위를 smoke_subset으로 수정하고 checkpoint_purpose=smoke를 저장한다. 외부scorer는 기본적으로smoke checkpoint를 거부한다.
4. train모드에서 공식집합 전체행수를 확인하고 arbitrary/partial평가를 구분한다. 기존학습목적식·모델·두단계갱신은 변경하지 않는다.

## 측정과 검증

수정전3개실패회귀·신규8개감사·기존32개역할+46개legacy, 합86개통과. source보존140개·runtime validate 통과.

## 결과

부호반전과smoke범위 결함을 재현·차단했다. 합성0/100%는 실제학습EER이 아니다. 이미끝난 실험의EER이개선됐다고 보고할 자료는 없다.

## 해석·실패한 부분·한계

올바른역할scorer로 수행한 실제학습도 다른이유로 나쁠수있다. 재현된코드결함과 실제성능악화의인과를 구분한다.

## 배운 점

역할·용도metadata를checkpoint와scorer에끝까지전달해야 한다. 결함재현만으로사용자실행의원인을확정하지 않는다.

## 실험 코드

- [소스: role_lifecycle.py](code/EBM/role_lifecycle.py)
- [소스: role_runtime.py](code/EBM/role_runtime.py)
- [소스: train_role_experiment.py](code/EBM/tools/train_role_experiment.py)
- [소스: test_role_code_audit.py](code/EBM/tests/test_role_code_audit.py)
- [소스: role_lifecycle.py](code/artifacts/reverse_ebm_code_audit_20261112/before/role_lifecycle.py)
- [소스: role_model.py](code/artifacts/reverse_ebm_code_audit_20261112/before/role_model.py)
- [소스: role_objectives.py](code/artifacts/reverse_ebm_code_audit_20261112/before/role_objectives.py)
- [소스: role_runtime.py](code/artifacts/reverse_ebm_code_audit_20261112/before/role_runtime.py)
- [소스: role_training.py](code/artifacts/reverse_ebm_code_audit_20261112/before/role_training.py)

코드의 파일별 원본 위치·SHA256은 [원본 파일 위치와 해시 목록](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [상세 설계·실험 기록](../../../ROLE_EBM.md)
- [검산 결과](../../_shared/references/artifacts/reverse_ebm_code_audit_20261112/validation.json)
- [설정·결과 근거](../../_shared/references/artifacts/reverse_ebm_code_audit_20261112/runtime_preflight.json)

## 연결

다른 관련 가설: [정상 다양성·도메인 지지와 정상쌍 학습](../../03_bonafide_diversity/README.md) · [잔여 방향·좌표·거리·기하와 변형 반응](../../05_residual_geometry/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
