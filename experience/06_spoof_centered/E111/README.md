# E111. 정상 중심과 스푸프 중심을 명시적으로 전환하는 원본240 기반 구현

**실행 상태:** 설계·로컬 CPU 구현 검증 완료; 실제 GPU 학습 미실행

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

정상 대신 알려진 위조를 한 중심과 전체쌍 BPL로 모으고 정상을 밀어내면 일반화에 도움이 되는가?

## 그렇게 설계한 이유

정상 지지 문제의 후속 가설로 compact class 자체를 바꾸되 데이터 라벨·원본240 입력·두 단계 갱신을 통제한다.

대조와 통제의 근거: 기존파일140개·원본snapshot·원격작업은 보존. 원래240과 별도RNG격리이므로 새bona통제arm도 동일격리를 적용해야 한다. 실제 음성·GPU/NCCL smoke는 수행하지 않았다.

## 모델·데이터·표본

원본240 Git10f3bcd의 전체38파일과 핵심5파일 보존. 실제 크기 backend3,744,224parameter와 중심160parameter. 로컬 합성 CPU/Gloo 검증만 수행.

## 실험 내용과 상세 절차

1. 원본 라벨bona1/spoof0과 실제클래스 OC가중치[5/9,5]를 유지하고 compact_label0에서 spoof당김0.9·bona밀기0.2로 역할을 전환한다.
2. spoof long×short 전체 로컬쌍을 mean으로 계산한다. 일반108×108=11,664쌍이며81배 계수를 추가하지 않는다.
3. backend OC+pair 갱신 후 갱신된 backend로 compact class long을 다시 계산하고 중심만 갱신한다. 두arm 중심단계 계수는5로 맞춘다. Adam 이동량이나 표본분산 동일성을 주장하지 않는다.
4. 공통 canonical 점수는 높은쪽이bona: bona중심+cos,spoof중심−cos다. 역할manifest·원자 checkpoint·source/RNG 검증·save-first 평가를 별도 경로로 구현한다.
5. 원본 손실·parameter·Adam moment의 일반/부분120·72·114행 회귀, 실제원본 sampler108batch, 합성108step→저장→세평가, 두CPU rank·실패전파·역할gradient 격리를 검사한다.

## 측정과 검증

최초 신규32+기존46=78검사, 평가연결감사 후 총86검사. 실제크기 양arm 초기tensor 동일. runtime validate는 launched=false.

## 결과

목적식 방향은 동시에 구성 가능하나 공동손실 단조감소·수렴·정상다양성 보장은 없다. 로컬검증은 통과했으며 실제EER은 없다.

## 해석·실패한 부분·한계

단일known-spoof 중심이 미지공격을 대표하는지는 미검증이다. bona반발만으로 bona간분산을 보장하지 않는다. 정규화cosine의 극점에는 잘못된클래스의gradient0 정지점도 가능하다.

## 배운 점

클래스역할·BPL대상·중심단계·score부호를함께바꿔야 한다. CPU회귀성공을 실제일반화성공으로 부르지 않는다.

## 실험 코드

- [EBM/role_lifecycle.py](code/EBM/role_lifecycle.py)
- [EBM/role_model.py](code/EBM/role_model.py)
- [EBM/role_objectives.py](code/EBM/role_objectives.py)
- [EBM/role_runtime.py](code/EBM/role_runtime.py)
- [EBM/role_training.py](code/EBM/role_training.py)
- [EBM/tools/train_role_experiment.py](code/EBM/tools/train_role_experiment.py)
- [EBM/tests/test_role_experiment.py](code/EBM/tests/test_role_experiment.py)

코드의 파일별 원본 위치·SHA256은 [source_manifest.json](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [docs/experiments/ebm_spoof_centered_design_20261112.md](../../_shared/references/docs/experiments/ebm_spoof_centered_design_20261112.md)
- [EBM/ROLE_EBM.md](../../../ROLE_EBM.md)
- [artifacts/reverse_ebm_implementation_20261112/validation.json](../../_shared/references/artifacts/reverse_ebm_implementation_20261112/validation.json)
- [artifacts/reverse_ebm_implementation_20261112/full_head_identity.json](../../_shared/references/artifacts/reverse_ebm_implementation_20261112/full_head_identity.json)

## 연결

다른 관련 가설: [정상 다양성·도메인 지지와 정상쌍 학습](../../03_bonafide_diversity/README.md) · [잔여 방향·좌표·거리·기하와 변형 반응](../../05_residual_geometry/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
