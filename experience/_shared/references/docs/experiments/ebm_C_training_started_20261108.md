> 표시 명칭·링크 이름은 설명형으로 정리했다. 원래 코드·경로·수치는 그대로 보존한다.

> 역사 문서 보존본. 원문 작성 당시의 상태·계획을 포함한다. 최신 상태와 확정 결과는 experience의 각 실험 README를 우선한다.

# EBM-FullContext: Long 원길이 학습의 단독 설정 변경

2026-11-08. 사용자 요청에 따라 EBM-NoRepeat 모델을 보존하고 EBM-FullContext 학습을 시작했다. 실행 이름 `EBM_e50_bs240_lr1e5_C_nativeLong_short303`. 학습 성능 결과는 아직 없으며, 아래는 변경·통제·실행 검증 기록이다.

## 변경한 핵심

- `EBM/arguments.py`: 이름을 EBM-FullContext로 변경하고 `cap_long=True→False`. 나머지 모든 설정은 EBM-NoRepeat와 같다.
- `EBM/data_loaders.py`: 공통 평가 로더에만 `cap_long=True`를 명시한다. 기존에는 하나의 flag가 학습과 평가 둘 다 바꿨다. EBM-FullContext에서는 학습 Long만 전체로 바꾸고 dev/target 입력은 EBM-NoRepeat와 같게 고정했다. EBM-NoRepeat 설정에서 이 연결 변경은 입력상 no-op이다.
- 모델·mask·OC/BPL 수식·sampler·optimizer·gradient replay·두 단계 갱신·logger 등 나머지 Python 27개 파일은 EBM-NoRepeat와 동일하다. 29개 Python 파일 syntax와 loader AST 차이 범위를 검증했다. 문서 `EBM/docs/CONTROL_C.md`를 함께 저장했다.

| 조건 | EBM-NoRepeat | EBM-FullContext |
| --- | --- | --- |
| 학습 Long | 앞64,600 sample, 최대4.0375초 | 원길이 전체 |
| 학습 Short | 최대48,440 sample 연속 random crop | 동일 |
| 짧은 음성 | 반복 없이0패딩+전체 mask | 동일 |
| 목적식 | long OC + 정상 long×short local BPL | 동일 |
| 보조 손실 | 둘 다0, wrapped 입력 없음 | 동일 |
| 전체/물리 batch | global240 / micro8, rank당12정상+108spoof | 동일 |
| 학습 | seed1, epoch50, lr1e-5, 두 단계 Adam | 동일 |
| 초기화 | seed 기반 새 모델 + frozen XLS-R | 동일; EBM-NoRepeat의 학습 checkpoint로 이어 학습하지 않음 |
| 증강 | codec-off·ACN-off·RawBoost0~8 | 동일 정책 |
| dev/최종 공통 평가 | 앞64,600cap·비반복·mask·batch8 | 동일 |

## 전체 연결을 확인한 범위

`main.py`의 seed/DDP·dataset·class weight·frozen XLS-R·model·optimizer 구성 → `get_loaders`의 class/length sampler와 TrainSet → Long cap/Short crop과 RawBoost → collate의 비반복 두 view → `native_steps.features`의 sample/feature mask → `B_Linear`의 시간/층 pooling과160차원 embedding → 원래 OC와12×12 BPL → 저장된 RNG의 embedding-gradient replay → theta 갱신 후 centroid 재계산/갱신 → epoch마다 scheduler → 5epoch별 dev·CLS/TrainState 저장까지 읽고 연결을 검증했다. Target set은 로더가 만들어지지만 학습 중 선택 기준에는 쓰이지 않는다. 실제 in-training 검증은 dev뿐이다.

최종 head는 cosine 하나를 사용한다. 입력 시퀀스에 고정 길이를 요구하는 flatten/절대길이 linear 계층이 없고 pooling mask가 실제 길이를 처리한다. Frozen frontend는 모든 hidden state25개를 반환하며 변경하지 않았다. Layer/time pooling과 dropout·noise를 포함한 runtime은 EBM-NoRepeat 그대로다.

## 실제 데이터와 검증

| class | N | Long이 EBM-NoRepeat 상한보다 긴 개수 | 해당 비율 | 최대 원길이 |
| --- | --- | --- | --- | --- |
| 정상 | 2,580 | 576 | 22.326% | 11.129375초 |
| spoof | 22,800 | 6,195 | 27.171% | 13.187938초 |

원음 전체 기준 EBM-FullContext의 Long 유효 sample 합은 EBM-NoRepeat보다 10.530% 많다. 이는 고유 학습 음성 기준이며 sampler oversampling·배치0패딩·GPU시간 증가율은 아니다. Short를 늘린 결과도 아니다.

입력 검사는 실제 10개 길이/class 사례×RawBoost9종=90쌍을 통과했다. Codec 및 wrap 호출을 금지하고 Long=EBM-FullContext원길이·EBM-NoRepeat cap, Short cap, 유한값을 검사했다. EBM-NoRepeat cap 이하 음성은 두 view가 정확히 같았고 무증강 Short도 같았다. Python의 crop/증강종류 선택 RNG 상태는 EBM-NoRepeat와 EBM-FullContext가 같았다. 두 rank의 **50epoch 전체 UID batch 순서**와12/108 quota가 일치하고 epoch당108 logical batch를 유지했다.

기존 컨테이너·CUDA_VISIBLE_DEVICES=0,1에서 양 GPU에 실제 최장 정상/스푸프를 반복 배치해 micro8의 상한 메모리를 확인했다. 이는 음성 반복을 학습 입력에 추가한 것이 아니라 검증용 batch에서 동일한 최장 원음 항목을 여러 슬롯에 넣은 스트레스 검사다. 본학습 sampler는 바꾸지 않았다. 각 GPU에서 실제120 logical batch의 OC/BPL·gradient replay·두 Adam step을 통과했고 classifier와 centroid가 갱신되고 frontend는 불변이었다. 최장 입력에서 최대 allocated 3,268.6MiB / reserved5,016MiB, solo/혼합길이 embedding 오차1.44×10⁻⁶ 미만이었다. 따라서 EBM-FullContext만 microbatch를 줄이는 변경은 하지 않았다. 기존 EBM-NoRepeat에서도 있던 DDP gradient-stride 경고는 수치 실패가 아니며 코드 변경하지 않았다.

## EBM-FullContext 효과를 어떻게 판정할 것인가

직접 비교는 과거240 대신 **현재 EBM-NoRepeat 고정50 vs EBM-FullContext 고정50**이다. 동일 trial 집합·공통EBM-NoRepeat입력·batch8·기본pooling noise1e-5 및 공통 RNG를 사용한다. 각 모델의19LA dev에서 같은 규칙(|FPR−FNR| 최소→동률낮은 FPR→높은 threshold)으로 운영점을 정하고 target EER·정상거부·spoof수락을 함께 본다. 원길이 구간은 [설정·결과 근거](../../artifacts/ebm_C_20261108/local_code_audit.json), [설정·결과 근거](../../artifacts/ebm_C_20261108/preflight.json)
- 실제 입력·sampler: [설정·결과 근거](../../artifacts/ebm_C_20261108/metadata_validation.json)
- 양GPU: [rank0](../../artifacts/ebm_C_20261108/verification_rank0.json), [rank1](../../artifacts/ebm_C_20261108/verification_rank1.json)
- 읽기전용 상태점검: [실험 구현·검증 소스](../../../workspace/artifacts/ebm_C_20261108/check_C_training.py), [최근 실제 상태](../../artifacts/ebm_C_20261108/last_check.json)

실행 중 서버 소스·branch·pull·중단·재시작을 하지 않는다. 기존에 요청된30분점검·종료후분석절차를 EBM-FullContext에 이어 적용해 전용 heartbeat `ebm-c-30-b` ACTIVE를 생성·확인했다. 예약 메타데이터는 `artifacts/ebm_C_20261108/monitor_schedule.json`이다. 로컬 예약은 PC와 앱·SSH 연결이 유지돼야 한다([공식 예약 안내](https://learn.chatgpt.com/docs/automations?surface=app)). 학습 종료 후에는 run.exit0·epoch50·CLS50/TrainState50·소스/설정/coverage를 먼저 검증한다. 단순 실행 시작을 효과 입증으로 읽지 않는다.


마지막 초기 진행 확인: 2026-11-08T06:17:48 KST, 1epoch 14/108배치, loss1=8.423/loss2=69.971. parent와직계두rank생존·오류없음·로그갱신·EBM-NoRepeat checkpoint보존확인. 초기5step의두손실/OC/BPL이 EBM-NoRepeat와float정확일치한 기록도 [설정·결과 근거](../../artifacts/ebm_C_20261108/initial_B_C_trace_comparison.json)에남겼다.
