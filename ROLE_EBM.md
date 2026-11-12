# 스푸프 중심 EBM 구현

2026-11-12. 원본240을 기준으로 OC·중심·BPL의 compact class를 바꾸는 별도 실행 경로다. 기본 compact class는 spoof이며 bonafide 모드로 원본 회귀와 통제 비교를 수행할 수 있다. 기존 `main.py` 기본 동작과 고정 원본은 변경하지 않았다.

## 평가 연결 재점검 수정

사용자의 높은 EER 보고 후 평가 연결의 두 문제를 재현하고 로컬에서 수정했다. 이 두 경로가 실제 사용자 실행에 사용됐는지는 실행 결과를 확보한 뒤 확인해야 한다.

첫째, 최초 tensor-only CLS는 기존 보나용 `B_Linear` 평가기에 `strict=True`로도 로딩됐다. 그 평가기는 신규 중심의 역할을 몰라 `+cosine`을 반환한다. 신규 `-cosine`과 반대이며, 부호만 분리한 합성 예제에서 EER0%가100%가 되는 상황을 확인했다. 이는 실제 학습 성능 수치가 아니다. 새 CLS는 역할·용도와 `role_state_dict`를 담은 `ebm_role_classifier_v2` 형식이고, TrainState도 `role_classifier`를 담은 전용 형식이다. 기존의 tensor-only/`classifier` fallback 로더는 새 파일을 조용히 잘못 평가할 수 없게 된다. 네트워크의 parameter key·값·손실·갱신 수식은 변경하지 않았다.

둘째, 최초 smoke의64개 subset 전체를 처리하면 결과 범위를 `full`로 기록했다. 이를 `smoke_subset`·`checkpoint_purpose=smoke`로 바로잡았다. 본학습 평가는 공식 집합별 전체 행 수를 확인하며, 임의·부분 집합은 각각 `custom`·`partial_*`로 표시한다. 외부 점수용 `load_scoring_head`는 기본적으로 epoch0 smoke 체크포인트를 거부하며, 실제 smoke 검사에서만 `allow_smoke=True`로 명시한다.

기존 v1 모델·점수·manifest는 덮어쓰거나 변환하지 않는다. 유효한 v1 역할 manifest가 있는 checkpoint는 신규 역할 대응 로더로 계속 읽을 수 있다. 기존 v1 파일을 기존 보나 평가기에 넣는 행위까지 소급해서 차단한 것은 아니므로, 과거 결과는 `score_bona` 열과 평가 경로를 확인해야 한다. 이 수정만으로 이미 완료된 실행의 EER이 개선됐다고 보고하지 않는다.

재현 전·후 검사와 범위는 [코드 감사 결과](experience/_shared/references/artifacts/reverse_ebm_code_audit_20261112/validation.json)에 기록한다. 현재 수정은 로컬이며 원격 snapshot·실행·checkpoint는 변경하지 않았다.

## 목적식과 이론 검토

데이터 라벨은 `bona=1`, `spoof=0` 그대로다. 스푸프 중심 모드는 다음을 적용한다.

```text
q = cosine(embedding, trained_spoof_center)

backend OC:
  spoof: (5/9) * softplus(20 * (0.9 - q))
  bona:     5  * softplus(20 * (q - 0.2))
  실제 OC 배치 전체에서 mean

backend pair:
  spoof long × spoof short의 모든 로컬 쌍
  mean(softplus(30 * (1 - pair_cosine)) - 0.3)

중심 단계:
  backend 갱신 후 spoof long을 다시 계산
  mean(5 * softplus(20 * (0.9 - q)))

높은 값이 bona인 최종 점수: -q
```

OC와 BPL이 요구하는 방향은 임베딩 공간에서 동시에 구성할 수 있다. 그렇다고 실제 신경망의 수렴이나 미지 공격의 일반화가 증명되는 것은 아니다. 특히 bona를 중심에서 밀어도 bona끼리의 다양성은 보장되지 않는다. 정규화 cosine의 극점에는 잘못된 분류에서도 embedding gradient가0인 정지점이 있을 수 있다.

이 최적화는 두 개의 조건부 목적식을 번갈아 갱신한다. 첫 단계는 중심을 고정한 `OC+pair`, 두 번째는 backend를 고정한 compact-class 중심 loss다. **단일 공동 손실이 매 단계 단조 감소한다고 주장하지 않는다.** 이전 단계의 embedding과 gradient는 다음 중심 갱신에 재사용하지 않는다.

중심 단계 계수5는 bona arm에서 원본과 같고 spoof arm에서도 같다. 이 통제는 손실 계수를 맞추는 것이며, Adam의 실제 이동량·표본 수·gradient 분산까지 같게 만드는 것은 아니다. 곱셈 후 평균하는 원본 연산 순서를 유지해 bona arm의 Adam moment까지 회귀 확인한다.

PyTorch의 동적 unused-parameter 계약에 따라 두 단계 DDP는 `find_unused_parameters=True`, `static_graph=False`를 사용한다. 고정 단계의 gradient는 `None`으로 지워 Adam 갱신에서 제외한다. 근거: [DDP 공식 문서](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html), [Adam 공식 문서](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html). 로컬 PyTorch2.13 CPU에서 양 역할·두 프로세스로 동작을 검증했으며 서버 PyTorch/CUDA/NCCL 조합은 별도 smoke 대상이다.

## 코드 구성

| 파일 | 역할 |
|---|---|
| `role_objectives.py` | immutable 역할 설정·검증, 원래 초기화의 OC 중심, backend/center 목적식과 점수 변환 |
| `role_model.py` | 기존 bridge·ASP·linear·LayerNorm 및 state key를 재사용하는 명시적 phase head |
| `role_training.py` | 원본 loader 순환·부분 배치, 전체 graph 두 단계 Adam, phase별 RNG 격리 |
| `role_lifecycle.py` | 역할 포함 원자 저장·hash commit·재로딩 검산, 공통 score 평가, save-first 학습 흐름 |
| `role_runtime.py` | 원본 dataset/sampler/loader 재사용, 데이터·source·frontend 고정, 실제 smoke와 학습 구성 |
| `tools/train_role_experiment.py` | 기본 read-only validate, 명시적 smoke/train, 기존 경로 덮어쓰기 거부·timeout·자신의 자손 정리 |
| `tests/test_role_experiment.py` | 원본 회귀·실제 sampler·부분 배치·합성108step·두 CPU rank·저장/평가/실패 검사 |

`baseline240_source`의5파일을 기존 manifest SHA로 확인하고 정상 import 가능한 `baseline240_source.data_loaders` 이름으로 사용한다. 실행 중인 원본·짧은 정상 음성의 3.0275초·2초 정렬 학습 snapshot, 원래 `ModelTrainer.train`, 기존 EBM-NoRepeat와 EBM-FullContext/native 경로는 건드리지 않는다. 이 경로는 동일한 원본 흐름을 명시적인 역할 선택으로 구현한 새 trainer이며 원본 함수를 무수정 호출한다고 설명하지 않는다.

실제 학습 입력은 원본64,600/48,440sample·부족하면 wrap·독립 RawBoost0~8·codec-off다. global240·두 GPU·보통12bona+108spoof/rank를 유지한다.106번째72행·마지막114행과 작은 loader 순환도 유지한다. 일반 spoof pair는108×108, 부분 spoof60일 때60×60이다. 전체 pair mean이므로81배 계수를 추가하지 않는다.

학습 backend phase별 RNG와 두 class loader generator를 분리해 중심 pass의 입력 수가 다음 epoch의 worker seed 등을 바꾸지 않게 한다. 이 RNG 격리는 역사적 단일 RNG 흐름과 다르므로 새 bona 통제 모델에도 똑같이 적용한다. 과거240 결과를 새 통제 모델과 동일한 실행으로 취급하지 않는다.

`train_rank*.jsonl`에는 클래스별 canonical score 평균·표준편차와 정규화 임베딩의 배치 내 분산 trace, backend/center gradient norm, 중심 norm·이동 norm·각도 변화를 기록한다. 배치 통계이므로 full-Train의 유효차원이나 일반화 성능을 대신하지 않는다.

## 읽기 전용 확인

`EBM/`에서 다음을 실행하면 설정과 manifest만 검사하고 종료한다. 출력의 `launched`는 `false`다. GPU·데이터 읽기·모델 다운로드·결과 폴더 생성은 없다.

```bash
python tools/train_role_experiment.py
python tools/train_role_experiment.py --compact-class bonafide
python tools/train_role_experiment.py --check-runtime
```

`--check-runtime`은 SoundFile·원본 loader·Transformers의 실제 lazy model import까지 확인한다. frontend weight를 읽거나 다운로드하지는 않는다. 로컬 검증에서는 기존 `artifacts/short32_code_validation_20261111/vendor`를 `PYTHONPATH`에 추가했다. 의존성을 전역 설치하거나 바꾸지 않았다.

## 실제 실행 진입 조건

이 구현 작업에서는 서버 실행을 하지 않았다. 실행 시 기존 작업·GPU 유휴와 전체 비용 범위를 확인한 뒤 기존 컨테이너, `CUDA_VISIBLE_DEVICES=0,1`에서 고유 root를 사용한다.

`--mode smoke`와 `--mode train`을 명시해야 GPU worker가 시작한다. 두 모드에는 `--run-root`와 정확한40자리 `--frontend-revision`이 필요하다. frontend는 `local_files_only=True`로 읽으므로 고정 모델이 cache에 없으면 중단한다. 기본 pipeline 시간 상한은168시간이며 `--timeout-hours`로 줄일 수 있다. 학습 재개·자동 재시도·대기열·앱 예약 기능은 없다.

smoke는 실제 원본 sampler의1·106·108번째 UID로120·72·114행의3논리 배치를 구성한다. 원본 dataset·증강을 사용하며 CPU에서 나머지105배치의 음성을 해독하지 않는다. 이어 CLS/TrainState/역할 정보 저장·복원, 두 rank 중심 동기화, 세 평가 집합에서 클래스×짧음/김 각16개씩 총64개/집합 추론을 검사한다. 각 rank의 peak CUDA memory도 기록한다.

본학습은 별도 새 root에서 `--smoke-receipt`를 요구한다. receipt의 역할·학습 seed/epoch·코드 SHA·frontend revision·loader 설정과 Train/eval protocol fingerprint가 같아야 한다. source가 바뀌었거나 smoke가 실패하면 다시 검증해야 한다. smoke의 optimizer와 checkpoint는 본학습에 재사용하지 않는다. 두 모드 모두 기존 폴더가 있으면 덮어쓰지 않는다.

원본 budget의50epoch와 seed1이 기본값이다. 명시적으로 다른 고정 epoch를 택하면 비교 양쪽도 그 budget을 맞춘다. train 중5epoch마다 저장 검증 후19LA를 평가하고, 학습 뒤 모든 저장점의21LA·DF를 평가한다. dev는 사용하지 않는다. 저장 실패는 다음 단계 진입을 막고, 평가 실패는 보존된 학습을 지우지 않으며 최종 상태를 `evaluation_incomplete`로 표시한다. 최소 EER는 사후값이고 checkpoint 선택의 독립 성능이 아니다.

## 저장과 평가 규약

```text
run_root/
  experiment.json
  model/CLS_5.pt
  model/TrainState_5.pt
  model/Model_5.json
  stages/checkpoint_5.complete.json
  stages/eval_5_19LA_scores.csv
  stages/complete.json
  run.exit.json
```

`Model_*.json`에 compact label·pair class·margin·가중치·center 계수·score 부호·architecture를 저장한다. `TrainState`에도 같은 metadata, optimizer·scheduler·rank RNG·source SHA를 저장한다. 세 파일을 읽어서 서로 일치하는지 검산한 뒤 SHA commit을 마지막에 쓴다. 완료 commit·역할 정보가 없거나 충돌하면 신규 로더가 거부한다.

신규 checkpoint는 **역할 대응 로더**로 평가한다. 기존 cosine 직접 계산 도구는 신규 중심 역할을 모르므로 그대로 넣어서는 안 된다.

```python
from role_lifecycle import load_scoring_head

head, manifest = load_scoring_head(run_root, epoch=50, device="cuda:0")
scores = head(hidden_states, mask=frame_mask, phase="score")
# scores["raw_center_cosine"]: 중심과의 원시 cosine
# scores["score_bona"]: 항상 높은 쪽이 bona
```

통제 평가는 앞64,600cap·비반복·mask·batch8·ASP1e-5다. 점수 파일에는 index·UID·raw cosine·canonical bona score·원래 label을 함께 기록한다. UID/index/label 중복·누락과 점수 부호를 검사한다. 같은 입력과 rank/step에서 양 역할은 같은 pooling RNG stream을 사용하며 평가가 학습 RNG를 바꾸지 않는다.

## 검증 범위

원본 bona 경로의 실제 `ModelTrainer.train`을 일반·부분 배치3개에서 실행해 손실·parameter·Adam state·scheduler를 정확히 대조했다. 신규 경로에서는 실제 원본 `DistributedSampler`/DataLoader의108배치 스케줄, 전체 합성108step의 학습→저장→세 평가, 양 역할의 Gloo 두 rank와 평가 중 한 rank의 고장 전파를 검사했다.

최초 구현의 신규32개와 기존 회귀46개, 총78개 검사가 통과했다. 이후 평가 연결 감사8개를 추가해 **최신86개 검사가 통과**했다. 이전 검사는 기존 보나 평가기로의 교차 로딩과 smoke 범위 표기를 놓쳤으며, 새 회귀 검사에서 이를 재현하고 차단했다. 최신 결과는 [코드 감사 결과](experience/_shared/references/artifacts/reverse_ebm_code_audit_20261112/validation.json)에 있다.

실제 크기인 backend3,744,224개와 중심160개도 양 역할 모두 원본과 초기 tensor가 일치했다. 기본 CLI와 runtime dependency 사전 검사도 통과했다. 최초 구현 당시 판정은 [이전 검증 결과](experience/_shared/references/artifacts/reverse_ebm_implementation_20261112/validation.json)에 보존했다.

결과와 로그는 `../artifacts/reverse_ebm_implementation_20261112/`에 있다. 실제 연구실 음성·XLS-R·CUDA/NCCL·GPU 메모리·수렴·EER 개선은 로컬 합성 검증으로 대체하지 않는다. 전체 설계와 비교 근거는 [설계서](experience/_shared/references/docs/experiments/ebm_spoof_centered_design_20261112.md)를 따른다.
