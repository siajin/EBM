# 원본240 + 원음2초 이하 정상의 L3,2 — 코드 검토용

2026-11-11 사용자 “학습 돌려봐”로 실행 승인을 받았다. 새 격리 snapshot은 기존 반복 진단의 정상 종료와 두 GPU 유휴 뒤 검증→50epoch 학습 순서로 시작하도록 준비한다. 아래0회/보류 설명은 구현 검증 시점의 기록이며 실제 실행 상태는 `../artifacts/short32_train_20261111/state.json`을 확인한다.
신규 지출 상한0USD, 서버 접속·GPU 작업·학습 대기 등록·실제 실험 optimizer 갱신0회.

## 원본 그대로 사용하는 부분

기준 commit: `10f3bcd057dc6a2ca687fbd5389ad6c9e15a3746`.
`baseline240_source/`의 main.py, train.py, data_loaders.py, model/b_linear.py,
model/loss.py는 해당 Git blob의 바이트 그대로 보관했다. manifest.json의 SHA256을
읽을 때 확인한다. 기존 EBM-FullContext/가변길이 코드와 공존하도록 opt-in 경로에서만 사용한다.

- `TrainSet` 및 `get_loaders()`를 그대로 호출한다. 정상·스푸프 **두 개의 실제
  torch DataLoader**와 각각의 DistributedSampler를 사용한다.
- Dataset 출력은 원래 `(long, short, label)` 세 값이다. custom collate, 새 batch
  sampler, 메타데이터를 추가한 batch, 길이 재묶기, 추가 원음 재표집을 사용하지 않는다.
- 원본 `ModelTrainer.train` 함수 객체를 직접 사용한다. 함수 본문을 바꾸거나 새로
  재현하지 않는다. 원본의 작은 loader 순환, 클래스 결합, 부분 배치도 그대로다.
- full graph의 backend 업데이트 → OC 중심 업데이트, 두 번의 Adam step, 원본
  scheduler, 클래스 가중치, 원래 OC/BPL 수식과 파라미터 key를 유지한다.
- 원래 long64,600 / short48,440 samples, 부족하면 wrap 반복, 각 view 독립
  RawBoost0~8, codec 변환 비활성을 그대로 사용한다.
- global240/두 rank이면 일반 배치는 GPU당120개(정상12/스푸프108), 106번째72개,
  마지막114개(정상6), 기존 BPL 쌍은 보통144개·마지막36개다.

이전 구현의 단일 loader 및 짧은 정상 음성의 3.0275초·2초 정렬 학습용 gradient replay 경로는 제거했다.
`native_steps.train_batch()`로 짧은 정상 음성의 3.0275초·2초 정렬 학습를 실행하려 하면 오류로 차단한다.

## 추가한 부분

`main.py`의 `short32_bpl=True` 분기 → `short32_original240.run()`으로 연결된다.
기본값은 False다. 원본 main의 모델·optimizer·scheduler 구성도 그대로 호출한다.

`Short32Hooks`는 DataLoader를 수정하지 않고 다음만 수행한다.

1. 정상 원음의 SoundFile header에서 실제 sample 수를 읽는다. 정상 원음
   **32,000 이하**만 대상이며,32,001부터 제외한다. 짧은 스푸프는 제외한다.
2. 원본 DistributedSampler의 순수한 epoch별 순서와 원래 batch 크기로 현재 정상
   행의 원음을 대응시킨다. sampler를 다시 열거해도 전용 Generator를 사용하므로
   실제 sampler 순서나 전역 난수를 소비하지 않는다. 작은 loader가 재순환하는
   경우도 같은 원본 batch 순서로 대응한다. batch 수·정상 수·라벨 순서를 검사한다.
3. 원본3초 frontend 입력을 관찰하고, 대상 행의 **이미 증강된 파형 앞32,000개**로
   2초 view를 만든다. 별도 원음 읽기, RawBoost, random crop을 추가하지 않는다.
4. frozen frontend에서2초 feature를 얻고, 같은 원본 backend로 embedding을 구한다.
   이 추가 frontend 호출은 unwrapped 모듈에 no_grad로 수행한다. rank마다 K가
   달라도 추가 DDP collective를 호출하지 않는다.
5. 기존 forward 결과에 조건부 BPL 보정만 더한다. 3초·2초 embedding 양쪽으로
   gradient가 전달되며, 기존 첫 번째 full-graph backward에 함께 포함된다.

원본 전체 정상 수를 N, 대상 정상 수를 K라 할 때:

```
A   = BPL(z4[:N], z3[:N])                # 기존 N×N 전체 쌍
A_E = BPL(z4[eligible], z3[:N])          # 대상 anchor K×N
B   = BPL(z3[eligible], z2)              # 대상 K×K 전체 양성쌍
L   = OC + A - K/(2N)*A_E + K/(2N)*B
```

이전에 선택한0.5/0.5 비중을 유지한다. K=N이면 `OC+0.5L43+0.5L32`,
K=0이면 원본 `OC+L43`과 정확히 같다. 단순히 계수1의 새 손실을 추가하는 안은 아니다.
혼합 배치에서는 대상4초 anchor행만 절반으로 줄인다. 같은3초 열이 다른 anchor에
기여하는 기존 항은 유지한다. K=1이면 L32는1쌍, K=2이면4쌍이다.

L4,2, 삼중 조합, BPL1, stop-gradient를 추가하지 않는다. 추가2초 backend의
dropout/ASP 난수는 rank·epoch·step에 따른 별도 fork_rng로 처리한다. 따라서
원래4/3 forward·중심 단계·DataLoader worker seed의 난수 흐름을 밀지 않는다.

## 평가·저장 — 저장 검증 후 단계별 실행

실제 학습 내부 함수는 그대로 두고 `short32_lifecycle.py`가 epoch 경계와 평가만
관리한다. dev 평가는 실행하지 않는다.

1. 시작 전 원본 소스 SHA·학습/평가/worker 모듈 import, 두 클래스의 오디오 header,
   원본 sampler·라벨·`module.loss.weight` 계약, 체크포인트 디렉터리 쓰기를 검사한다.
2.5/10/.../마지막 epoch가 끝나면 `CLS_E.pt`, `TrainState_E.pt`를 각각 같은 디렉터리의
   임시 파일에 기록·flush/fsync하고 최종 이름으로 교체한다. 두 파일을 CPU로 다시
   읽어 epoch·모델 tensor·유한성을 확인하고 SHA256/크기를 기록한다.
3. 마지막에 `short32_stages/checkpoint_E.complete.json`을 기록한다. 이 완료 표시와
   두 파일의 해시를 모든 rank가 확인하기 전에는 평가하지 않는다. 저장 실패는 모든
   live rank에 전달하고 다음 epoch나 평가로 진행하지 않는다. 완료된 이전 저장은 보존한다.
4. 해당 epoch의19LA를 평가한다. 평가가 실패하면 원인·rank·단계를 기록하고 다음
   epoch를 계속한다. 평가 오류를 학습 오류나 성공으로 처리하지 않는다.
5. 학습 종료 후 저장한 각 epoch를 순서대로 읽어 **21LA → 21DF**를 각각 평가한다.
   실패한 단계를 재시도하지 않고 다른 평가를 진행한다. 마지막 학습 모델 값은 복원한다.
6. `complete.json`에 단계별 성공/실패와 데이터셋별 최소 EER·해당 epoch를 기록한다.
   일부 평가가 실패하면 최종 상태는 `evaluation_incomplete`다. 최소 EER는 성공한
   평가 중 사후 최솟값이며 독립적인 모델 선택 성능이 아니다.

`TrainState`에는 모델·Adam·scheduler·epoch·필터링된 설정·원본 소스 manifest·
frontend revision 및 **rank별 Python/NumPy/Torch/해당 CUDA RNG 상태**를 함께 저장한다.
자동 학습 재시작/재개는 등록하지 않는다.

평가용 DataLoader는 기존 controlled 경로(비반복·앞64,600cap·mask·batch8)를 사용한다.
학습 DataLoader 두 개는 변경하지 않는다. 평가 head에만 state를 복사하고, frozen
frontend는 unwrapped로 호출한다. 배치마다 모든 rank가 오류를 공유하며, shard 크기가
달라도 같은 수의 상태 교환을 수행한다. 원음/worker/추론/검산 오류 때문에 한 rank만
나머지 rank의 결과를 기다리게 하지 않는다. 평가용 worker는120초 입력 timeout,
짧은 정상 음성의 3.0275초·2초 정렬 학습 분산 process group은 기본600초 timeout으로12시간 대기를 줄인다.
프로세스 강제 종료·GPU 드라이버 장애를 Python 예외 처리로 복구한다고 보장하지 않는다.

평가 전후에 학습 객체·모드·난수를 복원하고 optimizer는 갱신하지 않는다. 실행에 쓰지
않는 torchvision/오디오 증강 import를 main의 진입부에서 제거하고 transformers는
실제 실행 함수에서만 읽어, `--validate-only`가 GPU/모델 다운로드 없이 전체 진입부를
검사할 수 있게 했다. 동적 import 실패 시 부분 생성된 모듈 cache도 제거한다.

결과 위치(기존 결과 디렉터리 기준):

- `model/CLS_E.pt`, `model/TrainState_E.pt`
- `short32_stages/checkpoint_E.complete.json`
- `short32_stages/eval_E_19LA.json`, `eval_E_21LA.json`, `eval_E_21DF.json`
- `short32_stages/eval_E_PHASE_scores.csv`
- `short32_stages/complete.json`

## 검증 및 검토 위치

최신54개 검사 통과(저장/평가/오류 주입14 + 원본240/L32 18 + 기존 회귀22), 실패·오류·skip0개. 실제2worker 평가 모듈 import, 두CPU rank의 한쪽 평가실패·저장실패, unequal shard, 전체 합성 CPU 학습→저장→세집합 평가 연결을 포함한다. 이전40개 검사는 이전 구현의 기록이다.

- 실행 코드: `short32_original240.py`, 저장·평가: `short32_lifecycle.py`, 설정: `short32_bpl.py`.
- 소스 대조: `baseline240_source/manifest.json`, 원본 함수:
  `baseline240_source/train.py::ModelTrainer.train`.
- 회귀: `tests/test_short32_bpl.py`, `tests/test_short32_lifecycle.py`, 결과:
  `../artifacts/short32_robustness_20261111/final/validation.json`.
- 기준 상태·범위: 같은 artifacts 디렉터리의 `state.json`과 `plan.json`.

IRLab 디렉터리에서 다음 명령은 **검사만** 수행한다.

```powershell
python EBM/tools/validate_short32.py --vendor artifacts/short32_code_validation_20261111/vendor --report-dir artifacts/short32_robustness_20261111/review --patterns test_short32_lifecycle.py test_short32_bpl.py test_positive_bpl.py test_baseline240_mild.py test_conditional_bpl2.py
```

검증용 임시 CPU 모델에는 gradient/Adam 갱신을 수행한다. 실제 음성, XLS-R,
학습 checkpoint는 사용하지 않는다. 원본과 정확히 같은 K=0의 전체 epoch 업데이트,
활성 분기의 입력·난수 보존, 양방향 gradient, 원래 partial/cycling,
원본 RawBoost0~8 배열, 실제2worker DataLoader, K가 다른2rank CPU DDP 등을 확인한다.

실제 XLS-R·GPU 통합, GPU 메모리·속도, EER 개선 여부는 아직 검증하지 않았다.
메타데이터 대응은 원본의 정상적인 ordered loading을 전제로 한다. custom sampler나
collate로 교체하는 실험에는 그대로 재사용하지 않는다. 원본 소스의 일반적인 I/O
오류 처리 자체는 이번 변경 범위에서 수정하지 않았다.
