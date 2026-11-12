> 역사 문서 보존본. 원문 작성 당시의 상태·계획을 포함한다. 최신 상태와 확정 결과는 experience의 각 실험 README를 우선한다.

# EBM 원길이 학습과 3초 short crop 실험 설계

작성일: 2026-11-06. 상태: 설계 및 핵심 코드 구현 완료, 전체 학습 전. 후속 사용자 요청으로 실제 서버 코드에 반영하고 두 GPU의 제한된 update 검증을 통과했다. **실제 기본값은 기존 batch240·공식dev 전체를 보존했고 코덱 설정도 보존했다.** 아래 설계의 batch160·진단V14,983과의 차이 및 구현 범위는 [코드 변경 결과](ebm_native_length_short3_code_changes_20261106.md)를 기준으로 확인한다. 아래12run의 전체 학습은 아직 실행하지 않았다.

후속전체점검에서실시간codec두블록을주석처리해사용자의codec-off의도를반영했다. 확률값은유지했고실제호출0회를검증했다. 현재코드상태와실측계산비용은 [전체GitHub재점검](ebm_native_github_full_audit_20261106.md)에정리했다.

**Long-view와 평가 입력은 원길이를 보존하고, short-view는 기존의 최대 3.0275초 연속 crop을 유지한다.** 배치를 위한 0패딩은 XLS-R와 모든 시간 pooling에서 제외한다. 이를 구현 기준으로 삼고, 같은 short 구간의 반복·비반복 표현을 정렬하는 보조 손실은 대조군과 함께 검증한다. 기존의 정상 전체 교차 BPL은 유지한다.

목표는 짧은 정상의 반복 문맥에 대한 민감성을 줄이면서 스푸프 구분력을 유지하는 것이다. 정상 임베딩이 centroid로 가까워지는 현상만으로 개선을 판정하지 않는다. 원길이 전처리 변경의 효과와 반복 불변성 손실의 효과를 각각 측정한다.

## 1 고정한 결정과 검증할 가설

| 항목 | 이번 설계의 결정 |
|---|---|
| Short crop | `S = 303 * 160 - 40 = 48,440` samples, 16kHz에서 3.0275초. 정확히 3.000초로 바꾸지 않음 |
| 새 long-view | 원음 전체에 길이를 강제하는 crop·wrap을 적용하지 않음 |
| 새 short-view | 증강된 음성에서 길이 `min(실제 길이, S)`의 연속 구간을 선택. 부족한 길이를 반복하지 않음 |
| 평가 기본 입력 | 원음 전체, 발화당 long-view 점수 1개 |
| 배치 | 길이별 묶음, 논리적 전역 160개·GPU당 정상 8개+스푸프 72개 |
| 기존 BPL | GPU별 정상 long 8개 × short 8개의 전체 교차 비교 유지 |
| 추가 가설 | 같은 short 구간의 원길이/반복 입력을 정상 발화끼리 1:1로 정렬 |
| 반복 목표 길이 | 보조 branch에서만 `S`까지 wrap. 4.0375초로 늘리는 추가 조건은 이번에 넣지 않음 |
| 가중치 학습 | XLS-R 동결. Bridge·pooling·Linear·LayerNorm과 centroid는 기존 두 단계 방식으로 학습 |
| 기본 판정 | 50 epoch 고정. target 평가 결과로 checkpoint·손실 계수·seed 선택 금지 |

검증할 가설은 다음 세 가지다.

1. **H1 원길이 학습:** 반복·고정 절단을 제거한 학습으로 짧은 정상의 오류를 줄일 수 있는가?
2. **H2 반복 불변성:** 같은 short 구간의 반복/비반복 정렬 손실이, 동일한 branch에 분류 손실만 적용하는 것보다 도움이 되는가?
3. **H3 구분력 유지:** 정상 오류가 줄어도 스푸프 점수가 함께 높아져 판별력이 나빠지는 문제를 피할 수 있는가?

3.0275초가 최적이라는 가설은 세우지 않는다. 고정 3초와 상대 비율 crop의 비교, 새로운 초단 crop 생성, XLS-R fine-tuning은 이번 주 실험에서 제외한다.

## 2 설계의 근거와 범위

### 2.1 실제 데이터와 기존 관찰

학습 T는 원래 19LA train 25,380개다. 정상 2,580개 중 원음이 3초 미만인 것은 986개(38.22%), 2초 미만은 109개(4.22%), 1.5초 미만은 9개(0.35%)다. **3초 미만 정상 전체가 드문 것은 아니며, 특히 1~2초의 정상 노출이 적다.**

증강 전 `N < S`인 정상은 1,040개(40.31%)다. 이들이 short를 S까지 반복할 때 실제 입력 개입을 받는다. 원래 길이가 S 이상이면 short crop과 그 반복 branch는 같은 파형이므로 추가 정렬 손실에서 제외한다. 코덱 후 길이가 바뀌므로 실제 유효 쌍 수는 학습 중 다시 측정한다. 1,040개를 매 epoch의 확정 유효 쌍 수로 사용하지 않는다.

기존 고정 모델의 선택 표본 471개에서는 짧은 원음의 반복을 제거하면 일부 VCC 정상 표현이 크게 개선됐지만 스푸프 통과도 늘었다. 선택 표본 EER는 반복 3.57%, 비반복 9.57%였다. 이 비교에서 긴 음성은 여전히 첫 4.0375초를 사용했다. 따라서 이 값은 전체 원길이 모델의 성능이나 전체 DF 성능이 아니다. **정상에 대한 반복 불변성과 양 클래스의 분리 학습을 함께 검증해야 한다는 근거**로 사용한다.

학습 전체 파일의 헤더 검사는 16kHz·mono·샘플 수 일치를 확인했다. 원길이 0.4~29.312초 forward, frontend+26개 pooling mask, 실제 long/short waveform의 OC+BPL backward도 소규모 검사에서 통과했다. 단독/배치 cosine 점수 최대 차이는 약 5.96e-8이었다. 이 결과가 전체 DDP 학습·메모리·일반화 성능의 검증을 대신하지는 않는다.

### 2.2 추가 손실의 의미

예를 들어 증강 후 1.5초 short 파형을 `U`로 고정한다. 동일한 U를 반복해 S까지 늘린 파형이 `W`다. 같은 crop·같은 codec·같은 RawBoost 결과를 공유하므로 `U ↔ W`는 포함된 고유 내용과 증강의 종류를 고정한 입력 개입이다. 이 정렬은 반복 횟수와 총 문맥 길이의 변화에 대한 불변성을 요구한다. 반복 주기성만의 물리적 원인을 분리하는 실험은 아니다.

`U`와 W가 모두 이동 가능한 대칭 손실을 사용한다. 원길이 표현이 언제나 정답이라는 teacher 가정을 두지 않는다. 대신 OC 손실이 두 표현을 올바른 클래스 방향으로 유도한다. 정상 쌍만 정렬하고 스푸프에는 각 branch의 분류 손실을 적용해, 모든 스푸프를 서로 모으는 새 목적함수를 만들지 않는다.

마스크된 0패딩과 원길이의 정렬은 주 가설이 아니다. 올바른 mask에서는 이미 거의 같은 함수다. 여기서 W의 반복 sample은 **유효 입력**이며 마스크로 가리지 않는다. 무효로 표시하는 것은 batch 형상을 맞추는 0패딩뿐이다.

## 3 실제 EBM 구조와 유지할 학습 계약

### 3.1 Forward 경로

기준은 `/home/mentee/results/CM/LA/EBM_e50_bs160_lr1e5/backup/scripts`와 현재 `/home/mentee/code/EBM`의 핵심 소스다. 이번 읽기에서 `train.py`, `data_loaders.py`, `main.py`, `model/b_linear.py`, `model/loss.py`의 현재 hash는 기존 구조 검사와 같았다.

```text
waveform [B, N] + waveform_mask
    → frozen XLS-R 300M
    → 25 hidden states [B, 25, T, 1024] + frame_mask
    → 각 layer의 Linear1024→128 / ReLU / Dropout0.35 / Linear128→128 / LayerNorm
    → 25개 masked ASP가 layer별 요약 생성
    → layer attention으로 [B,T,128] 결합
    → 최종 masked ASP → [B,256]
    → Linear256→160 → LayerNorm160 → z [B,160]
    → cosine(z, centroid)와 OC/BPL 손실
```

Layer attention의 25개 layer 축 softmax는 유지한다. 그 attention을 만드는 각 시간 요약에 mask를 적용한다. 최종 시간 pooling까지 총 26개다. Linear와 LayerNorm은 frame 개수에 고정되지 않으므로 서로 다른 길이의 입력도 같은 160차원 z가 된다. 가중치 shape와 state dict key는 유지하고 forward에 mask와 embedding 반환 경로를 추가한다.

현재 ASP는 `sum(a_t*x_t)`와 `std(a_t*x_t + noise)`를 결합한다. 일반적인 weighted variance로 바꾸지 않는다. `noise_std=1e-5`, sample std의 correction=1, LayerNorm epsilon=`1e-12`를 보존한다.

### 3.2 BPL 구현과 논문의 구분

실제 `model/loss.py`는 `normalize(X) @ normalize(Y).T`의 모든 원소에 Softplus를 적용하고 평균한다. 원래 논문의 같은 발화 증강 쌍 설명과 달리 **현재 구현은 정상 전체 교차 쌍을 포함**한다. 이 기존 손실을 추가 1:1 손실로 대체하면 정상 전체 수축과 반복 불변성의 변경을 동시에 만들게 된다. 따라서 기존 손실을 보존하고 새 항을 분리한다.

현재 소스는 GPU 간 BPL embedding all-gather가 없는 baseline이다. 과거 전역 BPL 실험 archive에는 all-gather가 있다. 이번 기본 설계는 **GPU별 8×8 BPL**로 고정하고, 전역 16×16으로 조용히 변경하지 않는다. 전역 BPL은 별도 후속 비교로 둔다.

### 3.3 두 단계 optimizer

기존 `train.py`는 한 논리적 배치에서 다음을 순서대로 수행한다.

1. Centroid를 고정하고 mixed-class OC+BPL로 encoder θ를 갱신한다.
2. θ를 고정하고 **갱신된 θ의 정상 long embedding**으로 centroid c를 갱신한다.

새 설계에서도 순서와 갱신 횟수를 유지한다. 단계 2에 단계 1 전의 z를 재사용하지 않는다. XLS-R hidden states는 frozen이므로 계산을 재사용할 수 있지만, trainable backend의 정상 long forward는 다시 수행해야 한다. 새 원길이 조건의 centroid 갱신은 실제 평가와 같은 원길이 long 정상만 사용한다. 반복 보조 branch로 centroid를 직접 갱신하지 않는다.

## 4 입력 생성과 batch contract

### 4.1 증강과 crop 순서

E1/E2/E3에서 발화 i의 원음 x_i로부터 다음을 만든다. augmentation RNG는 `(seed, epoch, UID, occurrence_id, branch)`로 고정한다. 같은 UID가 배치 완성을 위해 재샘플링되면 occurrence_id를 달리한다.

1. **L:** 원음 전체를 읽고, 독립적인 codec 및 RawBoost를 적용한다. 기존 codec 전의 `1.2 * long_crop_size` 제한과 이후 64,600 crop/wrap을 제거한다. 증강 결과의 전체 길이를 유효 길이로 기록한다.
2. **U:** 원음을 다시 독립적으로 codec 증강한다. codec 결과에서 `min(N_aug, S)`의 연속 구간을 균일 random start로 선택한 후, 독립 RawBoost를 적용한다. 기존 short codec 전의 `1.2 * S` 사전 절단도 제거한다. RawBoost 후 실제 길이를 다시 확인한다.
3. **W:** 완성된 U를 그대로 공유한다. `len(U) < S`이면 `np.pad(U, (0,S-len(U)), mode='wrap')`, 그렇지 않으면 U와 동일하다. W에 별도 codec·RawBoost를 추가하지 않는다.

Codec 적용 확률 0.8, codec 후보/품질 분포, RawBoost의 0~8 선택 분포는 archive와 맞춘다. L과 U의 증강은 독립이며 U와 W의 waveform 증강은 공유한다. 코드의 `CodecAugmentationLight.convert()`는 전달받은 codec/quality 인자를 내부 선택으로 덮어쓰므로, seed 통제와 선택값 반환 경로를 구현해 실제 선택을 기록한다.

MP3/AAC의 디코딩 결과가 원음보다 길어지는 현상이 확인됐다. 이번 기본안은 증강 결과 길이를 사실대로 사용하며, 검증되지 않은 encoder delay 제거·무음 제거·끝부분 임의 보정을 추가하지 않는다. **원음 전체 보존은 고정 상한으로 버리지 않는다는 뜻이며, codec 증강 뒤 샘플 수까지 원본과 같다는 뜻은 아니다.** U는 codec 뒤 S 상한을 적용한다. 증강 없는 평가 입력은 원음 전체 그대로다.

E0은 archived TrainSet의 순서와 codec 사전 crop을 그대로 사용한다. 그러므로 E1−E0은 반복만의 효과가 아니라 고정 crop·wrap·증강 문맥 변경을 포함한 **입력 처리 묶음의 효과**다. 반복 정렬 손실 하나의 효과는 입력을 완전히 공유하는 E3−E2로 판단한다.

### 4.2 반환 항목과 마스크

```text
Dataset item:
    uid, occurrence_id, y, L, U, W,
    length_L, length_U, length_W,
    pair_active = (length_U < S),
    raw_samples, crop_start, codec_choice, augmentation_seed

Collate output, branch v ∈ {L,U,W}:
    input_v: float32 [B,max(length_v)]      # 오른쪽 0패딩
    sample_mask_v[b,t] = (t < length_v[b])
    labels: int64 [B]                      # 정상 먼저, 스푸프 나중
```

L/U/W는 각각 별도로 batch 최대 길이에 맞춘다. 서로 다른 branch의 time 길이를 같게 만들 필요는 없다. E2/E3에서는 세 branch 모두 정상·스푸프에 대해 생성해 label별 전처리 차이를 만들지 않는다. E0/E1도 기존 비교를 위해 두 branch를 전 클래스에 대해 계산한다.

W의 `length_W=S`이며 반복 영역도 mask=1이다. S보다 짧은 U의 batch 0패딩은 mask=0이다. 파형 값이 0인지로 mask를 추정하지 않는다. 원래 존재하는 0 sample도 유효 입력이다.

XLS-R의 실제 config는 `feat_extract_norm='layer'`, stable layer norm이다. waveform mask를 `attention_mask`에 전달한다. convolution-valid frame 수는 다음이며, 설치된 모델의 helper 결과와 대조한다.

```text
T(n) = floor((n - 400) / 320) + 1
frame_mask[b,t] = t < T(length_v[b])
```

`T≥2`, 즉 최소 720 samples를 요구한다. 현재 확인한 실제 데이터는 이를 만족한다. 예외 파일은 실패 로그에 남기고 원인을 확인하며, 음성 label·source에 따라 조용히 버리거나 반복으로 복구하지 않는다.

### 4.3 Masked ASP

시간 softmax에서는 tanh 이후 invalid logit을 `-inf`로 만든다. `weighted_t = a_t*x_t`, `q_t = weighted_t + noise_t`, 유효 frame 수 n에 대해 다음을 계산한다.

```text
mu = sum_valid(weighted_t)
q_bar = sum_valid(q_t) / n
sigma = sqrt(sum_valid((q_t - q_bar)^2) / (n - 1))
ASP = concat(mu, sigma)
```

padding에 noise를 생성하더라도 mask된 평균·분산에서 제외한다. 모든 frame이 유효한 입력은 원래 ASP와 같은 함수여야 한다. train의 noise와 dropout은 유지하고, 함수 동일성 QC에서만 noise=0·eval mode를 사용한다.

## 5 비교 조건과 손실의 정확한 정의

### 5.1 네 개 학습 조건

| 조건 | L / U 입력 | 분류 손실 | 기존 교차 BPL | 추가 반복 쌍 손실 |
|---|---|---|---|---|
| E0 고정 길이 대조 | archived long4.0375/short3.0275, 부족하면 wrap | L만 | 유지 | 없음 |
| E1 원길이 기준 | 전체 L / 최대3.0275초 U, wrap 없음 | L만 | 유지 | 없음 |
| E2 보조 분류 대조 | E1과 동일 + W | L/U/W 가중 평균 | 유지 | 없음 |
| E3 반복 정렬 후보 | E2와 동일 | E2와 동일 | 유지 | 같은 정상 U/W 쌍에만 추가 |

네 조건에 동일 UID batch schedule을 사용한다. E0도 길이별 공통 sampler를 사용하므로 과거 original sampler의 완전 재현 모델은 아니다. 기존 archive 모델은 역사적 참고로 따로 표시한다. 새 E0가 비교의 학습 대조군이다.

- E1−E0: 원길이 입력 처리 묶음의 효과.
- E2−E1: 추가 short/wrap 분류 학습 및 long 분류 가중치 재분배의 효과. 단일 요인 효과로 부르지 않음.
- **E3−E2: 같은 입력·분류 손실·기존 BPL에서 추가 반복 정렬 항의 효과. 주 비교.**

### 5.2 OC와 기존 BPL

`s(z,c)=cos(z,c)`, 정상 y=1, 스푸프 y=0이다. OC margin은 정상 0.9, 스푸프 0.2, alpha=20이다. class weight는 정상 5, 스푸프 5/9이며, 각 branch에서 weight를 곱한 per-example loss를 논리적 batch 80개로 평균한다.

```text
o(z,1;c) = 5   * softplus(20 * (0.9 - s(z,c)))
o(z,0;c) = 5/9 * softplus(20 * (s(z,c) - 0.2))
O_v = (1/80) * sum_i o(z_i^v, y_i; c)

g(t) = softplus(30 * (1-t)) - 0.3
B_old = (1/8^2) * sum_{i,j in local bonafide} g(cos(z_i^L,z_j^U))
```

E0/E1의 stage-1 loss는 `O_L + B_old`다. E0에서는 U 자리에 archived short branch가 들어간다. 기존 offset `-0.3`을 유지한다. 이 BPL의 최소값이 0이 아니라는 점도 유지한다.

### 5.3 추가 반복 쌍 손실

유효 정상 집합 `M_r={i: y_i=1 and length_U_i<S}`를 rank r마다 구한다. 새 손실은 **같은 UID·같은 occurrence의 U와 W만** 비교한다. 다른 UID와의 교차 조합을 만들지 않는다.

```text
h(t) = softplus(30 * (1-t)) - log(2)
P = mean_{i in all-rank active normal pairs} h(cos(z_i^U,z_i^W))
```

`-log(2)`는 cosine=1에서 penalty를 0으로 표시하기 위한 상수다. 기존 g와 cosine에 대한 gradient는 같다. 두 branch 모두 gradient를 받으며 stop-gradient·teacher·새 centroid는 도입하지 않는다. 전체 active 쌍이 0개인 step은 P=0으로 정의한다.

기본값은 `rho=0.25`, `lambda_pad=0.10`, 기존 BPL 계수=1이다. 이 계수는 신규 고정 가설값이며 최적화된 수치가 아니다. DF/21LA 결과를 본 뒤 수정하지 않는다.

```text
O_mix = (1-rho)*O_L + (rho/2)*O_U + (rho/2)*O_W
Loss_E2_stage1 = O_mix + B_old
Loss_E3_stage1 = O_mix + B_old + lambda_pad*P
```

분류 손실 계수의 합은 1이다. branch가 늘었다는 이유로 OC 전체를 세 배로 늘리지 않는다. E2/E3의 동일한 U/W 분류 학습이 정렬 방향을 정상 중심과 스푸프 분리 쪽으로 유도한다. 정렬 항 하나만으로 올바른 위치가 정해진다고 가정하지 않는다.

### 5.4 DDP reduction과 centroid 단계

기존 BPL은 rank별 8×8 평균, OC는 rank별 80개 평균이다. DDP 평균 후 이 두 항은 rank 평균 목적함수가 된다. 새 P만 유효 쌍 개수가 rank마다 다를 수 있어 아래 정규화를 쓴다.

```text
global_M = all_reduce_sum(local_M)
local_pair_sum = sum_{i in M_r} h(cos(z_i^U,z_i^W))
P_for_backward_on_rank_r = world_size * local_pair_sum / max(global_M,1)
```

DDP가 gradient를 world size로 평균하므로 위 식으로 전역 유효 쌍 평균 gradient가 된다. loss 표시에 같은 값을 그대로 전역 평균이라고 쓰지 말고, detached local sum/count를 모아 전역 P를 기록한다. `global_M=0`이면 graph에 연결된 `0*z` 형태의 0을 반환한다. 모든 rank가 count collective와 같은 수의 forward/backward에 참여해야 한다.

모든 조건의 stage 2는 해당 조건의 L에서 정상만 뽑아 `mean_bona o(z_L,1;c)`를 계산한다. 정상 8개의 weight=5와 갱신 전후 순서를 유지한다. 단계 1은 c gradient=None, 단계 2는 θ gradient=None으로 두고, 각 단계 전에 `optimizer.zero_grad(set_to_none=True)`를 호출한다. freeze한 파라미터에 남은 0 gradient나 Adam momentum으로 의도하지 않은 갱신이 생기지 않는지 검사한다.

## 6 길이별 배치와 메모리 설계

### 6.1 공통 sampler

모든 seed·epoch에서 T의 정상·스푸프 UID를 각각 shuffle한 뒤 전역 정상16/스푸프144, 162 step을 만든다. 모든 25,380개 발화를 한 번 이상 포함하고, 마지막 완성을 위해 정상12회·스푸프528회 재샘플링한다. class별 32 step 규모의 무작위 후보군 안에서 원음 길이로 정렬한 뒤 같은 길이 순위의 두 클래스를 결합한다. 완성된 step 순서는 다시 shuffle한다. rank별 정상8/스푸프72와 정상 먼저 순서를 보존한다.

E0~E3는 동일 seed의 epoch별 UID·occurrence schedule을 공유한다. 증강 결과 길이는 batch padding에 사용하고, source 또는 점수로 sampler를 조정하지 않는다. 각 half-second 구간에 1:9를 강제하면 정상이 없는 구간이 생기므로 고정 길이 bin별 강제 sampling은 하지 않는다.

같은 길이끼리 묶으면 기존 교차 BPL이 만나는 정상 길이 범위도 좁아진다. 이 효과를 공통 sampler로 통제한다. 원래 무작위 sampler와 길이별 sampler의 효과를 분리한 후속 비교는 가능하지만 기본 네 조건의 차이로 해석하지 않는다.

### 6.2 실행 메모리와 논리적 batch 보존

논리적 batch는 160으로 고정한다. 24GB GPU당 80개 원길이 입력의 학습 수용량은 아직 확인되지 않았다. 평균 padding이 줄어도 최장 학습 원음은 659 frames로 기존201보다 길다. 긴 음성을 몰래 4초·8초로 자르거나, 긴 batch만 정상 수를 줄여 해결하지 않는다.

물리 microbatch를 GPU당 최대8개로 시작한다. 필요한 경우 4 또는2로 줄이되 **네 조건에 같은 physical size와 같은 계산 경로**를 적용하고 아래 gradient-cache 방식을 검증한다. 단순히 microbatch별 BPL을 계산해 더하면 전체8×8 교차 항이 사라지므로 허용하지 않는다.

1. 고정된 waveform과 각 branch 길이를 준비한다. waveform 증강은 한 번만 만들고 서버 RAM 안에서 유지한다.
2. Classifier train mode의 no-grad pass로 microbatch별 L/U/W z를 모은다. backend 직전 CPU/CUDA RNG 상태를 저장한다. 소형 `[80,160]` 표현만 모으고 전체 SSL hidden states를 디스크에 저장하지 않는다.
3. z를 detached leaf로 두고 논리적 batch의 OC·전체8×8 BPL·유효 쌍 P를 한 번 계산해 `dLoss/dz`를 구한다.
4. 각 microbatch의 frozen XLS-R를 재계산하고, **backend 직전 저장한 RNG를 복원**해 같은 dropout/noise로 z를 재계산한다. 저장한 embedding gradient를 chain rule로 θ까지 역전파한다. 재계산된 z를 숫자로 곱한 surrogate만 미분하며 embedding gradient 자체로 다시 고차 미분하지 않는다.
5. 논리적 batch 전체의 θ gradient를 DDP 평균한 뒤 optimizer step을 한 번만 수행한다. no_sync를 쓴다면 forward와 backward 모두 그 context 안에 두고 마지막 replay에서 동기화한다. autograd의 default 평균에 추가 world-size 나눗셈을 중복하지 않는다.
6. 갱신된 θ에서 정상 L을 다시 계산하고 centroid 단계의 loss를 정상8개 전체 기준으로 평균해 별도 optimizer step을 한 번 수행한다.

이는 [gradient caching](https://aclanthology.org/2021.repl4nlp-1.31/)의 chain-rule 방식을 이 목적함수에 적용하는 설계다. 현재 EBM의 두 단계 optimizer와 DDP에서 검증된 구현이라는 뜻은 아니다. 먼저 작은 full-graph 기준과의 gradient·Adam update 일치를 검증해야 한다. 이 기준은 **같은 microbatch 경계·padding shape·RNG로 계산한 모든 graph를 유지한 뒤 전체 손실을 계산하는 기준**이다. dropout을 켜고 batch80을 한 번에 계산한 경우와 다른 shape의 RNG 배열이 자동 일치한다고 가정하지 않는다. frozen XLS-R 재계산 때문에 시간 비용은 늘 수 있다.

기존 OC 구현에는 `batch_size > 4`일 때만 class weight를 적용하는 조건이 있다. microbatch 크기로 이 조건을 판단하면 손실이 바뀐다. 새 helper는 **논리적 mixed batch80과 정상 batch8**을 기준으로 앞 절의 수식을 계산해 physical size와 무관하게 동일한 가중치를 사용한다. 고정된 논리적 배치를 바꾸려면 별도 프로토콜 변경으로 기록한다.

## 7 학습 설정과 초기화

| 항목 | 설정 |
|---|---|
| 학습 데이터 | T25,380개, 정상2,580/스푸프22,800 |
| 초기화 | 기존 캐시에서 식별한 XLS-R revision `1a640f32ac3e39899438a2931f9924c02f080a54`를 후보로 삼고, 실제 사용 파일 전체 hash·로컬 완전성을 실행 전에 고정. Backend와 centroid는 seed별 신규 초기화 후 네 조건에 동일 state를 복제 |
| 기존 epoch50 checkpoint | 구조·strict-load·함수 재현 QC용. 주 학습의 warm start로 사용하지 않음 |
| Seeds | 1, 2, 3. 모든 조건에 동일 seed 목록 사용 |
| Epoch | 50 고정, 조기 종료 없음, epoch50이 주 checkpoint |
| Optimizer | Adam, lr=1e-5, weight_decay=1e-4, betas=(0.9,0.999), eps=1e-8 |
| Scheduler | 기존 CosineAnnealingWarmRestarts, T_0=150, T_mult=1, eta_min=1e-5. 초기 lr와 min이 같아 실질적으로 상수 lr |
| Frontend | 50 epoch 모두 eval mode·requires_grad_(False)·no_grad. 원 코드의 `require_grad` 오타를 그대로 복제하지 않음 |
| Backend | 원래 dropout0.35, pooling noise1e-5, 160차원, OC/BPL 설정 유지 |
| DDP | 기존 컨테이너, CUDA_VISIBLE_DEVICES=0,1, world_size=2 |
| Precision | 최초 기준 FP32. TF32 off, AMP 미사용. 메모리 해결을 위해 조건별 precision을 다르게 바꾸지 않음 |
| Optimizer steps | 논리적162 step/epoch ×50=8,100회 encoder update와8,100회 centroid update, run당16,200회 |

주 비교는 네 조건×세 seed, 총12개 학습이다. seed1의 네 조건을 먼저 실행해 구현·자원 이상을 확인한 뒤 seed2/3을 이어간다. seed1 target 성능으로 조건을 삭제하거나 계수를 고치지 않는다. 자원·구현 문제로 프로토콜 변경이 필요하면 target 평가 전 새 버전을 만들고 모든 해당 조건에 같은 변경을 적용한다.

모델 초기화·데이터 증강·sampler·backend stochastic RNG를 분리한다. 같은 seed라도 time shape가 바뀌면 dropout mask가 같다는 보장은 없다. 동일 실행의 재현성과 gradient-cache pass/replay의 동일성을 보장하고, 조건 간 동일 seed만으로 완벽한 stochastic 통제가 됐다고 표현하지 않는다.

## 8 평가와 성공 판정

### 8.1 데이터 역할과 평가 시점

| 데이터 | 역할 |
|---|---|
| T 전체 | 유일한 가중치 학습 데이터, padding pair의 source |
| V14,983, 기존 19LA dev10화자 | 주어진 전처리의 구현 점검, epoch별 모니터링, epoch50 운영 threshold 고정 |
| Q9,861, 나머지 dev | 선택에 쓰지 않는 추가 진단. 과거에 이미 관찰한 자료이므로 새 독립 holdout으로 부르지 않음 |
| DF_pilot20,000, 정상2,000/스푸프18,000 | 주 성능 비교, 기존 고정 ID 재사용 |
| 21LA_pilot20,000, 정상2,000/스푸프18,000 | 동일 표본 규모의 전이 guardrail, 기존 manifest 재사용 |
| 19LA eval71,237 | 알려진 benchmark의 성능 보존 확인 |
| 전체 DF533,928 / 전체21LA148,176 | 설정·모델·threshold lock 후 보조 전체 평가. pilot과 합산해 독립 표본으로 세지 않음 |

50 epoch와 loss 계수는 고정한다. 학습 중 target 평가를 호출하지 않는다. 기존 `evaluation()`은 19LA/21LA/DF를 반복 평가하고 target 기준 best를 저장하므로 새 runner의 학습 선택 경로에서 제거한다. V만 확인하고 **12개 run의 설정·epoch50 checkpoint·V threshold를 모두 잠근 후** target을 평가한다. 이미 분석한 target 데이터이므로 전체 연구는 탐색적 후속 연구이며 미관측 target에 대한 확증으로 보고하지 않는다.

기존 F:의 학습 XLS-R 마지막 layer 캐시나160차원 h/z 캐시로 이번 학습을 대체하지 않는다. 새 view·25개 layer·trainable backend의 gradient가 필요하다. 원음은 기존 서버 data mount에서 읽는다.

### 8.2 전처리별 평가

모든 checkpoint를 두 정책으로 평가한다.

- **P_native:** 원음 전체 + 유효길이 mask. 새 모델의 실제 평가 정책이며 주 비교 E3−E2도 이 정책이다.
- **P_legacy:** 첫64,600 samples / 부족하면 wrap. 호환성과 반복 민감성을 보는 보조 진단이다.

E1−E0 등 학습 조건 차이를 계산할 때는 같은 평가 정책을 사용한다. E0/P_legacy와 E1/P_native의 차이는 train+test 처리 묶음의 효과로 별도 표시한다. E3는 추론에서 W나 short branch를 사용하지 않고, 모든 클래스에 동일한 P_native long 점수 하나를 출력한다.

### 8.3 지표와 threshold

주 지표는 DF_pilot의 E3−E2 EER 차이, 단위 percentage point다. seed별 값과 세 seed 평균을 모두 보고한다. 부 지표는 AUROC, V threshold에서 정상 FRR·스푸프 FAR, 두 오류 평균이다. EER가 좋아져도 운영 오류가 반대 방향이면 함께 표시한다.

각 checkpoint·평가 정책마다 V의 spoof FAR≤1%를 만족하는 가장 낮은 threshold를 고정한다. 후보는 spoof unique score의 `nextafter(score,+inf)`와 양끝 경계, accept는 `score>=threshold`다. 동점을 나누지 않고 실제 달성 FAR/FRR을 기록한다. DF EER threshold를 운영 threshold로 가져오지 않는다.

길이 `[0,1.5), [1.5,2), [2,3.0275), [3.0275,4.0375), [4.0375,∞)`별 두 클래스 오류를 모두 보고한다. DF는 ASV/VCC2018/VCC2020, codec, speaker group별로 함께 확인한다. 한 클래스가 없거나 극소수인 구간은 EER를 꾸며내지 않고 N·FRR·FAR와 NA 사유를 기록한다.

기계적 진단은 같은 U/W 쌍의 정상 cosine 거리, centroid 각도, `abs(score_U-score_W)`와 함께 스푸프 점수·클래스 간 분리도 변화를 측정한다. 평가 시 paired crop은 UID별 고정 random start를 사용한다. train에서는 원래 noise/dropout을 유지하고, 동일성 QC에서만 noise를 끈다. 실제 평가에서는 eval mode, noise1e-5와 발화별 고정 seed를 사용한다.

### 8.4 비교 불확실성과 채택 조건

주 비교 E3−E2에는 source별 speaker cluster paired bootstrap5,000회를 적용한다. 같은 draw를 모든 조건·seed에 사용하고, 각 draw에서 seed별 EER 차이와 그 평균을 다시 계산한다. seed를 재표집해3개만으로 광범위한 학습 모집단 CI를 만들지 않는다. 세 seed의 차이와 범위를 별도로 보이며 bootstrap은 **학습된 모델들을 고정한 평가 표본의 불확실성**으로 해석한다. 미해결 원발화 family와 두 target의 공유 의존성은 남는다.

E3를 유망한 후보로 판단하려면 다음을 함께 충족해야 한다.

1. DF_pilot의 평균 EER 차이 E3−E2가 음수이고 paired bootstrap95% 상한도0 미만이다.
2. seed별 방향을 모두 보고하며, 특정 seed 하나만 고른 결과가 아니다. 세 seed 중 역방향이 있으면 안정성 미확인으로 표시한다.
3. 21LA_pilot과19LA eval의 평균 EER 악화가 각각0.20pp 이내다. 이는 이번의 사전 engineering guardrail이며 통계적 비열등성 증명은 아니다.
4. V에서 고정한 threshold의 DF_pilot spoof FAR 악화가0.20pp 이내이고, 짧은 정상 FRR 개선 여부를 함께 확인한다. 정상 각도만 줄어드는 것으로 채택하지 않는다.

E1−E0, E2−E1과 세부 source/길이 분석은 부차적 탐색 비교로 표시한다. 주 비교 실패 후 유리한 부 지표를 주 가설로 바꾸지 않는다. 실패해도 수치와 비용을 모두 보존한다.

## 9 구현 파일과 변경 범위

별도 작업 경로와 결과명을 사용하고 원래 `/home/mentee/code/EBM`·기존 archive는 보존한다. 상대 경로 아래 파일명은 구현 예정안이다.

| 파일 | 구체적 변경 |
|---|---|
| `arguments.py` | S=48440, native_long, loss weights, local BPL, seed, logical/physical batch, eval policy를 명시. 원래 결과명 재사용 금지 |
| `data_loaders.py` | 하나의 class-balanced bucket batch sampler, UID/occurrence 포함 variable-length item, branch별 collate, 실제 길이 반환 |
| `data_processing/augmentation/codec_aug.py` | 선택 codec/quality와 출력 길이 관측 가능하게 분리. codec 전 고정길이 crop은 새 loader에서 제거 |
| `model/b_linear.py` | embedding 반환 경로, mask/mask_short, bridge25개+최종ASP의 masked sum/sample std. 기존 parameter shape/key 유지 |
| `model/loss.py` | 기존 all-pairs BPL 보존. `PairedPaddingLoss`는 UID가 일치하는1:1 쌍·active mask·sum/count를 반환. OC per-example helper 분리 |
| `train.py` | 논리적 batch 단위의 두 단계 업데이트, 필요 시 RNG replay gradient cache, c/θ freeze 검증, V 전용 모니터링 |
| `main.py` | XLS-R snapshot pin·freeze, 공통 initial state, DDP 및 optimizer 구성, target best-model 선택 제거 |
| 별도 `evaluate.py` | 정책별 score·threshold·길이/source 오류, 한 발화당 한 점수·중복 검수 |

`B_Linear`를 embedding-only 경로로 호출할 때 기존 forward 안의 OC/BPL이 중복 계산되지 않도록 API를 분리한다. c를 별도 head로 감싸더라도 state dict 변환표와 strict-load QC를 제공한다. DDP wrapper 바깥에서 loss를 계산하는 변경은 unused parameter 추적과 두 단계 freeze까지 검증한 뒤 사용한다.

## 10 구현 완료와 학습 시작 전 검사

아래는 앞으로 수행할 acceptance 기준이다. 기존 소규모 단독 GPU 검사를 전체 학습 검증으로 표시하지 않는다.

| 검사 | 통과 기준 |
|---|---|
| 원래 함수 보존 | 전체 valid mask·noise0·eval의 기존 checkpoint z/score 최대오차1e-5 이하 |
| 단독/배치·추가0·순서 | z 최대오차1e-4 이하, cosine score1e-5 이하. arbitrary invalid feature가 결과를 바꾸지 않음 |
| ASP 기준 | float64 ragged 출력/입력gradient 최대오차1e-10 이하, padded feature gradient0 |
| U/W 내용 통제 | W 첫len(U)가 U와 bitwise 동일, 나머지는 U의 순환, len(W)=S, 다른 codec/RawBoost 없음 |
| No-op 쌍 | len(U)=S면 pair_active=False. noise0·eval에서 P=0, pad 항 gradient0 |
| 빈 쌍과 rank 불균형 | rank별0/0,0/3,2/3 active의 loss와gradient가 전역 sum/count 기준과 일치, NCCL hang 없음 |
| sampler | 모든 T UID 포함, 전역162step, rank별8/72, 동일epoch schedule·중복발생수검증 |
| cache replay | 첫pass/replay embedding1e-5 이하. FP32 gradient와Adam update가 같은 microbatch 경계·padding·RNG의 full-graph 기준에 대해 atol1e-5/rtol1e-4 이내 |
| 두 단계 freeze | stage1 c 불변·θ 갱신, stage2 θ 불변·c 갱신, frozen XLS-R hash 불변 |
| 물리크기 독립 | physical8/4/2에서 같은 논리적 손실과 class weight, 전체8×8 BPL 보존 |
| 극단 길이 | T 최단/최장, eval29.312초 포함. 장문을 절단하지 않고 memory/time 기록 |
| checkpoint/score | strict-load, UID score1:1, eval rank padding 중복 제거, EER 독립 DET·AUROC rank 검산 |

연속 두 논리적 batch 이상에서 실제 DDP 두 단계 update와 gradient cache를 검사한다. 대표적인 긴 T batch와 short active가0인 batch를 모두 포함한다. 작동 여부 확인용 업데이트는 새 smoke 경로에서만 수행하고 본 학습 초기화로 재사용하지 않는다. 메모리 peak·남은 여유·step time을 측정해 physical size를 잠근 뒤 네 조건에 공통 적용한다.

## 11 실행 단계와 산출물

1. **현재 단계:** 설계와 근거 검수. 서버 원본 변경·학습 실행 없음.
2. **구현:** 별도 코드·원자적 run directory, config/manifest/source/pretrained hash 생성. acceptance 통과 후 seed1 네 조건의 학습 준비.
3. **학습:** E0→E1→E2→E3의 seed1, 이어 seed2/3. 매 실행 직전 서버·screen·컨테이너·지정GPU 실제 상태 확인. 다른 작업이 실행 중이면 중단·재시작하지 않음.
4. **평가 lock:** epoch50 모델 hash, V threshold, 평가 manifest를 고정. 그 뒤 pilot/19LA/전체 보조 평가와 통계 비교 수행.
5. **보고:** 모든 조건·seed의 성능·오류·정렬 효과·실측 자원 비용을 함께 기록. E3가 개선되지 않으면 그대로 보고하고 조건을 사후 교체하지 않음.

예정 run 이름은 `EBM_native_short303_<E0-E3>_s<seed>_<unique-id>`처럼 유일하게 만든다. 기존 이름으로 실행하면 삭제하는 logger 구조를 재사용하지 않고, 경로가 있으면 시작을 거부한다. config·epoch/step·optimizer·scheduler·RNG·sampler state를 저장해 재개 위치를 식별한다. 재개는 기존 프로세스 종료와 checkpoint를 확인한 별도 작업으로 취급하고 자동 재시작하지 않는다.

최소 산출물은 `protocol_lock.json`, UID/sampler manifest와 augmentation log, source/config hash, acceptance 결과, epoch50 model·optimizer·RNG state, epoch별 loss/active pair 수/gradient norm, 정책별 발화 점수와 V threshold, 최종 지표/CI, 자원 측정표다. loss는 `OC_L/OC_U/OC_W/B_old/P`, encoder·centroid update, 정상·스푸프별로 분리한다. P가0인 step 비율과 실제 active 정상 UID 수를 기록한다.

원음·연구실 원본 source·checkpoint를 무단으로 로컬이나 외부로 복사하지 않는다. XLS-R hidden states는 서버 디스크에 저장하지 않고 필요 시 RAM에서만 처리한다. 기존 F: 수신 작업이나 취소한 DF4×3 실험을 재개하지 않는다.

## 12 근거 자료와 설계의 한계

- [원길이 구현 가능성 검토](ebm_variable_length_feasibility_20261106.md): 실제 코드 위치, masked ASP·forward/backward 결과, 현재 local/global BPL 차이.
- [반복 문맥의 통제 실험](ebm_padding_context_mechanism_20261106.md): 정상과 스푸프의 상반된 변화, frontend 문맥의 영향.
- [학습 정상 길이 집계](../../artifacts/geometry_cache/local_analysis/variable_length_audit_20261106_v1/normal_duration_counts.json):3초미만38.22%,S미만40.31%.
- [반복과0패딩 양 비교](../../artifacts/geometry_cache/local_analysis/padding_budget_20261106_v1/comparison.json), [절단량 비교](../../artifacts/geometry_cache/local_analysis/padding_budget_20261106_v1/truncation_comparison.json): 증강 전 동일sampling 기준의 예상값. E2/E3의 추가 W 계산·gradient replay 비용은 포함하지 않음.
- [원 논문 3.1절](https://www.isca-archive.org/interspeech_2025/kim25g_interspeech.pdf), [공식 BPL 코드](https://github.com/kimho1wq/EBM/blob/main/model/loss.py): 논문 설명과 실제 교차 행렬 구현의 구분.
- [Wav2Vec2 문서](https://huggingface.co/docs/transformers/en/model_doc/wav2vec2): layer-norm frontend의 attention mask 사용.
- [Gradient caching 원 논문](https://aclanthology.org/2021.repl4nlp-1.31/): embedding gradient를 나누어 encoder에 전달하는 원리. 이번 EBM 적합성은 별도 acceptance 대상.

이 설계의 반복 branch는 S=3.0275초까지의 개입이므로 기존 long의4.0375초 wrap과 동일하지 않다. E3가 실패해도 모든 반복 불변성 학습의 실패로 확대하지 않는다. S미만 정상만 active이므로 매우 짧은 정상의 수 자체를 늘리는 실험도 아니다. 현재 알려진 source와 이미 관찰한 target에서의 결과이며, 특정 마이크·피치·주파수 원인을 확정하거나 미지 target 일반화를 보장하지 않는다.
