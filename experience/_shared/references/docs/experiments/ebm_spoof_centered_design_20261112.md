> 역사 문서 보존본. 원문 작성 당시의 상태·계획을 포함한다. 최신 상태와 확정 결과는 experience의 각 실험 README를 우선한다.

# 스푸프 중심 EBM 설계와 코드 검토

작성일: 2026-11-12 KST. 최초 설계 후 사용자 요청으로 이론 재검토와 로컬 구현을 진행했다. 최신 구현과 검증 범위는 [ROLE_EBM.md](../../../../../ROLE_EBM.md)에 기록했다. 실제 서버 학습과 두 GPU smoke는 아직 수행하지 않았다.

재검토에서 목적식 간 방향의 모순은 발견하지 않았다. 두 단계는 서로 다른 조건부 목적식을 줄이므로 공동 손실의 단조 감소·수렴을 보장하지 않으며, 중심 손실 계수 통제가 Adam 이동량까지 맞추는 것은 아니다. 구현에서는 보나 arm이 원본 Adam moment까지 재현하도록 곱셈 후 평균하는 연산 순서도 유지했다.

**스푸프를 단일 중심으로 모으고 보나를 그 중심에서 밀어내는 EBM은 수학적으로 구성할 수 있다.** 완전한 방향 전환에는 OC의 클래스 역할, 중심 갱신 대상, BPL 대상, 평가 점수 방향을 함께 바꿔야 한다. 주 실험은 원본240의 입력·표집·모델 구조를 기준으로 새로 초기화하며, `OC(spoof 중심) + spoof long×short 전체쌍 BPL`을 사용한다. 중심 단계의 손실 계수는 기존 보나 중심과 같은 5로 맞춘다.

이 목적식은 **보나가 스푸프 중심에서 멀어지도록** 한다. 보나끼리 서로 멀어지거나 분산이 유지되는 것은 보장하지 않는다. 알려진 스푸프들의 공통성을 학습하는 것이 미지 공격 탐지에 도움이 되는지는 실험으로 판단해야 한다.

## 1 검토 기준과 전체 코드 범위

원본240 기준은 로컬 Git에 보존된 `10f3bcd057dc6a2ca687fbd5389ad6c9e15a3746`이다. 로컬 현재 HEAD는 `cf199af03044c40837c08ce31d633e2c048bde08`이며 그 위에 기존 미커밋 확장들이 있다. 현재 `main.py`의 기본 경로는 원본240과 다르므로 그대로 실행해서 원본240의 역방향 실험이라고 부르지 않는다.

원본 Git tree의 **38파일 전체를 읽고 파일별 SHA·Python 정의·import를 목록화**했다. `baseline240_source`의 핵심 5파일은 보존 manifest의 바이트 SHA 및 원본 Git의 개행 정규화 텍스트와 모두 일치했다. 현재 문서·확장 포함 140파일도 목록화했다. 아래 핵심 경로와 관련 확장은 직접 흐름을 추적했다. 그 밖의 모든 분석 유틸리티까지 줄 단위로 검토했다고 주장하지 않는다. GitHub 최신 원격이나 현재 서버 프로세스 상태를 새로 조회한 작업은 아니다.

| 원본 영역 | 확인한 파일 | 판단 |
|---|---|---|
| 진입과 설정 | `main.py`, `arguments.py` | 모델·loss 생성, 클래스 비율 가중치, Adam, epoch scheduler, DDP, seed |
| 학습과 평가 | `train.py` | 클래스별 loader 순환, 보나 선두 결합, backend→중심 두 단계, EER, 저장 |
| 모델 전체 | `model/{__init__,b_linear,loss}.py` | bridge, layer attention, ASP, linear, LayerNorm, OC 중심, BPL |
| 입력 | `data_loaders.py` | 클래스 분리, DistributedSampler, 부분 배치, 64,600/48,440 입력, 반복, 독립 RawBoost |
| 데이터 정의 | `dataset/{__init__,ASVspoof2019_la_train,ASVspoof2019_la_eval,ASVspoof2021_la_eval,ASVspoof2021_df_eval}.py` | `bonafide=1`, `spoof=0`, 공식 eval 필터, attack 정보 |
| 증강 | `data_processing/__init__.py`, `augmentation/{raw_boost,codec_aug,augmentation,wav_read_functions}.py` | 실제 활성 RawBoost와 비활성 codec·별도 waveform augmentation 구분 |
| 로깅과 저장 | `logger/{__init__,logger,interface,local,neptune,wandb}.py` | 결과 경로·source backup·checkpoint, 외부 logger 사용 여부 |
| 분산과 임계값 | `utils/{__init__,ddp_util,get_threshold}.py` | 평가 gather와 점수 방향, 거리 단위 helper |
| scheduler | `scheduler/{steplr,cosine_warmup}.py` | 별도 제공 모듈이며 기준 main의 실제 scheduler는 PyTorch warm restarts |
| 환경과 문서 | `.gitignore`, `Dockerfile`, `docker_{build,run}.sh`, `README.md`, `LICENSE`, `docs/{CONTROL_B,RUN_BASELINE240_20261106,SERVER_BASELINE,WORKFLOW}.md` | source와 환경·역사적 설정의 구분 |

현재 확장에서는 `native_steps.py`, `positive_bpl.py`, 현재 `train.py`·`main.py`·`data_loaders.py`·`model/b_linear.py`, `short32_original240.py`, `short32_lifecycle.py`, `paired_evaluation.py`, 통제 평가 도구의 관련 호출도 확인했다. 특히 원본 학습, native/replay 학습, short32 hook 학습을 섞으면 다른 실험이 되므로 분리한다.

증거: [source_audit.json](../../artifacts/reverse_ebm_design_20261112/source_audit.json). 각 파일의 함수·클래스 위치도 포함한다.

## 2 원래 EBM이 실제로 최적화하는 것

동결 XLS-R의 hidden states를 학습 가능한 bridge·ASP·linear·LayerNorm이 임베딩 `z=fθ(x)`로 바꾼다. 학습되는 중심 parameter를 `c`라 하고, OC와 BPL 계산에서 각각 L2 정규화한다.

```text
u = z / ||z||
v = c / ||c||
q = uᵀv                  # cosine, 범위 [-1, 1]
y = 1: bonafide, y = 0: spoof
φ(t) = softplus(t)
```

이 중심은 단순한 보나 임베딩 산술평균이 아니다. 학습되는 OC parameter이며, 현재 두 단계 최적화의 두 번째 단계에서 보나만으로 갱신된다. 이 실험에서도 산술평균이나 사후 중심으로 교체하지 않는다.

### 2.1 기존 OC

[원본 loss.py](../../../../../baseline240_source/model/loss.py)의 OC는 다음과 같다.

```text
L_OC,bona = mean_i a[y_i] × {
    φ(20 × (0.9 - q_i)),  y_i = 1
    φ(20 × (q_i - 0.2)),  y_i = 0
}

a[0] = 1 / (2 × 0.9) = 5/9
a[1] = 1 / (2 × 0.1) = 5
```

`mean`의 분모는 해당 단계 실제 입력 행 수다. 손실 내부에 `batch>4`일 때만 가중치를 적용하는 분기가 있다. 원본의 실제 두 GPU 학습은 부분 배치까지 이 조건을 만족하지만, 소표본 테스트나 다른 배치 설정에는 그대로 일반화하면 안 된다.

정상적인 full batch에서 OC는 GPU당 보나12·스푸프108이다. 클래스별 총 계수는 각각 `12×5/120=0.5`, `108×(5/9)/120=0.5`다. 따라서 데이터 수가 9배라고 OC 총 기여도도 9배인 것은 아니다.

### 2.2 기존 BPL

```text
L_pair = mean_(i,j) [ φ(30 × (1 - cos(z_long_i, z_short_j))) - 0.3 ]
```

[원본 B_Linear.forward](../../../../../baseline240_source/model/b_linear.py)는 **보나 long 전체×보나 short 전체**를 사용한다. 일반 배치는 로컬 GPU당12×12=144쌍이며 다른 UID 간 쌍도 포함한다. 같은 발화의 두 view만 맞추는 diagonal loss가 아니다. 두 입력 묶음을 각각 재정렬해도 전체 평균은 같다. 같은 UID long·short는 독립 RawBoost를 적용한다.

정규화 cosine이 1이면 BPL도 `log(2)-0.3≈0.393147`이다. 학습 중 BPL 최솟값을 0이라고 기대하지 않는다. `-0.3`은 gradient에 영향을 주지 않으므로 역방향에서도 유지한다.

### 2.3 두 단계 Adam

[원본 train.py](../../../../../baseline240_source/train.py)의 한 논리 배치 흐름은 다음과 같다.

1. 중심 `c`를 동결하고 `L_OC,bona + L_pair,bona`로 backend `θ`를 갱신한다.
2. 갱신된 backend로 **보나 long만 다시 임베딩**하고, backend를 동결한 채 보나 OC로 중심 `c`만 갱신한다.

동결 frontend의 long hidden states는 재사용하지만, 중심 단계의 backend 출력은 θ 갱신 이후 다시 계산한다. 따라서 중심 갱신에 첫 단계의 낡은 임베딩을 그대로 쓰면 원래 방식과 다르다. classifier는 train 상태이므로 중심 단계에도 dropout·ASP 난수가 존재한다.

두 단계는 같은 Adam 객체를 사용한다. 논리 배치당 `optimizer.step()`은 두 번, 각 backend parameter와 중심 parameter는 각자의 단계에서 한 번씩 갱신된다. epoch scheduler는 epoch당 한 번이다.

## 3 역방향 목적식

### 3.1 OC의 클래스 역할 전환

새 중심 `c_S`에 대한 cosine을 `q_S`라 하면 다음을 사용한다.

```text
L_OC,spoof = mean_i a[y_i] × {
    φ(20 × (0.9 - q_S,i)),  y_i = 0   # spoof를 중심에 모음
    φ(20 × (q_S,i - 0.2)),  y_i = 1   # bona를 중심에서 밀어냄
}
```

데이터셋 라벨과 배치 비율은 그대로 `bona=1`, `spoof=0`, 12:108이다. OC 가중치도 **원래 데이터 클래스 기준** `[5/9,5]`를 유지한다. 라벨을 전역으로 `1-y`로 바꾸면 loader의 클래스 분리, BPL, sampler, EER가 연쇄적으로 바뀌기 때문에 loss 내부에서 역할을 명시한다.

기존 `r_real/r_fake` 이름은 의미 혼동을 만들므로 새 인터페이스는 `r_pull=0.9`, `r_push=0.2`, `compact_label=0`을 쓴다. margin을 단순히 `0.2/0.9`로 바꾸는 것이 아니다. 기존 학습은 `compact_label=1`로 같은 인터페이스를 사용한다.

**gradient 검토:** 모으는 항은 `dL/dq<0`이므로 gradient descent가 cosine을 높이고, 미는 항은 `dL/dq>0`이므로 cosine을 낮춘다. 단, 이는 score에 대한 방향이며 정규화된 임베딩의 정확한 평행·반평행 지점에서는 접공간 gradient가 0이 될 수 있다.

### 3.2 BPL을 스푸프 전체쌍으로 전환

```text
S = {i | y_i = 0}
L_pair,spoof = mean_(i,j in S×S)
    [ φ(30 × (1 - cos(z_long_i, z_short_j))) - 0.3 ]

L_backend = L_OC,spoof + 1.0 × L_pair,spoof
```

주 후보는 **스푸프 전체 long×short**다. 서로 다른 학습 공격의 스푸프도 같은 중심으로 모은다는 가설에 충실하다. 보나 BPL은 제거하고, 보나끼리 음의 BPL을 추가하지 않는다. BPL 계수에 마이너스를 붙이는 것은 입력 두 view를 서로 반대 방향으로 보내는 다른 목적식이다.

일반 배치 쌍 수는 `108×108=11,664`, 기존의 81배다. **mean reduction을 유지하므로 손실 계수를 81배로 키우거나 1/81로 줄이지 않는다.** 관여하는 원음 수·쌍의 의존성·gradient 분산·방향은 바뀌므로 수가 많아도 독립적인 감독 신호가 81배 생기는 것은 아니다.

기존 full-graph 코드는 이미 모든 보나·스푸프의 long·short frontend/backend 출력을 계산한다. 따라서 이 변경이 첫 단계의 frontend forward 수를 81배로 늘리지는 않는다. 다만 pair 행렬 연산과 역전파 대상은 늘고, 두 번째 단계의 backend 대상은12→108로 늘어난다. 전체 메모리·시간 증가는 실제 smoke에서 측정한다.

### 3.3 중심 갱신과 계수 통제

```text
θ 갱신 후:
    center_rows = long hidden states[y == 0]
    z_new = f_(θ_new)(center_rows)
    θ는 동결, c_S만 학습 가능
    L_center = 5.0 × mean_spoof φ(20 × (0.9 - cos(z_new, c_S)))
```

기존 중심 단계는 보나12개에 가중치5를 적용한다. 단순히 입력을 스푸프108개로 바꾸고 기존 OC class weight를 그대로 적용하면 중심 단계 손실의 계수는5/9가 된다. **같은 cosine 분포를 가정하면 계수와 gradient scale이 원래의1/9**다. Adam의 moment·epsilon·weight decay 때문에 실제 parameter 이동량까지 정확히1/9이라고 해석할 수는 없다.

주 비교에서는 `center_loss_multiplier=5`를 양쪽에 명시한다. 기존 보나 중심 arm에서는 실제 보나 배치6·12 모두 원본 수식과 같고, 새 스푸프 중심 arm에서는 클래스 수 보정이 중심 학습 강도까지 바꾸는 것을 막는다. 첫 단계 OC에는 원래 클래스 가중치를 계속 사용한다. 이는 **중심 단계 계수까지 기계적으로 복사한 버전과 구분한 통제 설계**다. 계수5/9 버전은 필요할 때 별도 보조 실험으로 비교한다.

추가 표집 없이 전체 스푸프 중심 행을 쓰므로, 중심 단계의 표본 수와 gradient 분산은 보나 arm과 다르다. 계수 통제가 이 차이까지 없애지는 않는다.

### 3.4 추론 점수

기존 EER는 `pos_label=1`, 높은 점수가 보나라는 규약이다. 이 규약을 모든 결과 파일에 유지한다.

```text
기존: raw_center_cosine = cos(z, c_B), score_bona = +raw_center_cosine
신규: raw_center_cosine = cos(z, c_S), score_bona = -raw_center_cosine

판정: score_bona >= threshold 이면 bona
```

`1-q_S`도 순위상 같지만, 변환 실수를 줄이기 위해 신규 canonical score는 `-q_S`로 고정한다. score 반전은 정확히 한 번만 수행한다. 원시 cosine과 canonical score를 별도 column으로 저장한다. EER가 높다는 이유로 실행 후 더 좋은 부호를 선택하지 않는다.

[paired_evaluation.common_scores](../../../../../paired_evaluation.py)는 모델 `forward()`를 거치지 않고 `z @ center`를 직접 계산한다. [short32_lifecycle.evaluate](../../../../../short32_lifecycle.py)는 새 OC 객체를 생성한 뒤 tensor만 복사한다. **새 loss의 inference 출력만 바꿔서는 이 경로들이 해결되지 않는다.** 새 실험 전용 공통 scorer가 checkpoint 역할 metadata를 읽어 부호를 적용하도록 설계한다. 취소된 기존 paired 작업의 snapshot·queue는 수정하거나 재활성화하지 않는다.

`get_threshold`는 `1-cos(각도 중간값)`을 반환하는 거리 단위 helper다. 원본 확인 경로에서 EER는 이 값을 사용하지 않는다. 이것을 신규 `-cos` 점수의 threshold로 그대로 가져오면 단위가 맞지 않는다. 예를 들어 기하학적 각도 중간값을 쓰면 기존 cosine은약0.613562, 신규 canonical score는약-0.613562이며 helper 반환값은약0.386438이다. 이 값들은 학습 margin 기반 참고값이며 검증된 운용 threshold가 아니다.

## 4 논리적으로 성립하는 부분과 실험으로 확인할 부분

### 4.1 기존 모델의 중심 부호만 바꾸는 것과 다르다

기존 `q_B`에 대해 `c_S=-c_B`라고 하면 `q_S=-q_B`다. 기존 OC를 이 변수로 다시 쓰면 spoof 당김 margin은 `-0.2`, bona 밀기 margin은 `-0.9`가 되어야 같은 식이다. 제안은 **당김0.9·밀기0.2를 스푸프 역할에 다시 부여**하고 BPL·중심 갱신 대상도 바꾼다. 따라서 저장된 기존 모델의 center나 최종 점수에 마이너스만 붙이는 것으로 구현할 수 없다. seed를 맞춰 처음부터 학습한다.

### 4.2 보나의 다양성 보존을 자동으로 보장하지 않는다

`cos(bona,c_S)<=0.2`는 중심으로부터의 각도 조건이다. 보나들이 서로 같은 반대 방향으로 모여도 만족한다. 정규화 cosine이므로 norm을 무한히 키워 거리를 벌리는 문제는 아니지만, **보나끼리의 각도·분산에는 직접 제약이 없다.** 보나 다양성 유지나 화자 불변성을 결과 전제로 쓰지 않는다.

반대로 스푸프가 `cos(spoof,c_S)>=0.9`를 만족한다면 중심과의 각도는약25.84도 이하이고, 그 조건만으로도 두 스푸프 간 cosine 하한은0.62다. BPL은 이를 더 강하게 한 방향으로 압축한다. 이는 의도된 스푸프 군집 수축이며, 클래스 전체가 하나의 출력으로 무너지는 실패와 구별한다.

margin은 hard constraint가 아니다. softplus는 margin을 넘겨도 작은 gradient를 남기므로 ‘0.9까지만 모으고 바로 멈춘다’는 설명도 정확하지 않다.

### 4.3 단일 스푸프 중심은 가능한 가설이지만 일반화 보장은 없다

학습 공격들에 공통적인 합성 흔적이 있고 backend가 이를 추출하면 알려진 스푸프를 모으는 정규화가 도움이 될 수 있다. 반면 공격별 차이가 크거나 새 공격에 그 흔적이 약하면, 미지 스푸프가 중심 밖으로 나가 높은 보나 점수를 받을 수 있다. 두 설명은 이 목적식으로부터 가능한 가설이며 현재 성능 사실이 아니다.

여러 공격을 한 중심에 모으는 것 자체가 수학적 모순은 아니다. 임베딩 공간에서 `spoof=+v`, `bona=-v`로 두면 OC와 spoof BPL의 목표를 동시에 만족하는 구성을 만들 수 있다. 합성 검산의 이 구성에서 신규·기존 backend 손실은 모두약0.456611이었다. 하지만 유한한 모델·데이터·증강 조건에서 그 매핑을 학습하고 본질적인 위조 신호를 보존하는지는 별개다. 첫 실험에서는 다중 중심·attack별 중심·새 대조손실을 넣지 않는다. 이들은 실패 원인을 본 뒤 따로 설계할 수 있다.

### 4.4 OC가 완전 붕괴를 무조건 방지하는 것은 아니다

양쪽 클래스가 모두 중심과 같은 임베딩이면, 밀어야 할 클래스의 OC loss는 크므로 좋은 목적함수 값이 아니다. 그러나 정규화 cosine의 정확한 극점에서는 embedding gradient가0인 정지점이 생길 수 있다. 합성 검산에서도 잘못된 클래스 loss가약80인데 gradient가0인 예제를 확인했다.

따라서 ‘OC를 넣었으니 붕괴 불가능’이라고 결론내리지 않는다. 학습 loss·유한성만으로 성공을 판정하지 않고 score 분리, 임베딩 분산, 중심 각도, gradient norm을 같이 측정한다. LayerNorm과 기존 ASP noise가 일반적인 수렴을 보장한다는 가정도 두지 않는다.

### 4.5 공격별 쌍 구성과 작은 배치의 차이

전체쌍 BPL은 학습 공격의 배치 내 출현빈도에 따라 attack×attack 쌍 비중이 정해진다. 이번에는 원래 sampler를 유지하므로 공격을 균등 재표집하지 않는다. 배치별 attack 구성과 cross-attack pair 비율을 계측해 어떤 쌍이 주로 학습되는지 보고한다.

보조 실험으로 쌍 수를12×12에 맞추고 싶다면 long용 스푸프 subset과 short용 subset을 **독립적으로** 뽑는다. 같은 subset을 양쪽에 쓰면 same-UID diagonal 비중이 전체108×108의1/108에서12×12의1/12로 커진다. 두 독립 균등 subset의 pair 평균은 고정된 전체 pair 행렬 평균의 불편추정량이지만, 단일 step의 gradient 분산과 실제 학습 경로까지 같지는 않다.

## 5 첫 실험에서 고정할 조건

주 비교를 새 보나 중심 기준 arm `A`와 새 스푸프 중심 arm `R_all`로 한다. 현재 short32나 mild/cache/가변 길이 정책을 결합하지 않는다. 아래50epoch는 **비교 설계 제안**이며 이번 요청으로 실행하거나 예약하는 설정이 아니다.

| 항목 | A 보나 중심 | R_all 스푸프 중심 |
|---|---|---|
| 데이터 label | bona1 / spoof0 | 동일 |
| OC compact class | bona | spoof |
| OC push class | spoof | bona |
| backend OC 가중치 | `[5/9,5]` | 동일 |
| OC pull/push margin |0.9 /0.2 | 동일 |
| BPL 대상 | bona long×short | spoof long×short |
| BPL 수식·계수 | alpha30, mean, 계수1 | 동일 |
| 일반 로컬 BPL 쌍 수 |144 |11,664 |
| 중심 갱신 대상 | bona long 전체 | spoof long 전체 |
| 중심 loss multiplier |5 |5 |
| canonical score |`+cos(z,c_B)` |`-cos(z,c_S)` |
| 초기화 | seed1 새 θ·c | 같은 초기 θ·c tensor, 학습 역할만 전환 |
| 입력과 증강 | 원본240 | 동일 |

공통 조건은 다음과 같다.

- Train 원음25,380개, bona2,580/spoof22,800. 원래 클래스별 두 `DistributedSampler`·DataLoader·작은 loader 순환·부분 배치를 유지한다. 전역 라벨 교환·클래스 비율 교환·50:50 재표집은 없다.
- global OC batch240, 두 GPU에서 보통rank당120=12+108, epoch당108 논리 배치. 학습50epoch·lr1e-5·weight decay1e-4·원래 epoch scheduler를 공통 적용한다. 각 단계의 실제 parameter step counter도 확인한다.
- train long64,600sample 앞부분, short48,440sample 원래 무작위 crop, 짧으면 원래 wrap 반복. 서로 독립인 기존 RawBoost0~8. codec·추가 waveform augmentation·HiFi-GAN·speed/tempo·VAD·short32 없음.
- frozen XLS-R, 기존 bridge·ASP·linear·LayerNorm 구조·parameter key/shape 유지. 일부 args의 `num_blocks`나 `linear_units`를 보고 다른 backbone을 만드는 일이 없도록 실제 생성자를 기준으로 한다.
- 비교 두 arm의 데이터 UID·crop·RawBoost 입력을 공통 seed 및 독립적인 loader RNG로 재현한다. center 대상 행 수가 달라 backend 난수 소비가 달라지므로 **같은 global seed 하나만으로 이후 난수까지 같다고 주장하지 않는다.**
- 새 공통 runner에서는 `(seed,epoch,step,rank,phase)`로 train backend 단계별 RNG stream을 분리하고 평가 RNG를 복원하는 방식을 양 arm에 똑같이 적용하는 것을 권한다. 이는 역사적 원본의 단일 RNG stream과 차이가 있으므로 기록한다. 먼저 원본 RNG state를 그대로 준 A 경로에서 원본 full-graph loss·gradient·update 일치 회귀를 통과한 뒤, 공통 RNG 격리를 적용한다. tensor shape가 다른 중심 단계의 dropout mask 자체는 같다고 주장하지 않는다.

원본 train에는 `param.require_grad=False`라는 속성명 오타가 있지만50epoch 조건의 frontend forward는 별도로 `set_grad_enabled(False)` 안에서 실행된다. 신규 runner는 `requires_grad_(False)`도 명시하고 gradient·parameter 불변을 확인한다. 또한 원래 설정은 `lr=lr_min=1e-5`이므로 scheduler가 있어도 학습률은 같은 값이다.

기존 저장240/B/C/short32 성능은 참고 자료로만 사용한다. 원래 실험들이 입력·난수·학습 이력까지 이 새 비교와 같다는 근거가 없으므로 `A`를 생략하고 신규 `R_all`과 과거 최소 EER만 비교해서 방향 전환의 효과라고 결론내리지 않는다.

### 5.1 부분 배치와 표본 수

원본 두 GPU의 **rank당** 정확한 스케줄은 다음과 같다.

| 위치 | bona | spoof | OC 합계 | A pair 수 | R_all pair 수 |
|---|---:|---:|---:|---:|---:|
| 보통 |12 |108 |120 |144 |11,664 |
|106번째 |12 |60 |72 |144 |3,600 |
|108번째 마지막 |6 |108 |114 |36 |11,664 |

107번째는 다시120개다. rank당 epoch 방문 수는 bona1,290·spoof11,616이고, 두 GPU 합계 bona2,580·spoof23,232다. spoof는 원음22,800개와 순환 추가 방문432개를 구분한다. pair 수는 rank당 epoch 합계 A15,444·R_all1,251,648이다. 일반 배치의81배와 epoch합계약81.0443배를 구분한다.

`bona_size`를 `spoof_size`로 이름만 바꾸고 prefix를 자르면 보나가 섞인다. 어떤 클래스가 앞에 있는지와 무관하게 `labels==compact_label`로 명시적으로 선택한다. 학습 계약상 보나 선두 배열은 검증하되 목적식이 그 위치에 의존하지 않게 한다. 새 추가 loss나 길이별 대상이 없으므로 K=0의 조건부 hook 경로는 필요하지 않다.

### 5.2 후속 원인 분리 실험

첫 비교와 함께 자동 실행하지 않는 보조 후보다.

| 후보 | 구분하려는 영향 |
|---|---|
| A_OC / R_OC | 양 arm에서 pair loss를 제거해 OC·중심 방향 전환의 영향과 pair 정규화의 추가 영향을 구분 |
| R_12 | step별 A의 bona수와 같은 수의 spoof를 long/short 각각 독립 선택하고, 중심도 같은 수로 균등 선택해 표본·쌍 예산 차이 진단 |
| R_literal | 중심 multiplier5/9로 두어 기계적 class-weight 재사용의 영향 확인 |
| 추가 공통 seed | 단일 초기화·표집 결과인지 확인 |

R_12는 일반12×12·마지막6×6이며 더 많은 스푸프 중 일부만 pair/중심 학습에 사용하므로 R_all과 동일한 학습이 아니다. OC 전체 스푸프는 계속 사용한다. 관측 EER를 보고 가장 유리한 조건만 주 결과로 승격하지 않는다.

## 6 수정 위치와 구현 구조

기존 고정 원본과 실행 snapshot을 보존하고 **기존 입력과 모델 구조를 재사용하는 새 opt-in 학습 경로**를 구현했다. 아래 표는 최초 변경 명세이며 실제 파일·실행 규약·검증 결과는 [구현 문서](../../../../../ROLE_EBM.md)를 따른다. scorer와 저장·평가 lifecycle은 `role_model.py`, `role_lifecycle.py`에 함께 구현했다.

| 수정 또는 추가 지점 | 필요한 변경 | 회귀 확인 |
|---|---|---|
| 새 `role_objectives.py` | `compact_label`, `r_pull/r_push`, class 기준 backend OC, 별도 center loss, 기존 BPL 호출 | `compact_label=1`의 원본 OC 값·gradient, `=0`의 당김/밀기 방향 |
| 새 `role_model.py` | 기존 B_Linear 구조·state key를 유지하는 head. long/short embedding 뒤 label mask로 pair 선택, backend/center/embed 모드 명시 | 모든 구조·초기 parameter 동일, 올바른 클래스에만 pair gradient |
| 새 `role_training.py` | 원본 두 loader·부분 배치·두 단계 full graph를 보존하고 center 행만 역할에 따라 선택 |120·72·114, 원래 UID 방문·Adam 갱신·scheduler 횟수 |
| 새 `tools/train_role_experiment.py` | 양 arm을 같은 생성자·설정 검증·명시적 역할로 실행, 기존 결과 경로 존재 시 중단 | 실제 entry point·worker import·잘못된 옵션 조합 사전 차단 |
| 새 역할 대응 scorer | `raw_center_cosine`와 `score_bona`, 역할 manifest 필수, 모든 평가 경로에서 동일 변환 | perfect/tied toy EER, sign 한 번만 적용, checkpoint reload 전후 동일 |
| 새 실험용 저장·lifecycle | save-first, CLS/TrainState+역할 sidecar+SHA commit, rank RNG, 단계별 exit | tensor뿐 아니라 역할까지 복원, 실패 평가와 완료 구분 |
| 새 테스트 파일 | 기존 원본 회귀와 신규 역할·DDP·저장 검증 | 아래 단계별 gate |

핵심 변경의 출발점은 다음 네 곳이다.

1. `baseline240_source/model/loss.py:51`의 보나/스푸프별 margin 적용을 역할 기반 식으로 일반화한다.
2. `baseline240_source/model/b_linear.py:40`의 `x[:bona_size]` BPL 선택을 role mask로 일반화한다.
3. `baseline240_source/train.py:139`의 보나 전용 중심 pass를 role mask와 별도 계수5의 center 목적식으로 바꾼다.
4. `train.py:test` 및 `paired_evaluation.py:common_scores`처럼 cosine을 반환·직접 계산하는 모든 새 평가 호출에서 canonical 방향을 일치시킨다.

위 원본 파일 자체는 보존한다. 현재 `native_steps.py`에는 보나 prefix와12개를 가정한 목적식·중심 loop가 있고, `positive_bpl.py`의 matched collate도 보나 오른쪽 입력12개를 가정한다. 이 두 경로는 첫 실험에서 호출하지 않는다. short32의 원본 함수 무수정 hook도 새로운 역할 변경에 억지로 사용하지 않는다. 신규 trainer에서는 원본으로부터 바뀐 동작을 명시적으로 검토할 수 있게 한다.

원본 `OCSoftmax.forward(label=None)`는 inference 때도 학습용 indexing 코드를 지나가며 불필요한 loss를 만든다. 신규 scorer는 label이 없을 때 **원시 cosine 계산 후 즉시 반환**하여 라벨 없는 in-place 변형이나 학습용 값의 혼용을 피한다. 이 정리는 새 경로에만 적용한다.

### 6.1 한 논리 배치 의사코드

```python
# 신규 runner 명세. 실제 실행 코드가 아님.
hidden_long, hidden_short = frozen_frontend(original_long, original_short)
mask = labels == compact_label
assert mask.any() and (~mask).any()

# Phase 1: theta만 갱신, center 동결
zero_grad(set_to_none=True)
freeze(center=True, backend=False)
loss1 = ddp_head(
    hidden_long, hidden_short, labels=labels,
    phase="backend", compact_label=compact_label,
)
# head 내부: weighted_role_OC(all rows) + original_BPL(z_long[mask], z_short[mask])
loss1.backward()
optimizer.step()

# Phase 2: 갱신된 theta로 다시 계산, center만 갱신
zero_grad(set_to_none=True)
freeze(center=False, backend=True)
loss2 = ddp_head(
    hidden_long[mask], labels=labels[mask],
    phase="center", center_loss_multiplier=5.0,
)
# head 내부: 5 * mean softplus(20 * (0.9 - cosine))
# pair loss 없음. center까지 no_grad로 감싸지 않음.
loss2.backward()
optimizer.step()
```

mask만 바꾸고 원본 `module.loss.weight`의 동결 규칙을 잘못 바꾸면 한 단계에서 양쪽이 모두 갱신되거나 중심이 전혀 갱신되지 않는다. tensor identity로 중심을 식별하고, 단계별 gradient 존재·parameter 이동을 검사한다. `zero_grad(set_to_none=True)`로 이전 단계의 gradient가 남아 동결 parameter에 Adam/weight decay가 적용되는 문제를 막는다.

센터 pass를 no-grad backend embedding과 differentiable center loss로 분리할 수는 있지만, DDP를 우회해 중심 gradient 동기화를 빠뜨려서는 안 된다. 기본안은 두 단계 모두 DDP head forward 안에서 수행하고 기존 `find_unused_parameters=True` 계약을 검증한다. global embedding gather를 추가하지 않고 BPL은 기존처럼 rank-local이다. 원본 스케줄에서 두 rank의 클래스 수가 같은지도 사전 검산한다.

### 6.2 checkpoint에 역할을 저장해야 하는 이유

`compact_label`, margin, class weight, score 부호 같은 값은 현재 OC 모듈의 일반 Python attribute다. `state_dict`의 weight shape만 같다고 역할까지 저장되지는 않는다. 예전 tensor-only 로더에 신규 CLS를 넣으면 에러 없이 보나 중심으로 오해할 수 있다.

새 저장물에는 다음을 함께 보존한다.

```json
{
  "schema_version": "ebm_role_v1",
  "compact_label": 0,
  "label_semantics": {"0": "spoof", "1": "bonafide"},
  "pair_class": 0,
  "r_pull": 0.9,
  "r_push": 0.2,
  "alpha_oc": 20.0,
  "alpha_pair": 30.0,
  "backend_class_weights": [0.5555555555555556, 5.0],
  "center_loss_multiplier": 5.0,
  "pair_reduction": "mean_all_local_cross_pairs",
  "score_bona_transform": "negative_center_cosine"
}
```

CLS와 TrainState를 원자 저장·재로딩 검산하고, 역할 sidecar와 source/config/frontend revision/RNG 및 파일 SHA까지 commit manifest에 묶는다. 신규 모델에 역할 metadata가 없거나 config와 충돌하면 평가를 거부한다. legacy model은 근거가 확인된 manifest에서만 보나 중심으로 등록한다. `strict=True` tensor loading만으로 역할 확인을 대신하지 않는다.

## 7 검증 단계와 통과 조건

### 7.1 이번에 완료한 로컬 검산

[audit_design.py](../../../workspace/artifacts/reverse_ebm_design_20261112/audit_design.py)를 CPU float64로 실행했다. 실제 원본 `OCSoftmax`·`BPL`을 불러오고 임베딩 수준에서 계산했다. optimizer를 만들거나 음성·XLS-R를 추론하지 않았다.

| 검사 | 결과 |
|---|---|
| 기존 OC 수식·gradient 재현 | 최대 gradient 오차0 |
| 양 역할의 score 미분 부호 | 당김 음수·밀기 양수 |
| 신규 임베딩 gradient 한 걸음 | spoof cosine0.303046→0.313324, bona0.303046→0.220026 |
| 중심 계수 비교 | 기존5 / 기계적 역방향5/9 / 통제5 확인 |
| spoof BPL gradient routing | 양 view의 spoof만 gradient, bona0, 순열 불변 |
| pair mean 계수 보존 | 입력 묶음을9배 복제해81배 쌍으로 늘려도 loss 동일 |
| 두 클래스 목표의 구성 가능성 | spoof=+v·bona=-v에서 OC와 BPL 동시 만족, 양 역할 loss약0.456611 |
| 독립 subset 추정 | 예제 전체 평균196=독립 subset 평균196, 같은 subset203.5 |
| 중심 부호만 바꾸는 등가성 | 같은 margin 역방향과 비등가 확인 |
| canonical score와 tie ROC | 올바른 부호EER0%, 잘못된 부호100%, 상수점수50% |
| 실제 현재 EER 함수 대조 | 독립 ROC와 완전분리·역방향·tie·혼합점수 오차0 |
| 원본 부분 배치·방문·쌍 수 |106·108번째 포함 일치 |
| 각도 극점의 정지 붕괴 반례 | 잘못된 클래스 loss약80, gradient0 |
| 보나 간 다양성 반례 | 같은 보나 방향도 push margin 만족 |
| 단계별 gradient 격리 | center-only와 backend-only 분리 확인 |

최초 설계의15개 모두 통과했다. [validation.json](../../artifacts/reverse_ebm_design_20261112/validation.json)에 환경과 수치를 기록했다. 이후 신규 runner의 CPU 구현 검사도 수행했으며 [구현 문서](../../../../../ROLE_EBM.md)에 구분했다. 실제 음성·두 GPU·성능 검증과 혼동하지 않는다.

### 7.2 구현 후 학습 전에 필요한 gate

1. **원본 회귀:** A의 공통 head·trainer가 같은 input/parameter/RNG에서 원본 full-graph의 loss1·loss2·θ·c·Adam state를 재현한다. 일반/부분 배치120·72·114를 포함한다. 최초 초기 tensor와 frontend 무갱신도 확인한다.
2. **역할 회귀:** pair·center에 bona가 섞이지 않고, 역할1/0에서 각 클래스의 score gradient가 맞는지 검사한다. BPL은 전체108×108·부분60×60, 기존 arm 마지막6×6을 확인한다. 순서 섞기·동일 UID·cross-attack·zero embedding/비유한 입력·빈 클래스 오류도 검사한다.
3. **실제 진입부:** 비Git 고정 snapshot과 실제 `__main__`·worker import·모든 runtime dependency를 검사한다. 과거 실패처럼 CPU unit만 통과하고 worker가 모듈을 못 찾는 상황을 막는다. validate-only는 logger·GPU·다운로드 전에 끝나야 한다.
4. **두 GPU 짧은 smoke:** 기존 자원 유휴 확인 후 고유 root에서3논리 배치만으로 full/부분 배치·optimizer각2회·중심각1회·rank 간 중심 동기화·유한 loss·peak GPU memory를 확인한다. smoke와 본학습 optimizer state를 재사용하지 않는다.
5. **저장과 평가 smoke:** save-first·CLS/TrainState/역할 manifest 일치·재로딩 점수·점수 부호 변환·공통 RNG·세 eval 집합의 두 클래스/길이·batch와 단독 점수 차이를 검사한다. 하나의 label별 `type`만 기록한21LA loader로 attack별 결과를 만들지 말고 UID로 공식 metadata를 연결한다.
6. **보존과 종료:** 새 snapshot SHA와 기존 source/model SHA를 확인하고, 단계 실패 시 본학습 진입을 막는다. retry0·무한대기 없음·본학습 새 seed 초기화·완료/실패/수동중단을 구분한다.

모든 필수 gate를 통과하기 전에는 본학습을 실행하지 않는다. 부족한 메모리에 대응해 신규 arm만 batch·pair·microbatch를 몰래 바꾸지 않고 설계를 갱신한다. smoke에서 측정할 때까지 전체 시간과 메모리의 증가율을 확정하지 않는다.

## 8 평가와 성공 판정

주 비교는 공통 고정 epoch50이며, 관측 최소 EER와 그 epoch는 사후 참고값으로 함께 보고한다. 같은 저장 스케줄5/10/.../50과 같은 후보 수를 적용한다.19LA를 보고 사용자 중단·설계 변경이 생기면 해당 이력을 기록하며,21LA/DF별 최솟값을 독립적인 모델 선택 결과로 해석하지 않는다.

최근 프로젝트 조건을 이어 **새 dev 평가는 넣지 않는다.**5epoch마다 저장 검증 후19LA 공식 eval 전체71,237개를 모니터링하고, 학습 후 같은 저장 checkpoint의21LA148,176개·21DF533,928개를 비교하는 설계를 제안한다. 평가량이 크므로 실제 실행 범위·시간 예산은 학습 실행 전에 명시한다.

공통 평가 입력은 앞 `min(original,64,600)`·비반복·mask·rank당batch8·seed1729·원래 ASP noise1e-5·동일 frozen frontend다. 두 모델에 같은 UID·파형·평가 난수를 사용하며, 학습한 각자 OC 중심으로 canonical score를 낸다. 신규 중심을 기존 중심으로 바꾸거나 eval 데이터로 중심을 다시 맞추지 않는다.

반복 평가는 별도 입력 민감도 진단으로만 구분한다. 학습 방향과 반복 여부를 동시에 바꿔 주 비교하지 않는다.

### 8.1 필요한 결과

- 각 집합의 EER, 같은 epoch의 차이, 고정50과 사후 최소·epoch·후보 수. 전체 UID/index 유일성·class 수·유한점수·독립 tie-aware ROC 재계산을 검산한다.
- bona rejection과 spoof acceptance를 함께 보고한다. dev 없는 조건에서 target EER 지점으로 만든 threshold는 **그 target의 설명용 threshold**이며 운용 성능이 아니다. 과거 bona 모델의 숫자 threshold를 신규 모델에 공통 적용하지 않는다. 기하학적 고정 threshold를 쓰면 그 근거와 학습 보정 부재를 표시한다.
- 21LA attack×codec·21DF attack/source·원래 길이 구간별 결과와 분모. 해당 subset에 양 클래스가 없으면 EER를 만들지 않고 누락률 등 가능한 지표만 쓴다. 새 공격을 중심에서 멀다고 bona로 받아들이는 사례를 확인한다.
- 길이·반복 비율·음성활동·화자·코덱에 따라 score가 어떻게 이동하는지 본다. 단순히 label shortcut을 뒤집어 배운 것과 공통 spoof 신호 학습을 구별하려는 진단이며, 사후 재가중만으로 원인을 확정하지 않는다.
- Train의 클래스별 중심 cosine 분포, class내 pair cosine·유효차원·정규화 임베딩 공분산 trace, class간 separation, OC/BPL gradient norm·각도, 중심 norm·angle 이동, backend/center update norm. 고정 Train probe를 이용해 dropout/noise 조건도 맞춰 측정한다. 클래스당 norm·표본 수가 다른 통계를 무보정 비교하지 않는다.

성공의 첫 기준은 통제된 A 대비 target EER 개선이다. Train 스푸프가 더 모인 사실만으로 성공이라고 판단하지 않는다. 특정 공격의 누락·보나 거부가 악화하면 평균 EER와 함께 보고한다. 단일 seed의 차이는 그 실행의 관측 결과로 한정하고, 일반적인 개선 주장은 같은 설정의 추가 seed나 사전 고정된 별도 holdout으로 확인한다.

## 9 비용과 자원 보존

이번 조사·설계의 새 지출 상한은 **0USD**다. 로컬의 제한된 foreground CPU 검산만 수행했고 원격 접속·GPU 추론·학습·새 대기열·앱 예약·리소스 생성은0회다. 기존 미커밋 변경과 모델·snapshot·프로세스는 건드리지 않는다.

후속 학습의 총 비용은 아직 견적하지 않았다. 실행 전 두 arm과 선택한 보조 단계·smoke·평가·가능한 재시도까지 묶은 상한을 작성한다. 기존 할당 자원으로 새 비용0USD 범위가 확인되면 그 범위에서 진행할 수 있고, 누적 새 지출 또는 안전 상한이 **10USD 이상이면 정확히10USD를 포함해** 비용 승인을 먼저 받는다. 이번 설계 요청은 그 학습 실행 지시를 대신하지 않는다.

새 실험은 기존 컨테이너와 지정 GPU의 실제 상태·기존 작업 완료를 실행 시점에 확인하고 고유 root에서만 수행한다. 자동 재시도는0, 전체 timeout과 단계별 exit를 둔다. 끝나면 자신의 worker만 종료·확인하고 임시 파일을 정리하며, 실패 로그·소스·checkpoint·검산 증거는 보존한다. 취소된 paired watcher나 중단된 기존 학습을 재가동하지 않는다.

이번 source SHA·Git status 보존 검사: [preservation.json](../../artifacts/reverse_ebm_design_20261112/preservation.json). 계획과 범위: [plan.json](../../artifacts/reverse_ebm_design_20261112/plan.json).
