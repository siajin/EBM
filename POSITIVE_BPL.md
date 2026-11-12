# 가변 길이 OC+BPL 입력과 통제 비교

최종 범위: **OC 손실과 OC 중심 학습은 유지한다. OC와 BPL의 입력을 맞춘 가변 길이로 변경한다.**
이전 BPL-only/Train 평균 중심 구현은 잘못 확대한 범위였으며 현재 실행 경로에서 제거했다.
기존 pilot/검증 결과는 과거 기록으로 보존하며 최종 OC+BPL 구성의 성능 근거로 사용하지 않는다.

## 2026-11-10 사용자 승인 50epoch 실행

**현재 활성 실행은 중간평가19LA로 바꾼 v2다.** 사용자 명시 재시작 승인 후 기존v1 진행분과 로그를 보관하고,
13:22:19 KST에 동일seed1로 처음부터50epoch를 시작했다.
`AuditedTrainer.evaluation`이 **5·10·…·50epoch마다19LA eval71,237개 전체**를 평가하며 dev 중간평가는 없다.
현재 서버 root는 `/home/mentee/analysis/ebm_coverage_e50_20261110_v2_mid19LA`, 결과명은
`EBM_e50_bs240_lr1e5_OC_BPL_coverage_activity_20261110_mid19LA`다.
활성 상태와 점검기는 `../artifacts/ebm_coverage_e50_mid19LA_20261110/state.json` 및 `check_run.py`다.
27개 테스트·사전검사·두GPU 무갱신/평가smoke 통과와 새 학습 첫배치의 기존 초기 손실 정확 일치를 확인했다.
최종 주비교는 두 고정50 모델의19LA/21LA/21DF 전체 EER이며 dev는 최종 보조threshold 계산에만 사용한다.
이전 완료시각 추정은 재시작과 중간평가량 증가를 반영해 갱신해야 한다.

아래 v1 기록은 이전 실행의 재현 정보이며 현재 활성 실행으로 해석하지 않는다.

사용자 요청으로 전수 순회+길이/VAD 정책의50epoch 학습과 배치 240 고정 길이 반복 입력 학습 고정50 모델의 동일 조건 비교를 시작했다.
운영 원음 전체 비반복 학습 저장소를 변경하지 않고 `/home/mentee/analysis/ebm_coverage_e50_20261110_v1/code`의
고정 소스 snapshot으로 실행한다. 소스/정책/체크포인트 SHA와 실행 설정은 그 폴더의 `experiment.json`이다.
결과명은 `EBM_e50_bs240_lr1e5_OC_BPL_coverage_activity_20261110`이다.
실제 진행과 완료 여부는 `../artifacts/ebm_coverage_e50_20261110/state.json` 및
`check_run.py`로 확인한다. 이 문서의 실행 시작 기록은 학습/평가 완료를 뜻하지 않는다.

- `tools/train_coverage_experiment.py`: 기존 main/trainer를 사용하며 loss/optimizer 수식은 변경하지 않는다.
  각 epoch의 실제 UID 사용률100%, 배치당144BPL쌍·backend/중심 각1회 갱신·유한 손실을 검사한다.
  epoch 경계에서 로그를 flush해 마지막 metric 누락을 방지한다.
- `tools/check_coverage_training.py`: 시작 전 소스·정책·기존 모델 불변과 GPU/결과 경로를 검사한다.
  완료 후 CLS50=TrainState50, 파라미터별 Adam5400회, 각rank의backend/중심5400회,
  모든 epoch의 원음 전수 사용과 metric5400행을 검사한다.
- `tools/evaluate_coverage_comparison.py`: 두 고정50 모델에 같은 앞64600 cap/비반복/mask/batch8,
  평가seed1729와 원래 ASP잡음1e-5·공통난수를 적용한다. 학습된 각자의 OC 중심 점수를 사용한다.
  frontend는 두 모델이 공유하며 별도 frontend 실행과 점수 차이1e-6미만을 확인한다.
  dev/19LA/21LA/21DF 총778,185개를 평가하고 전체 EER·dev기준 오류율·길이별 오류를 저장한다.
- 서버 `pipeline.sh`는 검사→GPU실행/평가smoke→학습50→체크포인트검산→전체통제평가→비교표 순서다.
  실패하면 다음 단계로 넘어가지 않는다. 기존 모델은 재학습하지 않고, target으로 epoch나 정책을 선택하지 않는다.
  별도 앱 예약을 생성하거나 과거 PAUSED 예약을 재활성화하지 않았다.

## 한 GPU의 최종 입력과 학습

1. 보나 원음12개·스푸프 원음108개를 클래스별 맞춘 비율로 선택한다.
   사용자 요청의 전수 순회 구성은 `coverage`, 이전 복원 추출 비교는 `replacement`다.
2. 각 원음의 화자/출처/원음길이 조건부 확률에서 길이를 선택하고 실제 조각을 뽑는다.
   OC는 이120개를 사용한다. BPL의 왼쪽 묶음은 이 OC 보나12개의 **동일한 임베딩**이다.
3. 같은 보나 분포에서 원음12개를 독립 추출해 길이·위치·증강을 선택한다. BPL 오른쪽 묶음이다.
4. 기존 수식 그대로 `OC(보나12+스푸프108) + BPL(보나12×보나12)`로 backend를1회 갱신한다.
5. 갱신한 backend로 OC 보나12개의 임베딩을 다시 계산하고 기존 OC 중심을1회 갱신한다.

GPU당 OC120개+추가BPL12개=132개 파형, 두GPU OC240개+추가24개=264개다.
`batch_size=240`은 **OC 배치**다. BPL은 GPU당144쌍이며 GPU 사이에 새로운 쌍은 추가하지 않는다.
두GPU gradient 평균은 기존 DDP를 사용한다. 한 논리배치의 Adam 호출2회(backend1/중심1)는 유지한다.

## 원음·길이·위치·증강 선택

| 길이(초) | 0.5 | 1 | 1.5 | 2 | 2.5 | 3 | 3.5 | 4 |
|---|---|---|---|---|---|---|---|---|
| 목표 확률 |10%|20%|30%|20%|8%|5%|6%|1%|

- 외부 정책의 Train 전용140개 클래스×화자×공격유형 조건부 표를 실제 Train과 대조한다.
- 원음을 먼저 선택하고 그 원음에서 가능한 길이를 선택한다. 짧은 음성에 불가능한 길이를 요구하지 않는다.
- 화자별 원음 가중치를 적용하며 화자 내부의 출처 비율은 유지한다. 기대값 기준으로 두 클래스의 길이 분포를 맞춘다.
- 음성활동 정책을 생략하면 방향은 앞/뒤50%, 방향 안의 조각 위치는 균등이다. 정책을 사용하면 같은 앞/뒤 격자의 후보에서 맞춘 확률로 선택한다. 파형을 이어 붙이거나 반복 연장하지 않는다.
- 각 조각에 원래 RawBoost0~8을 적용한다. 실제 배치 패딩과 frontend/backend mask를 유지한다.
- 기본 `replacement`에서 원음·길이는 복원 추출이므로 동일 UID나 동일 길이가 우연히 선택될 수 있다.
  매 배치의 길이 비율을 강제로 고정하지 않으며 누적 분포가 목표에 수렴한다.
- 원음/길이/위치/증강 RNG stream을 분리한다. 학습 seed·epoch·rank·slot으로 재현 가능하다.
- 실제 Train의 클래스/화자/공격유형/원음 수/길이 bucket/선택 확률이 정책과 다르면 실행을 거부한다.

## 매 epoch 원음 전수 순회

`--source-sampling coverage`는 OC 보나·OC 스푸프·BPL 오른쪽 보나의 세 stream마다
모든 원음을 한 번씩 순회한 뒤 부족한 배치 슬롯을 추가 추출하고 전체 순서를 섞는다.
두 GPU의 합집합에서 전수 순회를 보장하며 각 GPU가 모든 원음을 따로 보는 구성은 아니다.
BPL 오른쪽은 왼쪽과 독립된 순회/추가 추출 난수열을 사용한다.
`matched_same_source` 비교 조건에서는 OC 순회는 같고 오른쪽 UID만 왼쪽과 공유한다.

추가 추출을 단순 균등으로 하면 기존 화자/출처 목표가 조금 달라질 수 있다.
이를 막기 위해 총 슬롯 T, 원음 수 N, 기존 원음 선택 확률 p_i에 대해
추가 추출 확률을 `(T*p_i-1)/(T-N)`으로 보정한다.
각 원음의 기대 방문 횟수가 `1+(T-N)*추가확률=T*p_i`이므로
**기존 원음·화자·출처 비율과 길이×음성활동 기대분포를 그대로 유지**한다.

- 현재 두 GPU·108배치 기준 OC 보나2,580개+추가12회, OC 스푸프22,800개+추가528회다.
  독립 BPL 오른쪽도 보나2,580개+추가12회다. 입력 수와 최적화 횟수는 늘지 않는다.
- 모든 원음에서 `T*p_i >= 1`이어야 가능하다. 부족하면 사전검사에서 중단하고
  원음 선택 비율이나 epoch 길이를 자동으로 바꾸지 않는다.
- 현재 데이터의 최소 기대 방문 횟수는 보나1.00182, 스푸프1.01903이므로 가능하다.
- 길이·위치는 기존 정책에서 계속 무작위로 선택한다. 매 epoch의 길이/음성활동 비율을
  정수 개수로 강제하는 것은 아니며 유한 표본 오차는 남는다.
- 원음 UID를 모두 사용해도 원음의 모든 시간 구간이나 긴 연속 문맥을 한 epoch에 보는 것은 아니다.
- `replacement` 기본값과 기존 난수 동작은 재현용으로 유지한다.

### 전수 순회 구성의 실제 Train 검증

- 단위26개·Train-only 사전검사·두 worker의 실제2배치 로더 검사가 모두 통과했다.
  원음 선택50epoch, 실제 조각 전후각10epoch(각285,120조각)를 재현했다.
- 원음 선택50epoch 전부에서 OC 보나2,580개와 스푸프22,800개를 매번100% 사용했다.
  독립 BPL 오른쪽도 매 epoch 보나100%다. 보나의50epoch 방문 횟수 범위는
  OC26~75→50~54회, 스푸프는26~83→50~57회로 줄었다.
- 원음별 기대 선택 확률 최대 차이는 보나0·스푸프6.78e-21이었다.
  길이 및 음성활동 정책의 JSON/NPZ는 이전과 같은 파일을 재사용했다.
- 실제10epoch 누적의 목표 길이 확률 최대 오차는 OC보나0.347%p, OC스푸프0.102%p,
  BPL오른쪽 보나0.471%p다. OC 두 클래스의 길이TV는0.00594→0.00543이다.
  유한 표본 차이이며 모든 길이에서 전수 순회가 더 가깝다고 보장하는 결과는 아니다.
- 같은 누적 표본에서 길이별 두 클래스 VAD 평균 차이는0.11~0.66%p다.
  매 배치/epoch의 구간 개수를 고정하지 않으므로 기대분포와 달리 이 값이0이 되지는 않는다.
- epoch1 원음 시간 구간 합집합 사용률은 보나OC39.24→50.21%, 스푸프OC38.68→50.01%,
  보나OC+BPL61.19→71.70%로 늘었다. 모든 원음을 쓰는 것과 모든 시간 구간을 쓰는 것은 다르다.
- 50epoch 평균 화자 목표TV는 OC보나0.03476→0.00208, OC스푸프0.01171→0.00168이었다.
  정수 방문 횟수 때문에 개별 epoch에서 화자/출처 비율을 소수점까지 정확히 고정하지는 않는다.
- 상태/CSV/실행 구성은 `../artifacts/source_coverage_20261110/`, 서버 격리 검증 폴더는
  `/home/mentee/analysis/ebm_source_coverage_20261110_v1`이다.
  `--source-sampling coverage`를 명시한 구성이 이번 사용자 요청의 최종 구성이다.
  dev/eval·실제 모델 학습·optimizer 갱신은0회이며 성능 개선 여부를 측정한 결과는 아니다.

## 길이와 음성활동의 동시 통제

`--activity-policy`는 원음 선택 확률을 유지하고, 선택된 원음 안의 **길이 및 위치 확률**을 조정한다.
Train 전체의 앞/뒤 격자 후보를 중복 제거한 뒤 WebRTC VAD2(16kHz,20ms)로 증강 전 음성 프레임 수를 측정한다.
무음 판정은 별도 구간, 나머지는 음성 프레임 비율20% 간격으로 나눈다.
각 길이에서 보나/스푸프의 여섯 구간 확률과 평균 음성 비율을 맞춘다.

- 원음 선택 확률·화자/출처 비율·140개 클래스×화자×공격유형 그룹별 길이 목표는 유지한다.
- 원음별 조건부 길이 확률과 조각 위치 확률은 달라진다. 모든 원음은 계속 선택 가능하다.
- **화자별 음성활동 분포까지 같다는 보장은 없다.** 일부 화자는 해당 길이에 공통 후보가 없어 이 추가 조건을 동시에 충족할 수 없었다. 클래스 전체의 길이×음성활동을 통제한다.
- 구간별 확률과 평균을 맞추므로 연속적인 음성 비율의 모든 값이 같아지는 것은 아니다.
- VAD는 자동 추정이다. RawBoost 이후의 VAD 반응과 발음/언어/채널 특성까지 통제하지 않는다.
- 조각 후보의 다양성과 실제 선택된 시간 구간의 사용률이 변할 수 있다. 원음 UID 사용률은 같은 난수열에서 동일하다.
- VAD는 사전 계산만 한다. 학습 시 JSON/NPZ 확률표를 읽고 구간을 직접 해독하므로 매 조각 VAD 연산은 없다.
- 정책/배열 SHA, Train fingerprint, 원음 크기·수정 시각·UID 순서, 허용 구간과 확률 정규화·분포 제약을 시작 전에 검사한다.

확률표 생성과 검증은 Train만 사용하며 dev/eval 또는 모델 점수를 정책 선택에 사용하지 않는다.
`--validate-only`도 Train 프로토콜만 읽는다. 아래의 기존 학습 중 dev 평가 일정과 별개의 절차다.

### 2026-11-10 실제 Train 검증

- Train25,380개·중복 제거 후보734,966개, 단위22개·사전검사·두 worker의 실제2배치 입력 검사 모두 통과했다.
- 전체 후보의 기대분포에서 길이별 평균 음성 비율 차이6.30~8.04%p가0.000003%p 미만으로 감소했다.
  음성활동 여섯 구간 TV 거리0.107~0.259는모든 길이에서3e-9 미만이다.
- 실제epoch1의두rank·28,512조각에서 원음 UID와 증강seed 순서는 전후 완전히 같았다.
  층화 표본1,010개 실제 파형의 증강 전 VAD 프레임 수는 캐시와 전부 일치했다.
- 원음별 유효 후보 수(1/선택확률 제곱합)의 중앙값13.87→12.79개,
  2개 미만 비율0→0.366%다. 원음별 길이 확률 TV 변화의 중앙값은0.0388이다.
- epoch1 시간 구간 합집합 사용률: 보나OC39.39→39.24%, 스푸프OC38.66→38.68%,
  보나OC+BPL62.00→61.19%. 사용한 서로 다른 원음 수는 각각1,671/14,597/2,241개로 동일하다.
- 세밀한 음성 비율은 완전히 일치하지 않는다.4초 후보의20ms 프레임 수로 계산한KS는0.417→0.256이다.
  화자×길이별 평균 음성 비율 차이도 최대18.78%p 남는다. 증강 후 분포 일치나 성능 개선을 보장하지 않는다.
- 현재 정책은 서버의 격리 분석 폴더
  `/home/mentee/analysis/ebm_activity_joint_20261110_v5/activity_policy.json` 및 같은 폴더의
  `activity_sampling.npz`다. JSON SHA256은
  `0c0064b434fd6011b5c6ff5ba9a419c1fe769b9015fc7e59a6ccbeb126074a53`이다.
  입력 길이 정책은 같은 폴더의 `duration_policy.json`이다.
- 로컬 결과/CSV/그림은 `../artifacts/joint_activity_balance_20261110/`에 있다.
  서버 운영 저장소·기존 체크포인트는 불변이며 dev/eval·모델학습·optimizer 갱신은0회다.

## 기존 EBM과 달라지는 것과 유지하는 것

| 항목 | 기존 원본/4.0375초 상한 비반복 학습과 원음 전체 비반복 학습 | 최종 matched 구성 |
|---|---|---|
| 원음 선택 | 클래스별 shuffle; 4.0375초 상한 비반복 학습과 원음 전체 비반복 학습는 길이 sortish grouping | 클래스 quota/목표 비율 유지; coverage는 전수 순회+보정 추가 추출 |
| OC 입력 | 원본/4.0375초 상한 비반복 학습은 앞 구간 상한, 원음 전체 비반복 학습은 원길이 전체 | 보나/스푸프 모두0.5~4초 조건부 무작위 조각 |
| BPL 왼쪽 | OC 보나의 Long | OC 보나의 가변 길이 조각(임베딩 공유) |
| BPL 오른쪽 | 같은 원음 목록의 Short | 같은 분포에서 독립 추출한 보나 조각 |
| OC 손실/가중치 | 기존 OCSoftmax·클래스 가중치 | 동일 |
| BPL 수식/계수 | 기존 BPL·alpha30·전체12×12 | 동일 |
| 두 단계 학습 | backend OC+BPL → OC 중심 | 동일 |
| OC 클래스 배치 | GPU당 보나12·스푸프108 | 동일 |
| 모델/최적화 설정 | 기존 네트워크·frozenXLS-R·Adam·lr/weight decay | 동일 |
| 학습 일정 |50epoch·108논리배치/epoch | 동일 |
| 평가 점수 | 학습된 OC 중심과 cosine | 동일, 평가 난수/입력/지표 통제 추가 |

원본은 정상12개 전체 배치에서 Long12×같은 원음 Short12의144쌍이다.
대각12쌍은 같은 원음, 나머지는 묶음 내부 교차 원음이다. 원본의 불완전 마지막 배치는 예외다.
원음 전체 비반복 학습의 OC 입력도 원음 길이에 따라 가변이었지만 **맞춘0.5초 격자 분포의 조각**을 쓰던 것은 아니다.

기존과 최종 구성을 한 번 비교하면 OC 입력·길이 분포·BPL 원음 관계가 함께 바뀐다.
독립 원음 효과만 보려면 같은 OC 가변 입력을 쓰는 `matched_same_source`와
`matched_independent`를 비교한다. 두 조건의 OC 원음/조각/증강 stream은 동일하다.

## 평가 통제

- 두 구성 모두 **학습된 자기 OC 중심**으로 원래 cosine 점수를 계산한다.
  Train 평균 중심·추가 참조로더·ScoreReference 파일은 사용하지 않는다.
- `controlled_oc`에서 앞 최대64600sample(4.0375초), 한 조각, 반복 없음, 같은mask,
  GPU당batch8·CUDA_VISIBLE_DEVICES=0,1·평가seed1729를 사용한다.
- 원본 ASP의 eval 잡음은 유지하되 동일한 고정 RNG stream을 적용한다.
  평가 전후 Python/NumPy/CPU/CUDA RNG를 복원하고 모델/OC weight를 변경하지 않는다.
- 기존 full-ROC 선형 교차 EER(%), 보나=1·높은 점수=보나를 유지한다.
  trial 순서/수/SHA·frontend revision·점수 정의·seed·체크포인트 SHA와 개별 점수를 기록한다.
- dev를5epoch마다 및 마지막에 평가한다. target은 명시적인 평가전용 실행에서만 사용하고
  target으로 epoch/정책/threshold를 선택하지 않는다.
- 과거 난수 동작 재현용 `original_oc`도 보존한다. 비교 결과는 양쪽 모두 `controlled_oc`로 다시 계산한다.

## 실행과 통제 조건

```bash
export CUDA_VISIBLE_DEVICES=0,1
POLICY=/absolute/path/to/joint-policy.json
python main.py --bpl-pairing matched_independent --duration-policy "$POLICY" --validate-only
python main.py --bpl-pairing matched_independent --duration-policy "$POLICY" --name EBM_OC_BPL_matched
```

음성활동까지 맞춘 구성은 같은 폴더의 JSON과 NPZ를 함께 보존하고 명시적으로 지정한다.

```bash
ACTIVITY=/absolute/path/to/activity_policy.json
python main.py --bpl-pairing matched_independent --duration-policy "$POLICY" \
  --activity-policy "$ACTIVITY" --validate-only
python main.py --bpl-pairing matched_independent --duration-policy "$POLICY" \
  --activity-policy "$ACTIVITY" --name EBM_OC_BPL_activity_matched
```

전수 순회까지 적용하는 구성은 다음과 같다.

```bash
python main.py --bpl-pairing matched_independent --source-sampling coverage \
  --duration-policy "$POLICY" --activity-policy "$ACTIVITY" --validate-only
python main.py --bpl-pairing matched_independent --source-sampling coverage \
  --duration-policy "$POLICY" --activity-policy "$ACTIVITY" --name EBM_OC_BPL_activity_coverage
```

`activity_policy_path=None`과 `legacy_long_short` 기본값은 재현용으로 유지한다.
음성활동 정책은 matched 구성에만 적용하며 legacy 구성과 함께 지정하면 실행을 거부한다.
위 학습 명령은 사용 예시이며 이번 정책 생성·검증에서 학습을 실행하지 않았다.

- `legacy_long_short`: 기존 OC+BPL 학습 입력 경로. 기본 학습 설정으로 보존한다.
- `matched_same_source`: 양 클래스 OC 가변 입력, BPL 오른쪽은 OC 보나와 같은 원음에서 별도 조각.
- `matched_independent`: 동일 OC 입력/분포, BPL 오른쪽 원음까지 독립. 최종 목표 조건이다.

기존 체크포인트도 재학습 없이 같은 OC 평가를 사용할 수 있다.

```bash
python main.py --evaluate-checkpoint /absolute/path/to/old_CLS_50.pt \
  --evaluate-phases V --name EBM_old_controlled_OC
```

새 모델에도 같은 명령을 사용한다. 평가 전용은 optimizer update0회이며 기존 결과 경로는 덮어쓰지 않는다.
이전 잘못 구현한 BPL-only checkpoint는 학습된 OC 중심이 없으므로 이 비교의 대상이 아니다.

## 전체 변경 파일 (cf199af 대비)

| 파일 | 변경 |
|---|---|
| arguments.py | BPL 입력/길이·음성활동 정책·평가 모드/seed 설정 추가; 기존 키의 값 유지 |
| data_loaders.py | 명시적 matched 조건에서 새 로더로 분기; 기존 TrainSet/sampler/eval 클래스 유지 |
| dataset/ASVspoof2019_la_train.py | Train Item에 공격유형 메타데이터 및 Train 전용 프로토콜 reader 추가; UID/path/label 유지 |
| positive_bpl.py (신규) | 양 클래스 정책 검증·원음/길이/위치/증강 선택·120+12 sampler/collate·OC 평가 RNG/프로토콜 |
| activity_policy.py (신규) | 외부 음성활동 확률표 검증·선택, VAD의 학습 중 실행 없음 |
| tools/build_activity_candidates.py, fit_activity_policy.py, validate_activity_policy.py (신규) | Train 후보 VAD 측정·제약 확률 계산·기대분포/epoch1/실제 파형 검증 |
| tests/test_activity_policy.py (신규) | 분포 제약·원음/증강 난수 보존·캐시 무결성·Train 전용 reader 검사 |
| tests/test_source_coverage.py, tools/validate_source_coverage.py (신규) | 기대 원음 비율 보정·전수 순회·rank/OC 통제·실제 Train 사용률과 분포 검증 |
| native_steps.py | matched 입력의 OC+BPL 연결·가변 묶음 크기의 gradient replay; 기존 OC/중심2단계와 legacy 경로 유지 |
| main.py | CLI/사전검사·정책/입력 수 기록·OC 평가 프로토콜·체크포인트 평가전용 경로; OC 동결 제거 |
| train.py | 기존 훈련/CLS/TrainState 유지; OC 평가 난수 통제·프로토콜/개별 점수 기록 |
| tests/test_positive_bpl.py (신규) | 원본 보존·OC/스푸프 gradient·2단계 Adam·144쌍·양 클래스 분포·증강·OC 평가 회귀 검사 |
| tests/gpu_positive_smoke.py (신규) | 실제2GPU 각2논리배치·OC/중심 실제 갱신·두 조건 OC 입력 동일·평가/저장/불러오기 검사 |
| POSITIVE_BPL.md (신규) | 최종 동작·원본 대비 전체 변경·실행법·통제/한계 |

`model/loss.py`, `model/b_linear.py`, RawBoost 구현, logger는 수정하지 않았다.
codec-off/기존 masks/보조 손실0을 유지하며 추가 손실이나 collapse 방지 정규화를 넣지 않았다.
이전 별도 pilot의 RawBoost유형8 차이와 BPL-only 결과는 과거 검증 기록에 보존한다.
데이터/정책/모델/결과는 Git에 포함하지 않는다. 분포 일치만으로 성능 개선이나 숏컷 제거를 보장하지 않는다.

## 2026-11-10 추가: 속도 0.8~1.2배·최대 20% 크롭, 고정20epoch

사용자가 이전의 완만한 속도/크롭 분포 맞춤으로 20epoch 비교를 요청했다.
새 실행은 `tools/train_mild_duration.py`이며 OC+BPL 수식, OC 클래스 가중치,
backend→재계산한 OC 중심의 두 단계 Adam, frozen XLS-R, global OC240,
GPU당12×12 BPL, seed1, 108step/epoch를 유지한다. 기존 중단 학습을 재개하지 않는다.

`tools/build_mild_duration_policy.py`가 이전 randomized 계획에서 **19LA Train만** 추출해
UID·label·원본 샘플 수를 기존 검산된 메타데이터와 대조한다. 원음 선택은 클래스 내 균등하며,
전수 순회 후 보나12회/스푸프528회 추가 균등 추출로 매epoch 모든 원음을 사용한다.
OC 양 클래스와 독립 BPL 오른쪽 모두 방문마다 조건부 목표 길이를 무작위 선택한다.
이 실행은 기존 0.5~4초 길이표와 VAD 맞춤을 적용하지 않는다. 무음을 제거하지 않는다.

46,580개 변형을 서버 내에 미리 생성한다. 연속 원음 구간을 최소80% 남긴 뒤
SoX `tempo -s`(피치 보존 WSOLA)를 적용하고 float32 배열을 캐시한다.
정수 반올림 때도 속도범위와 크롭 상한을 지킨다. 목표마다 크롭 위치는 UID/목표/seed로 고정한
균등 위치이고, 방문별 무작위성은 목표 선택과 원본 RawBoost0~8에서 나온다.
출력 길이는 SoX의 실제 반환 길이와 검산하며, 수동 반복·패딩·길이 보정은 하지 않는다.
캐시의 모든 변형을 사전검사하고 학습 읽기 때도 파형 SHA를 검증한다.
반올림한 기대 길이 분포 겹침은 99.973684%지만 유한epoch 실제 분포에는 표본 잔차가 있다.

`positive_bpl.get_loaders`의 명시적 `mild_transform_cache` 옵션만 새 `MildTrainSet`을 선택한다.
기본/기존 조각 경로는 유지한다. `tools/check_mild_training.py`는20epoch 샘플 일정을
사전 재현해 원음 전수 사용 및 실제 길이 히스토그램을 기록하고, 최종 CLS/TrainState와
Adam2160step/매epoch 원음 사용/손실 유한성을 검산한다. 별도 GPU smoke2step은 본학습에 재사용하지 않는다.

5/10/15/20epoch 중간평가는 19LA 공식eval71,237개 전체이며 dev 평가/정책 적합/체크포인트 선택은 없다.
고정20을21LA148,176·21DF533,928개 전체에 평가하고, 기존 4.0375초 상한 비반복 학습/원음 전체 비반복 학습 20epoch 모델과 같은 입력·OC점수 정책으로 비교한다.
앞64600/반복 없음/mask/batch8/seed1729/원본 ASP잡음/각자 학습한 OC중심을 사용한다.
저장된 4.0375초 상한 비반복 학습과 원음 전체 비반복 학습 점수의 평가 난수는 다르므로 엄격한 동일 RNG 비교라고 부르지 않는다.
배치 240 고정 길이 반복 입력 학습의50epoch는 epoch가 다른 참고값이다. `tools/finish_mild_experiment.py`가 독립 EER 재계산과 비교표를 생성한다.

`tools/queue_mild_experiment.py`는 기존 DF 평가와 후속 비교분석이 성공적으로 끝나고
동일 컨테이너 셸과 GPU6/7이 실제 유휴일 때만 격리 snapshot을 실행한다.
대기8h/새pipeline48h/캐시20GiB 상한, 자동재시도0, 새 유료자원 및 지출상한0USD다.
기존 실행/모델/소스는 보존하며, 새 실행의 프로세스는 단계별 시간제한과 자체 정리를 갖는다.

## 2026-11-11 사용자 정정: 원본240 학습 전체 복원

사용자가 “학습 방식 전체를 복원해20epoch 처음부터 학습”을 선택했다.
앞 절의 독립 원음 BPL 대기는 학습 시작 전에 해제했으며 재활성화하지 않는다.
새 실행은 `tools/train_baseline240_mild.py`다. Git `10f3bcd`의 별도 고정
`baseline_code`에서 원본 main/model/loss/loaders와 **원본 ModelTrainer.train 그대로**를 호출한다.
기존 서버 CF branch나 실행 중인 평가 source는 변경하지 않는다.

변경되는 학습 데이터는 같은 Train-only 속도 0.8~1.2배·최대 20% 절단 변형 캐시다. 원본 클래스별 DistributedSampler가
원음 UID를 먼저 선택하고, 그 안에서 조건부 확률로 변형 하나를 선택한다.
`baseline240_mild.original_views`는 원본 TrainSet의 두 번의 음성 읽기만 같은 변형으로 대체한다.
원본의 long 앞64600sample/short 무작위48440sample·짧으면 wrap 반복·각 view 독립 RawBoost0~8을 그대로 실행한다.
따라서 최소80% 보존은 **속도 0.8~1.2배·최대 20% 절단 변형 생성 단계**의 제약이며, 원본 추가 crop 이후 네트워크 구간의 보장이 아니다.
학습 파형은 고정길이이며 mask·gradient replay·독립 오른쪽 보나 추가 입력은 사용하지 않는다.

원본 main의 2GPU global240/per-rank120과 원래 OC/BPL·클래스 가중치·두 단계 Adam을 사용한다.
기본108배치 중 각 rank의106번째는 보나12+스푸프60, 마지막은 보나6+스푸프108이다.
BPL은 처음107배치12×12, 마지막6×6이다. 매epoch 보나2580·스푸프23232회 방문하며
모든 원음 UID2580/22800을 사용한다. 데이터의 실제 소비만 기록하여 prefetch를 학습 사용으로 세지 않는다.
별도 GPU smoke는120/72/114 입력3배치로 전체 graph와 부분 배치를 확인하고, 본학습에는 재사용하지 않는다.

평가/저장은 사용자 지시대로5/10/15/20epoch의19LAeval전체, 고정20의21LA/DF전체다.
평가용으로만 같은 parameter key/shape의 mask 지원 backend에 학습 state를 복사한다.
학습 모델·OC중심·optimizer는 바꾸지 않고 평가 RNG를 보존한다.
반복 없는 앞64600/mask/batch8/seed1729/OCcosine/ASP잡음 조건으로 원본240의 보존20epoch도
재평가하여 같은 epoch·같은 평가 난수로 비교한다. 원본20에는 TrainState가 없으므로
그 과거 checkpoint에 대한 Adam-state 검증을 꾸며내지 않는다. 과거 codec 실행 불확실성은 남는다.

CPU 회귀4개는 원본 views, 클래스별 분산 sampler/부분 배치, 원본 train 함수와 직접 두 단계 Adam의
parameter 갱신 일치를 검증한다. 실제 캐시/2worker/GPU 전체 graph 검증은 기존 평가 종료 뒤 진행한다.
로깅의 기존 결과 삭제 분기는 같은 이름을 거부하도록 보호한다. 초깃값은seed1이며 저장240 가중치를 이어 학습하지 않는다.
