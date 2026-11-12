> 역사 문서 보존본. 원문 작성 당시의 상태·계획을 포함한다. 최신 상태와 확정 결과는 experience의 각 실험 README를 우선한다.

# EBM 채널 attention과 방향 손실 실험 명세

작성일: 2026년 11월 5일 KST. 버전: 1.0. 상태: 실험 설계 완료, 실험 구현 및 데이터 추출 전.

후속 실행 기록: 위 상태는 설계 작성 당시의 기록이다. 이후 T/V/DF pilot 캐시를 추출했으며, 2026-11-05 재추출 없는 네 가지 CPU 진단 결과는 [후속 결과](ebm_cached_core_results_20261105.md)에 정리했다. 본문 A~G 전체를 완료한 것은 아니다. 후속으로21LA전체cache와Q/19LA확장평가,P1/P2/P3-A의T/V전이비교를완료했다. 이들의지표별판정과진행중인내부CV/새추론은[최신통합분석](ebm_frozen_all_results_20261105.md)을따른다. 기존DF내부probe이득은전이EER개선과같은주장이아니다.

최종 발화 임베딩의 채널 attention과 방향을 반영하는 OC 손실을 검증한다. 기존 표현의 정보 조사, 정보의 일반화 확인, 새 학습 방식의 효과 확인을 별도 단계로 수행한다. A부터 G까지의 실험과 공통 데이터 추출 단계 P0를 정의한다. 목표는 특정 군집 모양을 만드는 것이 아니라, 기존 코사인 점수에서 이용하지 못하던 정보가 있고 제안 방법이 이를 미지 조건에서도 활용하는지 확인하는 것이다.

실행 기본값과 실험 목록은 [실험 설정](ebm_embedding_geometry_protocol.json), 검토 결과와 수학적 검증은 [검토 기록](ebm_embedding_geometry_review.md)에 둔다. 아래의 수치 설정은 사전등록할 설계 기본값이며, 이미 확인된 성능이나 보편적인 통계 기준이 아니다. 실험 ID는 구현과 결과를 연결하는 식별자다.

## 1 연구 질문과 가설

| 가설 | 연구 질문 | 핵심 실험 | 주된 반증 가능성 |
| --- | --- | --- | --- |
| H1 | 탐지에 유용한 정보가 특정 채널에 집중되는가 | A | 선택한 채널과 무작위 채널의 성능이 비슷함 |
| H2 | 발화별로 중요 채널을 바꾸는 것이 필요한가 | B | 입력별 gate가 고정 gate나 같은 크기의 MLP보다 좋지 않음 |
| H3 | 코사인과 벡터 크기를 넘어 방향 정보가 유용한가 | C | 방향 추가의 독립 평가 이득이 확인되지 않음 |
| H4 | spoof의 방향 구조가 공통인지 공격별인지 구별 가능한가 | D | 평균 방향 및 여러 모드가 독립 표본에서 재현되지 않음 |
| H5 | 방향별 정상 변동과 차원 간 상관관계가 탐지에 유용한가 | E | 공분산 기반 점수가 기존 점수보다 좋지 않음 |
| H6 | 위 정보가 화자나 수집 조건에만 의존하지 않는가 | F | 조건 통제 또는 미지 조건에서 이득이 사라짐 |
| H7 | 새 모듈 및 손실이 그 정보를 실제로 활용하게 만드는가 | G | 통제된 재학습에서 성능과 메커니즘 근거가 재현되지 않음 |

H1은 H2의 충분조건이 아니다. H3의 분포 차이도 H7의 성능 개선을 보장하지 않는다. H4에서 모든 spoof가 한 방향이어야 할 필요는 없다. 진단이 음성이어도 새 학습 방식의 가능성이 논리적으로 배제되는 것은 아니며, 현재 표현과 정한 probe의 범위에서 근거가 약하다는 의미로 기록한다.

## 2 프로젝트에 적용할 기준

### 2.1 확인한 구현과 탐색 기준

2026년 11월 5일 00시 28분 KST 전후의 읽기 전용 확인에서 현재 설정은 output_size 160, epoch 50, batch 240, lr와 lr_min 1e-5, alpha 20, alpha_cl 30, r_real 0.9, r_fake 0.2, T_0 150이었다. 학습 계정의 main.py 프로세스는 보이지 않았고 batch 160 및 240 실험에 50 epoch classifier checkpoint가 있었다. 이는 실행 시점 상태를 대신하지 않는다. GPU 처리 전 다시 상태를 확인한다.

기존 탐색 기준은 `/home/mentee/results/CM/LA/EBM_e50_bs160_lr1e5/model/BestCLS_50_21DF.pt`와 그 실험에 대응하는 frontend 상태 및 저장 소스다. 선택 이유는 미리 고정한 최종 epoch와 batch 160 기준을 사용하기 위해서다. 이름의 21DF는 저장 경로일 뿐 최적 checkpoint라는 근거로 쓰지 않는다. 파일 구조, 일치하는 frontend 상태, 전체 파라미터 로딩 여부를 P0에서 검증한다. 재사용 불가면 다른 epoch를 성능으로 고르지 말고 탐색 기준 미확정으로 처리한다.

batch 240 최종 checkpoint는 보조적인 표현 재현성 비교에만 쓴다. 서로 다른 batch의 결과로 attention이나 방향 손실의 효과를 주장하지 않는다. 기존 테스트셋 최저 epoch와 논문 수치는 새 실험의 모델 선택 기준으로 쓰지 않는다.

### 2.2 새 학습의 공통 조건

새 G 및 B의 완전 학습은 batch 160, epoch 50, lr=lr_min=1e-5, weight_decay=1e-4, 긴 입력 404 및 짧은 입력 303 frame 설정, 평가 404 및 num_seg=1을 기본으로 한다. 나머지 설정, optimizer 종류와 세부 인자, scheduler, 샘플 비율, 증강은 batch 160 저장 설정을 보존하고 해시를 기록한다. optimizer 종류와 decoder crop의 실제 sample 단위는 소스와 데이터 manifest 확인 전 임의로 추정하지 않는다.

frontend는 50 epoch 동안 기존 경로와 동일하게 고정한다. classifier는 학습한다. BPL은 GPU별 정상 긴/짧은 모든 쌍의 평균을 사용하는 기존 구현으로 유지한다. 전역 BPL, batch 확대, LR schedule 변경을 이번 요인과 섞지 않는다. GPU 두 장과 `CUDA_VISIBLE_DEVICES=0,1`을 사용한다. 새로운 기본값을 선택해야 하면 모든 비교군을 동일하게 변경하고 protocol 버전을 올린다.

소스 확인상 optimizer 갱신은 먼저 centroid를 고정한 classifier 단계, 다음 centroid만 학습하는 단계다. 새 gate와 metric은 첫 단계에서만 갱신하고 두 번째 단계에서는 고정한다. centroid 단계에는 bonafide만 기울기를 제공한다. 정확한 loss 결합과 class weight는 baseline의 실제 값을 따른다. 이름 비교로 requires_grad를 설정하는 기존 코드가 새 파라미터에 잘못 적용되지 않도록 파라미터 그룹과 gradient 검사를 추가한다.

### 2.3 실행 범위

원음·연구실 원본 소스·checkpoint는 서버에 둔다. 후속 사용자 명시 지시로 이번 실험의 파생 h/z·점수·metadata를 로컬 PC에 저장하고 재사용하는 범위가 허용됐다(MENTEE_SERVER.md22·29절). 현재 캐시 기반 통계와 표본 ID 잠금은 로컬 CPU에서 수행할 수 있다. 원본 음성·원본 코드·checkpoint의 전송은 이 범위에 포함하지 않는다. 기존 실험을 중단하거나 같은 이름으로 재실행하지 않는다. 로거의 기존 결과 삭제 경로를 재사용하지 않으며, 새 결과 경로가 존재하면 즉시 실패해야 한다.

## 3 데이터 분할과 평가 역할

### 3.1 고정할 데이터 역할

| 이름 | 출처와 역할 | 허용 작업 |
| --- | --- | --- |
| T | ASVspoof 2019 LA 공식 train | encoder 학습, probe 학습, 평균 및 공분산 추정 |
| V | 공식 dev에서 speaker 그룹으로 나눈 선택 부분 | probe와 metric 하이퍼파라미터 선택, 운용 threshold 설정 |
| Q | 공식 dev에서 남긴 잠금 부분 | 설계를 동결한 뒤 source 조건의 최종 확인 |
| R19 | 2019 LA 공식 eval | 기존에 관찰한 benchmark 재평가 |
| R21LA | 2021 LA 공식 eval | 기존에 관찰한 benchmark 재평가 |
| R21DF | 2021 DF 공식 eval | 기존에 관찰한 일반화 benchmark의 주 평가 |
| U | 사전 접근 이력을 점검한 별도 미사용 외부 평가셋 | 확보되고 잠금된 경우에만 새로운 외부 검증 |

V와 Q는 speaker 그룹을 seed 11로 무작위 순열한 뒤 절반씩 나눈다. 그룹 수가 홀수이면 V에 하나 더 둔다. 정상/spoof 클래스와 알려진 공격의 최소 표본 존재 여부를 점검한다. 분할이 불가능하면 자동으로 발화 단위 분할로 바꾸지 않는다. 다른 그룹 seed를 성능으로 고르지 않고 데이터 분할 재설계를 기록한다. 전체 manifest를 해시로 잠근다.

현재 실행 전에는 각 데이터의 경로, 접근 가능 여부, speaker 필드, 원발화와 파생 발화 관계를 확인해야 한다. 경로나 새 외부 데이터 접근 권한을 이 명세가 보장하지 않는다. U가 없으면 기존 benchmark에 대한 개선만 보고하고 새 blind 외부 검증이라고 부르지 않는다.

이미 관찰한 R19, R21LA, R21DF 결과는 이후 절차를 엄격히 고정해도 미관찰 test로 바뀌지 않는다. Q도 과거에 선택이나 분석에 사용했음이 발견되면 해당 오염 사실을 표시한다. encoder가 본 T 발화의 probe 성능은 학습 진단용이며 독립 평가가 아니다.

### 3.1.1 현재 캐시를 사용하는 진단의 표본 수

후속 사용자 요청에 따라 재학습 없는 캐시 실험에는 T25,380개(정상2,580/spoof22,800), V14,983개(1,351/13,632)를 유지하고 **DF20,000개와21LA20,000개를 각각 정상2,000/spoof18,000개로 고정**한다. 21LA 전체148,176개 캐시는 보존하되 반복 진단의 기본 입력으로 전부 사용하지 않는다. R2/R3에 대응하는 소규모 개입 확인은 각pilot 안에서 정상200/spoof1,800개, 합계2,000개씩이다.

구체적인 F0~F4/P1~P4/R1~R3별 수, score-independent SHA256 선택과 고정 ID 목록은 [동결 모델 명세3.1.1](ebm_frozen_model_experiments_20261105.md#311-신규-캐시-실험의-기본-데이터-수)에 있다. 이 표본 규칙은 현재 cached 분석의 후속 버전이며 과거 완료 결과의 표본을 바꾸지 않는다. 본문의 원래 B/G 공동 학습 및 공식 전체 benchmark 확인 계약과도 구분한다. 모델 학습/선택은 T/V이고, DF/21LA 내부 진단용 CV는 데이터셋별로 따로 수행해 전이 평가와 섞지 않는다. 전체21LA는 동결한 방법의 마지막 보조 평가에만 사용하며 pilot과 겹치므로 독립 확인으로 세지 않는다.

### 3.2 선택과 잠금 절차

1. P0 및 T와 V만으로 추출·분석 파이프라인을 검증한다.
2. A부터 F의 설정과 G 후보를 V에서 결정한다. 설정 탐색 횟수와 실패도 기록한다.
3. 기준 코드, 하이퍼파라미터, seed, 평가셋, 주 비교, 지표와 threshold를 해시로 잠근다.
4. Q와 R 평가를 한 번의 고정된 평가 묶음으로 실행한다. 실패 재실행은 같은 설정만 허용한다.
5. Q 또는 R을 보고 새 설정을 만들면 후속 버전의 탐색 결과로 분리한다. 이전 결과를 새로운 확인 결과처럼 재사용하지 않는다.

기본 checkpoint는 모든 새 학습에서 최종 epoch 50이다. dev-best checkpoint 선택은 이번 주 분석에서 사용하지 않는다. 모든 seed와 모든 비교군에 같은 epoch 규칙을 적용한다. 학습 중 Q 및 R을 반복 평가하지 않는다.

### 3.3 원본과 파생 표본 및 데이터 종속성

`utterance_id`와 별도로 `family_id`를 둔다. family는 동일 원발화의 crop, 증강, 재인코딩 및 확인 가능한 합성 파생본을 묶는다. `speaker_group`은 데이터셋별 의미가 확인된 화자 ID이며, 알 수 없는 값은 임의로 하나의 화자로 취급하지 않는다. 확인된 원발화 가족과 화자 연결이 있으면 의존성 graph의 연결 성분을 `dependency_group`으로 사용한다.

bootstrap과 분할의 원칙은 파생본 분리를 막고 알려진 종속성을 보존하는 것이다. 그룹 정의를 확인할 수 없는 평가셋은 utterance-family bootstrap만 사용하되 화자 종속성을 보정했다는 주장은 하지 않는다. 사용 가능한 독립 그룹이 매우 적으면 신뢰구간의 한계를 보고한다.

## 4 P0 임베딩 추출과 수치 검증

### 4.1 추출 대상과 저장 형식

기본 표현은 classifier의 마지막 LayerNorm 이후, OC 손실의 L2 정규화 이전 긴 발화 벡터 `z`다. LayerNorm 이전 벡터 `h`도 별도 hook으로 저장하여 보조 분석한다. 기본 차원은 160이다. `z`의 각 차원은 학습된 좌표이며 주파수 대역이라고 해석하지 않는다.

한 발화의 같은 crop에서 `h`, `z`, 원래 점수, 중심, 길이·padding metadata를 함께 얻는다. 가중 평균이나 짧은 branch를 기본 평가에 섞지 않는다. native 평가 loader를 재사용하고 crop 시작점, sample 수, padding 방식을 실제 값으로 기록한다. 설정의 404를 임의의 sample 수로 환산하지 않는다.

| 파일 | 필수 내용 |
| --- | --- |
| `manifest.parquet` | dataset, split, utterance_id, family_id, speaker_group, dependency_group, y, attack_id, condition_id, duration, crop_start, crop_samples, padding_fraction, extraction_seed, embedding_row |
| `embedding_z.npy` | 발화 순서와 일치하는 float32 N×160 |
| `embedding_h.npy` | LayerNorm 이전 보조 표현과 실제 차원 |
| `scores.parquet` | utterance_id, raw_score, cosine_s, raw_norm, tangent_norm, direction_valid |
| `model_metadata.json` | checkpoint 및 frontend SHA256, 코드 해시, 설정, key 누락·초과 목록, score polarity |
| `extraction_qc.json` | 누락·중복·finite 검사, 반복 추출 오차, 재현성 검사 결과 |

유도량은 서버에서 float64로 계산한다. 기본 label은 `y=1 bonafide`, `y=0 spoof`이며 정상 점수가 높다. public EBM 구현의 label convention도 이에 해당한다. 각 입력 protocol의 원래 문자열을 명시적으로 mapping하고 양·음성 뒤집힘 검사를 한다.

### 4.2 평가 난수와 crop

pooling에 `torch.randn` 호출 경로가 있으므로 `eval()`만으로 결정성이 보장된다고 가정하지 않는다. 원래 연산을 제거하지 않고 발화별 난수 상태를 재현하는 wrapper를 사용한다. 기본 추출 microbatch는 rank당 1발화이며 utterance 단위로 작업을 shard한다.

seed는 SHA256의 dataset, canonical utterance_id, condition_id, crop_id, extraction_seed로부터 생성한다. Python의 process별 hash()는 쓰지 않는다. 같은 키는 rank, 순서와 무관하게 같은 CPU 및 CUDA RNG를 쓰며, forward 전후 RNG 상태를 fork하여 복원한다. 설정과 환경, deterministic backend 상태도 기록한다.

기본 extraction_seed는 20261105이다. 추가 난수 민감도 분석은 V에서 세 개의 extraction seed 20261105, 20261106, 20261107로 고정한 최대 512발화에 수행한다. 기본 분석은 첫 seed만 사용하고, 차이가 수치 오차를 넘으면 label을 보기 전에 추출 방식을 재검토한다. 다른 seed 결과 중 좋은 것을 선택하지 않는다.

### 4.3 수식

`c`는 실제 학습한 centroid이고 `mu_B`는 T 정상 임베딩 평균이다. 둘을 서로 대체하지 않는다.

\[
\hat z=z/\|z\|_2,\quad \hat c=c/\|c\|_2,\quad s=\hat z^T\hat c,
\quad r=\hat z-s\hat c,\quad t=\|r\|_2.
\]

`t >= 1e-4`일 때만 `u=r/t`를 정의한다. 그 미만은 u를 0으로 저장하고 direction_valid=0을 별도 feature로 제공한다. 1e-5와 1e-3은 V에서의 민감도 분석이며 주 기준은 바꾸지 않는다. 작은 t 표본은 전체 탐지 평가에서 제거하지 않는다. 방향만의 군집 통계는 valid 표본에 한정하고 클래스별 제외 비율을 보고한다. z 또는 c가 1e-12 이하의 norm이면 추출 실패로 처리한다.

\[
\|r\|_2^2=1-s^2,\quad r^T\hat c=0,\quad
\|\hat z-\hat c\|_2^2=2-2s.
\]

잔차 norm의 클래스 차이를 새로운 방향 정보라고 주장하지 않는다. LayerNorm 때문에 z norm이 제한될 수 있으므로 norm 정보를 가정하지 않고 실제 분포를 기록한다. 학습 가능한 LayerNorm affine 파라미터 때문에 norm이 완전히 상수라고도 가정하지 않는다.

### 4.4 통과 기준과 실패 처리

모든 발화 ID·행 수 일치, 무한값과 NaN 0개, 로딩 누락·예상 밖 key 0개가 기본 조건이다. 같은 crop과 RNG 상태에서 native score와 hook으로 재계산한 cosine은 최대 절대오차 1e-5 이하여야 한다. 동일 발화의 반복 추출 및 다른 shard 순서의 score 차이도 1e-5 이하를 요구한다. float64 유도량 항등식은 최대 절대오차 1e-10 이내여야 한다. backend 비결정성이 남으면 환경과 실제 오차를 기록하고 허용 범위를 사후로 늘려 통과시키지 않는다.

DDP sampler padding으로 생긴 중복은 canonical ID와 condition으로 중복 여부를 검사한다. 다른 점수 중 좋은 것을 고르지 않고 한 번만 유효하게 처리한다. 기본 추출은 padding이 없는 shard 분할을 사용한다. score polarity는 bonafide와 spoof를 반대로 넘긴 합성 예제까지 검사한다.

## 5 공통 분석과 통계

### 5.1 점수와 지표

EER와 AUROC는 bonafide를 양성으로 계산한다. EER는 동일한 DET/ROC 구현으로 전체 표본과 모든 비교군을 처리하고 percent 단위로 저장한다. `delta_eer_pp = EER_baseline - EER_candidate`가 양수면 개선이다. pp와 상대 개선율을 함께 쓰되 둘을 혼동하지 않는다.

probe의 주 지표는 클래스 균형 log-loss다. `0.5 mean_B(-log p_B) + 0.5 mean_S(-log(1-p_B))`이며 p는 [1e-7, 1-1e-7]로 clip한다. 이는 동일 클래스 prior 아래의 예측 지표이며 실제 배포 확률 calibration을 입증하지 않는다. class-balanced AUROC, EER 및 표준 log-loss도 보조로 보고한다.

운용 오류는 V에서 baseline bonafide 오거부율 1%를 목표로 잡은 threshold로 계산한다. 각 모델은 자신의 V에서 같은 목표로 threshold를 한 번 설정한다. 동점이면 목표 오거부율을 넘지 않는 보수적 규칙과 실제 달성률을 기록한다. spoof 통과율과 정상 오거부율을 Q/R 및 변형 조건에서 보고한다. EER를 위해 test에서 threshold를 sweep하는 것은 평가 지표 계산으로 허용하되 그 threshold를 운용 threshold로 재사용하지 않는다.

공격별 EER는 각 공격 spoof와 그 평가셋의 동일한 전체 bonafide pool로 계산한다. 여러 공격에서 재사용된 정상 표본을 독립 반복으로 세지 않는다. 공격별 결과, macro 평균, 전체 utterance EER를 구별한다. 주 benchmark는 R21DF이며 R19와 R21LA는 guardrail 및 보조 benchmark다. U가 잠금되면 그 외부 결과를 별도로 주 검증으로 보고한다.

### 5.2 효과량과 판정

단일 seed의 evaluation CI는 같은 dependency_group을 두 모델에 같은 횟수로 resample하는 paired cluster bootstrap 2000회, seed 7301로 계산한다. 정상·spoof를 함께 포함하는 그룹은 그대로 보존한다. 한 클래스가 사라지는 replicate는 폐기하고 폐기율을 보고한다. 전체 그룹을 유지한 단일 계층 bootstrap을 기본으로 하고 임의의 nested resampling을 추가하지 않는다.

유효 dependency_group이 10개 미만이거나 클래스별 기여 group이 5개 미만이면 CI를 참고값으로만 표시하고 강한 supported 판정을 하지 않는다. 알려진 원발화 종속성을 묶다가 거대한 연결 성분만 남는 경우에도 그룹을 임의로 쪼개어 표본 수를 늘리지 않는다. A부터 F는 분포와 설계 선택을 위한 탐색 분석으로 95% CI를 사용하고, G의 사전등록된 다섯 비교와 같은 강도의 확인 결론으로 부르지 않는다. Q는 독립 평가 역할이지만 여러 진단에 재사용한 효과까지 전체 family-wise 오류가 제어된 것은 아니다.

seed 1, 2, 3은 screening 학습, 1, 2, 3, 4, 5는 최종 확인 학습이다. 최종 비교는 seed별 차이, 평균 차이, seed와 evaluation group을 함께 resample한 2000회 CI를 보고한다. seed 3개만으로 seed 모집단에 대한 정밀한 추론을 주장하지 않는다. 이 bootstrap은 평가 표본 및 initialization 불확실성을 근사하며 encoder 훈련 데이터 모집단 전체의 불확실성을 포함하지 않는다.

기본 실용 효과 기준은 R21DF EER에서 `delta_star=max(0.05 pp, 0.05×baseline_EER_percent)`다. 이는 이번 연구의 선택 기준으로 제안한 값이며 자연법칙이 아니다. 통계적 지지는 delta의 CI 하한이 0보다 큰 경우, 실용적 지지는 점추정이 delta_star 이상인 경우로 따로 기록한다. R19/R21LA에서 악화폭의 CI 상한이 `max(0.20 pp, baseline EER의 10%)` 이하인지 guardrail도 보고한다. 부정확한 CI는 개선이나 비열등이 아니라 불확실로 처리한다.

확인 비교군이 여러 개인 경우, 같은 R21DF에서 G의 다섯 주 비교에 대해 Bonferroni family-wise CI 99%를 추가한다. 다섯 비교는 G2 대 G0, G2 대 G1, G2 대 Gm, G3 native 대 G0 cosine, G3 native 대 G0에서 같은 절차로 fit한 posthoc metric이다. 실제 비교 tuple은 잠금 파일에 기록한다. G4 상호작용과 세부 공격·채널 분석은 탐색으로 구별한다. 채널별 탐색 p값을 제공한다면 BH-FDR q=0.05를 적용하되 순위 자체와 효과량도 보고한다.

### 5.3 Probe와 전처리의 공통 규칙

T에만 fit한 standard scaler와 class weight를 사용한다. logistic regression의 L2 C는 [0.01, 0.1, 1, 10]에서 V balanced log-loss 최소로 선택한다. 선형 probe와 1 hidden layer GELU MLP를 모두 사용한다. MLP hidden은 16 또는 32, Adam lr [1e-4, 1e-3], weight_decay [1e-4, 1e-2], batch 256, 최대 100 epoch, V patience 10이며 최저 V epoch를 저장한다. probe seed는 101, 102, 103이다. 동일 비교에서 같은 후보 수와 selection rule을 준다. test를 early stopping에 쓰지 않는다.

seed별 서로 다른 scaler를 혼합하지 않는다. feature 공간이 서로 다른 encoder끼리 채널 index나 prototype 각도를 직접 비교하지 않는다. 동일 checkpoint의 subset 비교에서만 좌표별 순위 상관을 사용한다. 다른 checkpoint의 비교는 예측 성능이나 회전 불변 스펙트럼을 사용한다. full training seed와 probe seed를 결합한 5×3 결과를 15개의 독립 full training run으로 세지 않는다. probe가 붙은 비교는 각 training seed 안의 동일 probe seed를 짝짓고, probe seed 변동을 따로 요약한다. joint CI가 필요하면 training seed, 그 안의 probe seed, evaluation dependency group을 순서대로 resample한다.

### 5.4 그림

기본 그림은 score histogram, norm histogram, PCA, UMAP, 공격×채널 heatmap, dimension ablation curve, covariance spectrum, delta EER forest plot이다. PCA와 UMAP은 T의 클래스 균형 최대 5000표본에 fit하고 다른 split에는 transform한다. UMAP seed 17, n_neighbors 30, min_dist 0.1, cosine metric을 고정한다. 그림의 클래스 균형 subsampling은 성능 평가 표본을 바꾸지 않는다. UMAP에서 새로 생긴 간격을 원래 공간의 밀도나 분리 증거로 쓰지 않는다 [4].

## 6 실험 A 채널 정보의 집중

### 6.1 입력과 계산

P0의 기존 checkpoint 임베딩을 사용하며 encoder는 고정한다. 기본은 z, h는 보조다. train 정상과 전체 spoof 및 공격 k별로 아래를 계산한다.

\[
J_{k,j}=\frac{(\mu_{B,j}-\mu_{S_k,j})^2}{\sigma^2_{B,j}+\sigma^2_{S_k,j}+10^{-8}}.
\]

J를 pooled spoof로 계산한 순위와 공격별 J의 동일 가중 평균 순위를 따로 저장한다. 주 순위는 공격 macro 평균이다. variance가 거의 0인 차원은 flag하고 분모 보정에 의존한 큰 값은 별도 보고한다. 선택 비율은 [0.05, 0.10, 0.25, 0.50, 1.00], d=160에서 개수는 [8, 16, 40, 80, 160]이다. 무작위 subset은 seed 200부터 219까지 20개를 만들고 좋은 subset 하나를 고르지 않는다.

### 6.2 비교와 절차

조건 A_all, A_top, A_random, A_drop_top, A_drop_random을 구성한다. 고정 score 비교에서는 z와 c에 동일 mask를 적용한 후 각각 재정규화해 cosine을 계산한다. masked c norm이 1e-12 이하면 해당 비교는 정의 불가다. 재학습 probe 비교에서는 남은 좌표로 같은 probe를 fit한다. norm 변화의 효과를 구분하기 위해 (s, norm)를 포함한 augmented probe 비교도 별도로 한다.

순위와 mask는 T에서 결정하고 V에서 probe 설정만 선택한다. Q/R에서 순위를 재계산하여 mask를 바꾸지 않는다. Q/R의 관찰 순위는 사후 안정성 그림으로만 보고한다. importance가 높다고 causal acoustic feature라고 명명하지 않는다.

### 6.3 결과와 판정

주 비교는 상위 25%와 무작위 25% probe의 Q balanced log-loss 차이, 보조는 R21DF delta EER다. top 제거 시의 악화가 random 제거의 20회 분포보다 큰지도 보고한다. 5%, 10%, 50%는 곡선으로 제시한다. 한 threshold만 좋은 경우 전체적으로 채널 집중이 입증됐다고 하지 않는다.

출력: `A/channel_statistics.parquet`, `A/masks.json`, `A/probe_scores.parquet`, `A/ablation_curves.csv`, `A/attack_channel_heatmap.png`, `A/decision.json`. 무작위 mask에 대한 변동과 evaluation group bootstrap 변동은 서로 다른 축으로 저장한다.

## 7 실험 B 발화별 채널 attention

### 7.1 고정한 구조

삽입 위치는 마지막 LayerNorm 이후이며 긴 branch와 짧은 branch에 같은 모듈을 사용한다. 기존 BPL 입력 위치를 유지하되 변환된 표현을 양 branch에서 동일하게 사용한다. 이는 BPL도 새 표현에 작용하는 설계다. gate를 BPL 뒤에만 넣는 변형은 이번 주 실험에 섞지 않는다.

정적 gate: `a=2 sigmoid(v)`, v∈R^160, v=0으로 identity 초기화한다. 동적 gate: `a(z)=2 sigmoid(W2 GELU(W1 z)+b2)`, hidden 16, W2와 b2를 0으로 초기화한다. 출력은 z'=a⊙z이고 OC 및 BPL 안에서 기존과 같이 정규화한다. 별도 LayerNorm을 추가하여 가중치를 다시 지우지 않는다. gate 출력은 항상 양수이므로 sign flip 모듈이 아니다.

대조 MLP는 `z'=z+W2 GELU(W1 z)+b2`, 같은 hidden 16과 identity 초기화다. 동적 gate와 MLP는 각각 5296개, 정적 gate는 160개 파라미터다. 실제 model summary로 검증한다. 기초 backbone 초기 state를 복제하고 새 모듈 전용 RNG를 사용하여 모듈 초기화가 backbone 또는 증강 난수 순서를 바꾸지 않게 한다.

### 7.2 비교군과 학습

B0=기존, B1=정적 gate, B2=동적 gate, B3=잔차 MLP다. G0, G1, G2, Gm과 정확히 같은 run을 재사용한다. B와 G를 별도로 중복 학습하지 않는다. 각 seed에서 동일한 데이터 순서와 증강 난수를 가능한 한 공유하고 DDP rank별 원발화 및 crop seed 해시를 비교한다.

기본 구조는 hidden 16 하나로 고정한다. hidden 8, 32 탐색을 추가한다면 B3에도 동일 예산을 주고 V에서만 선택하며 실행 수를 별도로 집계한다. 추가 하이퍼파라미터가 없는 B1과의 차이를 tuning 이득으로 과장하지 않는다.

### 7.3 gate 기능 진단

학습한 B2에서 원래 gate, 항상 1인 gate, T gate 평균, 다른 발화 gate를 비교한다. 다른 gate는 평가 라벨을 사용하지 않고 V에서 잠근 score·norm 구간 내에서 seed 301부터 320까지 20회 재배치한다. 단독 표본 구간은 제외하지 말고 unchanged로 표시한다. 이 개입은 latent 분포를 바꿀 수 있으므로 standalone 모델 superiority의 주 근거가 아니다.

gate 재배치 구간은 해당 모델의 gate 전 z와 learned c로 계산한 s의 T pooled 10분위, log(norm)의 T pooled 5분위의 교차 구간이다. 경계 중복은 합치며 label에 따라 구간을 바꾸지 않는다. 이 s는 gate 후 native score와 구별하여 저장한다. C의 방향 재배치에도 원래 checkpoint의 같은 10×5 규칙을 적용한다. MMD용 score overlap 병합 규칙과 재배치용 구간 규칙은 다른 목적이므로 혼용하지 않는다.

상위 gate 차원 제거와 같은 수의 무작위 차원 제거도 수행한다. a_j뿐 아니라 `abs(a_j z_j)`와 class별 gate 요약을 기록한다. 큰 gate 값이 곧 인과적 음향 중요도라는 해석은 하지 않는다. gate 분산이 큰 것만으로 H2 지지로 판정하지 않는다.

### 7.4 결과와 판정

주 비교는 B2 대 B1, B2 대 B3의 R21DF EER다. B2가 B0보다 좋아도 B1과 같으면 입력 적응성의 이득으로 주장하지 않는다. 원본·변형 발화의 gate 변화와 score 변화는 F에서 함께 분석한다.

출력: `B/model_comparisons.csv`, `B/gate_statistics.parquet`, `B/interventions.csv`, `B/condition_effects.csv`, `B/decision.json`. main 성능은 native gate만으로 계산하고 다른 gate 개입 결과 중 좋은 것을 채택하지 않는다.

## 8 실험 C 코사인을 통제한 방향 정보

### 8.1 가설의 정확한 범위

목표 가설은 `Y`와 `u`가 `(s, norm)`을 알고도 예측 관계를 갖는지다. 아래 probe 비교는 정한 모델 계열의 예측 개선을 확인하며 conditional mutual information의 정확한 추정이나 조건부 독립성의 완전한 증명은 아니다. 충분히 작은 score bin이라고 해서 s가 정확히 동일해지는 것은 아니다.

### 8.2 nested probe

| ID | 입력 |
| --- | --- |
| C0 | s |
| C1 | s, log(norm) |
| C1v | s, log(norm), direction_valid |
| C2 | s, log(norm), direction_valid, u의 160좌표 |
| Cnoise | C1v와 u 대신 label 독립인 160차원 unit noise |

direction_valid는 방향 안정성 정보 자체의 이득과 u의 이득을 분리한다. 주 비교는 C2 대 C1v이며 C1 대 C0는 norm의 기여다. Cnoise는 차원과 capacity의 보조 대조군이지 u와 완전히 동일한 분포의 null sample이 아니다.

각 입력에서 선형 probe와 공통 MLP grid를 사용한다. scalar baseline은 T quantile에 5개 knot를 둔 cubic spline logistic 후보도 허용하여 단순한 비선형 score 보정의 이득을 누락하지 않는다. 후보와 knot는 V까지로 고정한다. Cnoise는 id별 독립 PRNG로 만들고 원발화 파생본에는 같은 noise를 사용하여 독립 표본처럼 부풀리지 않는다. 의미가 있는 ID 문자열을 feature로 넣지 않는다.

### 8.3 score overlap과 분포 검사

T의 클래스 균형 pooled s quantile로 10개 bin 경계를 정한다. 경계 중복은 합치고 구간 끝은 ±infinity다. V에서 bin마다 클래스별 최소 20개 유효 방향, 클래스별 speaker 5개를 요구한다. 부족한 구간은 인접 구간과 결정적인 왼쪽 우선 규칙으로 합쳐 최종 경계를 동결한다. 화자 정보가 없으면 speaker 조건 충족을 주장하지 않는다. Q/R에서 부족한 구간은 통계 NA로 두고 병합·경계를 바꾸지 않는다. 각 클래스의 포함 비율과 score 분포 차이를 반드시 보고한다.

valid 방향의 Gaussian kernel MMD²는 bin별로 계산한다 [1]. diagonal 항을 제외한 unbiased U-statistic을 사용하고 음수 표본 추정값을 0으로 잘라내지 않는다. bandwidth는 T valid 방향의 고정 subset에서 pairwise Euclidean distance 중앙값, kernel은 exp(-||u-v||²/(2 bandwidth²))다. 중앙값이 0이면 positive distance 중앙값으로 바꾸고 그것도 없으면 kernel 비교를 NA로 처리한다. 각 bin에서 클래스별 최대 512표본을 고정 seed로 뽑으며 같은 원발화 가족의 반복본을 독립 표본으로 쓰지 않는다. 10개 subset 반복의 분포를 보고하되 10개 독립 실험이라고 세지 않는다. 전체 요약은 V에서 고정한 bin 가중치로 평균하고 평가 표본 수 가중으로 바꾸지 않는다.

finite bin MMD는 잔여 s·norm·화자 차이의 영향을 받을 수 있다. 효과량과 bin 내 불균형을 보고하고 정확한 conditional test p값을 붙이지 않는다. label 교환가능성이 정당화된 독립 group 검정을 별도 설계하지 않는 한 일반적인 발화 단위 permutation p값은 금지한다.

### 8.4 방향 재배치와 판정

C2 score를 원래 u와, 동결된 s·norm 구간 안에서 label 없이 섞은 u로 계산한다. 섞기 20회, seed 401~420이다. valid 상태가 같은 발화끼리만 섞고 동일 family는 donor에서 제외한다. donor 없음은 unchanged이며 바뀐 비율과 donor 간 s·norm 차이를 저장한다. 이는 방향 대응을 흐트러뜨리는 민감도 진단으로만 사용한다. 정확한 조건부 null을 생성했다는 주장은 금지한다.

주 지표는 Q balanced log-loss 개선, 보조는 R21DF delta EER와 error rescue/harm다. rescue와 harm은 V에서 고정한 각 모델 threshold로 계산한다. test EER threshold에서 분류한 rescue는 주 결과로 쓰지 않는다. C2가 개선되지 않으면 선형과 작은 MLP의 한계를 포함해 불확실 또는 미지지로 기록한다. 높은 MMD만으로 탐지 개선을 주장하지 않는다 [2].

출력: `C/probe_registry.json`, `C/bin_definitions.json`, `C/support_coverage.csv`, `C/mmd_by_bin.csv`, `C/probe_scores.parquet`, `C/rescue_harm.csv`, `C/direction_shuffle.csv`, `C/decision.json`.

## 9 실험 D 공통 방향과 공격별 방향

### 9.1 계산

valid u에서 공격별 평균 m_k, 집중도 R_k=||m_k||, 클래스별 covariance 및 2차모멘트 M_k=mean(u u^T)를 계산한다. `d_eff=(tr M)^2/tr(M²)`를 participation ratio로 명시하며, 다른 정의의 entropy effective rank와 혼용하지 않는다. 방향은 c에 수직이므로 160차원에서도 최대 유효 공간은 159차원이다. sample 수 한계와 tangent rank 제한을 분포 붕괴로 오해하지 않는다.

평균 방향의 정규화는 R_k>=0.1일 때만 수행한다. 이 수치는 prototype 불안정성을 피하기 위한 설계값이며 0.05와 0.2의 V 민감도도 기록한다. 낮은 R은 diffuse 분포와 반대 모드 상쇄 모두 가능하다. 공격 간 평균 방향 cos 행렬의 undefined 값은 0으로 채우지 않는다.

공격별 유효 방향이 20개 미만이거나 확인 가능한 speaker가 5개 미만이면 집중도·평균 방향은 탐색 참고값으로만 표시한다. R은 표본 수가 작아도 0보다 커질 수 있으므로, 같은 표본 수를 맞춘 반복 subsampling과 tangent 공간의 isotropic unit noise 1000회 분포를 함께 보고한다. 이 noise는 구조를 해석하는 기준 분포이지 실제 정상 분포나 정확한 조건부 null의 대체물이 아니다.

### 9.2 모델과 전이

T spoof valid 방향에 common prototype, 공격별 prototype, label-free spherical k-means K=[2,4,8]을 fit한다. K=1은 common과 연결된다. spherical k-means seed 501~505, 최대 100 iteration, 중심 norm이 0에 가까우면 고정 seed 재초기화하고 빈 cluster 횟수를 기록한다. score는 최대 prototype cosine이며 bona score 방향이 반대인 경우 부호는 V에서 한 번 고정한다. (s,norm,valid,prototype score)를 같은 probe에 입력한다. prototype만의 attribution 성능과 bonafide/spoof 탐지 성능을 구분한다.

각 K의 초기화 5개 중 T에서 cosine 합 목적이 가장 좋은 해를 대표 모델로 고른다. V 또는 test가 가장 좋은 초기화를 고르지 않는다. 나머지 초기화는 민감도 분석으로 남긴다. K와 prototype 유형은 공통 probe의 V balanced log-loss로 선택하고 동점이면 작은 K, 같은 K이면 common, label-free, attack-labeled 순으로 단순한 모델을 우선한다. common 평균 norm이 기준 미만이면 common prototype은 NA이며 임의 단위방향을 만들지 않는다.

공격별 prototype의 각도와 구조가 V와 Q에서 재현되는지 표본 bootstrap으로 평가한다. 다른 dataset의 attack_id 문자열이 같은 생성기를 뜻한다고 가정하지 않는다. 매핑이 문헌 또는 protocol로 확인된 경우만 동일 attack의 전이로 부른다.

### 9.3 미지 공격 범위

encoder가 이미 보았던 T 공격 하나를 probe에서만 제외하는 것은 `probe-only leave-one-attack-out`이다. 그 결과로 encoder 수준의 미지 공격 일반화를 주장하지 않는다. 공식 eval의 미지 공격은 학습 노출 기록을 확인하여 따로 표기한다.

완전한 leave-one-attack-out은 선택한 공격 k를 encoder 학습 T, V tuning, prototype 학습에서 모두 제외하고 baseline과 후보를 다시 학습한다. 제외 공격의 발화는 마지막 평가에만 사용하며 source-family도 분리한다. 알려진 모든 공격 family에 대해 같은 절차를 적용한다. 이 확장 실험은 본 G matrix에 포함되지 않으며 K_attack×모델 수×seed 수의 비용을 별도로 계산한다.

출력: `D/concentration.csv`, `D/direction_similarity.csv`, `D/spectra.csv`, `D/prototype_registry.json`, `D/prototype_scores.parquet`, `D/transfer_results.csv`, `D/decision.json`.

## 10 실험 E 방향별 정상 변동과 metric

### 10.1 동일 중심에서의 metric 비교

주 비교는 모든 방법에서 같은 learned unit centroid c_hat, 같은 unit z를 사용한다. T의 unit bonafide embedding으로 sample mean mu_B와 covariance Sigma_B를 계산한다. covariance는 mu_B를 기준으로 추정하지만 score 중심은 c_hat로 유지한다. 따라서 이 score는 정상 Gaussian likelihood라는 가정보다 고정 중심의 가중 거리로 해석한다.

\[
q_A=(\hat z-\hat c)^T A(\hat z-\hat c),\quad S_A=1-q_A/2.
\]

| ID | A |
| --- | --- |
| E0 | 기존 cosine |
| E1 | I, E0와 동일성 확인 |
| E2 | diagonal covariance inverse를 trace d로 정규화 |
| E3 | shrinkage full covariance inverse를 trace d로 정규화 |

Sigma_gamma=(1-gamma)Sigma+gamma tr(Sigma)/d I, gamma=[0.01,0.1,0.5,1.0]에서 V EER 최소로 선택한다. 동점이면 큰 gamma를 선택한다. E2와 E3의 gamma 후보 수는 같다. eigenvalue floor는 `1e-6×max(tr(Sigma)/d,1e-12)`다. covariance trace가 1e-12 이하이면 E2/E3을 degenerate로 기록하여 NA로 처리하고 임의 inverse를 만들지 않는다. E3은 solve/eigendecomposition으로 계산하고 무조건적인 inverse나 극소 고유값 증폭을 피한다. trace normalizing 때문에 gamma=1이면 E1으로 돌아와야 한다.

mu_B를 score 중심으로 쓰는 E_mu는 보조 요인 실험이다. E0 대 E_mu의 차이를 covariance 효과라고 하지 않는다. 중심 c_hat/mu_B와 metric I/diagonal/full을 모두 조합하여 중심 효과와 metric 효과를 구분한다. E_mu의 기준 Euclidean도 동일 mu_B로 계산한다.

### 10.2 출력과 해석

T와 V의 eigen spectrum, condition number, participation ratio를 기록한다. 주 성능 비교는 E3 대 E0, E3 대 E2의 R21DF delta EER이며 Q는 독립 source 평가다. gamma=1이 선택되거나 큰 비등방성만 있고 성능 이득이 없으면 방향별 분산의 유용성 근거가 약하다. full이 diagonal보다 좋으면 상관관계 활용의 근거가 되지만 causal artifact 방향을 식별했다는 뜻은 아니다.

단일 PSD metric은 잔차의 부호 반전을 제곱하므로 임의 잔차 v와 -v에 동일 값을 준다. 단, c_hat±v가 둘 다 unit sphere에 놓인다는 뜻은 아니다. 비대칭 spoof 방향 모델의 충분한 대체물로 해석하지 않는다.

출력: `E/metric_registry.json`, `E/covariance_spectrum.csv`, `E/metric_scores.parquet`, `E/center_metric_factorial.csv`, `E/decision.json`.

## 11 실험 F 화자와 채널 조건의 통제

### 11.1 평가할 모델

기존 checkpoint의 C1v와 C2, E0와 V에서 선택한 E metric, 학습 완료한 G0부터 G4를 평가한다. 모델 목록과 threshold는 변형된 Q/R 결과를 보기 전에 고정한다. 변형 표본으로 scaler, prototype, covariance, gate 또는 threshold를 다시 fit하지 않는다.

F는 `F_frozen`과 `F_trained` 두 평가 시점으로 나눈다. F_frozen은 C/E의 고정 encoder 결과만 사용해 G 전에 수행할 수 있다. F_trained는 G 종료 후 새 학습군을 평가한다. 전자의 완료를 위해 G가 필요하지 않으며, 후자의 결과로 G 설정을 다시 선택하지 않는다. B도 별도 학습 runner가 아니라 G0/G1/G2/Gm의 결과를 받아 H2를 평가하는 분석 단계다.

### 11.2 짝지은 변형 조건

| ID | 변형 | 고정 규칙 |
| --- | --- | --- |
| clean | 원본 | 기본 crop 및 원래 PCM |
| noise20 | white Gaussian noise SNR 20dB | source waveform RMS 기준, 발화별 고정 seed |
| noise10 | 같은 noise 방향 SNR 10dB | noise20과 같은 underlying noise, 크기만 변경 |
| gainminus6 | 진폭 -6dB | float32 PCM에서 곱하기 10^(-6/20) |
| mulaw8 | 8kHz G711 mu-law 왕복 | mono 8k encode/decode 후 native 16k로 복원 |
| opus16 | Opus 16kbps 왕복 | mono, target audio bitrate 16000, 20ms frame, decode 16k |

실제 source sample rate와 channels는 P0에서 확인하고 필요한 resampling을 모든 조건의 동일 공통 경로로 수행한다. codec 버전과 정확한 명령, delay 처리, output 길이, resampling backend를 manifest에 남긴다. 지연을 보정하는 backend 동작은 V에서 검증하여 잠근다. timestamp 기준 crop을 사용하며 codec 조건만 다른 구간을 잘라 쓰지 않는다. 변형은 원본 waveform에 적용한 뒤 crop과 padding을 한다.

noise RMS는 인위적으로 추가한 padding을 제외한 원본 유효 waveform에서 계산한다. 무음만 있는 입력은 조건을 생성하지 않고 이유를 기록한다. clipping은 적용하지 않는 float32 중간 신호를 쓰고 peak 및 abs>1 비율을 기록한다. decoder가 정수 PCM을 요구하면 공통 gain 처리와 saturation 검사를 V에서 결정하고 protocol 버전을 올린다. 실제 SNR 오차는 0.1dB 이내여야 한다. 길이 보정은 모든 모델에 같아야 하며 codec별 합격 tolerance는 V에서 측정한 sample 단위로 고정한다.

noise20과 noise10은 condition_id별로 별도 noise를 생성하지 않는다. 두 조건 모두 공통 `noise_base` key로 waveform을 한 번 생성하고 진폭만 바꾼다. 추출 RNG의 condition_id는 별도로 유지한다. native loader가 진폭을 재정규화한다면 gainminus6이 사라질 수 있으며, 이는 음량 변형 negative control로 해석하고 codec 강건성의 근거로 바꾸지 않는다.

codec 도구나 라이브러리가 없으면 해당 조건을 NA로 처리하고 평가 전 manifest에서 제외 여부를 잠근다. 다른 codec으로 조용히 대체하지 않는다. 평가 조건은 최소 clean, noise20, noise10, gainminus6를 반드시 포함한다. 학습 증강에서 이미 본 codec family는 미지 codec으로 부르지 않는다. 구현이 없던 조건을 성능 확인 후 추가하여 주 결론에 포함하지 않는다.

### 11.3 규모와 통제

V에서 pipeline QC를 수행한 뒤 Q 전체를 기본 paired perturbation panel로 사용한다. R21DF 변형 확장은 사전에 정한 클래스·공격별 최대 1000발화의 고정 subset에서만 수행하고 subset ID를 잠근다. clean 성능도 같은 subset으로 다시 계산하여 full R21DF 수치와 혼합하지 않는다. GPU 비용이 허용되면 전체 평가 확장을 별도 run으로 실행한다. 변형 수만큼 독립 발화 수가 늘었다고 세지 않는다.

가능한 metadata에 한해 speaker, duration 사분위, padding fraction 구간 [0,0.25), [0.25,0.5), [0.5,1]별로 효과를 보고한다. 구간은 T와 규칙에서 정하고 특정 평가 구간을 성능 때문에 제외하지 않는다. 성별·언어가 공식 metadata에 없으면 임의 추정 모델로 채우지 않는다. 화자와 클래스가 분리된 데이터에서는 within-speaker class 비교가 불가능하다는 점을 명시한다.

### 11.4 지표와 판정

모델별 조건 EER, V-fixed threshold의 bonafide 오거부율 및 spoof 통과율, baseline 대비 paired delta를 보고한다. score shift는 먼저 각 모델 V score의 robust scale로 정규화하여 서로 다른 score 범위를 구분한다. 음성 내용이 같은 paired 표본의 u와 gate 변화도 기록하되 그 변화가 무조건 작아야 한다고 요구하지 않는다. 필요한 것은 harmless 변형에서 탐지 결과가 무너지지 않는 것이다.

모든 모델에 같은 source-family bootstrap을 적용하고 한 bootstrap draw에서는 clean과 모든 변형을 함께 선택한다. 모델×조건 세부 비교는 탐색으로, 사전 잠근 noise20과 mulaw8의 baseline 대비 악화량은 주요 robustness 요약으로 보고한다. codec가 NA이면 noise20과 gainminus6 요약을 사용한다는 fallback도 사전 잠근다.

출력: `F/condition_manifest.parquet`, `F/transform_versions.json`, `F/transform_qc.csv`, `F/paired_scores.parquet`, `F/robustness.csv`, `F/subgroup_results.csv`, `F/decision.json`.

## 12 실험 G 학습 방식과 추론 점수의 효과

### 12.1 기본 방향 손실

주 후보는 채널별 가중 metric을 학습하는 `L_dir`이다. 기존 OC의 코사인 s를 아래 S_A로 대체하고 나머지 baseline loss reduction, class weight, 두 단계 centroid 갱신을 보존한다.

\[
a_j=\exp(2\tanh(v_j)),\quad
A=d\,\operatorname{diag}(a)/\sum_j a_j,\quad v_j=0,
\quad S_A=1-\tfrac12(\hat z-\hat c)^TA(\hat z-\hat c).
\]

\[
L_{dir}=\operatorname{WeightedMean은 baseline의 sample reduction과 class weight를 그대로 따른다. BPL의 계수와 구현도 유지한다. class mean의 단순 합으로 바꾸면 loss scale이 바뀌므로 주 비교에서는 변경하지 않는다. v의 weight decay는 0으로 하고 명시한 metric regularizer만 사용한다. 기존 가중치의 weight decay는 유지한다. 정규화 계수 0.01과 tanh 폭 2는 주 설계로 고정한다.

A=I이면 S_A=cosine이므로 baseline의 OC항으로 돌아온다. trace d로 전체 scale 확대를 제한한다. 이것은 분산 추정형 Mahalanobis가 아니라 교사학습으로 학습하는 대각 metric이다. raw norm을 새 손실에 섞지 않는다. S_A는 반드시 [-1,1]에 들어가지는 않으므로 clip하지 않고 native threshold를 V에서 정한다.

이 후보는 방향별 민감도를 바꾸지만 독립적인 spoof 방향을 하나 만드는 것은 아니다. H4의 비대칭 또는 여러 모드 가설과 구분하여 결론을 쓴다. 이 정의만으로 논문 신규성을 주장하지 않는다.

### 12.2 주 비교 matrix

| ID | 변환 | OC항 | BPL | 역할 |
| --- | --- | --- | --- | --- |
| G0 | identity | baseline | 기존 local | 기준 |
| G1 | static gate | baseline | 기존 local | B1과 같은 run |
| G2 | dynamic gate | baseline | 기존 local | B2와 같은 run |
| Gm | residual MLP | baseline | 기존 local | B3과 같은 run |
| G3 | identity | L_dir | 기존 local | 손실만 변경 |
| G4 | dynamic gate | L_dir | 기존 local | 결합 효과 |

모든 비교군은 같은 backbone 초기 상태에서 학습한다. G0 학습 checkpoint에서 G2만 추가 학습하는 비교는 주 matrix가 아니다. frontend는 같은 사전학습 상태를 고정한다. 추출, 검증, 저장 로직의 개선은 모든 군에 똑같이 적용한다.

### 12.3 gradient와 optimizer 계약

classifier 단계에서는 centroid만 고정하고 classifier, gate, metric을 갱신한다. centroid 단계에서는 나머지를 모두 고정하고 bonafide loss로 centroid만 갱신한다. spoof만 입력한 centroid gradient는 0이어야 한다. optimizer 갱신 전의 낡은 graph를 재사용하지 않으며 forward 횟수와 loss 결합은 baseline을 따른다.

BPL은 두 GPU 각각의 local 정상 쌍 평균이며 global gather를 추가하지 않는다. metric regularizer는 classifier 단계에 한 번만 넣고 BPL도 baseline과 같은 횟수로만 넣는다. 각 rank에 같은 정규화 항을 넣은 뒤 GPU 수를 추가로 곱하거나 나누지 않는다. loss reduction과 class weight의 DDP 기준 일치 여부를 합성 입력으로 검증한다.

backbone 초기 state를 seed마다 한 번 생성해 전 비교군에 복제한다. 새 모듈 초기화의 RNG를 분리한다. 데이터 및 증강은 epoch, utterance, view의 key로 맞추고 실제 해시를 기록한다. graph가 달라 전체 학습의 모든 난수가 같다고 보장하지는 않는다. 한 군만 OOM이라고 batch를 줄이지 않으며 변경 시 전 군의 비교 조건을 재설정한다.

### 12.4 학습과 score 효과의 분리

각 encoder에서 cosine score를 계산한다. 각 encoder의 T 특징과 learned c를 고정한 채 같은 L_dir 목적과 정규화로 posthoc diagonal metric도 학습한다. G0에 fit한 metric을 G0-post라고 부른다. Adam lr 1e-3, batch 256, 최대 100 epoch, V EER patience 10, seed 101~103을 쓴다. V EER 최소 epoch를 선택하고 동점이면 이른 epoch를 고른다. probe seed별 score를 ensemble하지 않고 개별 결과와 평균 지표를 보고한다. posthoc의 주 지표는 EER이며 미정의 확률 calibration을 추가하지 않는다.

| encoder | cosine | 같은 절차로 fit한 posthoc metric | joint training native metric |
| --- | --- | --- | --- |
| G0 | 기준 | G0-post | I이므로 cosine과 같음 |
| G3 | 표현학습 비교 | G3-post | G3-native |

G3-native 대 G0-cosine만으로는 학습 목적과 추론 점수 변경의 효과를 분리할 수 없다. G3-native 대 G0-post, G3-cosine 대 G0-cosine, G3-post 대 G0-post를 함께 해석한다. covariance 기반 E3과 supervised posthoc은 다른 목적이다.

### 12.5 학습 후 메커니즘 확인

같은 A/C/E 분석을 G0/G2/G3에 적용한다. 서로 다른 encoder의 동일 채널 번호가 같은 의미라고 가정하지 않고, 각 T에서 동일 절차로 fit한 분석의 독립 평가 성능을 비교한다. gate 앞뒤의 표현을 함께 저장한다. 같은 표현의 결정적 변환 후 probe가 좋아졌다고 정보이론적 정보량이 증가했다고 주장하지 않는다.

A의 trace와 고유값, gate 포화율, 클래스별 score, loss 항목과 V 지표를 기록한다. zero norm, NaN, 발산은 실패로 남기고 같은 run ID에 다른 설정을 덮어쓰지 않는다.

결합 효과는 각 seed에서 `I=EER_G2+EER_G3-EER_G0-EER_G4`로 계산한다. 양수이면 EER 차이의 가법 척도에서 상호작용이 있으나 통계 오차와 척도 의존성이 있다. G4가 가장 낮다는 사실만으로 상호작용을 주장하지 않으며 불리한 결과도 보고한다.

### 12.6 방향 구조에 따른 확장

full metric 확장은 `A0=diag(exp(2tanh(v)))+UU^T`, `A=d A0/tr(A0)`, rank 8로 정의한다. U는 고정 전용 RNG의 N(0, 1e-6), 즉 표준편차 0.001로 초기화한다. U=0이면 UU^T의 U-gradient도 0이므로 사용하지 않는다. 초기 A는 I와 정확히 같지 않으며 별도 G3r로 관리한다.

비대칭 prototype 후보 G3p는 unit 정상 중심 c와 K=[1,2,4]개의 unit spoof prototype p를 둔다. b=cos(z,c), f=max_k cos(z,p_k), S_p=b-f다. y=1에서는 softplus(20(0.2-S_p)), y=0에서는 softplus(20(0.2+S_p))에 기존 BPL을 더한다. classifier 단계에는 c를 고정하고 p와 classifier를, centroid 단계에는 p와 classifier를 고정하고 bonafide로 c만 갱신한다. p는 -c가 아니라 독립 초기화한다.

G3p는 비등방 거리와 별개 가설이다. 알려진 spoof를 외울 위험을 F와 완전 공격 제외 학습으로 검토한다. 주 확장에는 diversity regularizer를 넣지 않고 collapse와 실효 중심 수를 기록한다. K는 V에서만 선택한다. 주 후보로 승격하려면 R을 보기 전에 protocol과 다중 비교 family를 다시 잠그며 R 결과로 유리한 손실을 골라 바꾸지 않는다.

### 12.7 결과와 판정

주 비교 5개는 5.2절을 따른다. H7은 통제된 재학습 개선과 H6의 일반화 결과를 함께 평가한다. G3-native 대 G0-post가 불명확하면 posthoc 점수 변경만으로도 설명할 가능성을 남긴다. 고정 encoder probe 결과로 joint training 효과나 연구 신규성을 주장하지 않는다.

출력: `G/run_registry.json`, `G/epoch_metrics.csv`, `G/loss_components.csv`, `G/native_scores.parquet`, `G/posthoc_scores.parquet`, `G/seed_effects.csv`, `G/interaction.csv`, `G/decision.json`. checkpoint에는 classifier, gate, metric, centroid, frontend 참조와 optimizer state를 빠짐없이 저장한다. test 최고값만 저장하는 방식은 쓰지 않는다.

## 13 실행 수와 단계 판정

| 단계 | 내용 | 완전 학습 run 수 |
| --- | --- | --- |
| P0 | 기존 checkpoint 추출과 QC | 0 |
| A/C/D/E | 고정 encoder 진단과 작은 probe | 0, probe fit 수 별도 집계 |
| B/G screening | G0/G1/G2/Gm/G3/G4 각각 seed 1,2,3 | 18 |
| 주 효과 확인 | G0/G1/G2/Gm/G3에 seed 4,5 추가 | 10 추가, 합계 28 |
| 결합 효과 확인 | G4에도 seed 4,5 추가 | 2 추가, 합계 30 |
| G3r/G3p | full metric와 prototype 확장 | 설정과 seed에 따라 추가 |
| 완전 공격 제외 | encoder부터 공격 family 제외 | 공격 수 × 비교군 수 × seed 수 추가 |

과거 seed 1 run의 코드와 데이터 순서 등 비교 조건이 일치하지 않으면 G0 run으로 세지 않는다. GPU 측정 전 소요 시간을 단정하지 않는다. 추출도 frontend를 통과하므로 비용이 있으며 구현 단계 P0의 소수 표본 측정으로 시간과 GPU-hour를 추정한다.

P0 다음에는 C와 A를 먼저 수행한다. C가 불명확하면 h의 보조 probe를 조사한다. h에는 z의 learned centroid를 그대로 적용하지 않는다. (s_z, norm_z) 대 (s_z, norm_z, h)의 probe로 LayerNorm 이전 정보 손실 가능성을 확인한다. h의 좋은 결과가 z의 잔여 방향 가설을 지지하는 것은 아니다.

A가 지지되지만 B의 입력 의존성이 지지되지 않으면 static gate로 충분하다는 결론도 채택한다. 확인 학습의 진입과 중단 이유를 decision.json에 기록하고 불리한 seed를 제거하지 않는다. 진단이 음성이어도 새 표현학습이 불가능하다고 결론 내리지 않는다.

이번 의뢰는 명세 작성이다. 위 실행 수는 학습 시작 명령이 아니다. 구현 시 기존 실험 보호, 빈 결과 경로, 여유 용량, split manifest, checkpoint 대응, protocol lock을 확인한다.

## 14 구현 인터페이스와 산출물 계약

runner는 아직 구현하지 않았다. 다음 이름은 구현할 책임을 정의하며 이미 존재하는 명령으로 실행하지 않는다.

| 기능 | 입력 | 출력 | fit 허용 범위 |
| --- | --- | --- | --- |
| audit_inputs | checkpoint, protocol, 환경 | audit.json, split manifest | fit 없음 |
| extract_embeddings | audit 성공, checkpoint, crop 규칙 | P0 배열과 manifest | fit 없음 |
| fit_channel_rank | T 특징과 label | rank와 mask | T |
| fit_probes | T/V 특징과 grid | probe, scaler, registry | T fit, V 선택 |
| fit_metrics | T 정상 특징, V label, c | A, gamma, score 규칙 | T fit, V 선택 |
| fit_prototypes | T spoof 방향, V 특징 | prototype, K, probe | T fit, V 선택 |
| train_variant | G ID, seed, locked config | epoch 50 model과 이력 | T 학습, V 모니터링 |
| evaluate_locked | frozen registry, Q/R/U | 발화별 score와 통계 | fit 없음 |
| analyze_robustness | frozen model, paired conditions | F 결과 | fit 없음 |
| aggregate_results | score, dependency metadata | 효과량, CI, decision | fit 없음 |

resume은 run UUID, config hash, split hash, code hash, checkpoint hash가 같을 때만 허용한다. 완료 run을 새 설정으로 덮어쓰지 않는다. stdout, exit status, 시작/종료 KST, GPU 할당, software version을 기록한다. 임시 파일에서 atomic rename하고 마지막에 complete marker를 남긴다. 부분 score 파일을 완료 처리하지 않는다.

`decision.json` 필수 필드는 hypothesis, experiment, estimand, baseline, candidate, split, effect, units, CI, CI_level, independent_group_count, seed_count, multiple_comparison_family, support_coverage, protocol_deviations, status다. status는 supported, unsupported, inconclusive, not_run, invalid 중 하나다. unsupported는 효과가 정확히 0임을 증명한 뜻이 아니다.

연구실 데이터가 들어가는 산출물은 `/home/mentee/analysis/embedding_geometry/<run_uuid>/`에 저장하고 공간에 맞게 chunk화한다. 미생성 산출물은 not_run이다. 로컬에는 필요할 때 발화 ID를 포함하지 않는 집계만 보관한다.

## 15 검수 항목

| 검수 | 합격 조건 |
| --- | --- |
| checkpoint | frontend/classifier/centroid 대응, 모든 key 일치, hash 기록 |
| 데이터 | source-family 누수 없음, V/Q 분리, Q/R fit 금지, label 방향 명시 |
| 추출 | native score 일치, 발화별 seed, crop 일치, rank 중복 없음 |
| 수학 | tangent 항등식, A=I 동등성, PSD, trace, zero norm 처리 |
| gate | identity 초기화, 두 branch 공유, 파라미터 수, MLP 대조 |
| optimizer | 단계별 gradient, BPL 계수와 범위 유지 |
| 통계 | paired group bootstrap, seed 차이, pp 단위, 다중 비교 명시 |
| 조건부 분석 | finite bin 한계, valid율, overlap coverage, null 진단 범위 |
| 일반화 | encoder 공격 노출 기록, 재사용 benchmark와 새 U의 구분 |
| 보고 | 모든 seed와 실패 run, rescue/harm, guardrail, NA와 미실행 구분 |

## 16 근거와 적용 범위

[1] [Gretton 등의 A Kernel Two Sample Test](https://www.jmlr.org/beta/papers/v13/gretton12a.html)는 MMD 분포 비교의 근거다. 유한 score bin을 사용한 이 계획의 조건부 검정 타당성을 자동으로 보장하지 않는다.

[2] [Lopez Paz 등의 Revisiting Classifier Two Sample Tests](https://arxiv.org/abs/1610.06545)는 held-out classifier를 통한 분포 비교의 방법론이다. 이 계획의 nested probe와 일반화 조건은 별도로 설계했다.

[3] [Wu 등의 The Impact of Data Dependence on Speaker Recognition Evaluation](https://pmc.ncbi.nlm.nih.gov/articles/PMC5484007/)는 음성 평가의 의존성과 재표본화에 주의해야 하는 근거다. group 정의와 실용 효과 기준은 이번 설계의 선택이다.

[4] [UMAP 공식 문서 Using UMAP for Clustering](https://umap-learn.readthedocs.io/en/latest/clustering.html)는 밀도 보존과 겉보기 군집 분단의 한계를 설명한다.

[5] [EBM 공식 loss 구현](https://github.com/kimho1wq/EBM/blob/main/model/loss.py)은 공개판 정규화, BPL 전체 쌍 비교, label 방향의 참고다. 서버 구현 차이는 운영 기록과 P0 hash로 관리한다.

[6] `../../MENTEE_SERVER.md` (원자료 경로; 배포 제외 또는 미확보, 출처 목록 참조)은 설정, 두 단계 갱신, 저장 주의사항의 근거다. 과거 진행률은 현재 상태를 보장하지 않는다.
