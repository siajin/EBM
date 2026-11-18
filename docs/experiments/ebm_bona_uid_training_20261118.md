# 정상 BPL만 같은 UID로 변경하는 비교 실험

사용자 최종 요청은 정상 BPL의 all-cross 평균만 같은 UID의 long·short 평균으로 바꾸고 나머지 학습 조건을 유지하는 것이다. **학습 중에는5epoch마다19LA만 평가한다.** 학습 완료 후21LA·DF는 한 배치의 XLSR 출력을10개 저장 checkpoint에 함께 넣어 평가한다. 현재는 구현·검증 중이며 학습 시작과 구분한다.

## 목적식과 비교 조건

```text
L_backend = L_OC_original
          + mean_{spoof i,j} softplus(30*(1-cos(zL_i,zS_j)))-0.3
          + mean_{bona i}    softplus(30*(1-cos(zL_i,zS_i)))-0.3
```

정상 계수1·각 클래스 별도mean, 스푸프 pair1.0/pull0.9/push0.2, OC alpha20, BPL alpha30, 중심 단계 계수5는 유지한다. 기울기 투영·norm cap·Adam gate는 추가하지 않는다. backend160차원·XLSR 고정·원래 두 단계 Adam도 유지한다. Role v4에 정상 pairing을 기록하고 기존 v1/v2/v3의 metadata와 저장 모델 읽기는 보존한다.

## 배치와 학습 경로 검산

이전 정상 all-cross 실험의 보존 source payload와 현재 `role_training.py`, 원래 TrainSet/loader·train·model/loss를 byte 단위로 대조해 동일함을 확인했다. `role_runtime.build_trainer` AST도 동일하다. 옵션은 정상 역할 metadata와 평가 주기만 다르며 비목적식 학습 옵션은 모두 같다. 근거는 `artifacts/bona_uid_train_20261118/batch_contract.json`이다.

두 GPU 합계240, rank당 통상120=정상12+스푸프108이다. epoch당108논리 배치, backend108회·중심108회 갱신이다. rank별 정상1290개·스푸프11616개 방문이며, 원래 sampler 순환으로 스푸프216개가 반복되는 계약도 그대로다.

| 배치 | 정상 | 스푸프 | 전체 행 | 정상 쌍 | 스푸프 쌍 |
|---|---:|---:|---:|---:|---:|
| 일반 | 12 | 108 | 120 | 12 | 11664 |
| 106번째 | 12 | 60 | 72 | 12 | 3600 |
| 108번째 | 6 | 108 | 114 | 6 | 11664 |

long64600/front/wrap·short48440/random crop/wrap·독립 RawBoost·codec off, worker4, seed1, DistributedSampler와 기존 bounded cycle의 순서·난수 소비를 유지한다. 같은 UID 정렬은 같은 TrainSet item에서 함께 반환된 long/short 행을 같은 class mask로 선택해서 보장한다. 두 view를 서로 다른 순서로 섞지 않는다.

평가는 별도 head를 읽고 optimizer를 호출하지 않는다. 평가 전후 Python·NumPy·Torch·CUDA RNG를 복원하며, 학습 classifier/Adam 상태를 덮어쓰지 않는다. 추가 평가를 넣어도 다음 학습 step의 파라미터·Adam state가 평가를 생략한 경우와 같은지 CPU 회귀로 검사한다. 실제 GPU smoke에서도 원래108loader를 모두 읽고1·106·108번째 batch 갱신, 정상 short gradient, 중심 동기화 및 v4 저장·복원을 확인한 뒤 본학습에 진입한다.

## 19LA 중간 평가와 학습 후 특징 공유

5·10·…·50epoch의10개 checkpoint를 먼저 원자적으로 저장·검산하고, 각 시점에19LA 전체만 평가한다.21LA·DF는 학습 완료 전에는 평가하지 않는다. 기존 wrap64600/no-mask/batch8·matched_wrap seed1729·ASP1e−5·고정 frontend revision 조건을 유지한다. 평가 호출의 step/rank RNG도 바꾸지 않는다. 학습 후21LA·DF는 각 배치에서 XLSR을 한 번 호출하고 동일 hidden tensor를10개 저장 head에 공유한다.19LA의 완료 결과는 보존하며 전체 재평가하지 않는다.

재사용 가능한 것은 고정 XLSR의25개 layer 출력이며 학습되는 backend embedding은 checkpoint마다 다시 계산해야 한다. 입력64600의 출력 길이201·hidden1024·FP32 기준 음성당20,582,400byte, 세 집합753341음성 합계약15.51TB다. 현재 작업 디스크 여유는약157.7GB, 별도 데이터 디스크는약398.5GB로 전체 cache에는 부족하다. FP16으로 바꿔도 약7.75TB이며 이전 FP32 평가와 수치가 달라질 수 있어 자동 적용하지 않는다.

실제 기존 파일도 조사했다. F:의 `IRLab/XLSR/19LA_train_last_20261105_01`에는 완료된19LA train 마지막 레이어 출력 `[25380,201,1024]` FP32가 있다. 서버의 h/z는 과거 backend의160차원 표현이고 mild cache는 파형이다. 현재 필요한 평가 집합의25레이어 출력과 호환되는 전체 cache는 확인한 프로젝트 경로에서 찾지 못했다. 이를 대신 넣어 모델 구조·augmentation을 바꾸지 않는다.

사용자 최종 선택에 따라 전체 특징을 파일로 저장하지 않는다. 학습 완료 후 같은 batch의 frozen XLSR 출력 메모리를10개 head에 공유한 뒤 해제한다. 학습 중19LA는 각 checkpoint에서 계산한다. 학습 후 공유평가의 smoke는 세 집합에서 저장 점수·별도 frontend 계산과 동등성을 확인한 뒤21LA/DF 전체로 진행한다. 학습 전 smoke의 평가도19LA에 한정한다.

## 수학 및 성능 검토

정상 mean의 분모는 n, 스푸프 mean의 분모는 n²다. 정상 계수를1/n로 다시 줄이거나 pair 수 감소를 보상한다며 n배 올리지 않는다. 서로 다른 클래스의 positive pair는 만들지 않는다. 단위 스푸프=c·정상=−c·같은 UID long=short 구성에서 OC와 두 BPL의 하한을 동시에 달성해 제약의 모순이 없음을120/72/114행에서 수치 검산했다. 접선 gradient도 유한하고0에 가깝다. 근거는 `mathematical_review.json`이다.

성능 개선을 보장하지는 않는다. 정상끼리 모으던 제약을 없애면 유용한 클래스 응집까지 약해질 수 있고, 독립 crop/RawBoost 간 정합이 과도한 불변성을 요구할 수도 있다. 정상 OC의 응집 압력이 남아 anti-collapse 보장은 없다. 다만 배치·sampler·증강·optimizer·모델 구조의 변경이 함께 섞이지 않도록 코드·초기 tensor/RNG·부분 배치·CPU 갱신·실제 GPU smoke를 각각 검증한다.

## 실행과 비용

목표는 fresh seed1부터50epoch이며 기존 모델·점수·frozen source는 보존한다. 새 root와 별도 결과명을 사용하고 이미 완료된 진단을 재시작하지 않는다. 기존 할당 GPU6/7만 사용하고 신규 유료 자원·지출 상한0USD, 자동 재시도0, pipeline168h 상한이다. 예약·watcher는 만들지 않는다.

성능 개선이나 기울기 충돌 제거는 가설이다. 같은 epoch·고정50·공통15~50 구간 최소값을 기존240, 스푸프-only, 정상 all-cross와 비교한다. 최소값은 사후 선택이며 한 checkpoint의 동시 성능으로 해석하지 않는다.
