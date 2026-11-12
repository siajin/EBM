> 표시 명칭·링크 이름은 설명형으로 정리했다. 원래 코드·경로·수치는 그대로 보존한다.

> 역사 문서 보존본. 원문 작성 당시의 상태·계획을 포함한다. 최신 상태와 확정 결과는 experience의 각 실험 README를 우선한다.

# 진행 중인 EBM 동결 실험의 조건 점검

점검: 2026-11-05 18:15~18:35 KST. 대상은 현재 실행 중인 F/P/R 코드,저장된 분할·선택·scaler 기록,서버의 모델·소스·환경·QC다. **판정은 조건부 사용 가능이다. DF 내부 CV의 출처×클래스 커버리지에는 중요한 제한이 확인됐다.** 진행 중인 실험을 중단·재시작하거나 모델·hyperparameter를 바꾸지 않았다.

**2026-11-06 사용자 결정:** DF 4×3 보완 실험은 진행하지 않으며 이번 실행 범위에서 제외한다. 후보 분할만 보존하고 학습은 실행하지 않았다. 기존5×3의 출처×클래스 커버리지 제한은 그대로 명시한다. 이 아래의 점검 시각과 미실행 표시는 당시 기록이다.

## 1. 가장 중요한 발견: DF 5-fold는 모든 출처×클래스를 포함할 수 없다

DF_pilot의 실제 독립 화자 그룹 수는 다음과 같다.

| 출처 | 정상 발화/그룹 | spoof 발화/그룹 |
| --- | ---: | ---: |
| asvspoof | 1,038 /66 | 2,611 /48 |
| vcc2018 | 523 /12 | 6,374 /4 |
| vcc2020 | 439 /14 | 9,015 /10 |

**vcc2018 spoof가4그룹뿐이므로,화자를 겹치지 않게 나눈5개 outer-test fold에 이 조건을 모두 넣는 것은 불가능하다.** source×class stratification은 이 물리적 제약을 해결하지 못한다. 현재 코드가 이 joint-cell의 부족을 명시적으로 경고하지 않은 점은 보완이 필요했다. 실행 전에 확인했어야 하는 조건이다.

현재 DF 분할에서 확인한 누락은 다음과 같다. fold 번호는0부터 시작한다.

| 위치 | 빠진 출처×클래스 |
| --- | --- |
| outer0의 inner2 validation | vcc2018 spoof |
| outer1의 inner2 validation | vcc2018 spoof,vcc2020 spoof |
| outer2 test | vcc2018 spoof |
| outer3 test | vcc2018 spoof,vcc2020 spoof |
| outer4의 inner0 validation | vcc2018 spoof |
| outer4의 inner1 train | vcc2018 spoof |
| outer4의 inner2 validation | vcc2018 spoof |

모든 분할에는 정상·spoof 두 클래스와 각 출처 자체는 존재했다. 그러나 출처와 클래스를 교차하면 위 조건이 빠진다. 같은 fold를 모든 비교군에 재사용하므로 **현재 grouped OOF 비교 자체가 화자 누수로 무효인 것은 아니다.** 다만 “모든 fold가 동일한 출처·클래스 조건을 포함하는 검증”이나 “각 출처에서 반복 확인한 일반화”로 해석할 수 없다. 특히 source-specific 주장과 inner 선택의 도메인 구성에 주의해야 한다. 21LA의 source×class 두 cell은 현재 모든 분할에 존재했다.

이 제한은 P3-4.0375초 상한 비반복 학습과 그 fold를 재사용하는 P4에 적용된다. T/V 전이인 P1/P2/P3-A,기존 native 평가,F1 점수와 F3 거리의 계산 오류를 의미하지는 않는다. 기존 DF 내부 진단 역시5-fold라는 사실만으로 모든 출처×class의 완전 커버리지가 보장되지 않는다.

### 보완안은 준비했고 새 학습은 시작하지 않았다

DF에 대해4outer×3inner 분할 후보를 **metadata만으로** 만들었다. outer seed는13,inner seed는각outer순서로14/12/18/17이다. 사전에 정한 순서에서 source×class 커버리지를 만족하는 첫 분할을 택했으며,점수·임베딩·성능을 읽어 seed를 고르지 않았다.16개 split pair의 양쪽,즉32개 train/test/validation 부분에서6개 source×class cell이 모두 존재하고 화자가 겹치지 않음을 확인했다.

이 후보는 새 프로토콜 버전으로 사용할 수 있는 분할안이다. **현재5×3 실험을 교체하거나 새4×3 학습을 실행한 것은 아니다.** 새 분할을 사용하면 작은 probe는 다시 fit해야 하며 기존 EBM/음성 특징은 재추출할 필요가 없다. 또한 source당4개 spoof 그룹이라는 근본적인 적은 독립 표본 수가 사라지는 것은 아니다. 같은 이미 관찰한 자료의 보완 검증이며 새로운 blind 데이터 검증으로 바뀌는 것도 아니다.

근거: [현재 분할 및 완료 fit 점검](../../artifacts/geometry_cache/local_analysis/frozen_all_v1/condition_audit_20261105_v2/verification.json),[4×3 후보 설정](../../artifacts/geometry_cache/local_analysis/frozen_all_v1/CV4_coverage_candidate_v1/candidate_protocol.json),[후보의 metadata 검수](../../artifacts/geometry_cache/local_analysis/frozen_all_v1/CV4_coverage_candidate_v1/verification.json).

## 2. 누수·선택·공정성에서 확인한 부분

- T/V/DF/21LA 사이의 정확한 UID 중복을 검사했다. 각 내부CV의 train/test와 inner train/validation 사이에는 동일 화자 그룹이 없었고,모든 UID는 하나의 outer-test에만 속했다.
- DF/21LA의 기본 평가 수는 각각정상2,000/spoof18,000이다. 같은 seed·fold에서 모든 비교군이 같은 분할과 deterministic class sampler를 재사용한다.
- 점검 시점에 완료된569개 fit의 UID signature와 hyperparameter 기록,80개 outer 선택의 refit epoch와 inner 선택 기록을 확인했다. norm scaler34개를 표본 점검해 실제 train 부분의 평균/std와 대조했다. 아직 시작하지 않은 모델의 실제 scaler까지 검증했다고 주장하지 않는다.
- 전이 head는 T에서만 fit하고 V에서만 선택했다. 원래 EBM과 centroid는 고정됐고,후보를 DF/21LA 성능으로 다시 고르지 않았다.
- 모든 full batch는정상16/spoof144이며 weighted OC는 두 class loss의50:50 평균과 수치적으로 같았다. T의 전체 UID를 매 epoch 최소한 한 번 보며 끝부분은 class별 순환으로 채운다.
- static은160파라미터,dynamic과 잔차MLP는각5,296파라미터다. identity 초기화,변환L2 상한0.5,metric의양의대각·trace160·condition ratio상한을 확인했다.
- 실행한 head/probe/N0/common 소스는 시작 시 보존한 snapshot과 같았다. 모델 source를 실행 도중 수정하지 않았다.

이 점검은 known speaker grouping에 대한 것이다. 동일 원음의 합성·재인코딩 family가 모두 연결됐다는 증명이 아니며,DF와21LA가67화자 그룹을 공유한다는 제한도 남는다.

## 3. 조건은 지켰지만 부정 결과의 해석을 제한하는 설계

### 3.1 V가 매우 쉽다

V의 정상 최소 cosine은 spoof 최대 cosine보다 높고 native EER는0%다. covariance 후보도 V EER가 모두0%여서 큰gamma 우선 규칙으로 identity가 선택됐다. P1/P2는 고정 T 확률 링크에 대한 V LL로 선택하며 static/metric은 epoch0이 선택됐다.

규칙 위반은 아니다. 그러나 **분별력이 약한 V에서 identity가 선택된 결과를 attention/metric 전체의 무용성으로 해석할 수 없다.** 새로운 점수의 calibration과 ranking 개선도 동일한 질문이 아니다.

### 3.2 clean T에서 원래 OC의 추가 학습 여지가 작다

저장된 clean long-view T 표현에서 class-balanced OC는0.067425584다. 정상 cosine의 상한1과r_real0.9,alpha20으로 정해지는 손실 하한의 infimum은약0.063464006이다. 이 하한에 매우 가까우며,spoof class의 OC 항은 전체 class-balanced OC의약2.20%를 차지했다. T spoof22,800개 중원래fake margin0.2보다높은것은5개였다.

이는 입력별 gate가 수치적으로 학습되지 않았다는 뜻은 아니다. 실제 gradient와 optimizer step은 있었다. 다만 이미 학습된 깨끗한 T 표현에 **같은 OC**를 다시 적용하므로,미지 도메인의 잔여 방위를 새로 조직하라는 신호가 강하지 않다. 이 구조는 사용자가 제안한 새 방향 loss를 직접 시험한 것이 아니다. 원래 BPL·학습 증강·long/short 쌍을 재현한 공동 학습도 아니다.

### 3.3 현재 결과로 말할 수 있는 범위

현재 P1/P2의 결과는 bounded 후처리와 고정된 fit/선택 조건에 한정된다. 기존 DF 내부 방향 이득과 T/V 전이의 EER/LL 결과도 분리한다. source×class 커버리지가 빠진 현재 DF CV는 그 분할에 따른 정보 접근성 진단으로 남기고,조건별 재현을 강하게 주장하려면 보완 검증이 필요하다.

## 4. 기록·보고에서 실제 보완한 사항

### 4.1 실행 순서 기록 정정

완료 JSON의 `all_predictions_frozen_before_external_expansion=True`는 특징 추출까지 확장으로 읽으면 실제 순서와 맞지 않았다. Q 특징 추출 시작은epoch1793865644.70,마지막 head 상태 저장은1793865812.91로 병행됐다. Q에 고정 head를 적용한 평가 완료는1793866821.46이다.

원래 JSON을 별도로 보존하고 이 모호한 필드를 제거했다. 대신 **T/V만 fit/선택**,**특징 추출은 병행**,**R1 head 평가 전에 head 고정**,**확장 결과로 head 선택 안 함**을 명시했다. 학습된 상태·예측·수치 결과는 바꾸지 않았다.

### 4.2 P3-A의 V 운용점 보고 보충

P3-A의 원래 metric CSV에는 LL/EER/AUROC는 있었지만 공통V 고정 오류율이 없었다. 저장된 선택 모델·scaler로 V threshold를 계산하고 이미 저장된 평가 logit에 적용해 보충했다. fit·epoch/ridge 선택·음성 추론은 다시 하지 않았다. 이 보충값으로 모델을 다시 선택하지 않는다.

예를 들어선형 direction의seed평균에서DF 정상거부는38.95%,spoof통과는0.012963%였다. native는39.00%/0.011111%다.21LA는선형direction이11.3833%/0.016667%,native가12.40%/0.016667%였다. 이는 특정 V 고정 운용점의 결과이며 EER 개선과 동일시하지 않는다.

근거: [보충 운용점 표](../../artifacts/geometry_cache/local_analysis/frozen_all_v1/condition_audit_20261105_v2/P3_transfer_V_operating_points.csv),[정정 전 완료 기록](../../artifacts/geometry_cache/local_analysis/frozen_all_v1/condition_audit_20261105_v2/heads_complete_before_provenance_correction.json).

## 5. 서버와 아직 실행 전인 조건

현재 worker는 기존 컨테이너와 같은 mount namespace에 있고,실제 환경은 `CUDA_VISIBLE_DEVICES=0,1`이다. checkpoint·archived source28개·현재 manifest·추출기 hash가 준비 시점과 같았다. Q/19LA/DF QC128개의 h/z/score 차이는모두0,반복/GPU 간차이도0이었다. 현재 DF 전체 추출은 아직 최종 검수를 마친 것이 아니다.

두 GPU에서 SW Thermal Slowdown이 활성화돼 있고 온도는약87~88°원음 전체 비반복 학습였다. GPU6의낮은clock이 처리 시간을 늘린다. 현재 확인한 QC의 수치 불일치와는 구분하며,이 상태로 전체 추출의 최종 성공을 미리 보장하지 않는다.

R2/R3는 아직 prepare가 시작되지 않았다. 업로드된5개 UID파일의 hash와 sidecar 소스가 로컬 준비본과 일치했다. layer/time uniform조건,같은UID RNG,같은base noise,Opus32k 재압축 및 native 복제 검사는 코드에 준비돼 있다. **V 전처리QC·실제개입의native대응·모델state전후hash는 실행 후에 통과 여부를 판단해야 한다.**

R3의20/10dB는 원래 전체 발화의 RMS 기준이다. 모델에 들어가는 첫64,600sample 또는wrap-padding 구간의 실제SNR이 정확히그값이라는 보장은 없다. 두noise조건의base 파형과배율관계는같다. 이정의를명확히기록하며,이번점검으로원래입력변형정의를바꾸지않았다.

## 6. 검토 범위와 남은 판단

아래 분모는 이번 점검의 조건 영역 수이며,전체 실험 성공률이나 통계적 정확도 비율이 아니다.0/N은 남은 관찰 오류가 없다는 뜻이지 미래 실행까지 전부 검증했다는 뜻이 아니다.

### 실험 비교 설계와 활용성

| 범주 | 남은 관찰 문제 | 평가 |
| --- | --- | --- |
| 질문·비교의 완결성 | 1 /4 | DF source×class 완전커버리지를 위한 보완 검증이 남음;현재5×3은 제한된진단 |
| 해석의 명확성 | 0 /4 | 내부/전이,LL/EER,원래모델/작은probe,완료/가설지지를 구분 |
| 시각·상호작용 | N/A | 현재 실행 조건 검토이며 새시각화나UI변경 없음 |

### 정확성과 재현성

| 범주 | 남은 관찰 문제 | 평가 |
| --- | --- | --- |
| 원본·설정의 일치 | 0 /4 | cache/model/source/manifest hash 확인;R2/R3 실제QC는미실행 |
| 수식·가중치·용량 | 0 /4 | sampler·weightedOC·identity·bounded gate/metric 확인 |
| 지표와 비교의 일치 | 0 /3 | seed평균·고정V운용점·개별/다중CI를 구분;P3운용점 보충 |
| 실행·분석 기록 | 0 /3 | 실행순서 metadata정정,원본보존;미래finalizer에도조건제한을전달 |
| 데이터 분할 품질 | 1 /3 | 화자누수는미관찰;DF joint-cell 누락은남고21LA는해당누락없음 |
| 가설 지지의 범위 | 0 /4 | V의선택력·cleanT OC포화·동결모듈범위를명시;새loss학습효과미판정 |

수치 검사는2,101개 조건을 통과했지만 위 커버리지 한계가 해결된 것은 아니다. source×class cell 누락7개 partition을 별도로 남겼다. 결과 사용 준비도는 **제한을 붙인 탐색 분석**이며,모든 조건을 갖춘 확증 실험이라는 승인은 아니다. 4×3안의실제재학습과R2/R3의런타임검수는이번점검범위밖이다.
