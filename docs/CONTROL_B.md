<a id="b-상한을-유지한-반복-제거-통제실험"></a>

# 4.0375초 상한 비반복 학습: 상한을 유지한 반복 제거 통제실험

2026-11-06 기준 4.0375초 상한 비반복 학습 기본값이다. 전체 학습은 시작하지 않았고, 두 GPU에서 제한된 실제 갱신 검증만 수행했다.

| 항목 | 설정 |
|---|---|
| 이름 | EBM_e50_bs240_lr1e5_B_zeropad_short303 |
| Long | 앞부분 최대64,600sample, 4.0375초 |
| Short | 무작위 연속 구간 최대48,440sample, 3.0275초 |
| 짧은 음성 | 실제 길이 유지, 반복 없음 |
| 배치 정렬 | 물리 microbatch 최장 입력까지만 오른쪽0패딩, frontend와backend mask |
| 손실 | 원래 long OC + 정상 long×short 로컬 교차 BPL |
| 새 손실 | short_oc_weight=0, padding_loss_weight=0 |
| 보조 반복 입력 | 모델뿐 아니라 CPU collate에서도 만들지 않음 |
| 크기 | global240, GPU당120=정상12+spoof108, BPL12×12 |
| 기타 | epoch50, seed1, lr1e-5, micro_batch_size8 |
| 증강 | codec 두 호출 블록 주석, 확률값0.8 불변, ACN/추가 HiFi-GAN off, RawBoost0~8 유지 |
| 평가 기본값 | Long과 같은4.0375초 상한, 짧은 음성 반복 없음 |

설정은 arguments.py, 입력 처리는 data_loaders.py, 마스크는 model/b_linear.py,
분할 재계산은 native_steps.py에 있다. BPL 수식은 원본 그대로이며 GPU 간 embedding gather는 없다.

## 비교 조건

- A: native_length=False, cap_long=True, 두 보조 계수0. 기존 상한과 반복을 사용한다.
- 4.0375초 상한 비반복 학습: native_length=True, cap_long=True, 두 보조 계수0. 현재 기본값이다.
- 원음 전체 비반복 학습: native_length=True, cap_long=False, 두 보조 계수0. Long 전체를 사용하며 이번 대상이 아니다.

동일 구현의 반복 입력·비반복 입력 대조는 같은 현재 코드에서 새 실험 이름으로 각각 학습한다. 데이터, UID 배치 순서,
class 비율, 초기화 seed, codec-off, RawBoost 정책, physical batch와 optimizer를 맞춘다.
단일 seed의 결과를 일반적 개선의 증명으로 해석하지 않는다.

학습 효과의 최종 비교는 각50epoch checkpoint를 모두 같은 4.0375초 상한 비반복 학습 평가 설정으로 수행한다.
A 학습 설정 그대로 평가하면 반복 평가까지 함께 달라진다. target 성능으로 checkpoint나
threshold를 선택하지 않는다. 기존 batch240 결과와의 비교만으로 반복 제거의 단독 효과를
주장하지 않는다. 배치 240 고정 길이 반복 입력 학습의 보관 소스에는 활성 codec 호출이 있지만 당시 실행에서의
실제 적용 여부는 확인하지 못했다.

## 제한 검증

기존 컨테이너와 CUDA_VISIBLE_DEVICES=0,1에서 GPU별105개 검사를 통과했다.
실제25,380개 데이터의 동일 구현의 반복 입력·비반복 입력 대조 sampler UID 순서 동일, GPU당12/108 비율, 길이7사례×RawBoost9종,
codec/wrap 호출 금지, crop 상한, XLS-R mask와 단독/배치 embedding 오차3.73e-6을 확인했다.
실제 global240의 두 단계 갱신에서 기존 OC+BPL만 실행되고 frontend는 동결됐다.
이는 코드 연결과 수치 검증이며 정확도 개선 결과는 아니다.
