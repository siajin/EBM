> 역사 문서 보존본. 원문 작성 당시의 상태·계획을 포함한다. 최신 상태와 확정 결과는 experience의 각 실험 README를 우선한다.

# EBM B 조건: 길이 상한을 유지한 반복 제거 통제실험

2026-11-06 서버 `/home/mentee/code/EBM`에 반영했다. **이번 수정은 `arguments.py`, `data_loaders.py` 두 파일뿐**이다. 기존 마스크·메모리 분할·학습 연결 코드는 유지했다. 전체 학습은 시작하지 않았다.

## 현재 기본값

| 항목 | B |
|---|---|
| 실험명 | `EBM_e50_bs240_lr1e5_B_zeropad_short303` |
| Long | 원음 앞부분 최대64,600sample = 4.0375초 |
| Short | 원음에서 최대48,440sample = 3.0275초의 연속 무작위 crop |
| 상한보다 짧은 음성 | 실제 길이를 유지; 반복하지 않음 |
| 배치 크기를 맞추는 처리 | 물리 microbatch 안에서 오른쪽0패딩; waveform attention mask와 backend 시간 pooling mask 적용 |
| 보조 입력 | 사용하지 않으며 CPU에서도 wrapped 입력을 생성하지 않음 |
| 손실 | 기존 long OC + 정상 long×short의 로컬 교차 BPL |
| 새 보조 손실 | `short_oc_weight=0`, `padding_loss_weight=0` |
| Batch | 전체240; GPU당120 = 정상12 + spoof108; 로컬 BPL은12×12 |
| 기타 | epoch50, seed1, lr1e-5, physical batch8, 기존 OC/BPL 계수 유지 |
| 증강 | 현재 codec 호출 두 블록 주석 유지, 확률값0.8·변환 구현 불변; ACN off·추가 HiFi-GAN off 유지; RawBoost0~8 유지 |
| 현재 dev/eval 입력 | Long과 같은4.0375초 상한, 짧은 입력 반복 없음 |

여기서0패딩은 항상4.0375초/3.0275초까지 늘리는 방식이 아니다. 각 물리 microbatch의 최장 입력까지만 채운다. 마스크는 이 채움 영역을 구분한다. 원래 음성 안의 무음은 제거하지 않는다.

## 두 파일의 변경

- `arguments.py`: B용 새 결과 이름, `cap_long=True` 추가. 두 보조 손실 계수는 이미0이었으며 그대로 유지한다.
- `data_loaders.py`: `native_length=True`일 때도 `cap_long=True`이면 long/train과 평가 입력을 기존 상한까지만 자른다. 짧은 음성을 반복하지 않는다. Loader가 두 보조 손실 계수의 활성 여부를 collate에 전달하며, 둘 다0이면 wrapped 입력과 pair mask 자체를 만들지 않는다.

`native_steps.py`, `model/loss.py`, `model/b_linear.py`, `main.py`, `train.py`, logger는 이번에 수정하지 않았다. 따라서 이번 변경으로 전역 BPL, 새로운 손실, 추가 optimizer 단계가 도입되지 않는다. 보조 실험용 기존 코드는 남아 있지만 B에서는 실행되지 않는다.

## A와 B를 비교할 때의 통제

| 조건 | native_length | cap_long | 두 보조 손실 계수 | 학습 입력 |
|---|---|---|---|---|
| A | False | True | 둘 다0 | 기존 상한 유지 + 짧으면 반복 |
| B, 현재 기본값 | True | True | 둘 다0 | 같은 상한 유지 + 짧으면 실제 길이 |
| C, 이번 실행 대상 아님 | True | False | 둘 다0 | Long 전체 + Short 기존 상한 |

**A도 같은 현재 코드로 학습해야 반복 처리의 효과를 분리할 수 있다.** A/B에서 같은 데이터, sampler와 UID 배치 순서, class 비율, 초기화 seed, RawBoost 정책, codec-off, physical batch8, optimizer·scheduler, 고정50epoch를 사용한다. 이번 검증에서 A/B sampler의 전체 UID 배치 순서가 동일함을 확인했다. RawBoost의 입력 길이가 달라져 생기는 증강 결과 차이도 이 반복 처리 개입의 일부이므로, 이를 반복 경계 하나만의 효과라고 해석하지 않는다.

학습 효과만 비교하는 최종 평가는 **두50epoch checkpoint 모두 같은 B 평가 설정**으로 수행한다. A 학습 설정을 그대로 읽어 평가하면 A는 반복 평가, B는 비반복 평가가 되어 학습과 추론 처리를 함께 바꾼 결과가 된다. 현재 train은 target 성능에 따른 best 선택 없이 고정 epoch checkpoint를 저장하므로, A의 학습 중 dev 입력 정책 차이로 checkpoint가 선택되지는 않는다. 반복 평가 결과가 필요하면 별도의 보조 비교로 두 모델 모두 같은 반복 평가 설정에서도 평가한다.

전체 EER와 함께 길이별 정상 거부·spoof 수락을 확인한다. 클래스별 오류는 동일 dev 기준으로 정한 threshold를 사용하고, target 결과로 threshold나 checkpoint를 고르지 않는다. 한 seed의 개선은 해당 실행의 근거이며 일반적 개선 주장에는 여러 seed의 재현성과 효과 크기를 확인한다. B 구현 완료나 짧은 smoke update 통과만으로 판별 성능 개선을 주장하지 않는다.

## 기존 batch240 codec 기록 정정

사용자는 기존 `EBM_e50_bs240_lr1e5`에서도 codec을 껐다고 기억한다. 결과 폴더의 `backup/scripts/main.py`에는 ACN off가 있고, `backup/scripts/data_loaders.py`에는 확률0.8과 활성 codec 호출 두 곳이 있다. 백업 codec 구현의 convert도 실제 변환 함수로 연결된다. 그러나 codec 적용 횟수를 기록한 실행 로그는 확인하지 못했다. **확인 사실은 보관 소스의 상태이며, 당시 실행에서 codec이 적용됐다고 단정하지 않는다.**

따라서 이전의 ‘기존240은 codec-on이므로 다르다’라는 확정 표현을 철회한다. 당시 실행의 codec-off가 확인되더라도 이전 실험과 현재 코드 사이에는 sampler·마스크·재계산·평가 정책 차이가 남는다. 기존 결과와의 비교는 참고로 하고, 반복 제거 효과는 위의 동일 코드 A/B 비교로 판단한다.

## 검증과 보관

기존 컨테이너 안에서 `CUDA_VISIBLE_DEVICES=0,1`로 제한된 검증을 수행했다. 원래 실험을 중단하거나 전체 학습을 시작하지 않았다.

- 양 GPU 각각105개 검증 통과, 검증 프로세스 exit0.
- 실제 최소/중앙/최장 정상·spoof와 최장 DF 입력에서 A/B의 보존 원음·crop 상한 일치 확인. C는 long 전체를 보존하는 것도 확인.
- 7개 길이 사례 × RawBoost9종에서 길이 상한과 유한값 통과. 변환 시 실패하는 codec 함수를 넣고도 통과하여 codec 호출 없음 확인.
- 실제25,380개 학습 데이터와 loader 연결, A/B UID 순서 동일, GPU당 정상12/spoof108 확인.
- `np.pad(..., mode='wrap')` 호출 시 실패하도록 설정한 B 데이터 준비·collate 통과. wrapped/pair_active key 없음.
- 실제 XLS-R와 backend에서 혼합 길이0패딩 배치와 개별 입력의 embedding 최대 오차3.73e-6. 유효 feature mask 길이 일치.
- 실제 논리적 batch240에서 OC+BPL과 두 단계 optimizer update 통과. 본체와 centroid 갱신, frozen frontend 불변 확인. 보조 손실·활성 쌍0. 이 검증 배치의 peak allocated memory는 GPU별1,840/1,853MiB이며 전체 데이터 최악치 측정은 아니다.
- Python29파일 구문 검사 통과. 변경 파일은 두 개뿐이며 검증 전후 전체 Python hash 동일. Codec 활성 convert AST 호출0개.

서버 변경 전 원본: `/home/mentee/analysis/ebm_control_B_20261106_2118/before/`. 같은 폴더에 change.json과 GPU별 검증 결과를 보관했다. 로컬에는 원본 코드·음성·checkpoint 대신 [파생 검증 기록](../../artifacts/geometry_cache/local_analysis/control_B_20261106_v1/complete.json)만 가져왔다.
