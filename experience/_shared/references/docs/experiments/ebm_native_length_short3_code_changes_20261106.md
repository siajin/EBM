> 표시 명칭·링크 이름은 설명형으로 정리했다. 원래 코드·경로·수치는 그대로 보존한다.

> 역사 문서 보존본. 원문 작성 당시의 상태·계획을 포함한다. 최신 상태와 확정 결과는 experience의 각 실험 README를 우선한다.

# EBM 원길이와 3초 short 코드 변경 결과

**현재 기본값은 4.0375초 상한 비반복 학습:** Long4.0375초·Short3.0275초 상한을 유지하고 반복을 제거했다. CPU의 미사용 wrapped 생성도 껐다. 두 보조 손실0, 기존 OC+BPL 유지. 아래 내용은 이전 구현 기록이며 현재 설정과 통제 비교 기준은 [4.0375초 상한 비반복 학습 변경·검증 기록](ebm_control_B_20261106.md)을 따른다. 배치 240 고정 길이 반복 입력 학습 codec에 관한 확정 표현도 정정했다. 보관 소스에는 활성 호출이 있지만 당시 실행 적용 여부는 확인하지 못했다.

**최신 기본값 20:58 E1 전환:** 사용자가 이번에는 반복 입력 없이0패딩으로기존batch240결과와비교하려는의도를명확히해 `arguments.py`의 `short_oc_weight=0`, `padding_loss_weight=0`, 이름=`EBM_e50_bs240_lr1e5_native_short303_E1`로바꿨다. 모델에는long/short두branch만입력되고분류손실은기존long OC와로컬BPL만쓴다. 아래E3는이전구현당시기록이다. 미사용W가CPU collate에서생성될수는있지만모델forward나손실에는들어가지않는다. 배치 240 고정 길이 반복 입력 학습실험의보관코드에는onlinecodec확률.8/활성호출2개가있으므로현재codec-off결과와의단순비교로반복제거효과를단독입증할수없다.

2026-11-06 서버 `/home/mentee/code/EBM`에 반영했다. 변경 전 파일은 서버 안의 `/home/mentee/analysis/ebm_native_short303_patch_20261106_2005_03/before`에 보관했다. 별도 경로에서 검증한 후 실제 소스에 적용했고, 실제 경로를 다시 import해 두 GPU 검증을 통과했다. 전체50epoch 학습은 시작하지 않았다.

**후속 코덱 정정:** 이 문서는20:27반영판의기록이다. 이후전체GitHub비교에서사용자의추가HiFi-GAN/ACN주석과별개로실시간codec두호출이활성이었음을확인했다. 확률값을바꾸지않고두conditional block을주석처리해현재는실제codec호출0회를검증했다. 현재소스의전체차이·최장길이·속도측정은 [전체재점검결과](ebm_native_github_full_audit_20261106.md)를기준으로확인한다.

## 반영한 동작

**Long과 평가 입력은 원길이 전체, short는 최대48,440sample(3.0275초)의 연속 구간**이다. 부족한 short는 강제로 늘리지 않는다. 물리 microbatch를 만들 때만 오른쪽을0으로 채우며, XLS-R와26개 시간 pooling에서 해당 영역을 제외한다.

기존 BPL은 GitHub `kimho1wq/EBM`의 `model/loss.py`와 대조했다. 수정 직전 서버 파일은 이미 원본과 같은 SHA256(`164be4161b57a7899ef7b289e005ab80d12da373922d57e9e021844ea828b1ac`)이었고 전역 embedding all-gather가 없었다. 따라서 전체 저장소를 reset하지 않고 **원본 BPL 수식을 그대로 보존**했다. 현재 전체 batch240을 유지했으므로 GPU당 정상12개×short정상12개의 로컬 교차 비교다. 새 보조 손실의 유효 쌍 수와 합을 맞추는 scalar all-reduce는 BPL embedding을 공유하는 전역 BPL과 다르다.

사용자의 추가 지시에 따라 **코덱 확률·변환 호출·코덱 선택 설정을 바꾸지 않았다.** 기존 줄과의 일치를 검사했다. 원길이 모드에서는 long/short를 코덱 전에 일정 길이로 미리 자르던 제한만 해제해 원음 뒷부분이 사라지지 않도록 했다. 기존 데이터 경로와 사용자가 수정한 dataset 파일도 보존했다. 코덱이 실제 활성인지에 대한 새 설정 변경은 수행하지 않았다.

## 학습 손실과 메모리 처리

새 기본 설정은 설계의 E3다. 정상 전체 교차 BPL을 유지하고, 동일한 short 구간 U와 그것을3.0275초까지 반복한 W의 정상1:1 정렬을 별도로 추가했다. 실제로 반복된 정상만 이 손실을 받는다. W의 반복 영역은 유효 입력이고, batch용0패딩만 mask된다.

```text
기존 로컬 BPL: 정상 long 전체 × 정상 short 전체
OC_mix: 0.75*OC(long) + 0.125*OC(short) + 0.125*OC(wrapped_short)
추가 정렬: 0.10 * mean_active_normal[softplus(30*(1-cos(U,W)))-log(2)]
encoder loss: OC_mix + 기존 BPL + 추가 정렬
centroid loss: 업데이트된 encoder의 정상 long만 사용
```

Centroid를 고정한 encoder update 다음에, 갱신된 encoder로 정상 long을 다시 계산해 centroid를 update한다. 두 단계 Adam 방식과 기존 OC margin/class weight는 유지했다. XLS-R은 동결한다.

메모리를 위해 물리 microbatch8개로 계산하고, 모은 임베딩의 전체 논리적 손실 gradient를 같은 dropout/noise RNG로 재계산해 전달한다. **BPL을 microbatch별로 쪼개 평균하지 않는다.** 따라서 로컬12×12 쌍과 논리적 batch240이 보존된다. microbatch가 작아져도 OC class weight는 논리적 batch 크기로 판단한다.

## 파일별 변경

기존7개 파일을 수정하고, 반복 계산을 분리한131줄의 `native_steps.py` 하나를 추가했다. 조건별 실행기·12run 자동 실행 프레임워크·새 모델 계층은 추가하지 않았다.

| 서버 파일 | 변경 |
|---|---|
| `arguments.py:66` | 원길이 flag, physical/eval batch8, 보조 OC/정렬 계수 추가. 기존 하이퍼파라미터는 유지하고 실험명만 새 suffix 부여 |
| `data_loaders.py:24` | 정상·스푸프를 같은 길이 순위로 묶는 공통 sampler, 클래스 비율 보존. Long 고정 crop/wrap 제거, short 상한 유지, 동일 short의 보조 wrap 생성 |
| `data_loaders.py:65` | 평가 sampler에서 마지막 발화 중복을 제거. 발화 index와 길이를 collate로 전달 |
| `model/b_linear.py:24` | mask와 embedding 반환 경로 추가. 기존 state dict shape/key 유지 |
| `model/b_linear.py:137` | 25개 layer 요약과 최종ASP에 mask 적용. 기존 weighted-feature sample std 수식 유지 |
| `model/loss.py:41` | 기존 BPL 변경 없음. OC에 논리적 batch 크기를 전달하는 인자만 추가 |
| `native_steps.py:13` | XLS-R mask·frame mask 변환, 원래 BPL+새 보조 손실, embedding-gradient replay, 두 단계 갱신 구현 |
| `train.py:13` | 새 loader와 step helper 연결. 검증은 dev에만 수행하고 target best-model 선택 제거, 주기적 checkpoint 저장 |
| `train.py:44` | 실제 유효 길이로 평가. rank별 결과를 index와 함께 모아 누락·중복 검사 후 EER 계산 |
| `main.py:41` | rank별 현재CUDA device 지정. Frozen frontend의 불필요한DDP wrapper 제거, classifier DDP는 유지 |
| `main.py:183` | 신규 설정 검증·기존 결과 경로 재사용 거부 |
| `logger/local.py:28` | 같은 결과 폴더를 삭제하던 동작을 `FileExistsError`로 변경 |

기존 `train.py`의 중복된 두 클래스 결합·고정 길이 forward·target 선택 루프를 연결된 작은 함수로 대체했다. 원본288줄에서82줄로 줄였고 메모리 처리는 새 helper로 분리했다. dataset 두 파일을 포함한 무관한21개Python 파일은 수정 전 hash를 보존했다.

## 현재 실제 기본값

| 항목 | 반영값 |
|---|---|
| 전체 batch | **배치 240 고정 길이 반복 입력 학습 유지**, GPU당120=정상12+스푸프108 |
| Physical / eval batch | GPU당8 / 8 |
| Short | 303frames 설정 유지, 48,440samples |
| Long / 평가 | 전체 원길이 |
| 기존 BPL | 원본 로컬12×12, alpha_cl30 유지 |
| 새 손실 계수 | `short_oc_weight=.25`, `padding_loss_weight=.10` |
| 모델 출력 | 기존160차원, 새 학습 파라미터 없음 |
| 기타 기존 설정 | epoch50, seed1, lr/weight decay·margin·데이터 경로 유지 |
| 실험명 | `EBM_e50_bs240_lr1e5_native_short303_pair` |
| 학습 중 검증 | 기존 main에 연결돼 있던 **공식19LA dev 전체24,844개**. 과거 진단용V14,983개로 자동 대체하지 않음 |

앞선 설계의 batch160·V14,983과 위 두 항목은 다르다. 이번 요청의 기존 설정 보존과 핵심 연결 변경에 맞춰 현재 코드의 batch240·공식dev loader를 사용했다. 이전 V14,983 기준의 threshold나수치를 이 코드의dev 결과로 사용하면 안 된다.

기본값은 E3이며 `short_oc_weight=0`, `padding_loss_weight=0`이면 E1, `.25/0`이면 E2가 된다. E0는 `native_length=False`와 두 보조 계수0이 필요하다. 같은 이름으로 재실행하는 것은 거부하므로 조건별 실험명을 따로 지정해야 한다. 별도의 full-run 실행은 하지 않았다.

## 검증 결과

별도 stage와 실제 반영 경로에서 각각 기존 컨테이너·`CUDA_VISIBLE_DEVICES=0,1`로 통과했다.

- 기존 checkpoint strict-load, 원래 클래스와 all-valid mask forward 대조: z 최대 오차 약1.79e-7.
- 실제 음성 단독/배치 mask 비교: z 최대 오차 약2.03e-6. 원길이29.312초도 통과했다.
- Ragged ASP 출력·입력gradient, padding gradient0, 원래 로컬 BPL 수식 대조.
- 전체 graph 기준과 gradient replay의 **세 번 연속 두 단계DDP Adam update** 비교. 최대 parameter 차이 약1.82e-12. 한 rank만 유효 쌍이 있는 경우와 양 rank 모두0인 경우를 포함했다.
- E1/E2/E3 손실 경로, 실제 batch240(각GPU120)에서 encoder/centroid update·frozen frontend 검증.
- 원래 학습25,380개 전체 sampler coverage, epoch108step·GPU당12/108, 동일한 전역25920회 입력 횟수 확인.
- main→loader→학습→평가 연결 검사, rank별 평가 batch 수가 달라도 hang·발화 중복 없이5개를 정확히 집계하는 검사.
- 기존 결과 폴더 덮어쓰기 거부와 sentinel 보존.

실제 batch240의 제한된 smoke update에서 GPU별 peak allocated memory는 약2,909/2,689MiB였다. 이는 전체epoch의 최악 길이·처리속도 측정이나 성능 개선 결과가 아니다. 작은 모델의DDP 검사에서는 gradient stride에 대한 성능 경고가 있었지만 update 수치 검증은 통과했다.

Baseline archive28개 파일과기존checkpoint hash는 그대로다. 원본 소스·음성·checkpoint·SSL hidden을 로컬로 복사하지 않았고, 완료한 파생 검증JSON만 로컬에 보관했다. 최종20:27:38KST 확인에서 기존 셸은 유휴 상태, GPU6/7은 각6MiB·0%였다.

[최종 검증JSON](../../artifacts/geometry_cache/local_analysis/native_short303_code_patch_20261106_v1/final_verification.json), [파일별 변경량과hash](../../artifacts/geometry_cache/local_analysis/native_short303_code_patch_20261106_v1/installed.json), [GPU0 검수](../../artifacts/geometry_cache/local_analysis/native_short303_code_patch_20261106_v1/installed_rank0.json), [GPU1 검수](../../artifacts/geometry_cache/local_analysis/native_short303_code_patch_20261106_v1/installed_rank1.json).
