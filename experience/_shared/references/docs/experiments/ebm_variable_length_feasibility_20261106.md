> 역사 문서 보존본. 원문 작성 당시의 상태·계획을 포함한다. 최신 상태와 확정 결과는 experience의 각 실험 README를 우선한다.

# EBM 원길이 입력과 길이별 배치 적용 가능성

**현재 EBM은 원길이 음성을 처리할 수 있다. 길이가 비슷한 정상·스푸프를 묶고, 배치를 만들 때만 오른쪽에 0을 채운 뒤 XLS-R와 모든 시간 pooling에서 해당 부분을 제외하는 방식으로 수정하면 된다.** 기존의 반복 패딩만 삭제하는 수정은 부족하다. 데이터 로더, 두 클래스 배치 결합, 유효 길이 전달, pooling 및 long/short view 생성을 함께 바꿔야 한다.

실제 학습 파일 25,380개의 헤더를 검사하고, 기존 모델에 원길이 음성을 넣어 단독·배치 추론과 역전파를 확인했다. 아래 결과는 구현 가능성에 대한 검증이다. 원래 학습 소스와 checkpoint는 수정하지 않았고 optimizer update나 전체 재학습은 수행하지 않았다.

## 확인한 코드와 데이터

서버의 현재 `/home/mentee/code/EBM`에서 `data_loaders.py`, `train.py`, `main.py`, `model/b_linear.py`, `model/loss.py`, codec 및 RawBoost 증강을 읽었다. 추론과 gradient 검사에는 기존 epoch50/batch160 모델의 archived module과 checkpoint를 사용했다. 현재 `train.py`와 `model/loss.py`는 이 baseline archive와 SHA256이 같았다.

| 데이터 | 발화 수 | 원길이 중앙값 | 99백분위 | 최장 | 4.0375초 미만 |
|---|---:|---:|---:|---:|---:|
| 19LA 학습 T | 25,380 | 3.202초 | 8.087초 | 13.188초 | 18,609개, 73.3% |
| 기존 V 진단 분할 | 14,983 | 3.311초 | 8.339초 | 11.394초 | 10,561개 |
| 19LA eval | 71,237 | 2.818초 | 7.937초 | 13.026초 | 55,904개 |
| 21LA eval | 148,176 | 2.440초 | 7.380초 | 13.402초 | 127,978개 |
| 21DF 전체 | 533,928 | 2.880초 | 6.776초 | 29.312초 | 413,155개 |

T의 정상은 2,580개, 스푸프는 22,800개다. T 전체 25,380개 파일에서 16kHz·mono·실제 샘플 수가 메타데이터와 일치했다. V는 기존 캐시의 진단 분할이며 공식 dev 전체로 해석하지 않는다. 평가셋의 전체 길이 통계는 기존 메타데이터를 사용했고, DF에서는 선택한 극단 길이 4개 파일의 헤더를 추가 대조했다.

T에는 실제 샘플 길이가 20,338종 있고, 같은 길이로 가장 많이 모이는 경우도 6개뿐이다. 따라서 길이가 정확히 같은 발화만으로 GPU당 80개를 채우는 방식은 현실적이지 않다. 비슷한 길이끼리 묶고 작은 차이만 0으로 채우는 구성이 적합하다.

## 기존 코드에서 바꿔야 하는 부분

| 위치 | 현재 동작 | 원길이 방식의 변경 |
|---|---|---|
| `TrainSet.__getitem__`, 110–120행 | codec 증강 전에 최대 4.845초로 자를 수 있고, 이후 4.0375초로 반복 또는 절단 | long-view 전체를 보존하려면 앞단 codec crop과 뒤의 고정 crop·wrap을 함께 제거 |
| 같은 함수, 150–164행 | short-view를 다시 읽고 codec 증강 후 3.0275초로 반복 또는 random crop | `min(실제 길이, 3.0275초)`의 연속 구간을 사용하고 짧은 원음은 반복하지 않음 |
| `get_loaders`, 22–90행 | 정상·스푸프 별도 DistributedSampler, 기본 collate | 정상·스푸프 길이를 함께 고려하는 batch sampler와 길이를 반환하는 collate 필요 |
| `train.py`, 98–108행 | 클래스별 tensor를 `torch.cat` | 두 loader에서 따로 패딩하면 최장 길이가 달라 cat 실패. 합쳐서 한 번 collate하거나 결합 전에 재패딩 |
| `train.py`, 119–128행 | XLS-R에 waveform만 전달 | long/short 각각 waveform mask 전달 |
| `B_Linear.forward`, 24–35행 | bridge와 최종 pooling에 유효 길이 없음 | `mask`, `mask_short`를 받아 각 경로로 전달 |
| `Bridge_module.forward`, 79–83행 | 25개 layer 요약에서 모든 시간 위치 사용 | 25개 pooling의 softmax와 sample std에 frame mask 적용 |
| 최종 `AHP_ASP` | 최종 시간 pooling에도 모든 위치 사용 | 동일한 frame mask 적용. 총 26개 시간 pooling 수정 |
| `EnrollmentSet`, 210–216행과 `test()` | eval도 첫 4.0375초/wrap | 전체 원길이와 mask를 전달하고 발화 ID와 점수 대응 유지 |

XLS-R 설정은 실제 로딩 결과 `feat_extract_norm="layer"`, `do_stable_layer_norm=True`였다. 이 종류의 배치 입력은 attention mask를 전달하는 것이 [Hugging Face 공식 문서](https://huggingface.co/docs/transformers/en/model_doc/wav2vec2)의 지침과도 일치한다. waveform mask를 convolution 뒤 유효 frame mask로 변환해야 한다. 현재 모델의 frame 수는 `floor((샘플 수−400)/320)+1`이다.

현재 pooling은 일반적인 가중 분산이 아니라 `std(attention * feature + noise)`의 sample std를 사용한다. 이를 유지하면서 padding 위치를 softmax에서 제외하고, std의 평균과 분모에도 실제 frame 수만 사용해야 한다. 0을 곱한 뒤 padded frame까지 std의 분모에 넣으면 단독 입력과 달라진다. 모델 수식을 바꾸는 실험과 패딩을 제거하는 실험을 한 번에 섞지 않는 것이 해석에 유리하다.

## 길이별 묶음의 효율

T의 원길이 전체를 사용하고, 전역 배치 160·GPU당 정상 8개+스푸프 72개를 유지하는 메타데이터 시뮬레이션이다. 모든 원본 발화를 포함하고 마지막 배치 수를 맞추기 위해 정상 12회·스푸프 528회의 추가 샘플링을 했다. 같은 발화 재샘플링은 waveform을 이어 붙이는 반복 패딩과 다르다.

| 배치 구성 | 유효 frame / 배치에 할당한 frame | 유효 attention pair / 할당 pair |
|---|---:|---:|
| 무작위 길이 혼합 | 40.37% | 18.65% |
| 32개 배치 규모의 무작위 후보군 안에서 길이 정렬 | 92.50% | 84.04% |
| 전체 길이 정렬 뒤 배치 순서만 섞음 | 94.95% | 91.98% |

두 GPU 모두 162회씩 같은 수의 step을 받고, 모든 step에서 8:72 비율과 정상 먼저 배치를 유지했다. 정렬은 클래스별로 수행한 뒤 비슷한 길이 순위의 정상·스푸프를 결합했다. 이는 배치 구성의 참고 구현이며 실제 학습 loader에 연결한 결과는 아니다. 전체 정렬은 비교 정상의 길이를 과도하게 제한할 수 있어 무작위 후보군 안의 정렬을 우선 검토하는 편이 낫다.

0.5초 단위의 고정 구간에 1:9 비율을 강제하는 방식은 일부 구간에서 성립하지 않는다. 0.5–1.0초는 정상 0개/스푸프 93개이고, 1.0–1.5초는 정상 9개/스푸프 836개다. 이 스푸프를 버리거나 소수의 정상을 과도하게 재샘플링하기보다 가까운 길이의 정상을 함께 묶어야 한다.

코덱 증강 전 원길이에 대한 계산량 근사에서, 무작위 후보군 정렬은 기존 고정 201-frame 처리에 비해 전체 할당 frame 약 0.917배, attention pair 약 1.005배였다. 이는 실행시간 측정이 아니다. 평균 계산량이 비슷해도 긴 배치의 순간 메모리는 커진다. 증강 후 길이 변동을 반영한 실제 epoch 처리량은 별도 측정해야 한다.

## 실제 원길이 추론과 역전파 결과

기존 컨테이너에서 `CUDA_VISIBLE_DEVICES=0,1`로 두 GPU에 같은 검사를 수행했다. GPU는 각각 약 24GB의 TITAN RTX다. 선택은 점수와 무관하게 T 클래스별 최단·중간·최장 및 DF 부분집합별 최단·최장으로 고정했다. 선택 10개 중 8개의 서로 다른 원음으로 추론하고, 나머지 DF pilot 극단 2개는 헤더를 확인했다.

| 검사 | 관찰 결과 |
|---|---|
| 원길이 단독 입력 | 0.4초/19 frames부터 29.312초/1,465 frames까지 성공. 4초 이상의 원음도 자르지 않음 |
| 기존 classifier와 mask adapter, 모두 유효한 단독 입력 | 최종 z 최대 절대 오차 4.77×10⁻⁷ |
| 비슷한 길이·서로 다른 길이·긴 원음·DF 최단/최장 배치 | 10개 행 비교, z 최대 오차 2.03×10⁻⁶, cosine 점수 최대 오차 5.96×10⁻⁸ |
| 배치 오른쪽에 1초의 0을 더 붙임 | z 최대 오차 8.34×10⁻⁷ |
| 배치 순서 뒤집기 | z 오차 0 |
| backend의 padding feature를 모두 37로 바꿈 | mask가 있으면 z 오차 0 |
| XLS-R와 backend mask를 모두 생략 | 올바른 결과와 z 최대 차이 3.4763 |
| XLS-R mask만 쓰고 backend mask 생략 | z 최대 차이 0.7333 |

동일성 비교에서는 기존 pooling의 `1e-5` 확률적 noise를 0으로 설정했다. float32 계산 순서 차이에 따른 수치 오차는 남는다. 두 GPU의 위 비교 결과는 일치했다. 마스크 없는 배치도 실행 자체는 되지만, padding 양에 따라 임베딩이 달라지므로 원길이 추론으로 볼 수 없다.

실제 음성 5개에서는 long-view 전체와 최대 3.0275초의 연속 short-view를 각각 XLS-R에 통과시켰다. long tensor는 `[5,25,659,1024]`, short tensor는 `[5,25,151,1024]`였으며, 짧은 발화는 원음 그대로 두었다. 원래 크기의 pooling noise와 dropout을 켠 상태에서 OC+BPL backward가 성공했고 181개 학습 파라미터 tensor의 gradient가 모두 유한했다. 별도 검사에서 정상만 사용하는 centroid 단계의 gradient도 유한했다. optimizer step은 수행하지 않았다.

masked pooling을 각 발화의 실제 frame만 잘라 계산한 독립 ragged 기준과 float64로 비교하면 출력 최대 오차 2.22×10⁻¹⁶, 입력 gradient 오차 1.55×10⁻¹⁵였다. padding feature의 gradient는 정확히 0이었다.

작은 배치 추론의 peak allocated memory는 긴 학습 원음 2개에서 약 1,825MiB, 29.312초가 포함된 2개 배치에서 약 2,437MiB였다. 이 수치는 전체 학습의 optimizer·두 view·최적 모델 사본 등을 포함하지 않으며 GPU당 80개 원길이 학습의 수용 가능성을 보장하지 않는다. 최장 학습 음성의 659 frames는 기존 201 frames보다 attention pair 계산량이 발화당 약 10.75배다. 긴 배치는 별도 메모리 예산을 두어야 한다.

## 코덱 증강의 길이 변동

서로 다른 길이의 실제 음성 3개에 codec 5종을 적용했다. 현재 `convert()`는 전달받은 codec/quality 인자 대신 내부 `selectCodec()`를 사용하므로, 검사에서는 해당 인스턴스의 선택 함수를 고정했다. 원본 소스는 수정하지 않았다.

| codec | 이번 3개 음성의 출력 길이 변화 |
|---|---|
| MP3 | +1,537~1,726 samples, 약 0.096~0.108초 |
| OGG | 0 또는 +441 samples |
| AAC | +1,406~1,985 samples, 약 0.088~0.124초 |
| a-law / u-law | 0 또는 +1 sample |

RawBoost의 세 기본 연산은 9개 검사에서 길이를 보존했다. 코덱 결과는 증강 후 실제 샘플 수를 다시 측정해 mask를 만들어야 한다. 추가 길이의 시간 위치나 encoder delay를 이번 길이 측정만으로 확정할 수는 없다. 정확히 원래 시간 범위를 보존하려면 코덱 지연 처리와 길이 복원 정책도 명시적으로 검증해야 하며, 단순히 끝을 잘라서 해결했다고 가정하면 안 된다.

## BPL과 두 GPU 학습에서 유지할 조건

현재 BPL은 정상 long 임베딩 행렬 `X`와 정상 short 임베딩 행렬 `Y`에 대해 `normalize(X) @ normalize(Y).T`를 계산한다. **동일 발화의 long/short 대각 쌍뿐 아니라 서로 다른 정상 발화의 교차 조합도 포함한다.** 현재 원본 구현에서 GPU당 정상 8개이면 8×8 비교다.

과거 전역 BPL 실험의 archive에는 gradient를 보존하는 `all_gather`가 있지만, 이번에 읽은 현재 `/home/mentee/code/EBM/model/loss.py`에는 해당 코드가 없고 baseline과 같았다. 이번 검토에서 이를 복원하거나 바꾸지는 않았다. 전역 BPL 모델을 기준으로 새 실험할 경우 해당 archive를 별도 대상으로 삼아야 한다.

길이별 배치는 한 step 안에서 만나는 정상의 길이 범위를 좁힌다. 원래 다양한 길이의 정상 간 비교까지 학습에 기여했을 수 있으므로, 무작위 후보군 크기를 지나치게 작게 잡지 않고 batch 구성 효과를 별도로 확인해야 한다. 길이가 다른 rank별 입력 자체는 DDP 파라미터 shape를 바꾸지 않는다. 그러나 각 rank의 step 수와 정상 수, collective 호출 순서는 맞춰야 하며, gradient accumulation만으로 BPL의 전체 교차 조합이 자동 복원되지는 않는다.

실제 DDP 학습 loop와 GPU당 80개의 원길이 학습 메모리는 이번 검사 범위 밖이다. 두 GPU에서 독립적으로 동일한 추론·gradient 검사를 수행했다는 것과 구분한다.

## 적용안

long-view와 평가 입력은 원음 전체를 보존한다. short-view는 같은 발화의 실제 연속 구간을 사용하되 원음보다 길게 늘리지 않는다. 3.0275초 미만에서는 두 view가 같은 구간일 수 있으며, 학습에서는 각 view의 독립 증강을 유지할 수 있다. 이는 기존의 4초/3초 강제 입력 규칙을 바꾸는 학습 조건이다.

**short-view의 3.0275초 상한은 구조적 요구가 아니라 기존 학습 설정을 이어가는 비교 기준이다.** 이 상한이 원길이 두 view 또는 상대 비율 crop보다 성능이 좋다고 검증한 것은 아니다. [원 논문 3.1절](https://www.isca-archive.org/interspeech_2025/kim25g_interspeech.pdf)은 두 번째 정상 입력을 더 짧게 잘라 증강 전 변화를 만들고 임베딩을 정렬하는 목적을 설명한다. 두 view를 모두 원길이로 두고 독립 증강하거나, 원길이에 대한 비율로 연속 구간을 선택하는 것도 pooling 및 BPL의 입력 차원과 양립한다. 두 입력을 똑같이 만들면 길이 차이를 통한 변화는 없어지지만, 실제 코드의 교차 정상 쌍 BPL이 모두 0이 된다는 뜻은 아니다. 짧은 원음에서 두 view 모두 원길이가 되는 기준안과 이들 대안의 성능은 동일한 학습·평가 조건으로 비교해야 한다.

클래스 비율과 DDP step을 맞춘 길이별 batch sampler를 두고, long/short 각각 한 번씩 배치 최장 길이까지 0-padding한다. 증강 뒤 측정한 lengths로 waveform mask와 convolution-valid frame mask를 만든 뒤, XLS-R와 bridge의 25개 pooling 및 최종 pooling으로 전달한다. 160차원 출력, Linear, LayerNorm, centroid 차원은 그대로 사용할 수 있다.

현재 checkpoint의 평가 전처리만 바꾸면 기존에 학습한 반복 문맥과 달라진다. 이전 471개 선택 표본에서는 4초보다 짧은 원음의 반복을 없앴을 때 정상뿐 아니라 스푸프 점수도 올라 EER가 3.57%에서 9.57%로 악화됐다. 이 값은 전체 DF 결과가 아니고, 긴 발화는 첫 4.0375초를 유지한 이전 실험이다. 이번 전체 원길이의 구현 가능성과도 구분해야 한다. 성능 판단은 원길이 조건으로 학습한 모델과 기존 반복 조건을 비교해야 한다.

따라서 **길이별 배치와 마스크를 포함한 별도 원길이 학습 버전으로 수정할 수 있다.** 우선 현재 모델과 결과를 보존한 별도 경로에서 실제 sampler·collate·두 단계 DDP 학습 및 긴 배치 메모리를 확인한 후 재학습하는 순서가 적합하다. 고정 4초를 상한으로 남기는 선택도 가능하지만, 그 경우에는 전체 원길이 방식과 구분해 실험명을 정해야 한다.

## 재현 자료

로컬 파생 결과는 `artifacts/geometry_cache/local_analysis/variable_length_audit_20261106_v1/`에 있다. `metadata_audit.json`, `rank0_results.json`, `rank1_results.json`, `augmentation_probe/`, `independent_verification.json`, `final_server_verification.json`을 함께 확인한다. 메타데이터 배치·export hash·두 GPU 결과·실제 음성 gradient·ragged 기준 검산을 통과했다.

자체 코드는 `artifacts/geometry_experiments/`의 `audit_variable_length_metadata.py`, `variable_length_sidecar.py`, `variable_length_augmentation_sidecar.py`, `run_variable_length_audit.py`, `verify_variable_length_audit.py`다. 서버의 기존 모델을 import하며 연구실 원본 소스·음성·checkpoint를 로컬로 복사하지 않았다. waveform과 XLS-R hidden states는 파일로 저장하지 않았다.

서버의 두 새 검사 경로 `20261106_variable_length_01`, `20261106_variable_length_augmentation_01`은 모두 exit0이다. 2026-11-06 16:46:37 KST 확인에서 기존 컨테이너 셸은 유휴 상태이고 GPU6/7은 각각 6MiB·0%였다. baseline archive 28개 파일 및 checkpoint, 현재 핵심 소스 5개 파일의 hash를 보존했다.
