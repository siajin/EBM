<a id="e006-학습-길이-상한을-유지하면서-짧은-음성-반복을-제거한-학습b"></a>

# 학습 길이 상한을 유지하면서 짧은 음성 반복을 제거한 학습

**실행 상태:** 학습 50epoch 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

짧은 원음의 순환 반복을 제거하고 유효 길이를 마스킹하면 반복 의존을 줄일 수 있다.

## 그렇게 설계한 이유

긴/짧은 입력의 최대 길이는 그대로 두고, 원음이 짧으면 반복하지 않는 학습을 평가한다.

대조와 통제의 근거: 일반 물리 묶음은 8, 전체 OC 배치는 240. 긴 입력의 최대 길이만 4.0375초이며 모든 음성을 4초로 늘린 것이 아니다. 평가도 앞 64,600·비반복·mask 다.

## 모델·데이터·표본

2019 LA 학습 25,380개. 전체 배치 240, GPU 별 정상 12 +위조 108, 108배치/epoch. 새 seed 1초기화이며 EBM-Repeat 모델을 이어 학습하지 않는다.

## 실험 내용과 상세 절차

1. 긴 입력은 원음 앞 최대 64,600, 짧은 입력은 원음 전체에서 최대 48,440샘플의 연속 무작위 구간을 만든다. 원음이 더 짧으면 실제 길이를 유지한다.
2. 각 물리적 8개 묶음에서 최장 입력까지만 0을 붙이고, XLS-R 및 모든 시간 pooling 에서 유효 길이 마스크를 사용한다.
3. 정상/위조 비율과 길이를 고려하는 배치 샘플러, 특징의 분할 계산·기울기 재계산을 이용해 논리적 120개/rank의 OC +BPL과 두 단계 갱신을 수행한다.
4. OC는 긴 입력의 두 클래스, BPL은 정상긴 12× 짧은 12만 사용한다. 짧은 입력의 추가 분류 손실과 반복 관련 보조 손실은 0이다.
5. Adam·학습률 1e-5·weight decay1e-4·seed 1·50epoch. 코덱 비활성·RawBoost0~8을 적용하고 5epoch 간격 모델과 optimizer 상태를 저장한다.

## 측정과 검증

학습 종료/Adam5,400회/모델-학습 상태 일치와 전체 공식 평가·길이 별 오류를 확인했다. 최소 EER은 [두 반복 제거 학습 모델의 모든 저장 epoch 평가와 최소 EER 비교](../E010/README.md)에서 별도 추출한다.

## 결과

고정 50 공통 비반복 EER은 19LA 0.679810%,21LA 4.792117%,21DF 2.723788%. 전체 학습과 평가 완료.

## 해석·실패한 부분·한계

옛 240과 이 실행의 차이를 파형 반복 하나로 환원하지 않는다. 같은 새 코드의 순환 반복 학습 대조는 별도 실행 하지 않았다. 길이별 정상/위조 교환 관계도 [원음 길이별 오류와 임계값 민감도 분석](../../01_duration_bias/E011/README.md)에 기록한다.

## 배운 점

EBM-NoRepeat 모델은 같은 비반복 입력에서 EBM-Repeat보다 좋았으나 역사적 반복 평가까지 포함한 보편 우위는 아니다. 샘플러와 계산 방식도 달랐다.

## 실험 코드

- [소스: arguments.py](code/snapshots/EBM_e50_bs240_lr1e5_B_zeropad_short303/arguments.py)
- [소스: data_loaders.py](code/snapshots/EBM_e50_bs240_lr1e5_B_zeropad_short303/data_loaders.py)
- [training_snapshots/EBM_e50_bs240_lr1e5_B_zeropad_short303/docker_build.sh](code/snapshots/EBM_e50_bs240_lr1e5_B_zeropad_short303/docker_build.sh)
- [training_snapshots/EBM_e50_bs240_lr1e5_B_zeropad_short303/docker_run.sh](code/snapshots/EBM_e50_bs240_lr1e5_B_zeropad_short303/docker_run.sh)
- [소스: main.py](code/snapshots/EBM_e50_bs240_lr1e5_B_zeropad_short303/main.py)
- [소스: native_steps.py](code/snapshots/EBM_e50_bs240_lr1e5_B_zeropad_short303/native_steps.py)
- [소스: train.py](code/snapshots/EBM_e50_bs240_lr1e5_B_zeropad_short303/train.py)
- [소스: ASVspoof2019_la_eval.py](code/snapshots/EBM_e50_bs240_lr1e5_B_zeropad_short303/dataset/ASVspoof2019_la_eval.py)
- [소스: ASVspoof2019_la_train.py](code/snapshots/EBM_e50_bs240_lr1e5_B_zeropad_short303/dataset/ASVspoof2019_la_train.py)
- [소스: ASVspoof2021_df_eval.py](code/snapshots/EBM_e50_bs240_lr1e5_B_zeropad_short303/dataset/ASVspoof2021_df_eval.py)
- [소스: ASVspoof2021_la_eval.py](code/snapshots/EBM_e50_bs240_lr1e5_B_zeropad_short303/dataset/ASVspoof2021_la_eval.py)
- [소스: __init__.py](code/snapshots/EBM_e50_bs240_lr1e5_B_zeropad_short303/dataset/__init__.py)
- [소스: __init__.py](code/snapshots/EBM_e50_bs240_lr1e5_B_zeropad_short303/data_processing/__init__.py)
- [소스: interface.py](code/snapshots/EBM_e50_bs240_lr1e5_B_zeropad_short303/logger/interface.py)
- [소스: local.py](code/snapshots/EBM_e50_bs240_lr1e5_B_zeropad_short303/logger/local.py)
- [소스: logger.py](code/snapshots/EBM_e50_bs240_lr1e5_B_zeropad_short303/logger/logger.py)
- [소스: neptune.py](code/snapshots/EBM_e50_bs240_lr1e5_B_zeropad_short303/logger/neptune.py)
- [소스: wandb.py](code/snapshots/EBM_e50_bs240_lr1e5_B_zeropad_short303/logger/wandb.py)
- [소스: __init__.py](code/snapshots/EBM_e50_bs240_lr1e5_B_zeropad_short303/logger/__init__.py)
- [소스: b_linear.py](code/snapshots/EBM_e50_bs240_lr1e5_B_zeropad_short303/model/b_linear.py)
- [소스: loss.py](code/snapshots/EBM_e50_bs240_lr1e5_B_zeropad_short303/model/loss.py)
- [소스: __init__.py](code/snapshots/EBM_e50_bs240_lr1e5_B_zeropad_short303/model/__init__.py)
- [소스: cosine_warmup.py](code/snapshots/EBM_e50_bs240_lr1e5_B_zeropad_short303/scheduler/cosine_warmup.py)
- [소스: steplr.py](code/snapshots/EBM_e50_bs240_lr1e5_B_zeropad_short303/scheduler/steplr.py)
- [소스: ddp_util.py](code/snapshots/EBM_e50_bs240_lr1e5_B_zeropad_short303/utils/ddp_util.py)
- [소스: get_threshold.py](code/snapshots/EBM_e50_bs240_lr1e5_B_zeropad_short303/utils/get_threshold.py)
- [소스: __init__.py](code/snapshots/EBM_e50_bs240_lr1e5_B_zeropad_short303/utils/__init__.py)
- [소스: augmentation.py](code/snapshots/EBM_e50_bs240_lr1e5_B_zeropad_short303/data_processing/augmentation/augmentation.py)
- [소스: codec_aug.py](code/snapshots/EBM_e50_bs240_lr1e5_B_zeropad_short303/data_processing/augmentation/codec_aug.py)
- [소스: raw_boost.py](code/snapshots/EBM_e50_bs240_lr1e5_B_zeropad_short303/data_processing/augmentation/raw_boost.py)
- [소스: wav_read_functions.py](code/snapshots/EBM_e50_bs240_lr1e5_B_zeropad_short303/data_processing/augmentation/wav_read_functions.py)
- [소스: native_steps.py](code/EBM/native_steps.py)
- [소스: data_loaders.py](code/EBM/data_loaders.py)

코드의 파일별 원본 위치·SHA256은 [원본 파일 위치와 해시 목록](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [상세 설계·실험 기록](../../_shared/references/docs/experiments/ebm_control_B_20261106.md)
- [상세 설계·실험 기록](../../_shared/references/docs/experiments/ebm_B_final_results_20261108.md)
- [설정·결과 근거](../../_shared/references/artifacts/ebm_git_private_20261106/B_final_training_audit.json)
- [소스: native_steps.py](../../../native_steps.py)
- [소스: data_loaders.py](../../../data_loaders.py)

## 연결

다른 관련 가설: [길이·유효 발화량과 앞뒤 저에너지 편향](../../01_duration_bias/README.md) · [정상 다양성·도메인 지지와 정상쌍 학습](../../03_bonafide_diversity/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
