<a id="e007-반복-없는-학습에서-긴-입력의-상한만-제거한-학습c"></a>

# 반복 없는 학습에서 긴 입력의 상한만 제거한 학습

**실행 상태:** 학습 50epoch 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

반복을 없앤 상태에서 긴 원음의 문맥 상한을 해제하면 보존 정보가 늘어 일반화가 개선될 수 있다.

## 그렇게 설계한 이유

학습 시 긴 문맥을 더 사용하면 길이 상한을 둔 학습보다 개선되는지 본다.

대조와 통제의 근거: 4.0375초 상한 비반복 학습 대비 학습 긴 문맥만 변경한다. 정상 576개·위조 6,195개가 4.0375초 보다 길다. 새긴 입력의 고유 샘플 사용량은 4.0375초 상한 비반복 학습 보다 10.5302%많았다.

## 모델·데이터·표본

[학습 길이 상한을 유지하면서 짧은 음성 반복을 제거한 학습](../../02_repetition/E006/README.md)과 같은 학습 데이터·초기화 seed·배치·손실·증강·optimizer. 긴 입력만 원음 전체를 사용한다.

## 실험 내용과 상세 절차

1. 긴 입력의 4.0375초 상한을 해제한다. 짧은 입력은 기존 최대 3.0275초 연속 무작위 구간을 유지하고 짧은 원음 반복은 하지 않는다.
2. 평가용 DataLoader 에는 4.0375초 상한을 명시해 학습 상한 변경이 평가 문맥까지 바꾸지 않게한다.
3. 4.0375초 상한 비반복 학습의 학습 체크포인트를 이어 쓰지 않고 seed 1로 처음부터 50epoch를 학습한다. 같은 샘플러·마스크·미세배치 8·두 단계 Adam·codec-off·RawBoost를 사용한다.

## 측정과 검증

5~50epoch 체크포인트 전체를 4.0375초 상한 비반복 학습과 같은 비반복 입력·난수에서 평가한다.

## 결과

고정 50 EER은 19LA 0.584636%,21LA 5.163467%,21DF 2.881368%. 19LA는 4.0375초 상한 비반복 학습 보다 낮고 다른 두 집합은 높았다.

## 해석·실패한 부분·한계

긴 음성 비중·다른 평가 도메인에 따라 효과가 다르다. 최종 50과 여러 epoch 최소값은 다른 비교다. 단일 seed 결과를 보편 법칙으로 일반화하지 않는다.

## 배운 점

원음 전체 비반복 학습은 4.0375초 상한 비반복 학습보다19LA는 좋고21LA·DF는 나빴다. 더 긴 문맥을 쓰는 것이 항상 유리하지 않다.

## 실험 코드

- [소스: arguments.py](code/snapshots/EBM_e50_bs240_lr1e5_C_nativeLong_short303/arguments.py)
- [소스: data_loaders.py](code/snapshots/EBM_e50_bs240_lr1e5_C_nativeLong_short303/data_loaders.py)
- [training_snapshots/EBM_e50_bs240_lr1e5_C_nativeLong_short303/docker_build.sh](code/snapshots/EBM_e50_bs240_lr1e5_C_nativeLong_short303/docker_build.sh)
- [training_snapshots/EBM_e50_bs240_lr1e5_C_nativeLong_short303/docker_run.sh](code/snapshots/EBM_e50_bs240_lr1e5_C_nativeLong_short303/docker_run.sh)
- [소스: main.py](code/snapshots/EBM_e50_bs240_lr1e5_C_nativeLong_short303/main.py)
- [소스: native_steps.py](code/snapshots/EBM_e50_bs240_lr1e5_C_nativeLong_short303/native_steps.py)
- [소스: train.py](code/snapshots/EBM_e50_bs240_lr1e5_C_nativeLong_short303/train.py)
- [소스: ASVspoof2019_la_eval.py](code/snapshots/EBM_e50_bs240_lr1e5_C_nativeLong_short303/dataset/ASVspoof2019_la_eval.py)
- [소스: ASVspoof2019_la_train.py](code/snapshots/EBM_e50_bs240_lr1e5_C_nativeLong_short303/dataset/ASVspoof2019_la_train.py)
- [소스: ASVspoof2021_df_eval.py](code/snapshots/EBM_e50_bs240_lr1e5_C_nativeLong_short303/dataset/ASVspoof2021_df_eval.py)
- [소스: ASVspoof2021_la_eval.py](code/snapshots/EBM_e50_bs240_lr1e5_C_nativeLong_short303/dataset/ASVspoof2021_la_eval.py)
- [소스: __init__.py](code/snapshots/EBM_e50_bs240_lr1e5_C_nativeLong_short303/dataset/__init__.py)
- [소스: __init__.py](code/snapshots/EBM_e50_bs240_lr1e5_C_nativeLong_short303/data_processing/__init__.py)
- [소스: interface.py](code/snapshots/EBM_e50_bs240_lr1e5_C_nativeLong_short303/logger/interface.py)
- [소스: local.py](code/snapshots/EBM_e50_bs240_lr1e5_C_nativeLong_short303/logger/local.py)
- [소스: logger.py](code/snapshots/EBM_e50_bs240_lr1e5_C_nativeLong_short303/logger/logger.py)
- [소스: neptune.py](code/snapshots/EBM_e50_bs240_lr1e5_C_nativeLong_short303/logger/neptune.py)
- [소스: wandb.py](code/snapshots/EBM_e50_bs240_lr1e5_C_nativeLong_short303/logger/wandb.py)
- [소스: __init__.py](code/snapshots/EBM_e50_bs240_lr1e5_C_nativeLong_short303/logger/__init__.py)
- [소스: b_linear.py](code/snapshots/EBM_e50_bs240_lr1e5_C_nativeLong_short303/model/b_linear.py)
- [소스: loss.py](code/snapshots/EBM_e50_bs240_lr1e5_C_nativeLong_short303/model/loss.py)
- [소스: __init__.py](code/snapshots/EBM_e50_bs240_lr1e5_C_nativeLong_short303/model/__init__.py)
- [소스: cosine_warmup.py](code/snapshots/EBM_e50_bs240_lr1e5_C_nativeLong_short303/scheduler/cosine_warmup.py)
- [소스: steplr.py](code/snapshots/EBM_e50_bs240_lr1e5_C_nativeLong_short303/scheduler/steplr.py)
- [소스: ddp_util.py](code/snapshots/EBM_e50_bs240_lr1e5_C_nativeLong_short303/utils/ddp_util.py)
- [소스: get_threshold.py](code/snapshots/EBM_e50_bs240_lr1e5_C_nativeLong_short303/utils/get_threshold.py)
- [소스: __init__.py](code/snapshots/EBM_e50_bs240_lr1e5_C_nativeLong_short303/utils/__init__.py)
- [소스: augmentation.py](code/snapshots/EBM_e50_bs240_lr1e5_C_nativeLong_short303/data_processing/augmentation/augmentation.py)
- [소스: codec_aug.py](code/snapshots/EBM_e50_bs240_lr1e5_C_nativeLong_short303/data_processing/augmentation/codec_aug.py)
- [소스: raw_boost.py](code/snapshots/EBM_e50_bs240_lr1e5_C_nativeLong_short303/data_processing/augmentation/raw_boost.py)
- [소스: wav_read_functions.py](code/snapshots/EBM_e50_bs240_lr1e5_C_nativeLong_short303/data_processing/augmentation/wav_read_functions.py)

코드의 파일별 원본 위치·SHA256은 [원본 파일 위치와 해시 목록](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [상세 설계·실험 기록](../../_shared/references/docs/experiments/ebm_C_training_started_20261108.md)
- [설정·결과 근거](../../_shared/references/artifacts/ebm_C_20261108/state.json)
- [설정·결과 근거](../../_shared/references/artifacts/ebm_C_20261108/metadata_validation.json)
- [집계 평가 지표](../../_shared/references/artifacts/ebm_B_C_checkpoints_20261109/metrics.json)

## 연결

다른 관련 가설: [정상 다양성·도메인 지지와 정상쌍 학습](../../03_bonafide_diversity/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
