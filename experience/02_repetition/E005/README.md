<a id="e005-배치-240원래-반복-입력gpu-별-정상-쌍-학습"></a>

# 배치 240·원래 반복 입력·GPU 별 정상 쌍 학습

**실행 상태:** 학습 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

메모리 한도 안의 배치240은 더 많은 로컬 정상쌍으로 일반화를 개선할 수 있다.

## 그렇게 설계한 이유

최근 반복 제거·추가 손실 실험과 비교할 기존 배치 240 모델을 학습한다.

대조와 통제의 근거: 정상 쌍은 GPU 사이에서 공유하지 않는다. 마지막 부분 배치의 정상 6개는 6×6 =36쌍이다. 정확한 두 입력 길이는 4.0375/3.0275초다.

## 모델·데이터·표본

2019 LA 학습. 두 GPU, GPU 별 120 =정상 12 +위조 108. GPU 별 12×12 =144쌍, 한 epoch108배치.

## 실험 내용과 상세 절차

1. [배치 160·50epoch·GPU 별 정상 쌍 학습](../../03_bonafide_diversity/E002/README.md)와 같은 고정 길이긴/짧은 입력, RawBoost, OC +BPL 및 두 단계 Adam을 사용한다.
2. 전체 배치 240·학습률과 최소값 1e-5·weight decay1e-4·seed 1·50epoch로 학습한다.
3. 별도 정상/위조 DataLoader·DistributedSampler와 작은 loader 순환, 원래 부분 배치를 유지한다.
4. 학습 중 기록과 마지막 콘솔 평가를 구별하고, 보존된 15~50epoch8개 모델의 기록된 최소값을 취합한다.

## 측정과 검증

19LA·21LA·21DF 기록된 최소 EER과 epoch; 과거 반복 평가와 새 비반복 공통 평가는 별도 표다.

## 결과

기록된 최소는 19LA 0.217539%(25),21LA 1.052789%(50),21DF 2.396259%(15). 고정 50의 과거 반복 DF 3.477033%와최소 DF 값은 다르다.

## 해석·실패한 부분·한계

보관 코드에 코덱 경로가 있어도 실제 당시 코덱 적용을 확정하지 않는다. EBM-NoRepeat와 EBM-FullContext 와는 샘플러·마스크·메모리 분할 구현도 달라 순수 반복 학습 대조가 아니다.

## 배운 점

19LA·21LA·DF의 최적 epoch가 다르며 과거 반복 결과와 새 비반복 재평가를 같은 표에 섞으면 안 된다.

## 실험 코드

- [소스: arguments.py](code/snapshots/EBM_e50_bs240_lr1e5/arguments.py)
- [소스: data_loaders.py](code/snapshots/EBM_e50_bs240_lr1e5/data_loaders.py)
- [training_snapshots/EBM_e50_bs240_lr1e5/docker_build.sh](code/snapshots/EBM_e50_bs240_lr1e5/docker_build.sh)
- [training_snapshots/EBM_e50_bs240_lr1e5/docker_run.sh](code/snapshots/EBM_e50_bs240_lr1e5/docker_run.sh)
- [소스: main.py](code/snapshots/EBM_e50_bs240_lr1e5/main.py)
- [소스: train.py](code/snapshots/EBM_e50_bs240_lr1e5/train.py)
- [소스: ASVspoof2019_la_eval.py](code/snapshots/EBM_e50_bs240_lr1e5/dataset/ASVspoof2019_la_eval.py)
- [소스: ASVspoof2019_la_train.py](code/snapshots/EBM_e50_bs240_lr1e5/dataset/ASVspoof2019_la_train.py)
- [소스: ASVspoof2021_df_eval.py](code/snapshots/EBM_e50_bs240_lr1e5/dataset/ASVspoof2021_df_eval.py)
- [소스: ASVspoof2021_la_eval.py](code/snapshots/EBM_e50_bs240_lr1e5/dataset/ASVspoof2021_la_eval.py)
- [소스: __init__.py](code/snapshots/EBM_e50_bs240_lr1e5/dataset/__init__.py)
- [소스: __init__.py](code/snapshots/EBM_e50_bs240_lr1e5/data_processing/__init__.py)
- [소스: interface.py](code/snapshots/EBM_e50_bs240_lr1e5/logger/interface.py)
- [소스: local.py](code/snapshots/EBM_e50_bs240_lr1e5/logger/local.py)
- [소스: logger.py](code/snapshots/EBM_e50_bs240_lr1e5/logger/logger.py)
- [소스: neptune.py](code/snapshots/EBM_e50_bs240_lr1e5/logger/neptune.py)
- [소스: wandb.py](code/snapshots/EBM_e50_bs240_lr1e5/logger/wandb.py)
- [소스: __init__.py](code/snapshots/EBM_e50_bs240_lr1e5/logger/__init__.py)
- [소스: b_linear.py](code/snapshots/EBM_e50_bs240_lr1e5/model/b_linear.py)
- [소스: loss.py](code/snapshots/EBM_e50_bs240_lr1e5/model/loss.py)
- [소스: __init__.py](code/snapshots/EBM_e50_bs240_lr1e5/model/__init__.py)
- [소스: cosine_warmup.py](code/snapshots/EBM_e50_bs240_lr1e5/scheduler/cosine_warmup.py)
- [소스: steplr.py](code/snapshots/EBM_e50_bs240_lr1e5/scheduler/steplr.py)
- [소스: ddp_util.py](code/snapshots/EBM_e50_bs240_lr1e5/utils/ddp_util.py)
- [소스: get_threshold.py](code/snapshots/EBM_e50_bs240_lr1e5/utils/get_threshold.py)
- [소스: __init__.py](code/snapshots/EBM_e50_bs240_lr1e5/utils/__init__.py)
- [소스: augmentation.py](code/snapshots/EBM_e50_bs240_lr1e5/data_processing/augmentation/augmentation.py)
- [소스: codec_aug.py](code/snapshots/EBM_e50_bs240_lr1e5/data_processing/augmentation/codec_aug.py)
- [소스: raw_boost.py](code/snapshots/EBM_e50_bs240_lr1e5/data_processing/augmentation/raw_boost.py)
- [소스: wav_read_functions.py](code/snapshots/EBM_e50_bs240_lr1e5/data_processing/augmentation/wav_read_functions.py)

코드의 파일별 원본 위치·SHA256은 [원본 파일 위치와 해시 목록](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [설정·결과 근거](../../_shared/references/artifacts/ebm_B_C_checkpoints_20261109/dataset_minimum_comparison_pending_DF.json)
- [설정·결과 근거](../../_shared/references/artifacts/geometry_cache/local_analysis/B_final_analysis_20261108_v1/historical_legacy_final.json)
- [설정·결과 근거](../../_shared/references/artifacts/analysis/ebm-performance-diagnosis-20261104.json)

## 연결

다른 관련 가설: [길이·유효 발화량과 앞뒤 저에너지 편향](../../01_duration_bias/README.md) · [정상 다양성·도메인 지지와 정상쌍 학습](../../03_bonafide_diversity/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
