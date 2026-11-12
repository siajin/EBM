# E002. 배치 160·50epoch·GPU 별 정상 쌍 학습

**실행 상태:** 학습 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

로컬 정상쌍을 사용하는 배치160 EBM의 고정 표현을 분석하면 일반화가 무너지는 조건을 찾을 수 있다.

## 그렇게 설계한 이유

전체 배치 160의 기존 EBM을 학습하고 이후 기하·문맥 분석의 고정 모델로 사용한다.

대조와 통제의 근거: GPU 사이 BPL 특징 공유 없음. 일반 배치에서 같은 발화 쌍은 전체 BPL의 1/8이다. 원래 평가 입력은 64,600샘플 순환 반복/앞부분 절단이다.

## 모델·데이터·표본

2019 LA 학습 25,380개. 두 GPU, GPU 별 정상 8 +위조 72. 일반 BPL은 8×8 =64쌍이며 두 GPU의 별도 평균을 분산 학습으로 합친다.

## 실험 내용과 상세 절차

1. [E001](../E001/README.md)의 긴/짧은 입력 및 두 단계 손실 계산을 유지한다.
2. 전체 배치 160, seed 1, Adam, 학습률·최소 학습률 1e-5, weight decay1e-4, 50epoch로 학습한다. epoch당 162배치다.
3. 25,380개 원음 중 정상과 위조의 별도 DataLoader를 사용하고 작은 정상 loader를 순환해 1:9배치를 만든다.
4. 50epoch 모델을 고정한 뒤 최종 선형 변환 전후가 아닌, 마지막 선형 변환 출력과 그 뒤 LayerNorm 출력 각각 160차원을 저장해 후속 분석에 쓴다.

## 측정과 검증

기록된 최소 EER과 50epoch 고정 모델을 다시 평가한 수치를 별도로 기록한다.

## 결과

보존 최소는 19LA 0.231135%(25),21LA 1.016047%(35),21DF 2.421145%(25)다. 고정 50 캐시 평가의 전체 DF는 3.241828%로 다른 시점·경로의 값이다.

## 해석·실패한 부분·한계

최소값과 50epoch 캐시값을 혼용하지 않는다. 코덱의 실제 과거 적용과 frontend 전체 가중치의 과거 비트 동일성은 별도 로그로 확정하지 못했다.

## 배운 점

학습 중 최소와 고정50 재평가가 다르다. 분석용 checkpoint·입력·표본을 항상 같이 기록해야 한다.

## 실험 코드

- [training_snapshots/EBM_e50_bs160_lr1e5/arguments.py](code/snapshots/EBM_e50_bs160_lr1e5/arguments.py)
- [training_snapshots/EBM_e50_bs160_lr1e5/data_loaders.py](code/snapshots/EBM_e50_bs160_lr1e5/data_loaders.py)
- [training_snapshots/EBM_e50_bs160_lr1e5/docker_build.sh](code/snapshots/EBM_e50_bs160_lr1e5/docker_build.sh)
- [training_snapshots/EBM_e50_bs160_lr1e5/docker_run.sh](code/snapshots/EBM_e50_bs160_lr1e5/docker_run.sh)
- [training_snapshots/EBM_e50_bs160_lr1e5/main.py](code/snapshots/EBM_e50_bs160_lr1e5/main.py)
- [training_snapshots/EBM_e50_bs160_lr1e5/train.py](code/snapshots/EBM_e50_bs160_lr1e5/train.py)
- [training_snapshots/EBM_e50_bs160_lr1e5/dataset/ASVspoof2019_la_eval.py](code/snapshots/EBM_e50_bs160_lr1e5/dataset/ASVspoof2019_la_eval.py)
- [training_snapshots/EBM_e50_bs160_lr1e5/dataset/ASVspoof2019_la_train.py](code/snapshots/EBM_e50_bs160_lr1e5/dataset/ASVspoof2019_la_train.py)
- [training_snapshots/EBM_e50_bs160_lr1e5/dataset/ASVspoof2021_df_eval.py](code/snapshots/EBM_e50_bs160_lr1e5/dataset/ASVspoof2021_df_eval.py)
- [training_snapshots/EBM_e50_bs160_lr1e5/dataset/ASVspoof2021_la_eval.py](code/snapshots/EBM_e50_bs160_lr1e5/dataset/ASVspoof2021_la_eval.py)
- [training_snapshots/EBM_e50_bs160_lr1e5/dataset/__init__.py](code/snapshots/EBM_e50_bs160_lr1e5/dataset/__init__.py)
- [training_snapshots/EBM_e50_bs160_lr1e5/data_processing/__init__.py](code/snapshots/EBM_e50_bs160_lr1e5/data_processing/__init__.py)
- [training_snapshots/EBM_e50_bs160_lr1e5/logger/interface.py](code/snapshots/EBM_e50_bs160_lr1e5/logger/interface.py)
- [training_snapshots/EBM_e50_bs160_lr1e5/logger/local.py](code/snapshots/EBM_e50_bs160_lr1e5/logger/local.py)
- [training_snapshots/EBM_e50_bs160_lr1e5/logger/logger.py](code/snapshots/EBM_e50_bs160_lr1e5/logger/logger.py)
- [training_snapshots/EBM_e50_bs160_lr1e5/logger/neptune.py](code/snapshots/EBM_e50_bs160_lr1e5/logger/neptune.py)
- [training_snapshots/EBM_e50_bs160_lr1e5/logger/wandb.py](code/snapshots/EBM_e50_bs160_lr1e5/logger/wandb.py)
- [training_snapshots/EBM_e50_bs160_lr1e5/logger/__init__.py](code/snapshots/EBM_e50_bs160_lr1e5/logger/__init__.py)
- [training_snapshots/EBM_e50_bs160_lr1e5/model/b_linear.py](code/snapshots/EBM_e50_bs160_lr1e5/model/b_linear.py)
- [training_snapshots/EBM_e50_bs160_lr1e5/model/loss.py](code/snapshots/EBM_e50_bs160_lr1e5/model/loss.py)
- [training_snapshots/EBM_e50_bs160_lr1e5/model/__init__.py](code/snapshots/EBM_e50_bs160_lr1e5/model/__init__.py)
- [training_snapshots/EBM_e50_bs160_lr1e5/scheduler/cosine_warmup.py](code/snapshots/EBM_e50_bs160_lr1e5/scheduler/cosine_warmup.py)
- [training_snapshots/EBM_e50_bs160_lr1e5/scheduler/steplr.py](code/snapshots/EBM_e50_bs160_lr1e5/scheduler/steplr.py)
- [training_snapshots/EBM_e50_bs160_lr1e5/utils/ddp_util.py](code/snapshots/EBM_e50_bs160_lr1e5/utils/ddp_util.py)
- [training_snapshots/EBM_e50_bs160_lr1e5/utils/get_threshold.py](code/snapshots/EBM_e50_bs160_lr1e5/utils/get_threshold.py)
- [training_snapshots/EBM_e50_bs160_lr1e5/utils/__init__.py](code/snapshots/EBM_e50_bs160_lr1e5/utils/__init__.py)
- [training_snapshots/EBM_e50_bs160_lr1e5/data_processing/augmentation/augmentation.py](code/snapshots/EBM_e50_bs160_lr1e5/data_processing/augmentation/augmentation.py)
- [training_snapshots/EBM_e50_bs160_lr1e5/data_processing/augmentation/codec_aug.py](code/snapshots/EBM_e50_bs160_lr1e5/data_processing/augmentation/codec_aug.py)
- [training_snapshots/EBM_e50_bs160_lr1e5/data_processing/augmentation/raw_boost.py](code/snapshots/EBM_e50_bs160_lr1e5/data_processing/augmentation/raw_boost.py)
- [training_snapshots/EBM_e50_bs160_lr1e5/data_processing/augmentation/wav_read_functions.py](code/snapshots/EBM_e50_bs160_lr1e5/data_processing/augmentation/wav_read_functions.py)

코드의 파일별 원본 위치·SHA256은 [source_manifest.json](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [artifacts/analysis/ebm-performance-diagnosis-20261104.json](../../_shared/references/artifacts/analysis/ebm-performance-diagnosis-20261104.json)
- [docs/experiments/ebm_frozen_structure_audit_20261105.json](../../_shared/references/docs/experiments/ebm_frozen_structure_audit_20261105.json)
- [docs/experiments/ebm_frozen_all_results_20261105.md](../../_shared/references/docs/experiments/ebm_frozen_all_results_20261105.md)

## 연결

다른 관련 가설: [반복 입력과 학습·추론의 상호작용](../../02_repetition/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
