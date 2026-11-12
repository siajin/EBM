# E001. 배치 80·80epoch의 초기 반복 입력 학습

**실행 상태:** 학습 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

정상 중심과 정상 전체쌍 정렬로 미지 위조를 분리할 수 있으며 초기 반복 입력 모델을 후속 비교 기준으로 삼을 수 있다.

## 그렇게 설계한 이유

초기 학습 결과를 이후 배치·학습률 변경의 기준으로 보존한다.

대조와 통제의 근거: GPU 간 정상 특징 공유 없음. 정상 쌍 손실의 같은 발화 쌍 비중은 일반 배치에서 1/4다. XLS-R은 고정한다.

## 모델·데이터·표본

2019 LA 학습 25,380개. 두 GPU, 전체 배치 80 =GPU 별 정상 4 +위조 36. 일반 BPL은 GPU 마다 정상 4×4 =16쌍이다.

## 실험 내용과 상세 절차

1. 긴 입력 64,600샘플, 짧은 입력 48,440샘플을 원음에서 각각 생성한다. 긴 입력은 앞부분, 짧은 입력은 연속 무작위 구간이며 부족하면 순환 반복한다.
2. 각 입력에 독립 RawBoost를 적용하고 OCSoftmax +BPL로 본체, 정상 OCSoftmax로 기준 벡터를 순서대로 갱신한다.
3. seed 1, Adam, 학습률·최소 학습률 1e-6, weight decay1e-4, 80epoch를 사용한다. 한 epoch의 원래 반복 수는 323이다.
4. 보존 평가 기록은 15epoch부터 5epoch 간격이다. 80epoch 완료와 남아 있는 실제 평가 기록의 범위는 별도로 확인한다.

## 측정과 검증

19LA·21LA·21DF의 기록된 EER과 최소값·해당 epoch. 마지막 평가가 metric 파일에 없으면 해당 파일만으로 최종값을 만들지 않는다.

## 결과

보존 기록의 최소 EER은 19LA 0.280200%(70epoch),21LA 2.726782%(75),21DF 2.101303%(60)다.

## 해석·실패한 부분·한계

이후 실행은 학습률·학습 epoch·배치가 함께 달라진다. 단순 배치 크기 하나의 인과 효과로 비교할 수 없다. 과거 실제 코덱 적용 여부는 확정하지 못한다.

## 배운 점

배치뿐 아니라 학습률·epoch도 달랐으므로 초기 최소 EER을 후속 배치의 단일 원인 증거로 쓰지 않는다.

## 실험 코드

- [training_snapshots/EBM/arguments.py](code/snapshots/EBM/arguments.py)
- [training_snapshots/EBM/data_loaders.py](code/snapshots/EBM/data_loaders.py)
- [training_snapshots/EBM/docker_build.sh](code/snapshots/EBM/docker_build.sh)
- [training_snapshots/EBM/docker_run.sh](code/snapshots/EBM/docker_run.sh)
- [training_snapshots/EBM/main.py](code/snapshots/EBM/main.py)
- [training_snapshots/EBM/train.py](code/snapshots/EBM/train.py)
- [training_snapshots/EBM/dataset/ASVspoof2019_la_eval.py](code/snapshots/EBM/dataset/ASVspoof2019_la_eval.py)
- [training_snapshots/EBM/dataset/ASVspoof2019_la_train.py](code/snapshots/EBM/dataset/ASVspoof2019_la_train.py)
- [training_snapshots/EBM/dataset/ASVspoof2021_df_eval.py](code/snapshots/EBM/dataset/ASVspoof2021_df_eval.py)
- [training_snapshots/EBM/dataset/ASVspoof2021_la_eval.py](code/snapshots/EBM/dataset/ASVspoof2021_la_eval.py)
- [training_snapshots/EBM/dataset/__init__.py](code/snapshots/EBM/dataset/__init__.py)
- [training_snapshots/EBM/data_processing/__init__.py](code/snapshots/EBM/data_processing/__init__.py)
- [training_snapshots/EBM/logger/interface.py](code/snapshots/EBM/logger/interface.py)
- [training_snapshots/EBM/logger/local.py](code/snapshots/EBM/logger/local.py)
- [training_snapshots/EBM/logger/logger.py](code/snapshots/EBM/logger/logger.py)
- [training_snapshots/EBM/logger/neptune.py](code/snapshots/EBM/logger/neptune.py)
- [training_snapshots/EBM/logger/wandb.py](code/snapshots/EBM/logger/wandb.py)
- [training_snapshots/EBM/logger/__init__.py](code/snapshots/EBM/logger/__init__.py)
- [training_snapshots/EBM/model/b_linear.py](code/snapshots/EBM/model/b_linear.py)
- [training_snapshots/EBM/model/loss.py](code/snapshots/EBM/model/loss.py)
- [training_snapshots/EBM/model/__init__.py](code/snapshots/EBM/model/__init__.py)
- [training_snapshots/EBM/scheduler/cosine_warmup.py](code/snapshots/EBM/scheduler/cosine_warmup.py)
- [training_snapshots/EBM/scheduler/steplr.py](code/snapshots/EBM/scheduler/steplr.py)
- [training_snapshots/EBM/utils/ddp_util.py](code/snapshots/EBM/utils/ddp_util.py)
- [training_snapshots/EBM/utils/get_threshold.py](code/snapshots/EBM/utils/get_threshold.py)
- [training_snapshots/EBM/utils/__init__.py](code/snapshots/EBM/utils/__init__.py)
- [training_snapshots/EBM/data_processing/augmentation/augmentation.py](code/snapshots/EBM/data_processing/augmentation/augmentation.py)
- [training_snapshots/EBM/data_processing/augmentation/codec_aug.py](code/snapshots/EBM/data_processing/augmentation/codec_aug.py)
- [training_snapshots/EBM/data_processing/augmentation/raw_boost.py](code/snapshots/EBM/data_processing/augmentation/raw_boost.py)
- [training_snapshots/EBM/data_processing/augmentation/wav_read_functions.py](code/snapshots/EBM/data_processing/augmentation/wav_read_functions.py)

코드의 파일별 원본 위치·SHA256은 [source_manifest.json](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [artifacts/analysis/ebm-all-runs-comparison-20261104.json](../../_shared/references/artifacts/analysis/ebm-all-runs-comparison-20261104.json)
- [artifacts/analysis/ebm-performance-diagnosis-20261104.json](../../_shared/references/artifacts/analysis/ebm-performance-diagnosis-20261104.json)

## 연결

다른 관련 가설: [반복 입력과 학습·추론의 상호작용](../../02_repetition/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
