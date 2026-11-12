<a id="e003-배치-160에서-두-gpu의-정상-특징을-공유하는-학습"></a>

# 배치 160에서 두 GPU의 정상 특징을 공유하는 학습

**실행 상태:** 학습 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

GPU 사이 정상 특징을 공유하면 전체 배치를 늘리지 않고 더 다양한 정상 교차쌍을 학습할 수 있다.

## 그렇게 설계한 이유

전체 배치 크기를 늘리지 않고 정상 쌍을 GPU 사이까지 확장하면 어떤 결과가 나오는지 본다.

대조와 통제의 근거: 일반 배치의 BPL 쌍 수는 두 GPU 합산의 로컬 128쌍에서 전역 256쌍으로 증가한다. 메모리 큐·과거 특징을 사용하지 않는다.

## 모델·데이터·표본

[배치 160·50epoch·GPU 별 정상 쌍 학습](../E002/README.md)와 같은 데이터·배치·50epoch·학습률. 각 GPU의 정상 8개를 모아 정상 16개 대 16개,256쌍을 계산한다.

## 실험 내용과 상세 절차

1. BPL의 두 입력을 길이 1로 정규화한 후 gradient가 전달되는 분산 all-gather 로두 GPU의 특징을 모은다.
2. 정상 16×16의 동일 전역 BPL 평균을 각 GPU 에서 계산한다. 역전 파 합산과 DDP 평균을 고려해 BPL을 GPU 수로 다시 나누지 않는다.
3. 분류 손실은 각 GPU의 정상 8 +위조 72에 적용하고 나머지 두 단계 학습은 유지한다.
4. seed 1·50epoch·162배치/epoch를 사용한다. 마지막 정상 배치는 GPU 별 2개로 전역 4×4다.

## 측정과 검증

원본·단일 rank·두 GPU의 손실/입력 기울기/파라미터 기울기를 합성 기준과 대조한 뒤 실제 학습 EER을 기록했다.

## 결과

최소 EER은 19LA 0.255154%(25),21LA 0.992801%(35),21DF 2.333714%(25)다. [배치 160·50epoch·GPU 별 정상 쌍 학습](../E002/README.md) 대비 세 집합이 같은 방향으로 변하지 않았다.

## 해석·실패한 부분·한계

한 seed의 기록이다. 쌍 수만 늘린다고 모든 평가 집합이 개선되는 근거는 없다. 전역행렬을 GPU 별 독립 행렬과 같은 쌍 수로 세지 않는다.

## 배운 점

전역쌍 증가가 모든 집합의 일관된 개선으로 이어지지 않았다. gradient all-gather의 배율도 별도로 검산해야 한다.

## 실험 코드

- [소스: arguments.py](code/snapshots/EBM_e50_bs160_lr1e5_bpl_global/arguments.py)
- [소스: data_loaders.py](code/snapshots/EBM_e50_bs160_lr1e5_bpl_global/data_loaders.py)
- [training_snapshots/EBM_e50_bs160_lr1e5_bpl_global/docker_build.sh](code/snapshots/EBM_e50_bs160_lr1e5_bpl_global/docker_build.sh)
- [training_snapshots/EBM_e50_bs160_lr1e5_bpl_global/docker_run.sh](code/snapshots/EBM_e50_bs160_lr1e5_bpl_global/docker_run.sh)
- [소스: main.py](code/snapshots/EBM_e50_bs160_lr1e5_bpl_global/main.py)
- [소스: train.py](code/snapshots/EBM_e50_bs160_lr1e5_bpl_global/train.py)
- [소스: ASVspoof2019_la_eval.py](code/snapshots/EBM_e50_bs160_lr1e5_bpl_global/dataset/ASVspoof2019_la_eval.py)
- [소스: ASVspoof2019_la_train.py](code/snapshots/EBM_e50_bs160_lr1e5_bpl_global/dataset/ASVspoof2019_la_train.py)
- [소스: ASVspoof2021_df_eval.py](code/snapshots/EBM_e50_bs160_lr1e5_bpl_global/dataset/ASVspoof2021_df_eval.py)
- [소스: ASVspoof2021_la_eval.py](code/snapshots/EBM_e50_bs160_lr1e5_bpl_global/dataset/ASVspoof2021_la_eval.py)
- [소스: __init__.py](code/snapshots/EBM_e50_bs160_lr1e5_bpl_global/dataset/__init__.py)
- [소스: __init__.py](code/snapshots/EBM_e50_bs160_lr1e5_bpl_global/data_processing/__init__.py)
- [소스: interface.py](code/snapshots/EBM_e50_bs160_lr1e5_bpl_global/logger/interface.py)
- [소스: local.py](code/snapshots/EBM_e50_bs160_lr1e5_bpl_global/logger/local.py)
- [소스: logger.py](code/snapshots/EBM_e50_bs160_lr1e5_bpl_global/logger/logger.py)
- [소스: neptune.py](code/snapshots/EBM_e50_bs160_lr1e5_bpl_global/logger/neptune.py)
- [소스: wandb.py](code/snapshots/EBM_e50_bs160_lr1e5_bpl_global/logger/wandb.py)
- [소스: __init__.py](code/snapshots/EBM_e50_bs160_lr1e5_bpl_global/logger/__init__.py)
- [소스: b_linear.py](code/snapshots/EBM_e50_bs160_lr1e5_bpl_global/model/b_linear.py)
- [소스: loss.py](code/snapshots/EBM_e50_bs160_lr1e5_bpl_global/model/loss.py)
- [소스: __init__.py](code/snapshots/EBM_e50_bs160_lr1e5_bpl_global/model/__init__.py)
- [소스: cosine_warmup.py](code/snapshots/EBM_e50_bs160_lr1e5_bpl_global/scheduler/cosine_warmup.py)
- [소스: steplr.py](code/snapshots/EBM_e50_bs160_lr1e5_bpl_global/scheduler/steplr.py)
- [소스: ddp_util.py](code/snapshots/EBM_e50_bs160_lr1e5_bpl_global/utils/ddp_util.py)
- [소스: get_threshold.py](code/snapshots/EBM_e50_bs160_lr1e5_bpl_global/utils/get_threshold.py)
- [소스: __init__.py](code/snapshots/EBM_e50_bs160_lr1e5_bpl_global/utils/__init__.py)
- [소스: augmentation.py](code/snapshots/EBM_e50_bs160_lr1e5_bpl_global/data_processing/augmentation/augmentation.py)
- [소스: codec_aug.py](code/snapshots/EBM_e50_bs160_lr1e5_bpl_global/data_processing/augmentation/codec_aug.py)
- [소스: raw_boost.py](code/snapshots/EBM_e50_bs160_lr1e5_bpl_global/data_processing/augmentation/raw_boost.py)
- [소스: wav_read_functions.py](code/snapshots/EBM_e50_bs160_lr1e5_bpl_global/data_processing/augmentation/wav_read_functions.py)

코드의 파일별 원본 위치·SHA256은 [원본 파일 위치와 해시 목록](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [설정·결과 근거](../../_shared/references/artifacts/analysis/ebm-performance-diagnosis-20261104.json)
- [설정·결과 근거](../../_shared/references/artifacts/analysis/ebm-all-runs-comparison-20261104.json)
- `MENTEE_SERVER.md` — 원자료/운영기록 경로; 제외 사유는 전체 출처 목록에 기록

## 연결

다른 관련 가설: [반복 입력과 학습·추론의 상호작용](../../02_repetition/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
