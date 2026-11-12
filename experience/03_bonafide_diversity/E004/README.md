<a id="e004-전체-배치-320으로-늘린-실제-학습-시도"></a>

# 전체 배치 320으로 늘린 실제 학습 시도

**실행 상태:** 실패: 첫 배치 메모리 부족

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

배치320의 더 많은 정상쌍을 원래 full-graph 구조로 처리할 수 있다.

## 그렇게 설계한 이유

배치 160 기준에서 배치만 320으로 늘려 원래 전체 연산 그래프 학습이 가능한지 확인한다.

대조와 통제의 근거: 미세 배치 분할·특징 재계산을 새로 넣지 않았다. 정상 BPL은 예정 GPU 별 16×16이다.

## 모델·데이터·표본

같은 학습 데이터. GPU 별 160 =정상 16 +위조 144. 예정 50epoch·81배치/epoch.

## 실험 내용과 상세 절차

1. 기준 보관 Python28 파일과 실행 소스의 일치를 확인한 후 전체 배치와 결과 이름만 바꾼다.
2. 원래 긴/짧은 분기를 한 연산 그래프로 실행한다. 나머지 설정은 50epoch·학습률 1e-5·seed 1로 유지한다.

## 측정과 검증

실제 첫 분기의 메모리 사용과 종료 코드·완료 학습 갱신·저장 모델 존재를 확인한다.

## 결과

첫 배치 short 분기의 특징 결합에서 226MiB 추가 할당에 실패했다. GPU 사용 23.46GiB, 잔여 175.62MiB 기록. 완료 갱신·epoch·체크포인트·EER 없음.

## 해석·실패한 부분·한계

성능이 나쁜 실험이 아니라 성능을 측정하지 못한 실행이다. 이를 배치 320의 EER 0 또는 열등한 EER로 넣지 않는다.

## 배운 점

첫 배치 OOM으로 가설의 성능 부분은 평가하지 못했다. 메모리 가능성과 EER의 성공·실패를 구분해야 한다.

## 실험 코드

- [소스: arguments.py](code/snapshots/EBM_e50_bs320_lr1e5/arguments.py)
- [소스: data_loaders.py](code/snapshots/EBM_e50_bs320_lr1e5/data_loaders.py)
- [training_snapshots/EBM_e50_bs320_lr1e5/docker_build.sh](code/snapshots/EBM_e50_bs320_lr1e5/docker_build.sh)
- [training_snapshots/EBM_e50_bs320_lr1e5/docker_run.sh](code/snapshots/EBM_e50_bs320_lr1e5/docker_run.sh)
- [소스: main.py](code/snapshots/EBM_e50_bs320_lr1e5/main.py)
- [소스: train.py](code/snapshots/EBM_e50_bs320_lr1e5/train.py)
- [소스: ASVspoof2019_la_eval.py](code/snapshots/EBM_e50_bs320_lr1e5/dataset/ASVspoof2019_la_eval.py)
- [소스: ASVspoof2019_la_train.py](code/snapshots/EBM_e50_bs320_lr1e5/dataset/ASVspoof2019_la_train.py)
- [소스: ASVspoof2021_df_eval.py](code/snapshots/EBM_e50_bs320_lr1e5/dataset/ASVspoof2021_df_eval.py)
- [소스: ASVspoof2021_la_eval.py](code/snapshots/EBM_e50_bs320_lr1e5/dataset/ASVspoof2021_la_eval.py)
- [소스: __init__.py](code/snapshots/EBM_e50_bs320_lr1e5/dataset/__init__.py)
- [소스: __init__.py](code/snapshots/EBM_e50_bs320_lr1e5/data_processing/__init__.py)
- [소스: interface.py](code/snapshots/EBM_e50_bs320_lr1e5/logger/interface.py)
- [소스: local.py](code/snapshots/EBM_e50_bs320_lr1e5/logger/local.py)
- [소스: logger.py](code/snapshots/EBM_e50_bs320_lr1e5/logger/logger.py)
- [소스: neptune.py](code/snapshots/EBM_e50_bs320_lr1e5/logger/neptune.py)
- [소스: wandb.py](code/snapshots/EBM_e50_bs320_lr1e5/logger/wandb.py)
- [소스: __init__.py](code/snapshots/EBM_e50_bs320_lr1e5/logger/__init__.py)
- [소스: b_linear.py](code/snapshots/EBM_e50_bs320_lr1e5/model/b_linear.py)
- [소스: loss.py](code/snapshots/EBM_e50_bs320_lr1e5/model/loss.py)
- [소스: __init__.py](code/snapshots/EBM_e50_bs320_lr1e5/model/__init__.py)
- [소스: cosine_warmup.py](code/snapshots/EBM_e50_bs320_lr1e5/scheduler/cosine_warmup.py)
- [소스: steplr.py](code/snapshots/EBM_e50_bs320_lr1e5/scheduler/steplr.py)
- [소스: ddp_util.py](code/snapshots/EBM_e50_bs320_lr1e5/utils/ddp_util.py)
- [소스: get_threshold.py](code/snapshots/EBM_e50_bs320_lr1e5/utils/get_threshold.py)
- [소스: __init__.py](code/snapshots/EBM_e50_bs320_lr1e5/utils/__init__.py)
- [소스: augmentation.py](code/snapshots/EBM_e50_bs320_lr1e5/data_processing/augmentation/augmentation.py)
- [소스: codec_aug.py](code/snapshots/EBM_e50_bs320_lr1e5/data_processing/augmentation/codec_aug.py)
- [소스: raw_boost.py](code/snapshots/EBM_e50_bs320_lr1e5/data_processing/augmentation/raw_boost.py)
- [소스: wav_read_functions.py](code/snapshots/EBM_e50_bs320_lr1e5/data_processing/augmentation/wav_read_functions.py)

코드의 파일별 원본 위치·SHA256은 [원본 파일 위치와 해시 목록](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- `MENTEE_SERVER.md` — 원자료/운영기록 경로; 제외 사유는 전체 출처 목록에 기록
- [설정·결과 근거](../../_shared/references/artifacts/analysis/ebm-all-runs-comparison-20261104.json)

## 연결

다른 관련 가설: 주 가설에서만 분류한다.

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
