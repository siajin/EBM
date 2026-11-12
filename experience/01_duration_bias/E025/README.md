<a id="e025-완만한-변형-캐시를-원래-240-학습에-연결한-두-실행"></a>

# 완만한 변형 캐시를 배치 240 고정 길이 반복 입력 학습 학습에 연결한 두 실행

**실행 상태:** 1차 진입 실패·2차 평가 오류로 중단

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

완만한 변형만 배치 240 고정 길이 반복 입력 학습 흐름에 연결하면 강한 가변 crop 학습과 다른 결과를 얻을 수 있다.

## 그렇게 설계한 이유

[최대 20% 절단·0.8~1.2배 속도 변형의 학습 자료 정책과 실제 변형 캐시](../E024/README.md)의 데이터 변형만 적용하고 배치 240 고정 길이 반복 입력 학습 학습 방법을 복원해 효과를 평가하려 했다.

대조와 통제의 근거: 실패 당시의 고정 소스, 로그, 약 10.2GB 변형 캐시는 보존했다. 두 번째 실행은 읽기 전용 심볼릭 링크로 첫 캐시를 재사용했다. 당시에는 5epoch 직후 모델 저장보다 평가가 먼저인 경로였으므로 저장 모델이 남지 않았다.

## 모델·데이터·표본

배치 240 고정 길이 반복 입력 학습 보관 코드 commit 10f3bcd와 [최대 20% 절단·0.8~1.2배 속도 변형의 학습 자료 정책과 실제 변형 캐시](../E024/README.md) 캐시. 전체 배치 240·seed 1·처음부터 20epoch 예정·codec-off·기존 OC +BPL.

## 실험 내용과 상세 절차

1. 캐시 변형 하나를 선택해 원래 DataLoader가 읽는 원음으로 연결한다. 그 뒤 원래 긴 64,600/짧은 48,440 절단·순환 반복·독립 RawBoost를 그대로 한다.
2. 기존 별도 정상/위조 sampler·작은 loader 순환·원래 train 함수·본체 →중심 두 단계 Adam을 복원한다. 가변 입력·독립 BPL 오른쪽 묶음은 사용하지 않는다.
3. 1차 실행은 캐시·CPU·worker 검사는 통과했지만 실제 진입부에서 미정의 torch 변수 호출로 실패했다. GPU 학습 갱신 0회다.
4. 수정 snapshot의 2차 실행은 실제 본 학습을 시작했고 5epoch 후 첫 19LA 평가 worker 에서 controlled_data_loaders 모듈을 못 찾는 오류가 발생했다. 사용자의 종료 승인으로 남은 해당 프로세스만 정리했다.

## 측정과 검증

실제 optimizer 갱신·학습 로그·평가 오류·저장 파일 존재를 확인한다. 검증 통과와 본 학습/평가 완료는 구별한다.

## 결과

2차는 5epoch 직후 멈췄으며 저장 CLS/TrainState가 없고 완료 19LA·21LA·DF EER도 없다. 성능 저하나 성능 개선 수치를 이 실행에 부여하지 않는다.

## 해석·실패한 부분·한계

첫 실패는 메모리 부족이 아니라 runner 진입 오류, 두 번째는 평가 import 오류다. 실패 실행을 20epoch 완료로 보고하지 않는다.

## 배운 점

runner 진입과 평가 import 오류로 EER을 얻지 못했다. 실제 __main__·다중worker 검사와 평가 전 checkpoint 저장이 필요하다.

## 실험 코드

- [소스: activity_policy.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v1/activity_policy.py)
- [소스: arguments.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v1/arguments.py)
- [실험 구현·검증 소스](code/snapshots/ebm_baseline240_mild_e20_20261111_v1/baseline240_mild.py)
- [소스: data_loaders.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v1/data_loaders.py)
- [소스: main.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v1/main.py)
- [실험 구현·검증 소스](code/snapshots/ebm_baseline240_mild_e20_20261111_v1/mild_duration.py)
- [소스: native_steps.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v1/native_steps.py)
- [소스: positive_bpl.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v1/positive_bpl.py)
- [소스: train.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v1/train.py)
- [소스: ASVspoof2019_la_eval.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v1/dataset/ASVspoof2019_la_eval.py)
- [소스: ASVspoof2019_la_train.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v1/dataset/ASVspoof2019_la_train.py)
- [소스: ASVspoof2021_df_eval.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v1/dataset/ASVspoof2021_df_eval.py)
- [소스: ASVspoof2021_la_eval.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v1/dataset/ASVspoof2021_la_eval.py)
- [소스: __init__.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v1/dataset/__init__.py)
- [소스: __init__.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v1/data_processing/__init__.py)
- [소스: interface.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v1/logger/interface.py)
- [소스: local.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v1/logger/local.py)
- [소스: logger.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v1/logger/logger.py)
- [소스: neptune.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v1/logger/neptune.py)
- [소스: wandb.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v1/logger/wandb.py)
- [소스: __init__.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v1/logger/__init__.py)
- [소스: b_linear.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v1/model/b_linear.py)
- [소스: loss.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v1/model/loss.py)
- [소스: __init__.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v1/model/__init__.py)
- [소스: cosine_warmup.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v1/scheduler/cosine_warmup.py)
- [소스: steplr.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v1/scheduler/steplr.py)
- [소스: gpu_positive_smoke.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v1/tests/gpu_positive_smoke.py)
- [소스: test_activity_policy.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v1/tests/test_activity_policy.py)
- [실험 구현·검증 소스](code/snapshots/ebm_baseline240_mild_e20_20261111_v1/tests/test_baseline240_mild.py)
- [소스: test_intermediate_19la.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v1/tests/test_intermediate_19la.py)
- [실험 구현·검증 소스](code/snapshots/ebm_baseline240_mild_e20_20261111_v1/tests/test_mild_duration.py)
- [소스: test_positive_bpl.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v1/tests/test_positive_bpl.py)
- [소스: test_source_coverage.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v1/tests/test_source_coverage.py)
- [소스: analyze_train_crop_io.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v1/tools/analyze_train_crop_io.py)
- [소스: build_activity_candidates.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v1/tools/build_activity_candidates.py)
- [실험 구현·검증 소스](code/snapshots/ebm_baseline240_mild_e20_20261111_v1/tools/build_mild_duration_policy.py)
- [실험 구현·검증 소스](code/snapshots/ebm_baseline240_mild_e20_20261111_v1/tools/check_baseline240_mild.py)
- [소스: check_coverage_training.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v1/tools/check_coverage_training.py)
- [실험 구현·검증 소스](code/snapshots/ebm_baseline240_mild_e20_20261111_v1/tools/check_mild_training.py)
- [소스: compare_coverage_epoch20.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v1/tools/compare_coverage_epoch20.py)
- [두 비반복 학습의 전체 저장점 평가 코드](code/snapshots/ebm_baseline240_mild_e20_20261111_v1/tools/evaluate_B_C_checkpoints.py)
- [소스: evaluate_coverage_checkpoint.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v1/tools/evaluate_coverage_checkpoint.py)
- [소스: evaluate_coverage_comparison.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v1/tools/evaluate_coverage_comparison.py)
- [소스: evaluate_original240_reference.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v1/tools/evaluate_original240_reference.py)
- [실험 구현·검증 소스](code/snapshots/ebm_baseline240_mild_e20_20261111_v1/tools/finish_baseline240_mild.py)
- [실험 구현·검증 소스](code/snapshots/ebm_baseline240_mild_e20_20261111_v1/tools/finish_mild_experiment.py)
- [소스: fit_activity_policy.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v1/tools/fit_activity_policy.py)
- [실험 구현·검증 소스](code/snapshots/ebm_baseline240_mild_e20_20261111_v1/tools/prepare_mild_audio.py)
- [실험 구현·검증 소스](code/snapshots/ebm_baseline240_mild_e20_20261111_v1/tools/queue_mild_experiment.py)
- [실험 구현·검증 소스](code/snapshots/ebm_baseline240_mild_e20_20261111_v1/tools/train_baseline240_mild.py)
- [소스: train_coverage_experiment.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v1/tools/train_coverage_experiment.py)
- [실험 구현·검증 소스](code/snapshots/ebm_baseline240_mild_e20_20261111_v1/tools/train_mild_duration.py)
- [소스: validate_activity_policy.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v1/tools/validate_activity_policy.py)
- [소스: validate_source_coverage.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v1/tools/validate_source_coverage.py)
- [소스: ddp_util.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v1/utils/ddp_util.py)
- [소스: get_threshold.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v1/utils/get_threshold.py)
- [소스: __init__.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v1/utils/__init__.py)
- [소스: augmentation.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v1/data_processing/augmentation/augmentation.py)
- [소스: codec_aug.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v1/data_processing/augmentation/codec_aug.py)
- [소스: raw_boost.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v1/data_processing/augmentation/raw_boost.py)
- [소스: wav_read_functions.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v1/data_processing/augmentation/wav_read_functions.py)
- [소스: activity_policy.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v2/activity_policy.py)
- [소스: arguments.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v2/arguments.py)
- [실험 구현·검증 소스](code/snapshots/ebm_baseline240_mild_e20_20261111_v2/baseline240_mild.py)
- [소스: data_loaders.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v2/data_loaders.py)
- [소스: main.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v2/main.py)
- [실험 구현·검증 소스](code/snapshots/ebm_baseline240_mild_e20_20261111_v2/mild_duration.py)
- [소스: native_steps.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v2/native_steps.py)
- [소스: positive_bpl.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v2/positive_bpl.py)
- [소스: train.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v2/train.py)
- [소스: ASVspoof2019_la_eval.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v2/dataset/ASVspoof2019_la_eval.py)
- [소스: ASVspoof2019_la_train.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v2/dataset/ASVspoof2019_la_train.py)
- [소스: ASVspoof2021_df_eval.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v2/dataset/ASVspoof2021_df_eval.py)
- [소스: ASVspoof2021_la_eval.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v2/dataset/ASVspoof2021_la_eval.py)
- [소스: __init__.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v2/dataset/__init__.py)
- [소스: __init__.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v2/data_processing/__init__.py)
- [소스: interface.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v2/logger/interface.py)
- [소스: local.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v2/logger/local.py)
- [소스: logger.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v2/logger/logger.py)
- [소스: neptune.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v2/logger/neptune.py)
- [소스: wandb.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v2/logger/wandb.py)
- [소스: __init__.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v2/logger/__init__.py)
- [소스: b_linear.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v2/model/b_linear.py)
- [소스: loss.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v2/model/loss.py)
- [소스: __init__.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v2/model/__init__.py)
- [소스: cosine_warmup.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v2/scheduler/cosine_warmup.py)
- [소스: steplr.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v2/scheduler/steplr.py)
- [소스: gpu_positive_smoke.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v2/tests/gpu_positive_smoke.py)
- [소스: test_activity_policy.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v2/tests/test_activity_policy.py)
- [실험 구현·검증 소스](code/snapshots/ebm_baseline240_mild_e20_20261111_v2/tests/test_baseline240_mild.py)
- [소스: test_intermediate_19la.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v2/tests/test_intermediate_19la.py)
- [실험 구현·검증 소스](code/snapshots/ebm_baseline240_mild_e20_20261111_v2/tests/test_mild_duration.py)
- [소스: test_positive_bpl.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v2/tests/test_positive_bpl.py)
- [소스: test_source_coverage.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v2/tests/test_source_coverage.py)
- [소스: analyze_train_crop_io.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v2/tools/analyze_train_crop_io.py)
- [소스: build_activity_candidates.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v2/tools/build_activity_candidates.py)
- [실험 구현·검증 소스](code/snapshots/ebm_baseline240_mild_e20_20261111_v2/tools/build_mild_duration_policy.py)
- [실험 구현·검증 소스](code/snapshots/ebm_baseline240_mild_e20_20261111_v2/tools/check_baseline240_mild.py)
- [소스: check_coverage_training.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v2/tools/check_coverage_training.py)
- [실험 구현·검증 소스](code/snapshots/ebm_baseline240_mild_e20_20261111_v2/tools/check_mild_training.py)
- [소스: compare_coverage_epoch20.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v2/tools/compare_coverage_epoch20.py)
- [두 비반복 학습의 전체 저장점 평가 코드](code/snapshots/ebm_baseline240_mild_e20_20261111_v2/tools/evaluate_B_C_checkpoints.py)
- [소스: evaluate_coverage_checkpoint.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v2/tools/evaluate_coverage_checkpoint.py)
- [소스: evaluate_coverage_comparison.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v2/tools/evaluate_coverage_comparison.py)
- [소스: evaluate_original240_reference.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v2/tools/evaluate_original240_reference.py)
- [실험 구현·검증 소스](code/snapshots/ebm_baseline240_mild_e20_20261111_v2/tools/finish_baseline240_mild.py)
- [실험 구현·검증 소스](code/snapshots/ebm_baseline240_mild_e20_20261111_v2/tools/finish_mild_experiment.py)
- [소스: fit_activity_policy.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v2/tools/fit_activity_policy.py)
- [실험 구현·검증 소스](code/snapshots/ebm_baseline240_mild_e20_20261111_v2/tools/prepare_mild_audio.py)
- [실험 구현·검증 소스](code/snapshots/ebm_baseline240_mild_e20_20261111_v2/tools/queue_mild_experiment.py)
- [실험 구현·검증 소스](code/snapshots/ebm_baseline240_mild_e20_20261111_v2/tools/train_baseline240_mild.py)
- [소스: train_coverage_experiment.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v2/tools/train_coverage_experiment.py)
- [실험 구현·검증 소스](code/snapshots/ebm_baseline240_mild_e20_20261111_v2/tools/train_mild_duration.py)
- [소스: validate_activity_policy.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v2/tools/validate_activity_policy.py)
- [소스: validate_source_coverage.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v2/tools/validate_source_coverage.py)
- [소스: ddp_util.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v2/utils/ddp_util.py)
- [소스: get_threshold.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v2/utils/get_threshold.py)
- [소스: __init__.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v2/utils/__init__.py)
- [소스: augmentation.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v2/data_processing/augmentation/augmentation.py)
- [소스: codec_aug.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v2/data_processing/augmentation/codec_aug.py)
- [소스: raw_boost.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v2/data_processing/augmentation/raw_boost.py)
- [소스: wav_read_functions.py](code/snapshots/ebm_baseline240_mild_e20_20261111_v2/data_processing/augmentation/wav_read_functions.py)
- [실험 구현·검증 소스](code/EBM/tools/train_baseline240_mild.py)

코드의 파일별 원본 위치·SHA256은 [원본 파일 위치와 해시 목록](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [설정·결과 근거](../../_shared/references/artifacts/mild_duration_baseline240_20261111/state.json)
- [설정·결과 근거](../../_shared/references/artifacts/mild_duration_baseline240_20261111_v2/state.json)
- [설정·결과 근거](../../_shared/references/artifacts/mild_duration_baseline240_20261111_v2/user_stop_20261111_0956.json)
- [설정·결과 근거](../../_shared/references/artifacts/mild_duration_baseline240_20261111_v2/experiment.json)

## 연결

다른 관련 가설: [반복 입력과 학습·추론의 상호작용](../../02_repetition/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
