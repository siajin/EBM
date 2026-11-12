<a id="e021-정상위조-길이와-음성활동을-맞추고-독립-정상-묶음을-정렬한-학습"></a>

# 정상·위조 길이와 음성활동을 맞추고 독립 정상 묶음을 정렬한 학습

**실행 상태:** 사용자 중단:20epoch 저장

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

길이·음성활동·원음 방문률을 맞추고 독립 정상쌍을 학습하면 단음의 클래스 상관을 줄일 수 있다.

## 그렇게 설계한 이유

짧은 음성의 클래스 상관과 원음 미사용을 줄인 입력으로 OC +BPL 학습을 시험한다.

대조와 통제의 근거: OC 제거는 하지 않았다. 배치 240 고정 길이 반복 입력 학습 과는 입력 길이·쌍 선택·샘플러·원음 노출·음성활동 정책이 함께 달라진 복합 구성이다.

## 모델·데이터·표본

학습 자료 25,380개. 전체 OC 배치 240, GPU 별 정상 12 +위조 108. 추가 독립 정상 12개를 BPL 오른쪽에 사용해 GPU 별 파형 132개, 전체 264개다.

## 실험 내용과 상세 절차

1. 정상·위조 모두 0.5/1/1.5/2/2.5/3/3.5/4초 조각을 10/20/30/20/8/5/6/1% 목표 확률로 선택한다. 원음·길이·구간의 조건부 확률은 학습 자료에서만 추정한다.
2. 발화별 순환 반복을 하지 않고 유효 길이 마스크를 사용한다. 140개 클래스·화자·공격 그룹의 길이 정책은 [140개 클래스·화자·공격 그룹의 0.5~4초 공동 길이 균형](../E034/README.md), 음성활동 정책은 [목표 길이 안에서 정상·위조의 음성활동 비율까지 맞추는 정책](../E035/README.md), 매 epoch 원음을 빠짐없이 사용하는 추출 방법은 [매 epoch 모든 원음을 사용하면서 기존 기대 선택 비율을 유지하는 샘플러](../E037/README.md)을 따른다.
3. 각 epoch에 OC 정상·OC 위조·BPL 오른쪽 정상 세 stream 마다 모든 원음을 한 번 포함하고 부족 슬롯은 보정 확률로 추가한다.
4. BPL 왼쪽은 OC 정상 12개 의 같은 특징이고 오른쪽은 독립 정상 12개다. 손실은 기존 OC +BPL 이며 본체 →정상 기준 벡터의 두 단계 갱신을 유지한다.
5. seed 1·학습률 1e-5·50epoch 예정. 첫 실행은 1epoch 이후 평가 대상을 개발 →19LA로 바꾸기 위해 사용자 승인으로 중단했고, 새 실행은 같은 seed로 처음부터 시작했다.
6. 새 실행은 5epoch마다 먼저 저장하고 19LA 전체를 평가했다. 사용자가 높은 EER을 이유로 중단해 20epoch 완료,21epoch30/108에서 끝났다.

## 측정과 검증

매 epoch 원음 100%,144쌍/배치, 본체·중심 갱신 수와 유한 손실. 중간 19LA 5/10/15/20의 EER을 기록한다.

## 결과

19LA는 2.622022/2.927272/2.961711/2.708118%. 관측 최소는 5epoch2.622022%. 저장 모델 5/10/15/20 과 로그를 보존했으며 50epoch 완료가 아니다.

## 해석·실패한 부분·한계

이 구성의 LA 성능이 좋지 않았다는 근거다. 짧은 crop 하나의 독립 인과 효과, 길이 균형만의 효과를 식별하지 못한다. 첫 실행과 재시작을 독립 seed 반복으로 세지 않는다.

## 배운 점

복합 구성의 LA 성능은 나빴다. 길이 균형·짧은 crop·쌍 선택·노출 중 어느 변화가 원인인지는 분리하지 못했다.

## 실험 코드

- [소스: activity_policy.py](code/snapshots/ebm_coverage_e50_20261110_v1/activity_policy.py)
- [소스: arguments.py](code/snapshots/ebm_coverage_e50_20261110_v1/arguments.py)
- [소스: data_loaders.py](code/snapshots/ebm_coverage_e50_20261110_v1/data_loaders.py)
- [소스: main.py](code/snapshots/ebm_coverage_e50_20261110_v1/main.py)
- [소스: native_steps.py](code/snapshots/ebm_coverage_e50_20261110_v1/native_steps.py)
- [소스: positive_bpl.py](code/snapshots/ebm_coverage_e50_20261110_v1/positive_bpl.py)
- [소스: train.py](code/snapshots/ebm_coverage_e50_20261110_v1/train.py)
- [소스: ASVspoof2019_la_eval.py](code/snapshots/ebm_coverage_e50_20261110_v1/dataset/ASVspoof2019_la_eval.py)
- [소스: ASVspoof2019_la_train.py](code/snapshots/ebm_coverage_e50_20261110_v1/dataset/ASVspoof2019_la_train.py)
- [소스: ASVspoof2021_df_eval.py](code/snapshots/ebm_coverage_e50_20261110_v1/dataset/ASVspoof2021_df_eval.py)
- [소스: ASVspoof2021_la_eval.py](code/snapshots/ebm_coverage_e50_20261110_v1/dataset/ASVspoof2021_la_eval.py)
- [소스: __init__.py](code/snapshots/ebm_coverage_e50_20261110_v1/dataset/__init__.py)
- [소스: __init__.py](code/snapshots/ebm_coverage_e50_20261110_v1/data_processing/__init__.py)
- [소스: interface.py](code/snapshots/ebm_coverage_e50_20261110_v1/logger/interface.py)
- [소스: local.py](code/snapshots/ebm_coverage_e50_20261110_v1/logger/local.py)
- [소스: logger.py](code/snapshots/ebm_coverage_e50_20261110_v1/logger/logger.py)
- [소스: neptune.py](code/snapshots/ebm_coverage_e50_20261110_v1/logger/neptune.py)
- [소스: wandb.py](code/snapshots/ebm_coverage_e50_20261110_v1/logger/wandb.py)
- [소스: __init__.py](code/snapshots/ebm_coverage_e50_20261110_v1/logger/__init__.py)
- [소스: b_linear.py](code/snapshots/ebm_coverage_e50_20261110_v1/model/b_linear.py)
- [소스: loss.py](code/snapshots/ebm_coverage_e50_20261110_v1/model/loss.py)
- [소스: __init__.py](code/snapshots/ebm_coverage_e50_20261110_v1/model/__init__.py)
- [소스: cosine_warmup.py](code/snapshots/ebm_coverage_e50_20261110_v1/scheduler/cosine_warmup.py)
- [소스: steplr.py](code/snapshots/ebm_coverage_e50_20261110_v1/scheduler/steplr.py)
- [소스: gpu_positive_smoke.py](code/snapshots/ebm_coverage_e50_20261110_v1/tests/gpu_positive_smoke.py)
- [소스: test_activity_policy.py](code/snapshots/ebm_coverage_e50_20261110_v1/tests/test_activity_policy.py)
- [소스: test_positive_bpl.py](code/snapshots/ebm_coverage_e50_20261110_v1/tests/test_positive_bpl.py)
- [소스: test_source_coverage.py](code/snapshots/ebm_coverage_e50_20261110_v1/tests/test_source_coverage.py)
- [소스: analyze_train_crop_io.py](code/snapshots/ebm_coverage_e50_20261110_v1/tools/analyze_train_crop_io.py)
- [소스: build_activity_candidates.py](code/snapshots/ebm_coverage_e50_20261110_v1/tools/build_activity_candidates.py)
- [소스: check_coverage_training.py](code/snapshots/ebm_coverage_e50_20261110_v1/tools/check_coverage_training.py)
- [두 비반복 학습의 전체 저장점 평가 코드](code/snapshots/ebm_coverage_e50_20261110_v1/tools/evaluate_B_C_checkpoints.py)
- [소스: evaluate_coverage_comparison.py](code/snapshots/ebm_coverage_e50_20261110_v1/tools/evaluate_coverage_comparison.py)
- [소스: fit_activity_policy.py](code/snapshots/ebm_coverage_e50_20261110_v1/tools/fit_activity_policy.py)
- [소스: train_coverage_experiment.py](code/snapshots/ebm_coverage_e50_20261110_v1/tools/train_coverage_experiment.py)
- [소스: validate_activity_policy.py](code/snapshots/ebm_coverage_e50_20261110_v1/tools/validate_activity_policy.py)
- [소스: validate_source_coverage.py](code/snapshots/ebm_coverage_e50_20261110_v1/tools/validate_source_coverage.py)
- [소스: ddp_util.py](code/snapshots/ebm_coverage_e50_20261110_v1/utils/ddp_util.py)
- [소스: get_threshold.py](code/snapshots/ebm_coverage_e50_20261110_v1/utils/get_threshold.py)
- [소스: __init__.py](code/snapshots/ebm_coverage_e50_20261110_v1/utils/__init__.py)
- [소스: augmentation.py](code/snapshots/ebm_coverage_e50_20261110_v1/data_processing/augmentation/augmentation.py)
- [소스: codec_aug.py](code/snapshots/ebm_coverage_e50_20261110_v1/data_processing/augmentation/codec_aug.py)
- [소스: raw_boost.py](code/snapshots/ebm_coverage_e50_20261110_v1/data_processing/augmentation/raw_boost.py)
- [소스: wav_read_functions.py](code/snapshots/ebm_coverage_e50_20261110_v1/data_processing/augmentation/wav_read_functions.py)
- [소스: activity_policy.py](code/snapshots/ebm_coverage_e50_20261110_v2_mid19LA/activity_policy.py)
- [소스: arguments.py](code/snapshots/ebm_coverage_e50_20261110_v2_mid19LA/arguments.py)
- [소스: data_loaders.py](code/snapshots/ebm_coverage_e50_20261110_v2_mid19LA/data_loaders.py)
- [소스: main.py](code/snapshots/ebm_coverage_e50_20261110_v2_mid19LA/main.py)
- [소스: native_steps.py](code/snapshots/ebm_coverage_e50_20261110_v2_mid19LA/native_steps.py)
- [소스: positive_bpl.py](code/snapshots/ebm_coverage_e50_20261110_v2_mid19LA/positive_bpl.py)
- [소스: train.py](code/snapshots/ebm_coverage_e50_20261110_v2_mid19LA/train.py)
- [소스: ASVspoof2019_la_eval.py](code/snapshots/ebm_coverage_e50_20261110_v2_mid19LA/dataset/ASVspoof2019_la_eval.py)
- [소스: ASVspoof2019_la_train.py](code/snapshots/ebm_coverage_e50_20261110_v2_mid19LA/dataset/ASVspoof2019_la_train.py)
- [소스: ASVspoof2021_df_eval.py](code/snapshots/ebm_coverage_e50_20261110_v2_mid19LA/dataset/ASVspoof2021_df_eval.py)
- [소스: ASVspoof2021_la_eval.py](code/snapshots/ebm_coverage_e50_20261110_v2_mid19LA/dataset/ASVspoof2021_la_eval.py)
- [소스: __init__.py](code/snapshots/ebm_coverage_e50_20261110_v2_mid19LA/dataset/__init__.py)
- [소스: __init__.py](code/snapshots/ebm_coverage_e50_20261110_v2_mid19LA/data_processing/__init__.py)
- [소스: interface.py](code/snapshots/ebm_coverage_e50_20261110_v2_mid19LA/logger/interface.py)
- [소스: local.py](code/snapshots/ebm_coverage_e50_20261110_v2_mid19LA/logger/local.py)
- [소스: logger.py](code/snapshots/ebm_coverage_e50_20261110_v2_mid19LA/logger/logger.py)
- [소스: neptune.py](code/snapshots/ebm_coverage_e50_20261110_v2_mid19LA/logger/neptune.py)
- [소스: wandb.py](code/snapshots/ebm_coverage_e50_20261110_v2_mid19LA/logger/wandb.py)
- [소스: __init__.py](code/snapshots/ebm_coverage_e50_20261110_v2_mid19LA/logger/__init__.py)
- [소스: b_linear.py](code/snapshots/ebm_coverage_e50_20261110_v2_mid19LA/model/b_linear.py)
- [소스: loss.py](code/snapshots/ebm_coverage_e50_20261110_v2_mid19LA/model/loss.py)
- [소스: __init__.py](code/snapshots/ebm_coverage_e50_20261110_v2_mid19LA/model/__init__.py)
- [소스: cosine_warmup.py](code/snapshots/ebm_coverage_e50_20261110_v2_mid19LA/scheduler/cosine_warmup.py)
- [소스: steplr.py](code/snapshots/ebm_coverage_e50_20261110_v2_mid19LA/scheduler/steplr.py)
- [소스: gpu_positive_smoke.py](code/snapshots/ebm_coverage_e50_20261110_v2_mid19LA/tests/gpu_positive_smoke.py)
- [소스: test_activity_policy.py](code/snapshots/ebm_coverage_e50_20261110_v2_mid19LA/tests/test_activity_policy.py)
- [소스: test_intermediate_19la.py](code/snapshots/ebm_coverage_e50_20261110_v2_mid19LA/tests/test_intermediate_19la.py)
- [소스: test_positive_bpl.py](code/snapshots/ebm_coverage_e50_20261110_v2_mid19LA/tests/test_positive_bpl.py)
- [소스: test_source_coverage.py](code/snapshots/ebm_coverage_e50_20261110_v2_mid19LA/tests/test_source_coverage.py)
- [소스: analyze_train_crop_io.py](code/snapshots/ebm_coverage_e50_20261110_v2_mid19LA/tools/analyze_train_crop_io.py)
- [소스: build_activity_candidates.py](code/snapshots/ebm_coverage_e50_20261110_v2_mid19LA/tools/build_activity_candidates.py)
- [소스: check_coverage_training.py](code/snapshots/ebm_coverage_e50_20261110_v2_mid19LA/tools/check_coverage_training.py)
- [두 비반복 학습의 전체 저장점 평가 코드](code/snapshots/ebm_coverage_e50_20261110_v2_mid19LA/tools/evaluate_B_C_checkpoints.py)
- [소스: evaluate_coverage_comparison.py](code/snapshots/ebm_coverage_e50_20261110_v2_mid19LA/tools/evaluate_coverage_comparison.py)
- [소스: fit_activity_policy.py](code/snapshots/ebm_coverage_e50_20261110_v2_mid19LA/tools/fit_activity_policy.py)
- [소스: train_coverage_experiment.py](code/snapshots/ebm_coverage_e50_20261110_v2_mid19LA/tools/train_coverage_experiment.py)
- [소스: validate_activity_policy.py](code/snapshots/ebm_coverage_e50_20261110_v2_mid19LA/tools/validate_activity_policy.py)
- [소스: validate_source_coverage.py](code/snapshots/ebm_coverage_e50_20261110_v2_mid19LA/tools/validate_source_coverage.py)
- [소스: ddp_util.py](code/snapshots/ebm_coverage_e50_20261110_v2_mid19LA/utils/ddp_util.py)
- [소스: get_threshold.py](code/snapshots/ebm_coverage_e50_20261110_v2_mid19LA/utils/get_threshold.py)
- [소스: __init__.py](code/snapshots/ebm_coverage_e50_20261110_v2_mid19LA/utils/__init__.py)
- [소스: augmentation.py](code/snapshots/ebm_coverage_e50_20261110_v2_mid19LA/data_processing/augmentation/augmentation.py)
- [소스: codec_aug.py](code/snapshots/ebm_coverage_e50_20261110_v2_mid19LA/data_processing/augmentation/codec_aug.py)
- [소스: raw_boost.py](code/snapshots/ebm_coverage_e50_20261110_v2_mid19LA/data_processing/augmentation/raw_boost.py)
- [소스: wav_read_functions.py](code/snapshots/ebm_coverage_e50_20261110_v2_mid19LA/data_processing/augmentation/wav_read_functions.py)
- [소스: train_coverage_experiment.py](code/EBM/tools/train_coverage_experiment.py)

코드의 파일별 원본 위치·SHA256은 [원본 파일 위치와 해시 목록](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [상세 설계·실험 기록](../../../POSITIVE_BPL.md)
- [설정·결과 근거](../../_shared/references/artifacts/ebm_coverage_e50_mid19LA_20261110/experiment.json)
- [설정·결과 근거](../../_shared/references/artifacts/ebm_coverage_e50_mid19LA_20261110/user_stop_20261110.json)
- [설정·결과 근거](../../_shared/references/artifacts/ebm_coverage_e50_20261110/state.json)

## 연결

다른 관련 가설: [반복 입력과 학습·추론의 상호작용](../../02_repetition/README.md) · [정상 다양성·도메인 지지와 정상쌍 학습](../../03_bonafide_diversity/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
