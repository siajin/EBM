<a id="e026-원음-2초-이하-정상에만-30275초2초-손실을-더한-원래-240-학습"></a>

# 원음 2초 이하 정상에만 3.0275초·2초 손실을 더한 EBM-Repeat 학습

**실행 상태:** 사용자 중단:25epoch 저장·19LA 완료·후속21LA/DF 전체 평가 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

EBM-Repeat 모델을 유지하고 자연≤2초 정상에만3/2초 정렬을 추가하면 단음 정상의 길이 의존을 줄일 수 있다.

## 그렇게 설계한 이유

EBM-Repeat의 데이터로더·학습 방식을 유지하면서 짧은 정상에만 추가 길이 정렬 항을 시험한다.

대조와 통제의 근거: 정상 원음이 2초 이하이므로 추가 2초 입력은 원음 내용 전체를 포함하지만 증강과 반복은 유지된다. 4/2초 항·삼중 조합·화자별 추가 항·stop-gradient는 없다.

## 모델·데이터·표본

학습 자료 정상 2,580개 중 원음 ≤32,000샘플 109개만 추가 항 대상. 위조 22,800개는 추가 항 대상이 아니다. EBM-Repeat commit 10f3bcd의 보관 5 핵심 파일을 사용한다.

## 실험 내용과 상세 절차

1. 원래 두 DataLoader·DistributedSampler·원래 train 함수·긴 64,600/짧은 48,440 순환 반복·독립 RawBoost·codec-off를 그대로 사용한다.
2. 현재 정상 배치 행을 원래 sampler 순서와 대조해 대상 K개를 찾는다. 이미 증강된 3.0275초 파형 앞 32,000개로 2초 입력을 만든다. 별도 원음 재추출·새 RawBoost는 하지 않는다.
3. 고정 XLS-R로 2초 특징을 만들고 동일 분류기를 통과시킨다. 기존 4/3초 계산과 추가 3/2초 계산의 양쪽 특징에 기울기를 전달한다. 추가 난수는 별도 fork로 원래 학습 난수를 밀지 않는다.
4. 정상 수를 N, 그중 원음이 2초 이하인 대상 수를 K라 한다. A=BPL(4.0375초 전체 N개, 3.0275초 전체 N개), A_E=BPL(4.0375초 대상 K개, 3.0275초 전체 N개), B=BPL(3.0275초 대상 K개, 2초 대상 K개)로 두고, 총손실은 `OC + A − [K/(2N)]A_E + [K/(2N)]B`로 계산한다.
5. K=0이면 원래 `OC+BPL`과 같다. K=N이면 `OC+0.5BPL(4.0375초,3.0275초)+0.5BPL(3.0275초,2초)`가 된다. 0<K<N일 때는 대상에 해당하는 기존 긴 입력 행의 가중치만 절반으로 줄인다. 추가 항은 K×K 전체 교차 쌍이므로 K>1이면 서로 다른 원음도 비교한다.
6. 전체 배치 240·seed 1·학습률 1e-5·weight decay1e-4·처음부터 50epoch 예정.5epoch마다 모델과 optimizer 상태를 원자 저장·재로딩 검증한 뒤 19LA 전체를 평가한다. 학습 후 각 저장 epoch의 21LA →DF 평가가 예정돼 있다.

## 측정과 검증

서버 검사 57개, 두 GPU에서 원래 부분 배치까지 포함한 120/72/114개 입력의 세 논리 배치, 대상 수 0/전체/두 GPU 사이 불균형, 실제 4개 worker의 평가를 검증했다. 완료된 첫 5epoch 19LA 전체 71,237개 점수는 별도 ROC 계산으로 확인했다. 최신25epoch평가는 [짧은 정상 음성의 3.0275초·2초 정렬 학습 중단25epoch의21LA·DF 전체평가와 독립ROC 검산](../E113/README.md)에서 별도 기록한다. 50epoch완료·모든저장점21LA/DF평가 완료로 해석하지 않는다.

## 결과

50epoch 예정 중 사용자가25epoch 저장점 평가를 지정해 중단했다(run.exit143). 25epoch19LA EER1.2100611828687935%. 별도 고정25 전체평가의21LA EER11.308488302339532%, DF5.427399287107404%, run.exit0. 2026-11-12 독립 동점ROC·전체 index·UID유일성·class분모·유한점수·64600cap 검산 통과. 첫5epoch19LA2.437306%는 초기 이력으로 보존한다.

## 해석·실패한 부분·한계

첫 snapshot은 검사기의 Git 기준 경로 오류로 CPU 검사에서 멈춰 본 학습 갱신이 0회였다. 두 번째 snapshot에서 검사 경로를 수정해 시작했다. 이번 비교는 기존 4.0375초/3.0275초 항의 대상 행 가중치 감소와 3.0275초/2초 항 추가를 함께 바꿨으므로, 두 변화의 독립 효과를 분리한 실험은 아니다. 최신 완료25epoch는 EBM-NoRepeat/EBM-FullContext 25epoch 모델보다19LA·21LA·DF 모두 나쁘다. 같은 기본평가정책이지만 별도실행 난수이며, 한seed·다른학습구성의 비교다.

## 배운 점

25epoch 전체 평가에서 EBM-NoRepeat/EBM-FullContext 25epoch 모델보다 세 집합 모두 나빴다. 기존4/3 항 재가중과 새3/2 항 추가의 독립 효과는 분리되지 않았다.

## 실험 코드

- [소스: activity_policy.py](code/snapshots/ebm_short32_original240_e50_20261111_v1/activity_policy.py)
- [소스: arguments.py](code/snapshots/ebm_short32_original240_e50_20261111_v1/arguments.py)
- [실험 구현·검증 소스](code/snapshots/ebm_short32_original240_e50_20261111_v1/baseline240_mild.py)
- [소스: conditional_bpl2.py](code/snapshots/ebm_short32_original240_e50_20261111_v1/conditional_bpl2.py)
- [소스: conditional_bpl2_steps.py](code/snapshots/ebm_short32_original240_e50_20261111_v1/conditional_bpl2_steps.py)
- [소스: controlled_data_loaders.py](code/snapshots/ebm_short32_original240_e50_20261111_v1/controlled_data_loaders.py)
- [소스: data_loaders.py](code/snapshots/ebm_short32_original240_e50_20261111_v1/data_loaders.py)
- [소스: final_audit.py](code/snapshots/ebm_short32_original240_e50_20261111_v1/final_audit.py)
- [소스: main.py](code/snapshots/ebm_short32_original240_e50_20261111_v1/main.py)
- [실험 구현·검증 소스](code/snapshots/ebm_short32_original240_e50_20261111_v1/mild_duration.py)
- [소스: native_steps.py](code/snapshots/ebm_short32_original240_e50_20261111_v1/native_steps.py)
- [소스: positive_bpl.py](code/snapshots/ebm_short32_original240_e50_20261111_v1/positive_bpl.py)
- [실험 구현·검증 소스](code/snapshots/ebm_short32_original240_e50_20261111_v1/short32_bpl.py)
- [실험 구현·검증 소스](code/snapshots/ebm_short32_original240_e50_20261111_v1/short32_lifecycle.py)
- [실험 구현·검증 소스](code/snapshots/ebm_short32_original240_e50_20261111_v1/short32_original240.py)
- [소스: train.py](code/snapshots/ebm_short32_original240_e50_20261111_v1/train.py)
- [소스: wait_and_launch.py](code/snapshots/ebm_short32_original240_e50_20261111_v1/wait_and_launch.py)
- [소스: data_loaders.py](code/snapshots/ebm_short32_original240_e50_20261111_v1/baseline240_source/data_loaders.py)
- [소스: main.py](code/snapshots/ebm_short32_original240_e50_20261111_v1/baseline240_source/main.py)
- [설정·결과 근거](code/snapshots/ebm_short32_original240_e50_20261111_v1/baseline240_source/manifest.json)
- [소스: train.py](code/snapshots/ebm_short32_original240_e50_20261111_v1/baseline240_source/train.py)
- [소스: ASVspoof2019_la_eval.py](code/snapshots/ebm_short32_original240_e50_20261111_v1/dataset/ASVspoof2019_la_eval.py)
- [소스: ASVspoof2019_la_train.py](code/snapshots/ebm_short32_original240_e50_20261111_v1/dataset/ASVspoof2019_la_train.py)
- [소스: ASVspoof2021_df_eval.py](code/snapshots/ebm_short32_original240_e50_20261111_v1/dataset/ASVspoof2021_df_eval.py)
- [소스: ASVspoof2021_la_eval.py](code/snapshots/ebm_short32_original240_e50_20261111_v1/dataset/ASVspoof2021_la_eval.py)
- [소스: __init__.py](code/snapshots/ebm_short32_original240_e50_20261111_v1/dataset/__init__.py)
- [소스: __init__.py](code/snapshots/ebm_short32_original240_e50_20261111_v1/data_processing/__init__.py)
- [소스: interface.py](code/snapshots/ebm_short32_original240_e50_20261111_v1/logger/interface.py)
- [소스: local.py](code/snapshots/ebm_short32_original240_e50_20261111_v1/logger/local.py)
- [소스: logger.py](code/snapshots/ebm_short32_original240_e50_20261111_v1/logger/logger.py)
- [소스: neptune.py](code/snapshots/ebm_short32_original240_e50_20261111_v1/logger/neptune.py)
- [소스: wandb.py](code/snapshots/ebm_short32_original240_e50_20261111_v1/logger/wandb.py)
- [소스: __init__.py](code/snapshots/ebm_short32_original240_e50_20261111_v1/logger/__init__.py)
- [소스: b_linear.py](code/snapshots/ebm_short32_original240_e50_20261111_v1/model/b_linear.py)
- [소스: loss.py](code/snapshots/ebm_short32_original240_e50_20261111_v1/model/loss.py)
- [소스: __init__.py](code/snapshots/ebm_short32_original240_e50_20261111_v1/model/__init__.py)
- [소스: cosine_warmup.py](code/snapshots/ebm_short32_original240_e50_20261111_v1/scheduler/cosine_warmup.py)
- [소스: steplr.py](code/snapshots/ebm_short32_original240_e50_20261111_v1/scheduler/steplr.py)
- [실험 구현·검증 소스](code/snapshots/ebm_short32_original240_e50_20261111_v1/tests/test_baseline240_mild.py)
- [소스: test_conditional_bpl2.py](code/snapshots/ebm_short32_original240_e50_20261111_v1/tests/test_conditional_bpl2.py)
- [소스: test_positive_bpl.py](code/snapshots/ebm_short32_original240_e50_20261111_v1/tests/test_positive_bpl.py)
- [실험 구현·검증 소스](code/snapshots/ebm_short32_original240_e50_20261111_v1/tests/test_short32_bpl.py)
- [실험 구현·검증 소스](code/snapshots/ebm_short32_original240_e50_20261111_v1/tests/test_short32_lifecycle.py)
- [실험 구현·검증 소스](code/snapshots/ebm_short32_original240_e50_20261111_v1/tests/test_short32_runner.py)
- [실험 구현·검증 소스](code/snapshots/ebm_short32_original240_e50_20261111_v1/tools/run_short32_experiment.py)
- [실험 구현·검증 소스](code/snapshots/ebm_short32_original240_e50_20261111_v1/tools/train_baseline240_mild.py)
- [실험 구현·검증 소스](code/snapshots/ebm_short32_original240_e50_20261111_v1/tools/validate_short32.py)
- [소스: ddp_util.py](code/snapshots/ebm_short32_original240_e50_20261111_v1/utils/ddp_util.py)
- [소스: get_threshold.py](code/snapshots/ebm_short32_original240_e50_20261111_v1/utils/get_threshold.py)
- [소스: __init__.py](code/snapshots/ebm_short32_original240_e50_20261111_v1/utils/__init__.py)
- [소스: augmentation.py](code/snapshots/ebm_short32_original240_e50_20261111_v1/data_processing/augmentation/augmentation.py)
- [소스: codec_aug.py](code/snapshots/ebm_short32_original240_e50_20261111_v1/data_processing/augmentation/codec_aug.py)
- [소스: raw_boost.py](code/snapshots/ebm_short32_original240_e50_20261111_v1/data_processing/augmentation/raw_boost.py)
- [소스: wav_read_functions.py](code/snapshots/ebm_short32_original240_e50_20261111_v1/data_processing/augmentation/wav_read_functions.py)
- [소스: b_linear.py](code/snapshots/ebm_short32_original240_e50_20261111_v1/baseline240_source/model/b_linear.py)
- [소스: loss.py](code/snapshots/ebm_short32_original240_e50_20261111_v1/baseline240_source/model/loss.py)
- [소스: activity_policy.py](code/snapshots/ebm_short32_original240_e50_20261111_v2/activity_policy.py)
- [소스: arguments.py](code/snapshots/ebm_short32_original240_e50_20261111_v2/arguments.py)
- [실험 구현·검증 소스](code/snapshots/ebm_short32_original240_e50_20261111_v2/baseline240_mild.py)
- [소스: conditional_bpl2.py](code/snapshots/ebm_short32_original240_e50_20261111_v2/conditional_bpl2.py)
- [소스: conditional_bpl2_steps.py](code/snapshots/ebm_short32_original240_e50_20261111_v2/conditional_bpl2_steps.py)
- [소스: controlled_data_loaders.py](code/snapshots/ebm_short32_original240_e50_20261111_v2/controlled_data_loaders.py)
- [소스: data_loaders.py](code/snapshots/ebm_short32_original240_e50_20261111_v2/data_loaders.py)
- [소스: main.py](code/snapshots/ebm_short32_original240_e50_20261111_v2/main.py)
- [실험 구현·검증 소스](code/snapshots/ebm_short32_original240_e50_20261111_v2/mild_duration.py)
- [소스: native_steps.py](code/snapshots/ebm_short32_original240_e50_20261111_v2/native_steps.py)
- [소스: positive_bpl.py](code/snapshots/ebm_short32_original240_e50_20261111_v2/positive_bpl.py)
- [실험 구현·검증 소스](code/snapshots/ebm_short32_original240_e50_20261111_v2/short32_bpl.py)
- [실험 구현·검증 소스](code/snapshots/ebm_short32_original240_e50_20261111_v2/short32_lifecycle.py)
- [실험 구현·검증 소스](code/snapshots/ebm_short32_original240_e50_20261111_v2/short32_original240.py)
- [소스: train.py](code/snapshots/ebm_short32_original240_e50_20261111_v2/train.py)
- [소스: data_loaders.py](code/snapshots/ebm_short32_original240_e50_20261111_v2/baseline240_source/data_loaders.py)
- [소스: main.py](code/snapshots/ebm_short32_original240_e50_20261111_v2/baseline240_source/main.py)
- [설정·결과 근거](code/snapshots/ebm_short32_original240_e50_20261111_v2/baseline240_source/manifest.json)
- [소스: train.py](code/snapshots/ebm_short32_original240_e50_20261111_v2/baseline240_source/train.py)
- [소스: ASVspoof2019_la_eval.py](code/snapshots/ebm_short32_original240_e50_20261111_v2/dataset/ASVspoof2019_la_eval.py)
- [소스: ASVspoof2019_la_train.py](code/snapshots/ebm_short32_original240_e50_20261111_v2/dataset/ASVspoof2019_la_train.py)
- [소스: ASVspoof2021_df_eval.py](code/snapshots/ebm_short32_original240_e50_20261111_v2/dataset/ASVspoof2021_df_eval.py)
- [소스: ASVspoof2021_la_eval.py](code/snapshots/ebm_short32_original240_e50_20261111_v2/dataset/ASVspoof2021_la_eval.py)
- [소스: __init__.py](code/snapshots/ebm_short32_original240_e50_20261111_v2/dataset/__init__.py)
- [소스: __init__.py](code/snapshots/ebm_short32_original240_e50_20261111_v2/data_processing/__init__.py)
- [소스: interface.py](code/snapshots/ebm_short32_original240_e50_20261111_v2/logger/interface.py)
- [소스: local.py](code/snapshots/ebm_short32_original240_e50_20261111_v2/logger/local.py)
- [소스: logger.py](code/snapshots/ebm_short32_original240_e50_20261111_v2/logger/logger.py)
- [소스: neptune.py](code/snapshots/ebm_short32_original240_e50_20261111_v2/logger/neptune.py)
- [소스: wandb.py](code/snapshots/ebm_short32_original240_e50_20261111_v2/logger/wandb.py)
- [소스: __init__.py](code/snapshots/ebm_short32_original240_e50_20261111_v2/logger/__init__.py)
- [소스: b_linear.py](code/snapshots/ebm_short32_original240_e50_20261111_v2/model/b_linear.py)
- [소스: loss.py](code/snapshots/ebm_short32_original240_e50_20261111_v2/model/loss.py)
- [소스: __init__.py](code/snapshots/ebm_short32_original240_e50_20261111_v2/model/__init__.py)
- [소스: cosine_warmup.py](code/snapshots/ebm_short32_original240_e50_20261111_v2/scheduler/cosine_warmup.py)
- [소스: steplr.py](code/snapshots/ebm_short32_original240_e50_20261111_v2/scheduler/steplr.py)
- [실험 구현·검증 소스](code/snapshots/ebm_short32_original240_e50_20261111_v2/tests/test_baseline240_mild.py)
- [소스: test_conditional_bpl2.py](code/snapshots/ebm_short32_original240_e50_20261111_v2/tests/test_conditional_bpl2.py)
- [소스: test_positive_bpl.py](code/snapshots/ebm_short32_original240_e50_20261111_v2/tests/test_positive_bpl.py)
- [실험 구현·검증 소스](code/snapshots/ebm_short32_original240_e50_20261111_v2/tests/test_short32_bpl.py)
- [실험 구현·검증 소스](code/snapshots/ebm_short32_original240_e50_20261111_v2/tests/test_short32_lifecycle.py)
- [실험 구현·검증 소스](code/snapshots/ebm_short32_original240_e50_20261111_v2/tests/test_short32_runner.py)
- [실험 구현·검증 소스](code/snapshots/ebm_short32_original240_e50_20261111_v2/tools/run_short32_experiment.py)
- [실험 구현·검증 소스](code/snapshots/ebm_short32_original240_e50_20261111_v2/tools/train_baseline240_mild.py)
- [실험 구현·검증 소스](code/snapshots/ebm_short32_original240_e50_20261111_v2/tools/validate_short32.py)
- [소스: ddp_util.py](code/snapshots/ebm_short32_original240_e50_20261111_v2/utils/ddp_util.py)
- [소스: get_threshold.py](code/snapshots/ebm_short32_original240_e50_20261111_v2/utils/get_threshold.py)
- [소스: __init__.py](code/snapshots/ebm_short32_original240_e50_20261111_v2/utils/__init__.py)
- [소스: augmentation.py](code/snapshots/ebm_short32_original240_e50_20261111_v2/data_processing/augmentation/augmentation.py)
- [소스: codec_aug.py](code/snapshots/ebm_short32_original240_e50_20261111_v2/data_processing/augmentation/codec_aug.py)
- [소스: raw_boost.py](code/snapshots/ebm_short32_original240_e50_20261111_v2/data_processing/augmentation/raw_boost.py)
- [소스: wav_read_functions.py](code/snapshots/ebm_short32_original240_e50_20261111_v2/data_processing/augmentation/wav_read_functions.py)
- [소스: b_linear.py](code/snapshots/ebm_short32_original240_e50_20261111_v2/baseline240_source/model/b_linear.py)
- [소스: loss.py](code/snapshots/ebm_short32_original240_e50_20261111_v2/baseline240_source/model/loss.py)
- [실험 구현·검증 소스](code/EBM/tools/run_short32_experiment.py)
- [실험 구현·검증 소스](code/EBM/short32_original240.py)
- [실험 구현·검증 소스](code/EBM/short32_lifecycle.py)

코드의 파일별 원본 위치·SHA256은 [원본 파일 위치와 해시 목록](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [짧은 정상 음성의 길이 정렬 학습 안내](../../../SHORT32_ORIGINAL240.md)
- [실험 구현·검증 소스](../../../short32_original240.py)
- [실험 구현·검증 소스](../../../short32_lifecycle.py)
- [설정·결과 근거](../../_shared/references/artifacts/short32_train_20261111_v2/experiment.json)
- [설정·결과 근거](../../_shared/references/artifacts/paired_min_eer_20261111/current_eval5_validation.json)
- [설정·결과 근거](../../_shared/references/artifacts/experiment_methods_audit_20261111/server_inventory.json)
- [설정·결과 근거](../../_shared/references/artifacts/short32_train_20261111_v2/stop25_eval_status.json)
- [최신25epoch 전체점수 독립검산](evidence/independent_audit.json)
- [설정·결과 근거](evidence/metrics_21LA.json)
- [설정·결과 근거](evidence/metrics_21DF.json)
- [평가완료 표식](evidence/complete.json)

## 연결

다른 관련 가설: [반복 입력과 학습·추론의 상호작용](../../02_repetition/README.md) · [정상 다양성·도메인 지지와 정상쌍 학습](../../03_bonafide_diversity/README.md) · [잔여 방향·좌표·거리·기하와 변형 반응](../../05_residual_geometry/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
