<a id="e027-가변-길이-입력마스크기울기-계산의-실제-gpu-가능성-점검"></a>

# 가변 길이 입력·마스크·기울기 계산의 실제 GPU 가능성 점검

**실행 상태:** 구현 검증 완료·성능 학습 아님

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

유효 길이 마스크와 정확한 gradient 처리를 적용하면 기존 모델도 원음 길이를 유지할 수 있다.

## 그렇게 설계한 이유

원음 길이를 유지한 입력을 기존 특징 추출기와 분류기가 실제로 처리할 수 있는지 확인한다.

대조와 통제의 근거: 마스크는 분류기 최종 평균뿐 아니라 각 층의 pooling 에도 적용한다. 반복을 단순 삭제한 것과 다르다.

## 모델·데이터·표본

학습 자료 25,380개 header와 보존 개발/평가 메타데이터. 배치 160 고정 모델로 실제 최소·중앙·최장·혼합 길이 입력을 사용했다.

## 실험 내용과 상세 절차

1. 모든 원음 header 에서 길이·표본화 주파수·채널을 확인하고 클래스와 길이 구간별 배치 가능성을 계산한다.
2. 가변 파형을 오른쪽 0으로 채우고 XLS-R attention mask와 모든 pooling 마스크를 전달한다. 같은 음성을 혼합 배치와 단독으로 처리해 최종 특징/점수를 대조한다.
3. 원음 긴 입력과 최대 3.0275초 짧은 입력을 서로 별도 XLS-R 통과시키고, 원래 OC +BPL의 역전파 및 정상 기준 벡터 단계의 기울기를 검사한다.
4. 실제 RawBoost 각종과 배치 메모리·최장 평가 29.312초 등을 검사한다. 별도 실제 optimizer 성능 학습은 시작하지 않는다.

## 측정과 검증

길이별 단독/배치 score 최대 오차 5.96e-8, 특징 차이·유한 gradient·frontend 불변·두 GPU 동일 검사. 뒤 EBM-NoRepeat와 EBM-FullContext 연결 검사는 별도 E006/E007에 기록한다.

## 결과

원 길이 입력·역전파 가능성을 확인했다. 샘플러 전수·최악 길이 메모리 검사도 있지만 최종 EER 개선 결과는 아니다.

## 해석·실패한 부분·한계

작은 검증 배치 통과를 전체 학습 메모리 상한이나 일반화 성능으로 대체하지 않는다. 초기 short303 설정은 짧은 입력 303 프레임, 추가 짧은 분류 손실 0, 반복 관련 추가 손실 0으로 arguments.py만 바꾼 준비 단계였다. 해당 검증 표식은 full_training_started =false 다. 이후 실제 50epoch 학습은 [학습 길이 상한을 유지하면서 짧은 음성 반복을 제거한 학습](../../02_repetition/E006/README.md)에서 별도로 기록한다.

## 배운 점

단독·혼합 배치 일치를 검산해야 하며 작은 smoke 통과가 장기 학습 가능성이나 성능 향상까지 보장하지 않는다.

## 실험 코드

- [소스: variable_length_sidecar.py](code/artifacts/geometry_experiments/variable_length_sidecar.py)
- [소스: variable_length_augmentation_sidecar.py](code/artifacts/geometry_experiments/variable_length_augmentation_sidecar.py)

코드의 파일별 원본 위치·SHA256은 [원본 파일 위치와 해시 목록](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [상세 설계·실험 기록](../../_shared/references/docs/experiments/ebm_variable_length_feasibility_20261106.md)
- [소스: variable_length_sidecar.py](../../_shared/workspace/artifacts/geometry_experiments/variable_length_sidecar.py)
- [소스: variable_length_augmentation_sidecar.py](../../_shared/workspace/artifacts/geometry_experiments/variable_length_augmentation_sidecar.py)
- [설정·결과 근거](../../_shared/references/artifacts/geometry_cache/local_analysis/variable_length_audit_20261106_v1/metadata_audit.json)
- [설정·결과 근거](../../_shared/references/artifacts/geometry_cache/local_analysis/native_short303_E1_20261106_v1/verification.json)

## 연결

다른 관련 가설: [반복 입력과 학습·추론의 상호작용](../../02_repetition/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
