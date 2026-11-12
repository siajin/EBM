<a id="e051-같은-정상-기준-각도에서-손실이-구별하지-않는-방향의-수학-검사"></a>

# 같은 정상 기준 각도에서 손실이 구별하지 않는 방향의 수학 검사

**실행 상태:** 합성 수치·미분 검사 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

중심각을 고정한 위조의 방위 회전은 기존OC와 정상BPL이 직접 구별하지 않을 수 있다.

## 그렇게 설계한 이유

새 방향 손실을 논하기 전에 기존 OCSoftmax와 정상 쌍 손실의 실제 불변성을 확인한다.

대조와 통제의 근거: 정상도 모두 함께 회전하는 것과 일부 정상만 회전하는 것을 구별한다. 후자는 정상 쌍 손실을 바꿀 수 있다.

## 모델·데이터·표본

실제 손실 코드와 160차원 합성 벡터. 이 항목은 새로운 음성 성능 실험이 아니다.

## 실험 내용과 상세 절차

1. 단위 벡터 x를 정상 기준 방향 성분 sc와 옆 성분 r로 나눈다.
2. 크기와 s를 유지 한 채 c에 수직인 공간에서 r만 회전시켜 클래스 가중 OCSoftmax 값이 같은지 float64로 검사한다.
3. x와 c에 모두 수직인 방위 방향의 방향 미분과 원래 코사인·단위유클리드 항등식을 대조한다.
4. 정상 쌍 손실은 정상 특징의 상호 각에 의존하지만 위조만 회전 하면 그 항에 들어가지 않음을 실제 행렬 정의로 확인한다.

## 측정과 검증

loss 차이·직 교성·수학적 gradient·동점/극단 점 경계 검수.

## 결과

정상 기준 각을 고정한 위조의 옆 방위는 원래 분류 손실이 따로 식별하지 않는 자유 도임을 확인했다. 후속 실제 OC 방위 미분 최대 값은 5.55e-17 이었다.

## 해석·실패한 부분·한계

모든 회전 벡터가 실제 음성 encoder 출력으로 실현된다는 증명은 아니다. 실제 판별 정보 손실·성능 저하·새 손실 이득을 직접 증명하지 않는다.

## 배운 점

손실의 자유도는 확인했다. 실제 음성에서의 정보 손실이나 새방향 손실의 성능 이득은 별도 검증이 필요하다.

## 실험 코드

- [소스: verify_spec_math.py](code/docs/experiments/verify_spec_math.py)
- [소스: verify_native_preserving_math.py](code/artifacts/geometry_experiments/verify_native_preserving_math.py)

코드의 파일별 원본 위치·SHA256은 [원본 파일 위치와 해시 목록](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [상세 설계·실험 기록](../../_shared/references/docs/experiments/ebm_frozen_all_results_20261105.md)
- [소스: verify_spec_math.py](../../_shared/workspace/docs/experiments/verify_spec_math.py)
- [설정·결과 근거](../../_shared/references/docs/experiments/spec_validation.json)
- [설정·결과 근거](../../_shared/references/artifacts/geometry_cache/local_analysis/native_preserving_v1/loss_geometry_audit.json)

## 연결

다른 관련 가설: [정상 다양성·도메인 지지와 정상쌍 학습](../../03_bonafide_diversity/README.md) · [코사인 외 norm과 정규화 전 통계](../../04_norm_information/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
