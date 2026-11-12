# E015. 자연적으로 2초 이하인 음성에서 오류 전이·임계값 이동·공격별 회복 분석

**실행 상태:** [E014](../E014/README.md) 저장 점수 분석 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

반복 후 짧은 정상 오탐 증가는 점수 임계값 이동과 일부 공격의 위조 회복이 함께 만든 현상일 수 있다.

## 그렇게 설계한 이유

반복 후 정상 오탐 증가와 위조 누락 감소를 한꺼번에 확인하고 특정 공격의 기여를 분리한다.

대조와 통제의 근거: 후속 분석의 ≤2초는 정확한 정수 샘플 경계다. 임계값은 개발 자료에서만 정한다. 공격은 평가 metadata의 생성 방식 ID 다.

## 모델·데이터·표본

[E014](../E014/README.md)의 21LA 전체 점수. 원음이 32,000샘플 이하인 정상 571개·위조 49,193개를 분석한다. 원음이 정확히 2초인 정상 64개·위조 497개가 포함된다. 따라서 앞선 2초 미만 분석의 정상 507개·위조 48,696개와 분모가 다르다.

## 실험 내용과 상세 절차

1. 각 모델·입력에서 개발 EER 근사 임계값을 구한다. 높은 점수를 정상으로 수락한다.
2. 비반복 입력에서 정한 개발 임계값을 두 조건에 그대로 적용한 표와, 각 입력 조건에서 따로 정한 개발 임계값을 적용한 표를 만든다.
3. ≤2초와>2초, 정상/위조, 공격 × 코덱별 오류 수·분모·새로 틀림·회복을 합산한다.
4. 같은 길이 안의 EER도 따로 계산해 전체 임계값의 오류와 구별한다. A10·A11 누락 감소는 전체 위조 누락 감소를 분모로 계산한다.

## 측정과 검증

원래 240 native 개발 임계값 0.9927267432, 반복 개발 임계값 0.3411039114. 정상 거부·위조 누락을 각각 계산한다.

## 결과

짧은 정상 거부 27 →88/571(같은 임계값), 반복 기준으로 재설정하면 34/571. 위조 누락 10,717 →1/49,193(같은 임계값), 재설정하면 451/49,193. 전체 위조 누락 감소의 78.6736%는 A10·A11에서 발생했다.

## 해석·실패한 부분·한계

오탐 증가의 상당 부분은 임계값 이동에 민감하다. 오류 수 감소의 78.7%를 EER 개선의 78.7% 또는 학습 원인의 비율로 바꾸지 않는다.

## 배운 점

같은 임계값과 재보정 임계값의 결론이 다르다. A10·A11의 오류 기여율을 EER의 인과 기여율로 바꿀 수 없다.

## 실험 코드

- [EBM/tools/review_repetition_results.py](code/EBM/tools/review_repetition_results.py)

코드의 파일별 원본 위치·SHA256은 [source_manifest.json](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [EBM/tools/review_repetition_results.py](../../../tools/review_repetition_results.py)
- [artifacts/repetition_shortcut_20261111/reviewed_analysis/operating_points_21LA.csv](../../_shared/references/artifacts/repetition_shortcut_20261111/reviewed_analysis/operating_points_21LA.csv)
- [artifacts/repetition_shortcut_20261111/reviewed_analysis/paired_transitions.csv](../../_shared/references/artifacts/repetition_shortcut_20261111/reviewed_analysis/paired_transitions.csv)
- [artifacts/repetition_shortcut_20261111/reviewed_analysis/attacks_21LA.csv](../../_shared/references/artifacts/repetition_shortcut_20261111/reviewed_analysis/attacks_21LA.csv)
- [artifacts/repetition_shortcut_20261111/reviewed_analysis/length_specific_eer.csv](../../_shared/references/artifacts/repetition_shortcut_20261111/reviewed_analysis/length_specific_eer.csv)

## 연결

다른 관련 가설: [길이·유효 발화량과 앞뒤 저에너지 편향](../../01_duration_bias/README.md) · [정상 다양성·도메인 지지와 정상쌍 학습](../../03_bonafide_diversity/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
