<a id="e064-정상-예시를-더-잘-대표하는-중심이동정규화-점수-7-종"></a>

# 정상 예시를 더 잘 대표하는 중심·이동·정규화 점수 7 종

**실행 상태:** 정상만 이용한 기준 추정·평가 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

새출처 정상을 더 잘 대표하는 평균·다중중심·점수보정이 오류를 줄일 수 있다.

## 그렇게 설계한 이유

새출처 정상을 기준에 넣는 것만으로 오류를 해결할 수 있는지 본다.

대조와 통제의 근거: DF 정상 라벨로 기준을 추정하는 내부 진단과 Train만 사용한 3 중심 전이는 구별한다. 위조 라벨은 기준 fit에 쓰지 않지만 성능 계산에는 쓴다.

## 모델·데이터·표본

배치 160 고정 모델·DF 진단 20,000개. 이전 저장 출처 × 화자 5fold를 재사용한다. 학습 자료 정상 2,580도 별도 참조한다.

## 실험 내용과 상세 절차

1. 원래 학습 정상 기준 벡터의 코사인 점수와, 학습 정상에서 k-means로 구한 세 평균 기준 벡터 중 최대 코사인 점수를 사용하는 방법을 비교한다. k-means의 seed는 11, 초기화 횟수 n_init는 10이다.
2. DF 각 바깥 trainfold의 정상만으로 전체 평균 중심, 출처별 평균 중심,3개 평균 중심을 추정하고 held-outfold에 적용한다.
3. 출처별 평균 이동은 unit(x −출처 정상 평균 +학습 자료 정상 평균)와 원래 c의 cosine 이다.
4. 출처별 점수 표준화는 clip(s)의 atanh를 취한 뒤 학습 fold의 출처 정상 평균·표준편차로 표준화한다. 출처별 방법은 평가 시 출처 metadata가 필요하다.
5. 7 방법 전체를 선택 없이 보고하고 화자 군집 2,000회 seed 5519로 pairedEER 차이를 계산한다.

## 측정과 검증

EER/AUROC·paired 차이·정상 train 수·fold 정합성.

## 결과

기준 3.35%, 학습 자료 3 중심 3.05%, DF 전체 정상 중심 3.144444%, 출처 중심 3.20%, 출처이동 5.00%, 출처 점수 표준화 3.60%, DF 3 중심 12.50%.

## 해석·실패한 부분·한계

실패한 방법도 모두 기록한다. DF 내부 라벨·출처를 쓴 수치를 새 미지 출처배포 성능으로 제시하지 않는다. 단순 기준 교체만으로 일관된 해결을 보이지 못했다.

## 배운 점

몇몇 평균은 소폭 좋아졌지만 다중중심·이동은 악화했다. target 정상으로맞춘 내부 진단을 독립 전이 성능으로 쓰지 않는다.

## 실험 코드

- [소스: test_normal_reference_mechanisms.py](code/artifacts/geometry_experiments/test_normal_reference_mechanisms.py)

코드의 파일별 원본 위치·SHA256은 [원본 파일 위치와 해시 목록](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [소스: test_normal_reference_mechanisms.py](../../_shared/workspace/artifacts/geometry_experiments/test_normal_reference_mechanisms.py)
- [설정·결과 근거](../../_shared/references/artifacts/geometry_cache/local_analysis/normal_cause_audit_v1/normal_reference_lock.json)
- [설정·결과 근거](../../_shared/references/artifacts/geometry_cache/local_analysis/normal_cause_audit_v1/normal_reference_metrics.json)

## 연결

다른 관련 가설: [잔여 방향·좌표·거리·기하와 변형 반응](../../05_residual_geometry/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
