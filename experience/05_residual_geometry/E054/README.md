# E054. 원래 점수에 단조인 기본 식과 방향/잡음 잔차의 학습 자료 →평가 전이

**실행 상태:** 추가 분류기 학습·전이 평가 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

기존 점수에 단조인 기본식을 유지하면 잔여 방향의 전이 이득을 안정적으로 읽을 수 있다.

## 그렇게 설계한 이유

기본 점수 계수의 부호를 보장한 상태에서 방향 잔차가 실제로 전이되는지 본다.

대조와 통제의 근거: seed 3× 입력 5×ridge2의 30 후보 fit을 평가 전에 정한다. 내부 DF 교차검증과 다른 전이 실험이다.

## 모델·데이터·표본

학습 자료 25,380fit·개발 14,983 선택·DF/21LA 진단 각 20,000 평가. 배치 160 고정 특징.

## 실험 내용과 상세 절차

1. 기본 logit =(softplus(a)+1e-4)s +b +beta× 표준화 lognorm +eta× 방향 유효 표시로 정한다. norm 고정 시 s에 단조 증가다.
2. 조건은 기본 식만, 선형 방향 잔차, 동일 차원 선형 잡음 잔차,hidden16 방향 신경망 잔차, 동일 크기 잡음 신경망 잔차 5개다. 잔차 입력에는 s와 norm을 다시 넣지 않는다.
3. ridge1e-4/1e-2·seed 101/102/103·Adam lr0.001·최대 80epoch·10epochpatience를 쓴다. 모든 초기 잔차는 0이다.
4. 정규화 통계는 Train만 적합하고 개발 균형로그 손실에서 ridge/epoch를 선택한다.1e-8 이내 동률은 이른 epoch·큰 ridge 순이다.
5. 선택 후 두 평가 자료에 다시 fit 하지 않고 적용한다. 새로 개발 임계값을 계산한 추가 보고도 학습/모델 선택을 다시 하지 않는다.

## 측정과 검증

EER·AUROC·균형로그 손실·고정 개발 오류·10,000회 화자 paired 개별 95%구간.

## 결과

선형 방향 잔차의 DF 평균 LL은 0.563682 →0.561294로 소폭 개선했지만 EER 3.35 →3.40%로 악화.21LA EER 0.8%같음. 비영 잔차는 seed 101 에서만 선택됐다.

## 해석·실패한 부분·한계

로그 손실 개선과 EER 개선을 구별한다. 세 seed 에서 독립적으로 같은 개 선이 재현된 것이 아니며 개별 구간은 전체 탐색 보정이 아니다.

## 배운 점

로그손실의 작은 개선과 EER 악화가 공존했다. seed별 선택과 지표를 모두 보고해야 한다.

## 실험 코드

- [artifacts/geometry_experiments/frozen_probes.py](code/artifacts/geometry_experiments/frozen_probes.py)

코드의 파일별 원본 위치·SHA256은 [source_manifest.json](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [artifacts/geometry_experiments/frozen_probes.py](../../_shared/workspace/artifacts/geometry_experiments/frozen_probes.py)
- [artifacts/geometry_cache/local_analysis/frozen_all_v1/P3_P4/A/complete.json](../../_shared/references/artifacts/geometry_cache/local_analysis/frozen_all_v1/P3_P4/A/complete.json)
- [artifacts/geometry_cache/local_analysis/frozen_all_v1/P3_P4/A/DF_pilot/metrics.csv](../../_shared/references/artifacts/geometry_cache/local_analysis/frozen_all_v1/P3_P4/A/DF_pilot/metrics.csv)
- [docs/experiments/ebm_frozen_all_results_20261105.md](../../_shared/references/docs/experiments/ebm_frozen_all_results_20261105.md)

## 연결

다른 관련 가설: [코사인 외 norm과 정규화 전 통계](../../04_norm_information/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
