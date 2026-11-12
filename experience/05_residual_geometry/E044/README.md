# E044. 기존 코사인에서 좌표를 제거하거나 1~80개만 남기는 실험

**실행 상태:** 캐시 점수 개입 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

일부 좌표를 가린 코사인의 성능을 보면 기존 점수의 좌표 의존을 진단할 수 있다.

## 그렇게 설계한 이유

고정 160차원 표현의 어떤 좌표 부분 집합이 현재 점수에 기여하는지 분석한다.

대조와 통제의 근거: 전체 160차원 encoder·LayerNorm을 먼저 계산한 후 점수 좌표만 가린다. 모델 차원·연산 량을 줄여 재학습한 실험이 아니다.

## 모델·데이터·표본

배치 160 고정 모델. 순위는 학습 자료 raw post-LayerNorm z, 개발·DF 진단에서 평가. 새 분류기 fit 없음.

## 실험 내용과 상세 절차

1. [E041](../../04_norm_information/E041/README.md)의 공격별 평균 Fisher 순위를 학습 자료에서만 고정한다. z와 정상 기준 벡터 c의 동일 좌표를 같이 남긴 뒤 각각 재정규화해 코사인을 계산한다.
2. 기본 제거 실험은 상위 8/16/40/80개 제거와 seed 200~219의 무작위 제거 20개다.
3. 후속 유지 실험은 80/70/60/50/40/30/20/10개, 이어 5/3/2개, 마지막 1개를 계산한다. 각 k 에서 상위·하위·중첩 무작위 20개를 모두 보고한다.
4. 1개 좌표에서는 수학적으로 sign(z_j)sign(c_j)의 두 점수만 가능하다. float 나눗셈 반올림으로 동점을 가짜 세분화하지 않고 모든 이진 운영점을 열거한다.
5. 1개 실험 v1/v2는 검수기의 ROC 끝 점/내부 점 누락을 발견해 중단했고 v3에서 검수기를 고쳤다.3개를 독립 성능 반복으로 세지 않는다.

## 측정과 검증

EER/AUROC·기존 개발 임계값 오류·점수 변화·화자 군집 2,000회 개별 95%구간. 무작위 20개 범위는 설계 변동이지 CI가 아니다.

## 결과

하위 10/5/3/2개 유지의 DFEER은 2.9444/2.9500/2.9222/2.9222%, 하위 1개는 4.345939%. 하위 2개 의 AUROC와 개발 EER은 전체보다 악화했다.

## 해석·실패한 부분·한계

특정 부분 집합의 낮은 EER은 좌표 대부분이 불필요하거나 모델 본질 차원이 2 라는 증거가 아니다. ROC 곡선이 교차하며 동등성/비열등성 검정은 하지 않았다.

## 배운 점

소수 좌표의 낮은 EER이 모델 본질 차원이나 연산 축소를 뜻하지 않는다. 동점 ROC와 곡선 교차도 중요하다.

## 실험 코드

- [artifacts/geometry_experiments/channel_keep_one.py](code/artifacts/geometry_experiments/channel_keep_one.py)
- [artifacts/geometry_experiments/channel_keep_sweep.py](code/artifacts/geometry_experiments/channel_keep_sweep.py)
- [artifacts/geometry_experiments/channel_keep_tiny_sweep.py](code/artifacts/geometry_experiments/channel_keep_tiny_sweep.py)

코드의 파일별 원본 위치·SHA256은 [source_manifest.json](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [artifacts/geometry_cache/local_analysis/channel_keep_sweep_v1/protocol_lock.json](../../_shared/references/artifacts/geometry_cache/local_analysis/channel_keep_sweep_v1/protocol_lock.json)
- [artifacts/geometry_cache/local_analysis/channel_keep_tiny_v1/protocol_lock.json](../../_shared/references/artifacts/geometry_cache/local_analysis/channel_keep_tiny_v1/protocol_lock.json)
- [artifacts/geometry_cache/local_analysis/channel_keep_one_v3/protocol_lock.json](../../_shared/references/artifacts/geometry_cache/local_analysis/channel_keep_one_v3/protocol_lock.json)
- [artifacts/geometry_experiments/channel_keep_one.py](../../_shared/workspace/artifacts/geometry_experiments/channel_keep_one.py)
- [docs/experiments/cached_core_20261105/02_channel_reduction_to2.md](../../_shared/references/docs/experiments/cached_core_20261105/02_channel_reduction_to2.md)

## 연결

다른 관련 가설: 주 가설에서만 분류한다.

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
