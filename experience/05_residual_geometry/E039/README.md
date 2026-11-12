<a id="e039-고정-모델의-정규화-전후-특징-추출전체-평가-캐시-구축"></a>

# 고정 모델의 정규화 전후 특징 추출·전체 평가 캐시 구축

**실행 상태:** 추출·검증 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

동일 forward의h/z·점수·메타데이터를 잠근 캐시는 재추론 없이 다양한 기하 가설을 검사할 수 있다.

## 그렇게 설계한 이유

이후 분석이 같은 모델·같은 발화의 특징과 점수를 재사용하도록 기준 자료를 만든다.

대조와 통제의 근거: 작은 2,000개는 각 20,000개 안에,21LA 진단은 전체 21LA 안에 있다. DF 전체는 진단과 나머지의 서로소 합집합이다.

## 모델·데이터·표본

배치 160·50epoch 고정 모델. 학습 자료 25,380, 개발 선택 14,983, DF 진단 20,000을 먼저 추출하고,21LA 전체 148,176·개발 나머지 9,861·19LA 전체 71,237·DF 나머지 513,928을 확장했다.

## 실험 내용과 상세 절차

1. 원래 평가 전처리인 앞 64,600/부족하면 순환 반복을 유지하고 최종 선형 출력 h·LayerNorm 출력 z 각 160개 와 정상 기준 벡터를 저장한다.
2. 발화별 ID·라벨·원음 길이·패딩 비·공격·화자·출처·코덱을 특징 행에 연결한다. 기존 점수와 z의 정규화 내적을 대조한다.
3. 원래 1e-5 pooling 잡음의 발화별 seed를 고정해 단독·재실행·두 GPU QC128개를 확인한다.
4. 추출 완료 표식·행수·유한 값·각 파일 SHA 확인 후에만 완료 캐시로 등록한다. SSL 중간 전체 특징·원음·모델 파일은 로컬로 가져오지 않는다.

## 측정과 검증

기준 cosine·EER/AUROC 재계산, 파일·행 정렬·checkpoint/frontendrevision 확인.

## 결과

기준 EER은 DF 진단 3.35%,21LA 진단 0.80%,21LA 전체 1.053539%,19LA 전체 0.435177%, DF 전체 3.241828%, 개발 나머지 0.011542%였다.

## 해석·실패한 부분·한계

특징 추출은 검출기 재학습이 아니다. 과거 여러 로컬 해시 폴더는 같은 캐시의 재검산 버전일 수 있으므로 새 표본 실험으로 중복 계산하지 않는다.

## 배운 점

추출 완료·UID 정렬·모델 revision을 먼저 검산해야 한다. 겹치는 진단·전체 표본을 독립 반복으로 세지 않는다.

## 실험 코드

- [소스: frozen_r1_extract.py](code/artifacts/geometry_experiments/frozen_r1_extract.py)
- [소스: la21_extract.py](code/artifacts/geometry_experiments/la21_extract.py)
- [소스: geometry_cache.py](code/artifacts/geometry_experiments/geometry_cache.py)

코드의 파일별 원본 위치·SHA256은 [원본 파일 위치와 해시 목록](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [상세 설계·실험 기록](../../_shared/references/docs/experiments/21la_cache_20261105.md)
- [상세 설계·실험 기록](../../_shared/references/docs/experiments/ebm_frozen_all_results_20261105.md)
- [상세 설계·실험 기록](../../_shared/references/artifacts/geometry_experiments/README.md)
- [설정·결과 근거](../../_shared/references/artifacts/geometry_cache/catalog.json)

## 연결

다른 관련 가설: 주 가설에서만 분류한다.

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
