<a id="e068-위조-특징의-주축과-두-위조-사이-연속-경로의-정상-최근접-진입"></a>

# 위조 특징의 주축과 두 위조 사이 연속 경로의 정상 최근접 진입

**실행 상태:** 기하 분석·연속 검산 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

위조 사이 구면경로를 보면 정상분포와 연결되는 방향의 구조를 진단할 수 있다.

## 그렇게 설계한 이유

위조 표현이 타원 같은 주축을 갖는지, 위조끼리 연결한 중간 경로에 정상이 더 가까워지는지 본다.

대조와 통제의 근거: 최종 판정에는 위조 양 끝도 포함한다. 옆 방향 공간은 최종 코사인 성분을 제거한 공간이므로 전체 특징 겹침과 다르다.

## 모델·데이터·표본

배치 160 고정 모델의 학습 자료 25,380·개발 14,983·DF/21LA 진단 각 20,000. 전체 unit 특징 x와 점수를 제거 한 단위 옆 방향 u를 따로 분석한다.

## 실험 내용과 상세 절차

1. 각 자료·공간에서 위조만으로 표준화 없는 PCA를 적합하고 정상도 같은 축에 투영한다. 분산·주축각·상위좌표계수·공격 간/공격 내 분산을 계산한다.
2. 최근접·20 번째 이웃·점수 차 ≤0.02 인 근접 이웃·무작위·위조 PC1 하위 10%와 상위 10%끝 점 간 쌍을 각 600개 목표로 고정 난수 선택하고 중복 무방향 쌍을 제거한다.
3. 구면최단호의 양 끝 10%를 제외한 연속 구간에서 가장 가까운 정상 거리가 모든 관찰 위조 거리보다 작아지는 위치가 있는지 검사한다.
4. 초기 5 번째 이웃 반경 공 진입, 보수적 참조/반경 검사, 최종 전체 관찰 표본의 해석적 연속 최근접 판정까지 구분해 보존한다. 격자 중간 점만으로 부재를 선언하지 않는다.

## 측정과 검증

PC 분산·90%설명 차원·자료간축 내적·연속 경로 진입 수·실제 160 좌표반례 검증.

## 결과

진단 DF/21LA 옆 방향의 90%분산에는 37/32차원 이 필요했다. 최근 접 위조 경로는 정상 최근접 진입 0 건, 긴축 양 끝은 DF 전체 특징 13/600·옆 방향 1/600,21LA 11/600·0/600 이었다.

## 해석·실패한 부분·한계

실제 보간 음성의 정답이나 위조 경로 전체가 지지된다는 증명이 아니다. 화면 2차원 모양을 고 차원 전부로 읽지 않는다.

## 배운 점

최근접위조 경로는 정상최근접 진입이 없었지만 긴축 일부는 있었다. 화면의 타원 모양을 전체160차원으로 일반화하지 않는다.

## 실험 코드

- [소스: spoof_path_geometry.py](code/artifacts/geometry_experiments/spoof_path_geometry.py)
- [소스: continuous_spoof_path_labels.py](code/artifacts/geometry_experiments/continuous_spoof_path_labels.py)

코드의 파일별 원본 위치·SHA256은 [원본 파일 위치와 해시 목록](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [상세 설계·실험 기록](../../_shared/references/docs/experiments/spoof_path_geometry_20261105.md)
- [소스: spoof_path_geometry.py](../../_shared/workspace/artifacts/geometry_experiments/spoof_path_geometry.py)
- [소스: continuous_spoof_path_labels.py](../../_shared/workspace/artifacts/geometry_experiments/continuous_spoof_path_labels.py)
- [설정·결과 근거](../../_shared/references/artifacts/geometry_cache/local_analysis/spoof_path_geometry_v1/protocol.json)

## 연결

다른 관련 가설: [정상 다양성·도메인 지지와 정상쌍 학습](../../03_bonafide_diversity/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
