# E069. 위조 연결 경로 주변의 정상·위조 밀도와 빈 구간

**실행 상태:** 기하 밀도 분석 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

정상최근접 진입 부재가 관측위조의 밀도 지지까지 뜻하지는 않을 수 있다.

## 그렇게 설계한 이유

최근 접 정상이 없다는 것과 실제로 위조 표본이 지지하는 것을 구별한다.

대조와 통제의 근거: 경로마다 같은 가중치로 내부 길이 비중을 평균한다. 경로 중 몇 개가 완전히 빈지와 다른 통계다.

## 모델·데이터·표본

[E068](../E068/README.md)의 같은 캐시·잠근 연결 쌍. 각 클래스의 모든 관찰 표본을 반경 안 카운트에 사용한다.

## 실험 내용과 상세 절차

1. 최대 2,000개 씩의 클래스 균형 표본에서 혼합 5 번째 다른 이웃 거리의 중앙값을 기본 반경으로 정하고 0.5/1/2배를 모두 본다.
2. 연결의 위조 양 끝 두 개와 경로 양 끝 10%를 제외한다. 구면 호와 각 표본 반경 공의 진입·이탈 사건을 해석적으로 구해 연속 길이와 이웃 수를 적분한다.
3. 주변 정상 수 n_B/N_B, 위조 수 n_S/(N_S −2)를 계산해 위조 표본이 많은 효과를 보정한다.
4. 둘 다 0, 클래스 균형 환산 지지 5 미만, 정상 비율 ≥위조 2배, 위조 비율 ≥정상 2배, 나머지 혼재의 5 상태로 나눈다. 지지 기준 1/5/10도 보존한다.
5. 정상 참조 수 N_B, 위조 참조 수 N_S와 반경 안의 개수 n_B,n_S에 대해 b =n_B/N_B, f =n_S/(N_S −2)로 클래스별 밀도를 보정한다. N_B(b +f)<5는 희소 구간, 관측이 전혀 없으면 빈 구간으로 구분하고, 그 밖에서는 2배 우세 기준으로 정상 우세·위조 우세·혼합을 나눈다. 두 위조 끝 점은 참조에서 제외한다.

## 측정과 검증

반경별 5 상태 비중·평균/중앙 이웃 수·고유 tube 표본 수·반경 확대 단조성·독립 직접 거리 검증.

## 결과

기본 반경에서 DF 최근접 경로는 위조 우세 87.22%·둘 다 없음 1.44%,21LA는 89.52%·1.70%. 반경 0.5배에서는 빈 구간이 약 79~83%로 늘었다.

## 해석·실패한 부분·한계

반경·관찰 표본에 민감한 탐색 정의다. 관찰된 빈 구간이 실제 음성 분포의 공백이라는 증명은 아니다. 보간 학습의 효능도 시험하지 않았다.

## 배운 점

반경을 줄이면 빈경로가 크게 늘었다. 관측 밀도와 실제 음성분포의 지지를 구분해야 한다.

## 실험 코드

- [artifacts/geometry_experiments/spoof_path_density.py](code/artifacts/geometry_experiments/spoof_path_density.py)

코드의 파일별 원본 위치·SHA256은 [source_manifest.json](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [docs/experiments/spoof_path_density_20261105.md](../../_shared/references/docs/experiments/spoof_path_density_20261105.md)
- [artifacts/geometry_experiments/spoof_path_density.py](../../_shared/workspace/artifacts/geometry_experiments/spoof_path_density.py)
- [artifacts/geometry_cache/local_analysis/spoof_path_geometry_v1/path_density_v1/density_metrics.csv](../../_shared/references/artifacts/geometry_cache/local_analysis/spoof_path_geometry_v1/path_density_v1/density_metrics.csv)

## 연결

다른 관련 가설: [정상 다양성·도메인 지지와 정상쌍 학습](../../03_bonafide_diversity/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
