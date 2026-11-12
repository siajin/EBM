# E052. 고정 특징 위에서 정적·입력별 좌표 가중치와 같은 크기 신경망 학습

**실행 상태:** 추가 모듈 학습·전이 평가 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

기존 표현의 좌표 가중치를 작게 학습하면 encoder 재학습 없이 전이를 개선할 수 있다.

## 그렇게 설계한 이유

원래 특징 추출기를 바꾸지 않고 마지막 160 좌표를 입력별로 조절하면 전이 성능이 나아지는지 본다.

대조와 통제의 근거: 정적 160 파라미터 대 입력별/잔차 신경망 5,296 파라미터. 추가 모듈 optimizer는 있지만 원래 EBM·XLS-R·기준 벡터는 동결이다.

## 모델·데이터·표본

학습 자료 25,380에서 추가 모듈만 학습, 개발 선택 14,983에서 epoch 선택. DF/21LA 진단 각 20,000과 전체 21LA에 적용. 원래 배치 160 모델·정상 기준 벡터 고정.

## 실험 내용과 상세 절차

1. 정적 가중치 160개는 g =1 +0.5tanh(v), 입력별 가중치는 160 →16 →160의 GELU 신경망으로 g(x)=1 +0.5tanh(F(x))를 만든다.
2. 두 경우 최종 점수는 normalize(g ⊙x)와 원래 c의 내적이다. g는 0.5~1.5 범위다.
3. 같은 파라미터 수 5,296개 의 대조 신경망은 x +0.5/sqrt(160)tanh(F(x))를 만들고 정규화한다. 변화 norm 상한 0.5를 검사한다.
4. seed 101/102/103, Adam lr0.001·weight decay0·명시적 변형 규제 0.01, 정상 16 +위조 144·162배치/epoch로 최대 80epoch 학습한다. 기존 BPL은 추가하지 않는다.
5. 모든 모듈은 처음에 항등 함수다. epoch0을 포함해 고정 학습 자료 확률 보정의 개발 균형로그 손실로 선택하고,1e-8 이내 동률은 이른 epoch,10epoch 무개선이면 중단한다.
6. 학습된 입력별 가중치·학습 자료 평균 가중치·가중치 없음·학습 자료 점수 10구간 ×norm5구간 내 다른 발화 가중치로 순환 교환한 조건도 계산한다.

## 측정과 검증

seed 별 EER·균형로그 손실·개발 고정 오류. DF 주 5 비교는 10,000회 paired 화자 재표집의 개별 99%로 family95%를 보수 통제한다. seed 예측 앙상블이 아닌 seed 별 차이 평균이다.

## 결과

정적은 epoch0, 입력 별은 epoch1, 잔차 신경망은 11/9/11epoch 선택. DF 평균 EER 3.350000 →3.342593%, 개선 구간[−0.054719,0.065027]%p로 0 포함.21LA 진단 EER 0.8%로 같다.

## 해석·실패한 부분·한계

변형 폭이 제한된 후처리와 기존 OC 목적의 결과다. 특징 추출기와 attention을 공동 재학습한 모든 방법을 기각한 것이 아니다. 가중치 교환은 실현 불가능한 중간 조합일 수 있다.

## 배운 점

선택된 입력별 가중치의 EER 이득은 확립되지 않았다. 후처리 실패가 모든 attention 공동학습의 실패는 아니다.

## 실험 코드

- [artifacts/geometry_experiments/frozen_heads.py](code/artifacts/geometry_experiments/frozen_heads.py)

코드의 파일별 원본 위치·SHA256은 [source_manifest.json](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [artifacts/geometry_experiments/frozen_heads.py](../../_shared/workspace/artifacts/geometry_experiments/frozen_heads.py)
- [artifacts/geometry_cache/local_analysis/frozen_all_v1/head_protocol_lock.json](../../_shared/references/artifacts/geometry_cache/local_analysis/frozen_all_v1/head_protocol_lock.json)
- [artifacts/geometry_cache/local_analysis/frozen_all_v1/head_metrics.csv](../../_shared/references/artifacts/geometry_cache/local_analysis/frozen_all_v1/head_metrics.csv)
- [artifacts/geometry_cache/local_analysis/frozen_all_v1/head_paired_effects.json](../../_shared/references/artifacts/geometry_cache/local_analysis/frozen_all_v1/head_paired_effects.json)

## 연결

다른 관련 가설: [정상 다양성·도메인 지지와 정상쌍 학습](../../03_bonafide_diversity/README.md) · [코사인 외 norm과 정규화 전 통계](../../04_norm_information/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
