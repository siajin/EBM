# E059. 기존 OC gradient·방향 정규화·정규화 전 특징 복원·변형 반응의 네 후속 분석

**실행 상태:** 캐시 계산·검산 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

gradient·잔여 크기·LayerNorm 복원을 함께 보면 후처리 전이 실패의 가능한 구조를 찾을 수 있다.

## 그렇게 설계한 이유

고정 후처리의 전이 한계를 실제 학습 신호와 표현 기하 관점에서 검사한다.

대조와 통제의 근거: 네 분석의 별칭 H3/H4/H5/H1은 근거 파일 이름에만 사용한다. 새 음성 forward·detectorfit0회다. affine 계수수치 복원은 새 검출기 학습이 아니다.

## 모델·데이터·표본

배치 160 모델의 학습 자료 25,380·개발 14,983·DF/21LA 진단 각 20,000, 총 80,363 고유 발화. 기존 [E058](../../05_residual_geometry/E058/README.md) 변형캐시와 저장 분류기를 재사용.

## 실험 내용과 상세 절차

1. 실제 PyTorch softplus의 threshold20 분기까지 맞춰 클래스 균형 OC 손실과 z·정적/입력별 가중치·거리 파라미터 gradient를 클래스별로 계산한다. 원래 학습 갱신은 하지 않는다.
2. 방향 정규화 민감도는 log ||u_changed −u_native ||를 log ||r_native ||·log ||x_changed −x_native ||·출처 × 클래스 절편으로 기술하고 화자 2,000회 재표집한다.
3. Train의 h/z 만으로 LayerNorm affine 계수를 정답 없이 복원하고 h를 평균·크기·정규화 방향의 상호 작용으로 재구성한다. 저장된 h 분류기의 예측 분해 일치를 검사한다.
4. 기존 평가 자료 내부 분류기를 새로 fit 하지 않고 20/10dB·Opus 특징에 적용해 기본 점수 대비 방향 이득이 유지되는지 본다.

## 측정과 검증

손실/gradient 분율·조건 보정 회귀·방향 크기별 민감도·복원 오차·저장 분류기 LL/EER. gradient 분율은 벡터 합의 기여 비율과 다르다.

## 결과

cleanTrain 위조의 균형 OC 손실 비중 2.2004%, 표본 z-gradient 평균 비중 1.0281%. 방향 정규화의 작은 잔여 크기 증폭과 h의 방향 × 스케일 상호 작용을 확인했다. DF 방향 이득은 변형에서 대체로 남았지만 21LA 10dB는 EER 이득이 역전됐다.

## 해석·실패한 부분·한계

원래 학습의 RawBoost·짧은 분기·BPL을 포함한 전체 gradient를 복원한 것이 아니다. 작은 잔여 방향 변화가 곧 부동 소수 점 오류나 새 gate의 효과라는 증명도 아니다.

## 배운 점

방향×스케일 상호작용과 작은잔여의 증폭이 남았다. clean OC gradient를 원래 증강+BPL 전체 gradient로 대체하면 안 된다.

## 실험 코드

- [artifacts/geometry_experiments/check_cached_hypotheses.py](code/artifacts/geometry_experiments/check_cached_hypotheses.py)

코드의 파일별 원본 위치·SHA256은 [source_manifest.json](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [docs/experiments/ebm_cache_hypotheses_20261106.md](../../_shared/references/docs/experiments/ebm_cache_hypotheses_20261106.md)
- [artifacts/geometry_experiments/check_cached_hypotheses.py](../../_shared/workspace/artifacts/geometry_experiments/check_cached_hypotheses.py)
- [artifacts/geometry_cache/local_analysis/cache_hypotheses_20261106_v2/protocol.json](../../_shared/references/artifacts/geometry_cache/local_analysis/cache_hypotheses_20261106_v2/protocol.json)
- [artifacts/geometry_cache/local_analysis/cache_hypotheses_20261106_v2/complete.json](../../_shared/references/artifacts/geometry_cache/local_analysis/cache_hypotheses_20261106_v2/complete.json)

## 연결

다른 관련 가설: [길이·유효 발화량과 앞뒤 저에너지 편향](../../01_duration_bias/README.md) · [정상 다양성·도메인 지지와 정상쌍 학습](../../03_bonafide_diversity/README.md) · [잔여 방향·좌표·거리·기하와 변형 반응](../../05_residual_geometry/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
