<a id="e073-파형을-먼저-반복하는-경우와-추출한-특징을-뒤에서-반복하는-경우의-8조건"></a>

# 파형을 먼저 반복하는 경우와 추출한 특징을 뒤에서 반복하는 경우의 8조건

**실행 상태:** 실제 내부 개입·검산 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

backend길이를 같게 해도 파형반복과후기 특징복제가 다르면 frontend문맥이 중요할 수 있다.

## 그렇게 설계한 이유

분류기 입력 프레임 수를 같게한 채 XLS-R이 반복 문맥을 봤는지의 차이를 검사한다.

대조와 통제의 근거: 층 0은 첫 hidden state,1~24는 encoder 출력 순서다. 층 혼합은 실제한 음성 forward 에서 나오지 않는 내부 대조일 수 있다.

## 모델·데이터·표본

[출처별 같은 원음의 순환 반복·반사채움·비반복과 긴 원음 절단 대조](../E071/README.md)의 같은 471개·배치 160 모델. 파형 반복의 XLS-R 출력은 201 프레임이다.

## 실험 내용과 상세 절차

1. 원음을 64,600까지 반복한 뒤 추출한 25층 특징과, 원음을 그대로 넣고 추출한 특징을 시간 축에서 반복/끝 절단해 201 프레임으로 맞춘 특징을 만든다.
2. 층 0~8,9~16,17~24의 세 블록을 정하고 파형 반복 특징에서 후기 특징 복제 값으로 교체하는 2³ =8 조합을 모두 계산한다.
3. 각 조합은 같은 backend 길이 201·같은 가중치·발화 seed를 사용한다. 긴 원음 144개는 입력이 원래 같아 모든 조건 점수 0차이인지 확인한다.
4. 세 블록 교체의 6 순서 평균 Shapley 기여와 층별 원래 공통 prefix 특징 의상대 RMS·평균 cosine을 계산한다.
5. 짧은 정상 207개 의 실제 h(original)−h(wrap)와 z 차이를 원래 160차원에서 비중심 SVD 해 변화 에너지 집중을 측정한다.

## 측정과 검증

조건별 점수·각도·블록 Shapley·SVD 제곱 합·긴 원음 동일 입력 검사.

## 결과

짧은 VCC2018 정상 각도는 파형 반복 111.232°,9~16만 교체 14.749°, 모든 층 후기 복제 9.200°. 중간 블록은 내부 분해 점수 변화의 78.3%를 차지했다. h 변화 첫 SVD 축은 99.696%였다.

## 해석·실패한 부분·한계

중간층 하나가 유일한 원인이라거나 그 SVD 축이 물리적 마이크/음색 축이라는 증명은 아니다. 동일 backend 길이에서도 반복 위치가 영향을 준다는 근거다.

## 배운 점

동일201프레임에서도 차이가 컸다. 중간층교체 분해를 유일한물리 원인으로 읽을 수 없다.

## 실험 코드

- [소스: frontend_context_sidecar.py](code/artifacts/geometry_experiments/frontend_context_sidecar.py)

코드의 파일별 원본 위치·SHA256은 [원본 파일 위치와 해시 목록](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [소스: frontend_context_sidecar.py](../../_shared/workspace/artifacts/geometry_experiments/frontend_context_sidecar.py)
- [상세 설계·실험 기록](../../_shared/references/docs/experiments/ebm_padding_context_mechanism_20261106.md)
- [설정·결과 근거](../../_shared/references/artifacts/geometry_cache/local_analysis/padding_causal_20261106_v1/analysis_complete.json)

## 연결

다른 관련 가설: [길이·유효 발화량과 앞뒤 저에너지 편향](../../01_duration_bias/README.md) · [잔여 방향·좌표·거리·기하와 변형 반응](../../05_residual_geometry/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
