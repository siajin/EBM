<a id="e071-출처별-같은-원음의-순환-반복반사채움비반복과-긴-원음-절단-대조"></a>

# 출처별 같은 원음의 순환 반복·반사채움·비반복과 긴 원음 절단 대조

**실행 상태:** 실제 추론·독립 검수 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

같은 짧은 정상의 반복정책을 바꾸면 출처별 큰점수 이동을 직접 재현할 수 있다.

## 그렇게 설계한 이유

짧은 새 출처 정상의 큰 점수 이동을 동일 음성 개입으로 재현한다.

대조와 통제의 근거: 원음 음량 정규화·속도 변화·무음 제거 없음. 반사는 시간 역전 내용이 들어가는 민감도 대조이며 자연적인 추가 발화가 아니다.

## 모델·데이터·표본

배치 160 고정 모델. DF nocodec 에서 ASVspoof/VCC2018/VCC2020× 정상/위조 × 원음 길이 층화 471개(정상 303·위조 168).

## 실험 내용과 상세 절차

1. ID 해시·화자별 번갈아 선택으로 고정한다. 정상<2/2~4.0375/≥4.0375초는 출처별 13/40/32,40/40/32,34/40/32; 위조는 세출 처 모두 20/20/16개다.
2. 각 원음에서 앞 64,600 이상은 자르고 짧으면 순환 반복 64,600·반사 64,600·실제 원음 길이의 세 조건을 만든다.
3. 긴 원음 144개는 원래 앞 64,600 창안의 앞/뒤 1.5초·2초를 취해 각 세 처리를 적용한다. 최근 336개 진단과 달리 뒤 구간이 원음 전체 끝은 아니다.
4. 같은 발화·seed·원래 pooling 잡음·고정 가중치로 계산한다. 원래 반복 조건 h/z/점수가 기존 캐시와 오차 0 인지 확인한다. 총 3,141개 조건 행이다.
5. 첫 12개 주 비교를 추론 전 잠그고 출처별 화자 군집 5,000회·Bonferroni family95%구간을 계산한다. 후속 문맥/분산 기전은 결과를 본 뒤의 탐색으로 구별한다.

## 측정과 검증

paired 점수/각도·원래 긴 음성 동일 입력 대조·같은 471개 EER/AUROC. 층화 표본의 EER을 DF 전체 EER로 부르지 않는다.

## 결과

짧은 VCC2018 정상은 비반복 −반복 각도 −98.413°, VCC2020 −57.678°로 정상 중심에 가까워졌다. 그러나 471개 EER은 반복 3.571429%→비반복 9.570957%·반사 11.221122%로 악화했다.

## 해석·실패한 부분·한계

정상 점수 구제와 전체 탐지력 개선을 구별한다. 길이·문맥·반복 경계가 함께 바뀌며 주기성 하나의 효과를 식별하지 못한다.

## 배운 점

정상각도 회복과표본전체 EER악화가 공존했다. 한클래스 점수 구제를 탐지력개선으로 대체하지 않는다.

## 실험 코드

- [소스: padding_causal_sidecar.py](code/artifacts/geometry_experiments/padding_causal_sidecar.py)

코드의 파일별 원본 위치·SHA256은 [원본 파일 위치와 해시 목록](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [상세 설계·실험 기록](../../_shared/references/docs/experiments/ebm_padding_context_mechanism_20261106.md)
- [소스: padding_causal_sidecar.py](../../_shared/workspace/artifacts/geometry_experiments/padding_causal_sidecar.py)
- [설정·결과 근거](../../_shared/references/artifacts/geometry_cache/local_analysis/padding_causal_20261106_v1/protocol.json)
- [설정·결과 근거](../../_shared/references/artifacts/geometry_cache/local_analysis/padding_causal_20261106_v1/analysis_verification.json)

## 연결

다른 관련 가설: [길이·유효 발화량과 앞뒤 저에너지 편향](../../01_duration_bias/README.md) · [잔여 방향·좌표·거리·기하와 변형 반응](../../05_residual_geometry/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
