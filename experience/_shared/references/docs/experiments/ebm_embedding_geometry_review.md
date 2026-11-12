> 역사 문서 보존본. 원문 작성 당시의 상태·계획을 포함한다. 최신 상태와 확정 결과는 experience의 각 실험 README를 우선한다.

# EBM 임베딩 실험 명세 검토 기록

> 이 문서는00시대 명세 작성 당시의 검토 기록이다. 아래 “추출·학습 미실행”은 그 시점에 한정된다. 후속 동결 실험의 현재 수행·통계·해석은 [통합 결과 분석](ebm_frozen_all_results_20261105.md)을 따른다. 공동 재학습A~G 전체를 완료한 것은 아니다.


검토일: 2026년 11월 5일 KST. 대상: [실험 명세](ebm_embedding_geometry_spec_20261105.md)와 [설정 파일](ebm_embedding_geometry_protocol.json).

가설 H1부터 H7까지를 실험 A부터 G에 연결하고 공통 추출 단계 P0를 추가했다. 명세는 구현 가능한 계산·대조·판정·저장 계약을 정한 상태다. 실제 연구실 데이터 추출, GPU 평가, 새 학습은 수행하지 않았다. 아래 검증은 문서의 논리와 합성 입력의 수학에 한정된다.

## 1 첫 검토에서 반영한 프로젝트 조건

| 확인 사항 | 명세에 반영한 내용 |
| --- | --- |
| 기존 결과는 여러 test셋과 epoch에서 관찰됨 | R19/R21LA/R21DF를 미관찰 blind test라고 부르지 않음 |
| 기존 실행의 dev 선택 미사용 | 새로운 V/Q 역할과 최종 epoch 50 규칙을 분리 |
| 저장 이름 BestCLS가 최적 모델의 증거가 아님 | epoch 규칙과 checkpoint 대응 검증을 우선 |
| classifier와 centroid가 다른 단계에서 갱신됨 | 새 gate/metric의 gradient 경로와 두 단계 optimizer 계약 명시 |
| BPL은 GPU별 정상 전체 쌍 평균임 | global gather와 batch 변경을 이번 요인에 섞지 않음 |
| 평가 경로에도 난수 호출 가능성이 있음 | 발화별 RNG, microbatch 1, native score 일치 검사 추가 |
| source 코드와 연구 데이터를 외부로 복사할 수 없음 | 원본 및 임베딩은 서버에, 로컬에는 명세와 합성 검증만 저장 |

현재 상태의 근거는 MENTEE_SERVER.md와 읽기 전용 SSH의 프로세스·checkpoint 존재 확인, AST 요약이다. 실험 시작 시에는 상태와 환경을 다시 점검해야 한다. 원래 학습 소스·설정은 수정하지 않았다.

## 2 두 번째 검토에서 보완한 방법론

| 위험한 해석 또는 설계 | 보완 |
| --- | --- |
| 정규화 유클리드의 차이를 새 방향 정보로 해석 | 코사인 동등성 검사를 대조군으로 명시 |
| tangent 잔차 크기를 추가 정보로 해석 | 잔차 norm이 sqrt(1-s²)임을 검증하고 방향 u를 분리 |
| 거의 0인 tangent를 정규화해 잡음 증폭 | valid 기준과 별도 indicator, 제외 비율, 전체 표본 유지 |
| 평균 방향이 0이면 구조가 없다고 판단 | 반대 모드 상쇄 반례와 covariance·MMD 검사 포함 |
| finite score bin의 permutation을 정확한 조건부 검정으로 주장 | MMD와 shuffle을 보조 진단으로 한정 |
| attention이 단순한 모델 용량 증가로 좋아짐 | 파라미터 수가 같은 residual MLP 대조군 포함 |
| 큰 attention 값이 인과적 중요도라는 주장 | 제거·재배치 개입과 성능 비교를 보조 근거로 사용 |
| 새 loss의 성능을 score 변경 효과와 혼동 | baseline encoder의 supervised posthoc metric과 비교 |
| full covariance를 쓸 때 중심도 바꿈 | 주 분석 중심 고정, 중심×metric 보조 요인 분리 |
| full metric의 low-rank factor를 0으로 초기화 | factor gradient가 0이 되는 점을 피하는 비영 초기화 |
| trace만으로 모든 방향의 발산을 막는다고 가정 | bounded diagonal, 명시적 regularizer, spectrum 검사 |
| encoder가 본 공격을 probe에서만 빼고 미지 공격이라고 주장 | probe-only와 encoder부터 제외한 실험을 구분 |
| 다른 학습 seed의 채널 index를 직접 비교 | 독립 좌표계는 예측 성능과 회전 불변 지표로 비교 |
| pre-LayerNorm h에 z centroid를 그대로 사용 | h는 원래 score에 추가하는 별도 probe로 조사 |

새 방향 손실의 주 모델은 비등방 대각 metric이다. 비대칭 spoof prototype은 다른 가설의 확장이며, 결과를 본 뒤 주 모델을 바꿀 수 없도록 protocol lock을 분리했다. 신규 손실의 성능이나 신규성은 아직 확인하지 않았다.

## 3 세 번째 검토에서 보완한 실행과 통계

| 항목 | 최종 규칙 |
| --- | --- |
| 반복 표본과 동일 화자 | dependency group 기반 paired bootstrap |
| training seed와 probe seed | 서로 다른 반복 축으로 보고하고 5×3을 15개 full run으로 세지 않음 |
| 다중 확인 비교 | G의 사전등록 비교 5개, family-wise 99% CI |
| 작은 그룹 수 | 전체 10개 미만 또는 클래스 기여 5개 미만이면 강한 supported 판정 보류 |
| 원래 benchmark 재사용 | 탐색 및 재평가의 범위를 명시하고 새로운 외부 U와 구분 |
| 비교 방향과 단위 | baseline EER minus candidate EER, percentage point |
| threshold | V에서 목표 정상 오거부율 1%로 고정, test EER threshold 재사용 금지 |
| B와 G 중복 학습 | B는 G0/G1/G2/Gm run을 재사용 |
| F와 G의 순환 의존 | F_frozen과 F_trained를 분리 |
| noise 20dB와 10dB | 같은 noise_base waveform에 크기만 변경 |
| prototype 초기화 | T 목적함수로 restart 선택, V로 seed를 고르지 않음 |
| 완전 학습 수 | screening 18, 주 효과 확인 28, 결합 확인 30 |
| 실행되지 않은 결과 | not_run으로 표시, 예정 경로를 실제 결과처럼 제시하지 않음 |

코덱 도구와 데이터 metadata, 대응 frontend checkpoint는 실행 전 점검 사항이다. 불가능한 조건을 다른 조건으로 자동 교체하지 않으며 변경 이력과 분석 범위를 다시 잠근다.

## 4 합성 수치 검증

[검증 코드](../../../workspace/docs/experiments/verify_spec_math.py)를 로컬 CPU에서 합성 NumPy 배열로 실행했다. **25개 검사가 통과했다.** 개별 결과와 대상 파일 SHA256은 [검증 결과](spec_validation.json)에 저장했다.

검사 범위는 정규화 거리 항등식, tangent 직교성, identity metric과 OC loss의 동등성, 라벨 방향, PSD와 trace, shrinkage 극한, 같은 코사인의 방향별 점수 차이, 평균 방향 상쇄 반례, gate 파라미터 수와 identity 초기화, low-rank 초기화, 다중 비교 수준, run 수, 가설 누락과 split 오염 설정, 의존성 DAG, 문서 구조 및 언어다.

이 검사는 실제 EBM forward, PyTorch gradient routing, GPU DDP 동등성, 실제 데이터 분할, 임베딩 또는 탐지 성능을 검증하지 않는다. 이 항목은 P0와 새 구현의 통합 검수에서 수행해야 한다.

## 5 구현 시작 전 필요한 입력

1. 실제 데이터 경로와 공식 protocol, source-family·speaker metadata의 가용성.
2. 기준 classifier와 대응 frontend checkpoint의 key 및 hash 일치.
3. 새 runner의 발화별 RNG·crop 재현성과 native score 일치.
4. 학습 단계별 gradient와 2 GPU BPL reduction의 합성 통합 검사.
5. V/Q manifest와 주 비교 목록을 포함한 protocol lock.

설계상의 선택은 수치 기본값까지 명시했다. 위 입력을 확인하기 전에는 실제 실험 결과나 실행 가능 환경이 확보되었다고 보고하지 않는다.
