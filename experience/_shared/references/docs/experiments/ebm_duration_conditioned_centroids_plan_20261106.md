> 역사 문서 보존본. 원문 작성 당시의 상태·계획을 포함한다. 최신 상태와 확정 결과는 experience의 각 실험 README를 우선한다.

# EBM 동결 상태에서 원음 길이별 정상 센트로이드의 효과를 검증하는 실험 설계

작성일: 2026-11-06 KST  
상태: **설계안 작성 완료. 본 문서의 신규 성능 실험은 미실행.**  
범위: 기존 파생 임베딩·메타데이터를 사용하는 CPU 사후 분석. EBM 가중치 갱신, 새 음성 forward, 서버 실험 등록은 포함하지 않는다.

## 1. 검증할 주장과 결론의 범위

검증 질문은 다음과 같다.

> 동일한 동결 EBM 표현을 사용할 때, 패딩 전 원음 길이에 따라 정상 센트로이드를 선택하면 기존 단일 중심보다 정상·스푸프 구분 성능이 좋아지는가?

주장을 세 단계로 구분한다.

| 단계 | 검증할 내용 | 필요한 증거 |
| --- | --- | --- |
| H1: 사후 성능 개선 | 길이별 정상 중심으로 재채점하면 DF 전체 EER가 감소한다 | 학습셋에서만 중심을 만든 방법이 동일 DF 발화에서 기존 점수를 능가 |
| H2: 중심 방향의 유용성 | 단순 길이별 점수 보정이나 중심 재추정보다 길이별 방향을 쓰는 것이 낫다 | 전체 평균 중심·스칼라 보정 대조군 대비 개선 및 같은 길이 안의 순위 개선 |
| H3: 일반화 범위 | 이득이 특정 출처 또는 평가셋에만 한정되는지 확인한다 | 출처×길이별 결과, 19LA/21LA 평가셋에서의 변화 |

이 설계는 **동결 표현의 사후 readout 효과**를 검증한다. 길이 조건을 넣어 EBM을 공동 재학습했을 때의 효과, 원음 길이의 물리적 인과효과, 무음의 영향까지 증명하는 실험은 아니다. 사용자가 배제한 무음 분포 가설은 실험 요인에 넣지 않는다.

DF는 이미 원인 분석에 사용한 benchmark다. 따라서 아래 절차를 결과 계산 전에 고정하더라도 **새로운 독립 확증 실험 또는 과거 관찰 이전의 사전등록**이라고 부르지 않는다. 문서 고정 이후의 설정 변경과 결과 선택을 막는 탐색적 검증 계획으로 사용한다.

## 2. 실제 EBM 구현에서 확인한 사항

### 2.1 기준 모델과 소스

2026-11-06 01:21 KST에 서버 상태와 기준 실험의 보관 소스를 읽기 전용으로 확인했다. 이 확인은 학습·추론 실행이 아니다. 그 시점의 GPU는 유휴였으나, 이후 서버 작업에는 다시 상태 확인이 필요하다.

기준은 **epoch 50 / batch 160 / 원래 rank-local BPL 실험**이다. 전역 BPL이나 batch 240 결과와 섞지 않는다.

- 보관 소스: 서버 `/home/mentee/results/CM/LA/EBM_e50_bs160_lr1e5/backup/scripts`
- 기준 checkpoint: 같은 실험의 `model/BestCLS_50_21DF.pt`
- checkpoint SHA256: `104873703a57cad98b686d770e0860e031a2cf9abef79b72668780c1660e0b24`
- frontend: `facebook/wav2vec2-xls-r-300m`, 캐시 기록 commit `1a640f32ac3e39899438a2931f9924c02f080a54`
- 기존 구조 감사: [구조 감사 JSON](ebm_frozen_structure_audit_20261105.json)

| 보관 파일 | 이번 읽기에서 확인한 SHA256 | 설계에 관련된 코드 위치 |
| --- | --- | --- |
| `model/b_linear.py` | `c2cbcf898f1905c9c6ee8b20ef4514c4af7cb343abbf35e2dbd1fd526ad3459f` | forward 24–44행, ASP noise 159–166행 |
| `model/loss.py` | `164be4161b57a7899ef7b289e005ab80d12da373922d57e9e021844ea828b1ac` | BPL 12–23행, OCSoftmax 41–62행 |
| `train.py` | `b7b98a182a4946c479f477b7110e1c4a49be53ee6b6e89ef1b4bc4785374873f` | 두 단계 갱신 129–141행, 평가 210–238행 |
| `data_loaders.py` | `e46ae2e171df2ecf7665baf84ce38661fe92fd277dc8c1bd2d92470b39ad3556` | 학습 증강 103행 이후, 평가 입력 197–218행 |
| `arguments.py` | `567cc9b95a69dda73052100ecd4d76fbf420d1abf46e5d09ef208577953552e3` | epoch·차원·프레임·loss 설정 |

원본 소스·원음·checkpoint를 로컬로 복사하지 않았다. 현재 작업 폴더의 기본 설정 대신 **기준 실험 보관 설정과 캐시 identity**를 따른다. 과거 학습 당시 frontend 전체 가중치의 별도 비트 해시는 검증되지 않았다는 기존 한계도 유지한다.

### 2.2 실제 평가 점수와 사용할 표현

실제 경로는 다음과 같다.

```text
고정 길이 음성
→ XLS-R의 25개 hidden states
→ 레이어별 projection + 레이어 attention
→ 시간 attention / ASP
→ Linear(256→160): h
→ 학습된 LayerNorm(160, eps=1e-12): z
→ L2 정규화: u
→ 정규화된 loss.weight와 내적: s0
```

기존 점수는 다음과 같다.

\[
u_i=\frac{z_i}{\|z_i\|_2},\quad
c_0=\frac{w}{\|w\|_2},\quad
s_0(i)=u_i^\top c_0.
\]

- `y=1`은 정상, `y=0`은 스푸프다. **점수가 클수록 정상**이다.
- 주실험은 마지막 LayerNorm **이후**의 160차원 `z`를 사용한다. `h`나 pooling 전 특징으로 대체하지 않는다.
- OC의 `alpha=20`, 정상 margin 0.9, spoof margin 0.2와 class weight는 학습 손실에 사용된다. **평가 점수에 softplus·margin·class weight를 다시 적용하지 않는다.**
- 새 점수도 정답 label 없이 계산한다. 정상·스푸프 모두 관측 가능한 길이만으로 중심을 선택한다.
- 기존 `OCSoftmax`에 여러 중심을 넣으면 현재 단일 열 점수·label indexing과 평가 `.item()` 전제가 바뀐다. 원본 모듈을 바꾸지 않고 캐시 위 별도 scorer로 구현한다.

### 2.3 학습된 중심은 정상 평균과 다르다

실제 iteration은 두 단계다.

1. `loss.weight`를 고정하고 표현 모듈을 OC+BPL로 갱신한다.
2. 표현 모듈을 고정하고 **정상 샘플의 OC 손실**로 `loss.weight`를 갱신한다.

따라서 `c0`는 단순 정상 평균이 아니다. 정상의 margin 충족 정도가 gradient에 영향을 미치고, 표현 자체는 스푸프를 포함한 OC 목적에 맞춰 학습됐다. 평균 중심으로 교체한 결과에는 **중심 추정법 변경 효과**가 섞인다. 이를 분리하기 위해 단일 평균 중심 대조군과 기존 중심을 보존하는 보조 실험을 반드시 포함한다.

BPL은 정상 long/short의 정규화 임베딩 사이 **모든 쌍**의 유사도를 높인다. 같은 발화의 두 view만 대응시키는 손실이 아니다. 기준 alpha는 30이다. 기존 모델은 이미 정상 길이 변이를 압축하는 압력을 받았으므로, 길이별 중심 차이가 작거나 새 중심의 분별력이 떨어질 수도 있다. 이 경우를 실패 결과로 그대로 보고한다.

### 2.4 입력 길이와 난수 경로

기준 설정은 `num_test_frames=404`, sample 수는 `404*160-40=64600`이다. 16 kHz에서 **4.0375초**다. 평가 로더는 짧으면 `np.pad(..., 'wrap')`, 길면 처음 64,600 sample을 사용한다. 평가 시 segment는 1개다.

원음 길이와 입력 길이를 구분한다.

\[
d_{orig}=N_{source}/f_s,\qquad d_{input}=4.0375,\qquad
d_{unique}=\min(d_{orig},4.0375).
\]

모든 평가 입력의 길이는 같지만 원음 길이는 다르다. 본 실험의 조건은 캐시의 **패딩·절단 전 `duration`**이다. 4.0375초 이상은 하나의 구간으로 묶어, 입력에 포함되지 않은 뒷부분의 길이를 세분화해 활용하지 않는다.

학습 short view는 48,440 sample, 3.0275초다. 학습에는 long/short별 codec·RawBoost 증강도 있었다. 원음이 짧은 학습 정상의 개수는 전체 짧은 view 노출량과 같지 않다.

ASP는 `.eval()`에서도 `1e-5 * torch.randn(...)`를 생성한다. 따라서 방법마다 음성을 다시 forward하면 미세한 표현 차이가 생길 수 있다. **모든 비교는 동일 발화의 동일 캐시 `z`로 수행**한다. attention, pooling, LayerNorm을 재실행하거나 교체하지 않는다.

## 3. 데이터와 정보 사용 규칙

### 3.1 역할을 고정한다

| 자료 | 정상 / 스푸프 | 용도 | 허용하지 않는 사용 |
| --- | --- | --- | --- |
| `T` | 2,580 / 22,800 | 정상 중심·정상 점수 평균 추정 | 평가 발화로 중심 보충 |
| `V` | 1,351 / 13,632 | 방법별 운영 임계값 계산, 구현 점검 | V 최저 EER를 근거로 주 방법 교체 |
| `DF_pilot` | 2,000 / 18,000 | 저장·정렬 점검과 고정 pilot 결과 | pilot 결과를 보고 구간·수식 수정 |
| `DF_remaining` | 12,869 / 501,059 | pilot과 합쳐 DF 전체 평가 | 독립 holdout이라고 주장 |
| DF 전체 | **14,869 / 519,059** | 주 평가, 총 533,928개 | 정상만 제외하거나 재가중한 값을 주 EER로 사용 |
| `19LA_eval` | 7,355 / 63,882 | 기존 도메인 성능 변화 | 결과를 본 뒤 주 방법 선택 |
| `21LA_eval` | 14,816 / 133,360 | 추가 전이 평가 | DF와 독립 표본으로 단순 합산 |
| `Q` | 1,197 / 8,664 | 선택적 참고 결과 | 새 독립 확증셋으로 표시 |

DF 전체는 `DF_pilot`과 `DF_remaining`을 **발화 ID 기준 서로소 합집합**으로 구성한다. `load_embeddings("DF_full")`라는 미등록 API를 가정하지 않는다. 두 실제 별칭을 읽고 합치는 전용 조립 함수를 만든다.

기존 benchmark와 발화·화자·원천 콘텐츠가 중복될 수 있으므로 데이터셋별 결과를 독립적으로 보고한다. 특히 19LA와 21LA를 합쳐 표본 수를 늘린 신뢰구간을 계산하지 않는다.

### 3.2 길이 구간을 고정한다

경계는 초 단위로 정확히 `[2.0, 3.0, 64600/16000]`을 사용한다. 표시에 반올림한 `4.037`을 계산 경계로 사용하지 않는다.

| 구간 | 원음 길이 | T 정상 수 | V 정상 수 |
| --- | --- | ---: | ---: |
| B0 | `0 < d < 2.0` | 109 | 46 |
| B1 | `2.0 <= d < 3.0` | 877 | 397 |
| B2 | `3.0 <= d < 4.0375` | 1,018 | 506 |
| B3 | `d >= 4.0375` | 576 | 402 |

위 개수는 문서 작성 시 기존 로컬 캐시 메타데이터로 재확인했다. 신규 분류 성능 결과는 아니다. 학습 정상 중 1.5초 미만은 9개이므로 별도 학습 중심을 만들지 않는다. 평가에서는 `<1.5초`를 보조 오류 구간으로 표시할 수 있지만 scorer의 구간은 바꾸지 않는다.

길이 결측·0 이하·비유한 값이 있으면 해당 발화를 조용히 제외하지 않고 실행을 중지한다. T에서 어느 구간의 정상 수가 30 미만이면 현재 계획과 캐시가 다른 것이므로 중지하고 원인을 기록한다. DF 결과를 본 뒤 구간을 합치는 fallback은 허용하지 않는다.

### 3.3 평가 데이터 누수 방지

- 중심·점수 보정 통계는 T의 `y=1`에서만 계산한다. T 스푸프는 성능 진단에만 사용할 수 있다.
- V는 운영 임계값 계산에만 사용하며, 주 방법·구간·수축 강도 선택을 하지 않는다.
- DF/19LA/21LA의 label은 최종 지표 계산에서만 사용한다. 센트로이드 생성·선택·점수 방향 결정에는 사용하지 않는다.
- source·codec은 결과 분해와 군집 통계에 사용한다. 배포용 scorer의 routing 입력에는 넣지 않는다.
- 정답을 이용해 스푸프에는 기존 중심, 정상에는 길이별 중심을 적용하는 oracle은 만들지 않는다.
- 사용자가 취소한 **DF 4×3 보완 교차검증은 수행·등록하지 않는다.** 이번 설계에는 DF 내부 중심 fit이나 새 DF 교차검증이 없다.

## 4. 추정량과 비교 방법

모든 연산은 캐시를 읽은 뒤 float64로 수행한다. `unit(v)=v/||v||2`로 정의한다. 정상 평균은 정규화 이전의 `z` 평균이 아니라 **발화별 L2 정규화 후 평균**이다.

\[
\mu=\frac1{N_B}\sum_{i\in T,y_i=1}u_i,\quad
\mu_k=\frac1{n_k}\sum_{i\in T,y_i=1,b(d_i)=k}u_i,\quad
g=unit(\mu),\quad g_k=unit(\mu_k).
\]

각 평균 norm이 `1e-8` 이하이거나 비유한 값이면 해당 추정량을 식별 불가로 표시하고 중지한다. 0벡터를 임의의 정상 중심으로 대체하지 않는다.

### 4.1 필수 비교군

| ID | 정의 | 역할 |
| --- | --- | --- |
| N | `u @ c0` | 기존 EBM 점수 |
| G | `u @ g` | 중심을 정상 평균으로 바꾼 효과 |
| **D** | **`u @ g[b(d)]`** | **주실험: 원음 길이별 정상 중심** |
| N-OFF | 기존 점수에서 T 정상의 길이별 평균 차이만 제거 | 기존 점수의 길이 편향 보정 |
| G-OFF | G 점수에서 T 정상의 길이별 평균 차이만 제거 | D와 G의 방향 차이를 스칼라 보정과 비교 |
| R | 무작위로 만든 길이 구간과 같은 크기의 정상 중심 | 중심 수·작은 집단 평균 추정에 대한 무작위 대조 |

N-OFF와 G-OFF는 각각 기준 점수 `q ∈ {N,G}`에 대해 다음과 같이 계산한다.

\[
a_{q,k}=mean_{T,B,k}(s_q),\quad a_q=mean_{T,B}(s_q),\quad
s_{q\text{-OFF}}(i)=s_q(i)-a_{q,b(d_i)}+a_q.
\]

보정량은 정상에서만 계산하며 정규분포·표준편차·스푸프 사전확률을 가정하지 않는다. 이 방법은 길이 구간 내부에서는 상수만 더하므로 **구간 내부 AUROC/EER를 바꾸지 못한다.** D가 구간 내부 순위를 바꿔 개선하는지 확인하는 핵심 대조군이다.

주실험 D는 여러 중심의 `max`나 평균을 취하지 않는다. 입력의 원음 길이에 해당하는 중심 하나를 선택한다. `max`는 어느 중심이든 가까우면 정상 점수가 커지는 다른 검출 규칙이므로 이번 핵심 주장과 구분한다.

### 4.2 기존 학습 중심을 보존하는 보조 실험 A

평균 중심 교체가 기존 중심의 장점을 잃는 경우를 조사하기 위해, 학습된 방향을 기준으로 길이별 **접평면 방향 차이**만 추가한다.

\[
P_0=I-c_0c_0^\top,\quad
\rho_k=\frac{n_k}{n_k+100},\quad
\delta_k=\rho_k P_0(\mu_k-\mu),\quad
c_{A,k}=unit(c_0+\delta_k),\quad
s_A(i)=u_i^\top c_{A,b(d_i)}.
\]

- 수축 상수 100, 보정 계수 1을 문서에서 고정한다. V/DF에 맞춰 탐색하지 않는다.
- `P0`는 실제로 160×160 행렬을 만들 필요 없이 `v - (v @ c0)*c0`로 계산한다.
- 정상 표본 수가 적은 구간의 보정을 줄이고, native 방향에 직교하는 길이 차이만 사용한다.
- 이 수식은 **본 설계의 사후 추정량**이며 원래 EBM 손실 또는 QAMO 구현이라고 부르지 않는다.
- A가 좋아지고 D가 실패하면 “기존 방향을 보존한 길이 조건 보정은 유망”이라고 보고한다. 주실험 D의 성공으로 바꿔 기록하지 않는다.

추가로 D의 표본 수 민감도를 보기 위한 고정 수축 평균 S를 계산한다.

\[
\widetilde\mu_k=\frac{n_k\mu_k+100\mu}{n_k+100},\quad
s_S(i)=u_i^\top unit(\widetilde\mu_{b(d_i)}).
\]

A와 S는 보조군이다. 결과가 가장 좋은 방법을 사후에 새 주 방법으로 지정하지 않는다. 계수 sweep, 길이 spline, 새 학습형 gating은 이번 범위에 넣지 않는다.

### 4.3 무작위 중심 대조 R의 정확한 정의

199개 반복을 사용하고 seed는 `202611060100 + r`, `r=0,...,198`로 고정한다.

1. T 정상의 실제 bin label을 발화 사이에서 무작위 순열한다. 각 그룹의 크기는 `[109,877,1018,576]`으로 유지한다.
2. 순열된 label별로 정상 중심 4개를 계산한다. 표현·정답 label은 바꾸지 않는다.
3. 각 평가셋에서도 실제 bin label 배열을 별도로 순열하고 그 label로 중심을 선택한다. 평가셋의 bin 총수는 유지한다. 정답 label로 층화하지 않는다.
4. T/V/평가셋의 순열 난수 스트림은 분리하고 발화 ID 정렬 후 적용한다. 행 순서·rank에 결과가 의존하지 않게 한다.
5. 평가셋의 길이 배열을 사용하는 R은 **배치 단위 무작위 대조군**이며 배포 후보가 아니다. 평가 label이나 임베딩으로 순열을 고르지 않는다.

199개 EER를 전부 저장하고 중앙값·2.5/97.5 분위수·D의 위치를 보고한다. 최상의 seed를 골라 D와 비교하지 않는다. 이 분포를 음성 길이의 인과효과에 대한 엄밀한 randomization p-value라고 해석하지 않는다. R은 반복 하나씩 계산해 중심·seed·지표를 저장하며, 전체 평가 점수의 199배 행렬을 메모리에 보관하지 않는다.

## 5. 평가 지표와 임계값

### 5.1 주 지표: DF 전체 EER

- 모든 방법에 **동일한 533,928개 발화**를 사용한다.
- 기존 [EER 함수](../../../workspace/artifacts/geometry_experiments/fast_analyze.py)의 tied ROC crossing 선형 보간을 따른다. `pos_label=1`, 큰 값이 정상이다.
- 지표 단위는 `%`, 방법 간 차이는 **percentage point(pp)**로 구분한다.
- `ΔEER(B,D)=EER(B)-EER(D)`로 정의한다. 양수면 D 개선이다.
- EER는 평가 label로 계산하는 순위 지표다. EER에서 역산한 임계값을 운영 임계값이라고 부르지 않는다.
- full DF를 주 결과로 사용하고 pilot은 별도 줄로 표시한다. pilot과 remaining을 독립 반복 실험처럼 평균내지 않는다.

### 5.2 운영 지표: V에서 고정한 방법별 임계값

큰 점수가 정상일 때 다음과 같이 정의한다.

\[
FRR(t)=P(s<t\mid y=1),\qquad FAR(t)=P(s\ge t\mid y=0).
\]

각 방법별로 V에서 **spoof FAR ≤ 1%**를 만족하는 가장 낮은 임계값을 선택한다. 후보는 V spoof의 각 고유 점수보다 바로 큰 float64 값 `nextafter(score,+inf)`와 양끝 경계다. 동점은 통째로 처리하며 label을 이용해 동점을 나누지 않는다. 이 규칙은 목표 FAR 아래에서 허용되는 가장 낮은 threshold를 선택한다.

- V에서 실제 달성한 FAR/FRR와 오류 개수를 저장한다.
- threshold를 고정한 채 DF/19LA/21LA의 FAR/FRR·오류 개수를 함께 보고한다.
- V에서 FAR를 맞춰도 평가셋 FAR가 같아지는 것은 아니다. “동일 평가 FAR에서 정상 오탐이 감소했다”는 표현에 사용하지 않는다.
- 서로 다른 score 정의에 기존 cosine 숫자 threshold를 그대로 재사용하지 않는다.
- 기존 `frozen_common.threshold()`와 `df_screen.threshold_record()`는 **정상 FRR 1% 기준**이다. 이번 FAR 1% helper로 그대로 재사용하지 않는다.
- 기존 분석의 `19LA_near_EER=0.8619597055730815`도 이번 V-FAR 기준과 다르다. 운영 주 결과에 혼용하지 않는다.

R의 각 반복을 운영 지표로 표시하려면 같은 순열 방식의 V 점수로 해당 반복의 임계값을 별도로 고정한다. R의 주 용도는 EER 분포 비교이며, 199개 운영 결과 중 좋은 것만 고르지 않는다.

### 5.3 같은 길이·같은 출처 안의 평가

다음 결과를 빠짐없이 낸다.

1. B0–B3별 AUROC/EER, 각 클래스의 N과 화자 그룹 수.
2. 출처 `asvspoof / vcc2018 / vcc2020` × B0–B3별 AUROC/EER와 운영 FAR/FRR.
3. `<1.5초`, `[1.5,2초)` 정상의 운영 오탐률 및 오류 수. 새 scorer bin을 만들지는 않는다.
4. codec별 결과는 보조 분해로 제공하고 codec을 routing 입력으로 사용하지 않는다.
5. 원래 틀렸던 정상 중 회복한 수와 원래 맞던 정상 중 새로 틀린 수, 스푸프에서도 대응하는 오류 전환표.

양 클래스가 없는 셀은 AUROC/EER를 `NA`로 표시한다. 클래스별 N이 50 미만 또는 어느 클래스의 관측 화자 그룹이 5 미만인 셀은 수치를 표시하되 **작은 표본**으로 명시하고 통계적 우월성 주장을 하지 않는다.

정상 점수만 높아지는 것은 성능 개선의 충분조건이 아니다. 짧은 스푸프 점수도 함께 올라가면 FAR가 나빠질 수 있으므로 두 클래스를 함께 평가한다. 같은 길이 안의 순위 개선도 출처 혼합 효과일 수 있어 출처×길이 결과까지 함께 해석한다.

## 6. 불확실성과 성공 판정

### 6.1 평가 발화의 짝지은 군집 부트스트랩

DF에서는 검수된 `dependency_group`을 기본 군집으로 사용한다. 실제 메타데이터에서 source×speaker와의 대응을 확인하고, codec 변형을 다른 독립 발화 군집처럼 나누지 않는다. 같은 군집의 정상·스푸프는 함께 뽑는다. 재표집 전에 군집 ID가 여러 source를 잘못 합치지 않는지 검사한다.

- 반복: **10,000회**, seed `2026110601`.
- DF는 source별로 군집을 재표집하고, source 내 원래 군집 개수를 유지한다.
- 모든 방법에 동일한 군집 multiplicity를 적용한다. EER를 각 재표집에서 재계산하고 차이를 구한다.
- 분류 점수·V threshold·T 중심은 이 부트스트랩에서 재추정하지 않는다.
- 주 비교는 D 대 N, G, N-OFF, G-OFF의 **4개**다.
- 각 비교의 일반 95% 구간과, 4개 비교에 대한 Bonferroni **98.75% 구간**을 함께 보고한다. 후자는 분위수 `[0.00625,0.99375]`를 사용한다.
- 양 클래스가 사라진 반복은 무효로 기록한다. 유효 반복률이 99% 미만이면 성공 판정을 보류하고 군집 커버리지 문제를 보고한다. 발화 단위 bootstrap으로 바꿔 유리한 구간을 만들지 않는다.

이 구간은 **추정된 중심을 고정한 조건부 평가 불확실성**이다. T 중심 추정·V threshold 추정의 변동, 미해결 원천 발화 family의 의존성, 미관측 출처에 대한 불확실성은 포함하지 않는다. 특히 source가 3종이라는 제한은 화자 수가 많아도 사라지지 않는다.

계산은 score를 한 번 정렬하고 chunk 단위 weighted ROC 또는 군집별 정렬 점수의 누적 질량을 이용한다. `10000 × 533928 × 방법 수` 전체 배열을 메모리에 만들지 않는다. 통계 반복 수를 결과에 따라 늘리거나 줄이지 않는다.

### 6.2 중심 추정의 안정성

T의 정상 화자를 단위로 **500회**, seed `2026110602`로 재표집해 중심만 다시 계산한다. 성능을 보고 방법을 선택하지 않는다.

각 길이 구간의 표본 수, 화자 수, 평균 resultant norm `||μk||`, 원래 추정 중심과의 각도 분포, global 중심과의 각도를 저장한다. 어떤 재표집에서 bin이 비면 해당 횟수를 기록하고 각도 계산에서 `NA`로 둔다. 이 절의 각도 안정성은 6.1의 성능 신뢰구간에 fit 불확실성을 추가한 결과가 아니다.

### 6.3 판정표

| 결과 | 허용하는 결론 |
| --- | --- |
| D 대 N의 ΔEER 보정 구간 하한이 0 초과 | DF 전체에서 동결 길이별 중심의 개선을 지지 |
| 위 조건에 더해 D가 G, N-OFF, G-OFF도 모두 보정 구간 하한 > 0 | 중심 재추정·스칼라 보정을 넘어선 길이별 방향 활용의 이득을 지지 |
| 전체 이득과 함께 B0의 AUROC가 G보다 높고, 충분한 표본의 출처×B0에서도 같은 경향 | 짧은 정상 문제와 연결된 방향 정보라는 해석이 강화됨. 보조 CI는 탐색적으로 표시 |
| 전체 EER만 개선되고 bin 내부 순위 이득이 없음 | 길이 간 점수 정렬·보정 효과 가능성. H2는 미확인 |
| 정상 FRR만 감소하고 FAR 증가 또는 EER 미개선 | 정상 수용 범위 확대에 따른 교환. 전체 성능 개선으로 부르지 않음 |
| A/S만 개선, D 실패 | 사전 고정한 보조 방법의 탐색적 근거. 주실험 성공으로 대체하지 않음 |
| 개선 구간이 0을 포함 | 현재 표본·설계로 개선 확정 불가 |
| D 실패 | 현재 표현과 평균 중심 기반 readout의 한계. 길이 조건 공동 재학습 전체를 기각하지 않음 |

주효과는 상대 감소율과 절대 pp를 모두 보고하고, 아주 작은 차이를 실질적으로 큰 성능 개선처럼 표현하지 않는다. 임의의 최소 개선폭이나 비열등성 허용폭은 이 문서에서 사후 도입하지 않는다. 19LA/21LA에서 악화되면 수치와 구간을 그대로 제시하고 “전반적 개선”이라는 표현을 제한한다.

R과 A/S, 세부 셀의 결과는 탐색적 분석이다. 이들의 수많은 95% 구간 중 유리한 것만 골라 확증적 주장을 하지 않는다.

## 7. 구현 계약과 검수 항목

### 7.1 캐시 API와 정렬

기존 [geometry_cache.py](../../../workspace/artifacts/geometry_experiments/geometry_cache.py)의 `resolve_run`, `validate_run`, `load_embeddings`를 재사용한다. source manifest나 원음 경로를 추가로 가져오지 않는다.

각 별칭에서 다음을 확인한다.

- catalog 상태 `ready`, 완료·QC marker, checkpoint identity 일치.
- 입력 파일 SHA256·byte 수, catalog fingerprint.
- `utterance_id` 유일성 및 `rank`, `embedding_row`와 배열 행 대응.
- `z.shape == (N,160)`, finite 값, `||z|| > 1e-12`.
- 모든 자료의 정규화 `c0`가 일치하고 `| ||c0||-1 | < 1e-10`.
- float64 재계산 `unit(z) @ c0`와 캐시 `cosine_s` 최대 차이 `<1e-10`.
- 원래 float32 score와의 대응은 기존 추출 QC 허용 오차를 별도로 따른다. float64 항등식 기준과 섞지 않는다.
- `crop_samples=64600`, `crop_start=0`, `duration`은 전처리 전 원음 길이라는 추출 provenance 확인.
- 가능하면 `padding_fraction == max(0,1-duration/4.0375)`를 sample 반올림 오차 범위에서 확인한다.
- T/V와 평가셋의 canonical speaker 교차 및 발화 ID 중복을 재검사한다. 알려진 평가셋 간 중복은 별도 기록한다.
- DF pilot/remaining의 합집합 크기·클래스 수·ID 서로소성 확인. 판독 순서는 `utterance_id` 정렬로 고정한다.

캐시 검사 실패 시 해당 샘플을 빼고 계속하지 않는다. 데이터 수정이 필요하면 별도 버전과 원인을 기록하고 설계 적용 여부부터 다시 확인한다.

### 7.2 계산 의사코드

```python
# 학습 정상의 통계량만 fit. 실제 코드는 ID/shape/hash 검사를 먼저 수행한다.
edges = np.array([2.0, 3.0, 64600 / 16000], dtype=np.float64)
bin_id = lambda d: np.searchsorted(edges, d, side="right")
unit = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)

ut = unit(T.z.astype(np.float64))
is_bona = T.rows.y.to_numpy() == 1
bt = bin_id(T.rows.duration.to_numpy())
mu = ut[is_bona].mean(axis=0)
mu_k = np.stack([ut[is_bona & (bt == k)].mean(axis=0) for k in range(4)])
g, g_k = unit(mu), unit(mu_k)

def score(rows, z):
    u = unit(z.astype(np.float64))
    b = bin_id(rows.duration.to_numpy())  # 정답 label을 사용하지 않는다.
    return {
        "N": u @ c0,
        "G": u @ g,
        "D": np.einsum("nd,nd->n", u, g_k[b]),
    }
```

평가 label이 바뀌어도 scorer 출력은 불변이어야 한다. 같은 발화를 어느 행·rank에서 읽어도 점수는 같아야 한다.

### 7.3 필요한 수치 검사

단순 문서 생성 테스트를 추가하지 않는다. 향후 scorer를 구현할 때 결론을 바꿀 수 있는 다음 검사만 수행한다.

1. 길이 `2.0`, `3.0`, `4.0375` 경계와 바로 앞뒤 값의 routing 확인.
2. native score를 캐시로 독립 재계산해 일치 확인.
3. 모든 bin 중심을 같게 주면 D와 단일 중심 점수가 일치하는지 확인.
4. `μk=μ`이면 A가 N으로 환원되는지 확인.
5. 같은 bin 안에서 N-OFF/G-OFF의 순위 및 AUROC가 N/G와 일치하는지 확인.
6. label 열을 제거하거나 순열해도 새 점수가 동일한지 확인.
7. method별 발화 집합이 완전히 같은지 확인.
8. 별도 tied-score 예제로 기존 EER와 독립 구현이 일치하는지 확인. bootstrap의 multiplicity가 모두 1인 경우 원래 EER를 재현해야 한다.
9. 운영 threshold가 실제 V FAR ≤ 1%를 만족하는지 동점 포함 검사.
10. 원본 코드·모델·캐시가 수정되지 않았고 새 optimizer·음성 forward가 사용되지 않았는지 기록.

## 8. 실행 순서와 산출물

이 절은 **향후 실행 절차**다. 문서 작성 요청만으로 다음 계산이나 서버 대기열 등록을 수행하지 않는다.

1. 설계 파일 SHA256, 코드 revision, Python·NumPy·pandas·scikit-learn 버전을 기록한다.
2. 새로운 결과 디렉터리를 `exist_ok=False`로 생성한다. 같은 이름으로 재실행해 기존 결과를 지우는 로거를 사용하지 않는다.
3. 입력 catalog·hash·ID·native 점수와 길이 메타데이터를 검수한다.
4. **T만 읽어** N/G/D/N-OFF/G-OFF/A/S의 추정량과 R의 중심을 저장한다. 소스·평가 label을 fit 함수 인자로 받지 않는다.
5. V 점수와 각 운영 threshold를 계산하고 모델·설정·threshold lock 파일을 저장한다. 주 방법 D를 바꾸지 않는다.
6. 이후 pilot, full DF, 19LA, 21LA의 고정 scorer 결과를 계산한다. 점수는 생성 완료 즉시 ID와 함께 저장한다.
7. 전체·길이·출처×길이 지표, paired 군집 CI, R 분포, T 중심 안정성을 계산한다.
8. 독립 검수 후 성공·실패·판정 보류와 모든 대조군을 함께 보고한다. 결과를 보고 설정을 바꾸면 새 탐색 버전으로 분리한다.

제안 구현 파일은 `artifacts/geometry_experiments/duration_conditioned_centroids.py`다. **이 문서 작성 시 해당 신규 runner는 만들거나 실행하지 않는다.** 실제 구현 때는 기존 캐시 API와 검증된 EER helper를 사용하고 별도 가중치 파일을 로딩하지 않는다.

결과 루트는 다음 이름을 기본으로 하되 이미 있으면 고유 suffix를 붙인다.

```text
artifacts/geometry_cache/local_analysis/duration_centroids_20261106_v1/
  protocol_lock.json         # 설계 hash, 상수, 방법, primary 비교, 데이터 역할
  input_validation.json      # identity/hash/UID/shape/native score 검수
  fit_summary.json           # T만 사용했다는 기록, bin별 N·화자·norm
  centroids.npz              # c0, g, gk, A/S 중심 및 보정 통계량
  V_thresholds.json          # 방법별 FAR 목표와 실제 달성치
  scores_<dataset>.parquet   # UID, method 점수, bin, 평가 metadata
  metrics_overall.csv
  metrics_source_duration.csv
  error_transitions.csv
  paired_intervals.json      # 일반·다중비교 보정 구간, 유효 반복률
  random_controls.csv       # 199개 결과 모두
  centroid_stability.json
  verification.json
  conclusions.json
```

추정량 저장물은 캐시에서 계산한 파생 분석 결과다. 연구실 원본 소스·원음·checkpoint·인증 정보는 산출물에 넣지 않는다. CPU 스레드는 기본 2개로 제한하고 GPU·SSH가 필요 없는 경로를 우선한다. 원음 입력의 변경 실험, 원본 EBM 수정, 새 GPU 작업은 본 설계와 분리한다.

## 9. 기존 결과와 혼동하지 않을 사항

- 기존 DF 내부 정상 reference 실험은 DF 정상으로 fit한 진단이었다. 본 실험은 **T 정상만으로 fit**한다.
- 기존 KMeans 다중 중심의 최대 유사도 결과는 원음 길이 routing 실험이 아니다. 성공·실패를 그대로 전이해 해석하지 않는다.
- 기존 학습 중심, T 단일 평균 중심, T 길이별 평균 중심을 모두 비교해야 기존 방향의 장점과 길이 조건의 효과를 분리할 수 있다.
- 원래 짧은 정상에 오류가 많았다는 관찰만으로 D가 좋아질 것을 보장하지 않는다. T의 짧은 정상은 DF의 새로운 출처를 충분히 대표하지 않을 수 있다.
- 단순 중심 방향 변화로 회복할 수 없는 표현 중첩이 있으면 재채점은 실패할 수 있다. 실패는 encoder·BPL 공동 학습의 가능성을 소진한 결과가 아니다.
- 음성 길이와 wrap 반복률은 현 평가 경로에서 연결돼 있다. 본 실험의 성공도 둘 중 무엇이 물리적 원인인지 분리하지 못한다.

## 10. 근거 자료

- [EBM 동결 실험 및 실제 구조 명세](ebm_frozen_model_experiments_20261105.md): 원래 학습과 캐시 readout의 차이.
- [구조 감사 기록](ebm_frozen_structure_audit_20261105.json): 기준 checkpoint·보관 소스 위치·코드 hash.
- [캐시 접근 구현](../../../workspace/artifacts/geometry_experiments/geometry_cache.py): 행 정렬·identity·배열 읽기.
- [기존 정상 reference 진단](../../../workspace/artifacts/geometry_experiments/test_normal_reference_mechanisms.py): DF 내부 fit과 이번 T-only 설계의 구분.
- [원음 길이 집계](../../artifacts/geometry_cache/local_analysis/normal_cause_audit_v1/training_duration_coverage.csv), [입력 길이 설정](../../artifacts/geometry_cache/local_analysis/normal_cause_audit_v1/training_input_lengths.json), [전체 DF EER tail](../../artifacts/geometry_cache/local_analysis/normal_cause_audit_v1/EER_tail_complete.json).
- [QAMO, arXiv:2509.20679](https://arxiv.org/html/2509.20679v1): 음질 조건별 정상 중심과 학습 목적을 사용하는 관련 연구. 본 설계의 사후 길이 조건 추정량과 동일하지 않다.
- [SAMO, arXiv:2211.02718](https://arxiv.org/abs/2211.02718): 화자별 정상 attractor를 사용하는 다중 중심 선행연구.
- [Duration-aware self-attention, Interspeech 2026](https://www.isca-archive.org/interspeech_2026/tu26_interspeech.html): 길이·구간 offset 조건을 attention에 반영하는 관련 연구. 길이별 중심의 직접 근거로 혼용하지 않는다.

**최종 보고 문장의 기준:** “고정 EBM의 동일한 캐시 표현에서, 학습 정상만으로 추정한 원음 길이별 중심이 어떤 대조군에 비해 얼마만큼 개선/악화됐는가”를 실제 결과대로 쓴다. 재학습 이득·음향 원인·독립 확증까지 확대하지 않는다.
