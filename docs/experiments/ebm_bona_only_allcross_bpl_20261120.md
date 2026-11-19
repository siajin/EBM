# 스푸프 BPL 제거 · 정상 전체 교차 BPL 유지

비교 기준은 스푸프 중심 OC + 스푸프/정상 전체 교차 BPL의 v3 모델이다. 새 버전은 스푸프 BPL 계수만 1에서 0으로 바꾸며 정상 전체 교차 BPL 계수1과 나머지 조건을 유지한다. 스푸프 중심을 정상 중심으로 바꾸는 실험이 아니다.

## 목적식과 두 단계 업데이트

정규화한 long/short 임베딩을 각각 u_i, v_i, 정규화한 스푸프 중심을 c, 정상 인덱스를 B, 스푸프 인덱스를 S라 한다. 원래 데이터 라벨0=spoof/1=bonafide를 유지한다.

```
L_backend = (1/N) sum_i w_i softplus(20 * d_i)
          + (1/|B|²) sum_{i,j in B} [softplus(30*(1-u_i·v_j)) - 0.3]
d_i = 0.9-u_i·c   (spoof)
      u_i·c-0.2   (bonafide)
w_i = 5/9         (spoof), 5 (bonafide)

L_center = (5/|S|) sum_{i in S} softplus(20*(0.9-u_i_new·c))
score_bona = -u·c
```

backend 단계에서는 중심을 동결하고 기존 Adam으로 backend만 갱신한다. center 단계에서는 backend를 동결하고 갱신된 backend로 스푸프 long 임베딩을 다시 계산해 중심만 갱신한다. 두 단계와 Adam 상태·weight decay 처리는 원래 방식이다. 위 수식의 c와 u는 정규화된 방향이며 실제 파라미터를 단위벡터로 투영하거나 새 optimizer를 넣지 않는다.

정상 BPL은 원본 EBM의 alpha30 전체 long×short mean과 값/기울기가 같다. 같은 UID와 다른 UID를 모두 포함한다. 정상 pair 수는 로컬120행에서144, 부분72행에서144, 부분114행에서36이다. 스푸프 pair는 모두0이며 제거한 항 대신 다른 손실 계수를 늘리거나 정상/스푸프 pair를 통합 평균하지 않는다.

## 논리·수학 검토

정상을 -c, 스푸프를 +c로 두면 정상 전체 교차 BPL과 OC의 개별 최소를 함께 달성할 수 있다. 따라서 '정상을 서로 모으면 스푸프 중심에서 멀어질 수 없다'는 구조적 모순은 없다. softplus 목적식은 유한한 구면 범위에서 아래로 유계이며, -0.3은 상수이므로 기울기를 바꾸지 않는다.

스푸프 short에는 OC가 원래 적용되지 않았으므로 스푸프 BPL 제거 후 직접 기울기는0이어야 한다. 반면 스푸프 long은 OC를 통해 계속 학습되고, 중심도 스푸프 long으로 계속 갱신된다. 정상 short의 BPL 기울기는 일반적인 비정렬 입력에서 양수 norm이어야 한다. 정상 임베딩이 완전히 정렬된 해에서는 이 norm이0일 수 있으므로 검증은 고정 비정렬 입력에서 수행한다.

정상/스푸프는 backend 파라미터를 공유하므로 정상 BPL만 남겨도 OC와의 국소 기울기 충돌 가능성은 남는다. 스푸프 OC 자체의 수축 압력도 그대로다. 따라서 이번 변경은 스푸프 BPL에 의한 항을 제거하는 ablation이며 anti-collapse나 일반화/EER 개선을 보장하지 않는다. 서로 다른 실험의 손실 총합이 낮아졌다는 이유만으로 성능 향상을 판단하지 않는다.

## 비교 조건과 구현

두 전체 long/short forward와 모든 입력 행을 유지한다. 모델은 행 사이 BatchNorm이 없는 LayerNorm 구조이고 dropout/ASP 난수 소비도 유지한다. 비활성 스푸프 BPL은 계산 자체를 건너뛰어 NaN을0으로 곱하는 우회가 없다. 가중치1인 기존 경로는 원래 연산 순서를 유지한다. 기본값1과 v1~v5 metadata를 그대로 보존하고, 계수가 다른 모델은 `ebm_role_v6`에 스푸프 계수를 명시해 잘못된 목적식으로 재시작/평가하지 못하게 한다.

```
python tools/train_role_experiment.py --spoof-bpl-weight 0 --spoof-bpl-pairing all_cross --bona-bpl-weight 1 --bona-bpl-pairing all_cross --pair-target-cosine 1 --r-pull .9 --eval-input front64600_wrap --final-eval-policy deferred_shared
```

위 명령의 기본 mode는 validate이며 학습을 실행하지 않는다. 새 학습은 seed1에서50epoch로 시작하고5epoch마다19LA만 평가한다. 종료 후10개 저장 head가 batch별 XLSR 출력을 공유해21LA/DF를 평가한다. 기존240 및 dual all-cross와 동일 epoch·공통15~50 범위·고정50·관측최저/후보수 모두 비교한다. target별 관측최저는 사후 선택이다.

이번 작업은 구현·CPU 회귀·실제 두 GPU 사전 검증까지 진행하고 본학습은 시작하지 않는다. smoke epoch0은 실제 학습 모델과 구분하며 새 학습 초기화에 재사용하지 않는다. 검증 결과는 `experiments/bona_only_bpl_20261120`에 코드·설정과 함께 기록한다. 원음·가중치·음성별 점수는 GitHub에 올리지 않는다.

## 학습 후 공유 평가 경로

이전 실험의 `finalize.py`에는 v5·스푸프 same UID 역할과 `spoof_uid` 결과 열이 고정돼 있으므로 새 모델에는 재사용하지 않는다. 보존된 검증 파이프라인의 reference test용 파일로만 남긴다. 새 버전은 `finalize_bona_only.py prepare_shared ROOT RESULT`로 정확한 v6 역할을 확인하고19LA10개 결과를 보존하며21LA·DF의10head 공유 평가 manifest를 만든다. 전체 평가 후 `finalize_bona_only.py compare ROOT RESULT`로 기존6개 모델의 결과를 유지한 채 새 `bona_only` 열을 추가한다.

정상/스푸프 전체교차 v3와의 차이를 주 비교로 두고, 스푸프 same UID v5 결과를 덮어쓰지 않는다. 추가 통합 검사는 v6 manifest 준비·잘못된 역할 거부·실제 v6 head의 공유 점수/RNG·합성 전체평가 집계를 확인한다. 합성 점수는 임시 테스트 데이터이며 실제 모델 성능 근거로 사용하지 않는다.
