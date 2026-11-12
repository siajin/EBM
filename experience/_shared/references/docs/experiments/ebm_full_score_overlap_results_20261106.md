> 역사 문서 보존본. 원문 작성 당시의 상태·계획을 포함한다. 최신 상태와 확정 결과는 experience의 각 실험 README를 우선한다.

# 같은 EBM 원점수의 정상·스푸프: 전체 캐시와 기존 변형 결과 확인

2026-11-06 KST. 사용자 요청 ‘확인해봐’에 대한 로컬 CPU 분석이다. 전체 원점수 캐시를 조회하고 기존 변형 점수와 UID로 대조했다. SSH·서버 상태 조회·음성 forward·새 분류기fit·학습은 없다. 현재 서버가 유휴라는 뜻은 아니다. 취소한 DF4×3 CV도 실행하지 않았다.

**결론: 전체 캐시에서는 동일 원점수의 비교 표본을 더 확보할 수 있다. 그러나 해당 발화의 변형 반응 대부분이 저장되지 않아 기존 결과만으로 큰 표본의 조건부 반응 검증까지 완료할 수는 없다. 현재 저장된 작은 matched subset에서는 layer attention 균등화가 스푸프만 선택적으로 정상 방향으로 이동시킨다는 근거를 찾지 못했다.**

## 1. 자료와 매칭 정의

- 같은 epoch50/batch160 checkpoint `104873703a57cad98b686d770e0860e031a2cf9abef79b72668780c1660e0b24`.
- 전체DF533,928(정상14,869 / spoof519,059), 전체21LA148,176(정상14,816 / spoof133,360).
- 각20,000개에 R2-00/R2-10/R2-01/R2-11과 R3-noise20/noise10/Opus32k의 점수가 저장돼 있다. 각 조건의UID 집합이같고 native score도 전체cache와1e−10 이내로일치한다.
- 추가 padding471개·frontend/context471개·pooling471개는 각기 다른 조건이다. 같은UID라는 이유로 이를R2-10이나R3의반응으로 대체하지 않았다. 긴무반복crop은원래144개만있다.
- 엄격: source·speaker·codec·길이bin 일치, 원길이차이≤0.25초, native cosine차이≤0.02.21LA는transmission도일치한다.
- 완화: 위에서speaker일치만해제한다. 다른화자차이가남으므로엄격조건의증거와같이해석하지않는다.
- 길이bin은[0,2),[2,3),[3,4.0375),[4.0375,6),[6,∞)초다. caliper0.01/0.005도같은규칙으로계산했다.

매칭은반응을사용하지않고native score·metadata만사용했다. 먼저비복원쌍수를최대로만든뒤,그최대개수안에서원점수차이의합을최소화했다. 결과의최대쌍수는최소vertex cover와의일치로검증했다. 이전greedy숫자는가능한쌍수의하한이었으며이번이최대수다.

## 2. 전체native 표본과 기존반응의 실제범위

| 자료 | 조건 | 전체native에서가능한최대쌍 | 기존R2/R3반응이양쪽모두있는표본에서최대쌍 |
|---|---|---:|---:|
| DF | 엄격 | 515 | 6 |
| DF | 화자조건완화 | 2,041 | 53 |
| 21LA | 엄격 | 106 | 3 |
| 21LA | 화자조건완화 | 230 | 16 |

두열은각표본집합에서독립적으로최적화한쌍수다. 전체matching중기존반응이있었던쌍의개수나확률이아니며차이를직접빼누락쌍수로해석하지않는다. DF완화53쌍은이전greedy52쌍과알고리즘이다르다.

원점수차이caliper를좁혔을때:

| 자료/조건 | ≤0.02 전체/기존반응 | ≤0.01 전체/기존반응 | ≤0.005 전체/기존반응 |
|---|---:|---:|---:|
| DF엄격 | 515 / 6 | 348 / 2 | 226 / 0 |
| DF완화 | 2,041 / 53 | 1,628 / 31 | 1,255 / 17 |
| 21LA엄격 | 106 / 3 | 74 / 3 | 62 / 2 |
| 21LA완화 | 230 / 16 | 187 / 13 | 155 / 10 |

전체DF엄격515쌍은ASVspoof38/VCC2018149/VCC2020328쌍이며18개화자에한정된다.21LA엄격106쌍은27개화자다. 많은발화가독립화자수의부족을해결하지않는다. 원발화family관계도여전히미해결이다. DF엄격18화자는앞선설계의20화자실행가능성참고기준보다작으므로,이결과를검정력충족이라고부르지않는다.

전체DF엄격matching의평균native score차이는0.007745,21LA는0.006622다. 여전히완전동일점수가아니며caliper민감도와score조건보정이필요하다. source별차이·길이bin내차이·미관측조건을완전히제거한것도아니다.

## 3. 기존변형반응에서실제로나온값

`d=s_after−s_before`, `D=d_spoof−d_bona`다. 아래는caliper0.02의별도원점수최적화matching에대한점추정이며확증적검정결과가아니다.

### layer attention 균등화(R2-10)

| 자료/조건 | 쌍 | 보나mean(d) | spoof mean(d) | mean(D) |
|---|---:|---:|---:|---:|
| DF엄격 | 6 | +0.973411 | +0.973318 | −0.000093 |
| DF완화 | 53 | +1.365048 | +1.360899 | −0.004148 |
| 21LA엄격 | 3 | +1.962254 | +1.287154 | −0.675100 |
| 21LA완화 | 16 | +0.973535 | +0.842041 | −0.131494 |

DF 엄격 6쌍을 원점수 구간별로 나눈 결과는 다음과 같다. 구간은 pair의 원점수 평균으로 정의했고, 반응을 보고 구간을 선택하지 않았다.

| pair의 native score 평균 구간 | 쌍 | 보나 mean(d) | spoof mean(d) |
|---|---:|---:|---:|
| [−1,−0.9) | 3 | +1.928290 | +1.927621 |
| [0.9,1] | 3 | +0.018532 | +0.019015 |

**이작은DF표본에서는낮은원점수의양클래스가같이크게이동하고,높은원점수의양클래스는같이작게이동했다.** 따라서전체클래스평균차이보다원점수별반응을봐야한다는사용자의지적과일치한다. 다만구간당3쌍이므로전모집단에서조건부정보가없다는증명이아니다. 중간점수구간의반응은이엄격표본에없다.

그밖의R2-01/R2-11·잡음·Opus조건도같은pair로재계산했다. R2-01의D는DF완화+0.084476/21LA완화−0.386414로방향이다르고,잡음·Opus에서도현재작은표본만으로두도메인의‘정상유지/spoof정상이동’을확인하지못했다. 전체숫자는아래CSV에보존했다. 무작정유의하지않다는것을효과0이나동등성으로해석하지않는다. 군집/원발화문제와작은overlap을고려해이번에는효과유의성·CI를주장하지않는다.

## 4. 다음검증에필요한범위

확인한것은C0(native overlap와기존반응coverage)이며,C1의확대된반응효과는아직측정하지않았다. 전체음성을새로추론할필요는없다. 잠근matched pair들의누락발화에서만후보변형을계산하면된다.

현재저장한엄격caliper0.02의최대쌍·최소원점수차이목록에서R2/R3반응이없는endpoint는DF935개,21LA187개,합계1,122개다. 이는**이선택된pair목록기준**이며추론량최소화까지최적화한절대최소값은아니다. 해당UID·원점수·metadata만포함한manifest를저장했으며작업을실행·예약하지않았다. native점수는이미있고,새로운추론이필요한것은반응이다.

515/106쌍의반응을측정하더라도화자수·source-family한계가남는다. 실제독립검정력과추론군집을점검한뒤반응을추출해야한다. 원점수조건을풀거나화자를섞어서표본수를키운결과를엄격결과처럼대체하지않는다. 이미관찰한DF/21LA의탐색적후속이라는성격도유지한다.

## 5. 재현과검수

- 실행기: [check_full_score_overlap_20261106.py](../../../workspace/artifacts/geometry_experiments/check_full_score_overlap_20261106.py).
- 독립검수: [verify_full_score_overlap_20261106.py](../../../workspace/artifacts/geometry_experiments/verify_full_score_overlap_20261106.py).
- 정식결과v2: [overlap_and_coverage.csv](../../artifacts/geometry_cache/local_analysis/full_score_overlap_20261106_v2/overlap_and_coverage.csv), [existing_matched_responses.csv](../../artifacts/geometry_cache/local_analysis/full_score_overlap_20261106_v2/existing_matched_responses.csv), [score구간별반응](../../artifacts/geometry_cache/local_analysis/full_score_overlap_20261106_v2/existing_response_by_native_score.csv).
- `../../artifacts/geometry_cache/local_analysis/full_score_overlap_20261106_v2/missing_endpoints_for_locked_strict_pairs.parquet` (원자료 경로; 배포 제외 또는 미확보, 출처 목록 참조).
- [검수결과](../../artifacts/geometry_cache/local_analysis/full_score_overlap_20261106_v2/independent_verification.json).

v1은최대개수만최적화한중간결과로보존했고v2가최대개수내native score차이까지최소화한정식결과다. 두버전의최대개수는같고개별pair선택·반응점추정은달라질수있다. v2의원점수차이합은모든조건에서v1이하임을확인했다.

24개조건별pair파일의UID·label·중복·원점수gap·길이gap·stratum을독립검산했다. 서로겹치는설정을합친6,983개pair행을확인했으나이를독립표본6,983개로세지않는다. 반응산술최대오차0,입력hash불변이며작은그래프50개에서완전열거로최대개수와최소원점수차이의두목적을검수했다. 기존checkpoint·음성·cache는수정하지않았다.
