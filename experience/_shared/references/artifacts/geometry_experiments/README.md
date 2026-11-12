> 역사 문서 보존본. 원문 작성 당시의 상태·계획을 포함한다. 최신 상태와 확정 결과는 experience의 각 실험 README를 우선한다.

# EBM 기하 분석 첫 실행

2026-11-05 KST, 사용자 요청에 따라 재학습 없는 빠른 진단부터 실행했다. 전체 명세는 [실험 명세](../../docs/experiments/ebm_embedding_geometry_spec_20261105.md)에 있으며 이 실행은 그 축소판이다. **01:29 KST에 추출·축소 분석·출력 검증의 정상 완료를 확인했다. 전체 명세 완료 또는 가설 확인을 뜻하지 않는다.**

후속 상태: **11:29 KST 21LA 전체 eval 캐시 추가·독립 검수까지 완료했다.** 현재 재사용 가능한 자료는 T/V/DF pilot/21LA eval 총208,539개이며, 하단의 로컬 캐시 절과 [21LA 기록](../../docs/experiments/21la_cache_20261105.md)을 따른다.

후속 분석의 기본 표본 수는 **DF_pilot20,000개와21LA_pilot20,000개(각정상2,000/spoof18,000)**다. 21LA 전체148,176개 캐시는 보존하고 마지막 보조 평가에만 사용한다. R2/R3 초기 확인은 각pilot 안의2,000개(200/1,800)씩이며, 각 실험별 수와 고정 ID 읽기는 [동결 명세3.1.1](../../docs/experiments/ebm_frozen_model_experiments_20261105.md#311-신규-캐시-실험의-기본-데이터-수)을 따른다. 분석 표본 이름은 raw cache API의 새 별칭이 아니며 manifest의 cache_row로 원본 배열을 선택한다.

최신 후속 결과: **2026-11-05 09:06 KST 재추출 없는 네 가지 CPU 진단 완료.** [결과 문서](../../docs/experiments/ebm_cached_core_results_20261105.md)와 `geometry_cache/local_analysis/cached_core_v1/`을 참고한다. DF 내부 nested 화자 교차검증에서는 방향 추가가 scalar·잡음 대조보다 좋았고, 고정 코사인의 T-Fisher 상위 채널 제거는 무작위 제거보다 크게 악화되지 않았다. LayerNorm 전후와 조건별 방향 군집 비교도 포함한다. DF 내부 학습 결과를 T→DF 전이 성능으로 해석하지 않는다.

채널 축소 후속: **09:59 KST 80→10개 유지 실험 완료.** [가설·방법·결과·해석](../../docs/experiments/cached_core_20261105/02_channel_reduction_to10.md), 코드 `channel_keep_sweep.py`, 결과 `geometry_cache/local_analysis/channel_keep_sweep_v1/`에 있다. 하위10개 EER2.9444%, 상위10개5.0500%, 무작위10개20회의 중앙값3.6694%다. 전체160개는3.3500%다. 기존160차원 표현에서 점수 계산 좌표만 줄인 실험이며 새 추출·학습은 없다.

더 작은 채널 수: **10:13 KST 5·3·2개 유지 완료.** [실험 문서](../../docs/experiments/cached_core_20261105/02_channel_reduction_to2.md), 코드 `channel_keep_tiny_sweep.py`, 결과 `geometry_cache/local_analysis/channel_keep_tiny_v1/`이다. 하위 유지 EER2.9500/2.9222/2.9222%, 상위5.6500/6.8500/6.8500%, 무작위 중앙3.6194/4.6472/5.5750%다. 하위2개는DF AUROC와V EER가악화돼EER만으로전체성능개선을주장하지않는다. 기존10개23조건은오차0재현, 새실행267조건독립검증pass다.

1개까지 완료: [1개 결과와 2개 실험의 쉬운 설명](../../docs/experiments/cached_core_20261105/02_channel_reduction_to1_and_explanation.md). 코드 `channel_keep_one.py`, 완료 결과 `geometry_cache/local_analysis/channel_keep_one_v3/`다. 하위1개 EER4.3459%, 상위1개14.8255%, 무작위중앙13.5020%이며 정확한1차원코사인의±1동점을보존했다. 기존2개23조건오차0재현, 새135조건검증pass. v1/v2는독립DET검증의trimming문제를발견해미완료로보존했다. 실제두좌표와한좌표분포그림도포함한다.

서버 루트: `/home/mentee/analysis/embedding_geometry/20261105_fast_p0_01`.

| 단계 | 범위 | 확인 상태 |
| --- | --- | --- |
| prepare | 공식 protocol 점검, T/V/Q 화자 분할 | exit 0 |
| QC | T 128개, 반복·GPU 간 일치·native score 확인 | exit 0 |
| extract | T+V 40,363개, h/z 각 160차원 | exit 0, 약20분55초 |
| fast analysis | C/A/E/D 일부, 아래 계약 | exit 0, 약27초 |
| output verification | 점수 범위, 행 정렬, log-loss·거리 항등식 재계산 | exit 0 |

`fast_extract.py`와 `fast_analyze.py`는 이번 작업에서 새로 작성한 runner다. 서버 원본 연구 코드를 복사한 파일이 아니다. 원음·원시 임베딩·원래 checkpoint·발화별 metadata와 예측은 서버에만 둔다. 두 파일은 서버에 `fast_extract_v1.py`, `fast_analyze_v1.py`로 업로드했다.

## 데이터와 추출

T는 19LA 공식 train 전부(25,380개), V는 공식 dev에서 seed 11로 나눈 10명 화자(14,983개)다. 나머지 10명 Q(9,861개)는 이번 임베딩 추출과 분석에서 열지 않는다. train/dev 화자 중복, 중복 발화 ID, 누락 파일 경로는 0이다. source-family 파생 관계는 protocol만으로 확인할 수 없어 미해결이다.

기준 checkpoint는 기존 batch160의 epoch50 `BestCLS_50_21DF.pt`다. 이름을 최적 checkpoint의 근거로 해석하지 않는다. strict classifier load, offline XLS-R reload, archived native 평가 모듈, 발화별 seed, microbatch 1, 고정 crop/padding을 사용한다. frozen frontend의 과거 실제 가중치와 별도 해시 일치를 증명한 것은 아니다.

QC 결과: 정상 32개와 A01~A06 각각16개, 총128개. 반복 및 GPU 간 z 차이 0. native cosine 최대 오차 `1.4797872960325975e-07`. tangent 항등식 및 유한값 점검 통과. 모델 로딩을 제외한 QC 처리량은 총30.727개/초이며 전체 소요 시간의 보장값이 아니다.

## 축소 분석 계약

1. **C**: spline(s), log(norm), valid 지시자를 대조한 뒤 tangent 단위 방향 u의 160좌표를 추가한다. 같은 차원의 label 독립 unit noise를 보조 대조군으로 넣는다. L2 logistic probe만 사용한다.
2. **A**: T에서 공격별 Fisher score를 계산해 macro 평균으로 순위를 정한다. 전체160채널, 상위40채널, 무작위40채널 20개를 logistic probe로 비교한다. 채널 attention을 학습하는 실험은 아니다.
3. **E**: T 정상의 unit z 공분산만 사용해 고정 learned centroid 주변 대각/전체 metric을 만든다. shrinkage 0.01/0.1/0.5/1.0을 모두 보고한다. gamma=1은 정규화 Euclidean과 cosine의 동등성 대조군이다. 후보 중 V 최저값은 독립 검증 성능이 아니다.
4. **D**: T의 공격별 평균 방향 크기와 uncentered 2차모멘트 participation ratio를 기술한다. 평균 방향 크기만으로 다봉 구조 또는 구조 부재를 판정하지 않는다. V 재현성과 표본 수가 같은 isotropic 대조는 아직 수행하지 않는다.

probe/scaler/순위/covariance는 T에서만 fit한다. C/A 정규화 계수 C는 0.01/0.1/1/10 중 V의 held-out 화자 이외로 선택해 합친다. 10명 화자에 대한 V 내부 5-fold 교차 선택 결과다. C의 log-loss 차이에는 2,000회 화자 bootstrap을 붙이지만, 개발 결과를 완전히 독립적인 검정처럼 해석하지 않는다. 정상/스푸프 각각50% 가중 log-loss이며 y=1과 큰 score는 정상이다.

전체 명세의 MLP, 반복 seed, 나머지 mask 비율, 조건부 MMD, 방향 shuffle, Q/외부 평가, perturbation F, attention 및 손실 재학습 B/G는 실행하지 않는다. 이번 결과로 미지 공격 일반화, attention의 효과, 새 loss의 효과를 확인했다고 주장하지 않는다.

## 재접속 시

먼저 `../../MENTEE_SERVER.md` (원자료 경로; 배포 제외 또는 미확보, 출처 목록 참조)와 현재 서버 상태를 확인한다. 서버 루트의 `extract/rank*_progress.json`, `extract.exit`, `fast_analysis.exit`를 읽고, 실패했을 때만 해당 로그 마지막 부분을 확인한다. 성공은 종료 코드0과 단계별 `complete`/`summary.json`으로 확인한다. 실행 중인 launcher를 중복 실행하지 않는다. 원래 EBM 로거와 `main.py`를 호출하지 않는다.

수학 검토: 기존 명세의 합성 25개 검사는 [결과](../../docs/experiments/spec_validation.json)에 있다. 이번 runner도 컴파일 후 EER의 완전/동률/역순, balanced log-loss 방향, tangent 직교성, 화자 교차 선택의 전체 표본 예측, paired bootstrap 개선 방향을 합성 입력으로 확인했다. 이는 실제 성능 검증을 대체하지 않는다.

## 1차 결과와 해석

V에서 기존 cosine EER가 이미0%, AUROC1이다. 정상 최소 cosine0.509365와 스푸프 최대-0.290336 사이에0.799701의 간격이 있어, 별도 점수 범위 계산도 완전 분리를 확인했다. C 다섯 probe와 E 여덟 후보도 EER0%다. 무작위40채널20개 중19개도 EER0%여서 이 V로 추가 탐지 이득을 비교하기 어렵다.

| C 비교 | V balanced log-loss |
| --- | ---: |
| C0: cosine spline | 0.0029737963 |
| C1v: cosine spline + norm + valid | 0.0042803381 |
| C2: C1v + tangent 방향 | 0.0003809358 |
| Cnoise: C1v + 독립 unit noise | 0.0004038119 |

방향 추가의 log-loss는 감소했지만 noise 대조도 비슷한 값이다. 방향 고유의 정보나 새 loss의 효과가 입증됐다고 해석하지 않는다. 상위40채널의 log-loss0.0000774292는 무작위20개 중앙값0.0001763974보다 작지만, attention 학습이나 미지 공격의 개선은 아직 확인하지 않았다.

D에서 학습 공격별 평균 방향 크기0.9342~0.9559, 2차모멘트 participation ratio1.1954~1.3083을 얻었다. 이 통계만으로 서로 다른 공격 방향이나 정상과의 방향 차이를 확인한 것은 아니다. centered covariance 차원과도 구분한다.

서버의 `verify_fast_outputs_v1.py`는 자체 작성한 [검증 코드](../../../workspace/artifacts/geometry_experiments/verify_fast_outputs.py)다. 정상/스푸프 점수 범위, 전체 행과 예측 정렬, C log-loss 독립 재계산을 확인했다. 정규화 Euclidean과 cosine의 최대 차이는6.66e-16이다. 서버 `output_verification_v1.json`에 집계 검증 결과, `fast_analysis/summary.json`에 분석 집계가 있다. 원시 임베딩과 발화별 결과는 계속 서버에만 둔다. 마지막 확인 시 지정 GPU는 유휴였다.

## 후속 DF pilot — 완료

사용자가 DF에서 비교하도록 요청해 `/home/mentee/analysis/embedding_geometry/20261105_df_fast_01`에서 별도 실행했다. **2026-11-05 01:52 KST 기준 prepare/freeze/QC/extract/evaluate 모두 정상 완료됐다.** 추출은 모델 로딩 제외약9분58초, CPU 평가는약15초다.

원래 평가 코드와 [공식 DF metadata](https://github.com/asvspoof-challenge/2021/blob/main/eval-package/README.md#df)의 `eval`533,928개에서 정상2,000/스푸프18,000개를 SHA256 발화ID 우선순위로 클래스 내 균등 추출했다. 점수와 무관하고 행 순서에도 불변이다.110개 공격·9개 codec을 모두 포함한다. 이2만 개 수치를 전체 DF 성능이라고 부르지 않는다. 기존 추출 속도로 pilot10~11분, 전체 약4시간40분이 예상된다.

T에서 분석용 분류기와 통계량을 fit하고 V 전체에서 설정과 threshold를 고정한 뒤 DF를 평가한다. DF에서 새로 fit하거나 설정을 선택하지 않는다. 기존 V 내부 교차 선택 예측과는 달리 각 probe의 C를 V 전체로 하나 선택한 모델을 보존한다. 기존 E의 V EER 동률 규칙에 따라 주 metric은gamma1=cosine으로 선택됐다. 사전에 고정한 나머지 covariance 후보는 민감도 분석으로 모두 보고하되 DF 최저값을 새 주 모델로 채택하지 않는다.

자체 작성 runner는 [DF 추출](../../../workspace/artifacts/geometry_experiments/df_extract.py), [설정 고정과 평가](../../../workspace/artifacts/geometry_experiments/df_screen.py), [합성 수치 검증](../../../workspace/artifacts/geometry_experiments/verify_df_math.py)이다. 원래 연구실 코드를 수정하지 않는다. 기존 컨테이너의 GPU0/1을 사용하고, 발화별 난수·native 평가 경로·최종 LayerNorm 전후 추출을 유지한다. DF QC128개는 반복/GPU 간 오차0, native score 오차1.00e-7 이하로 통과했다. source×speaker93그룹으로 paired bootstrap을 산출하지만 source-family 매핑은 미해결이다.

`run_DF_pilot_v1.sh`가 prepare→freeze→qc→extract→evaluate를 자동으로 이어간다. 종료 코드 `extract.exit`, `evaluate.exit`와 `DF_evaluation/complete`, `DF_evaluation/summary.json`으로 완료를 확인한다. `registry.json`, `frozen_models.joblib`은 DF 추출 전에 고정됐고 Q는 열지 않았다. DF는 이미 관찰한 benchmark이며 이 실행은 탐색용 pilot이다. 상세 인수인계는 `../../MENTEE_SERVER.md` (원자료 경로; 배포 제외 또는 미확보, 출처 목록 참조)에 기록했다.

## 노트북 PC의 결과 캐시와 오프라인 재분석

2026-11-05 후속 사용자 요청으로 **이번 실험에서 생성한 파생 결과를 로컬 PC에도 보관**하도록 저장 정책을 확장했다. 이 절의 노트북은 Jupyter 문서가 아니라 현재 Windows PC를 뜻한다. 원래 연구실 소스·원음·checkpoint를 복사하는 절차는 포함하지 않는다. 실험의 split, 모델 선택, 통계 규칙은 그대로다.

캐시 폴더는 `artifacts/geometry_cache/`, 색인은 `catalog.json`이다. **T25,380개·V14,983개·DF pilot20,000개·21LA eval148,176개, 합계208,539개를 저장했다.** 파생 결과53파일306,968,382바이트(약307MB)다. Q·19LA eval·DF 전체는 아직 캐시가 없다. 이전 절의 서버 전용 저장 설명은 사용자 저장 요청 전의 기록이다.

21LA 별칭은 `21LA_eval`이다. `load_embeddings("21LA_eval")`로 h/z 각[148176,160]과 native score·codec·transmission metadata를 읽는다. 같은 기준 checkpoint이고 원래 단일 발화와 QC128개의 h/z/score 오차가0이다. 전체 native EER1.0535393%, AUROC.997452884를 독립 검수했다. `verify_la21_cache.py`와 `local_analysis/21la_cache_validation_v1/verification.json`을 참고한다. 기존 C/A probe를21LA에 새로 fit/evaluate한 것은 아니며 `analysis_complete=false`는 그 추가 분석의 미실행을 뜻한다. native 캐시는 완료·검증됐다.

저장 항목은 LayerNorm 전후 h/z, 정규화 centroid, 발화별 점수·label·분석 metadata, 기존 예측, 집계 결과다. 원래 음성 파일 경로를 포함하는 source manifest, 원본 소스, checkpoint, 인증 파일, 실행 로그, pickle 모델을 가져오지 않는다. 모델 SHA256·추출 코드 SHA256·manifest SHA256·frontend commit·실행 식별자로 캐시 버전을 구분하며, 각 파일의 크기와 SHA256을 서버/로컬에서 대조한다. 과거 frontend 가중치의 별도 비트 단위 해시 일치를 증명한 것은 아니며 그 한계도 기록한다.

프로젝트 폴더에서 새 완료 결과를 받을 때:

```powershell
python -X utf8 artifacts/geometry_experiments/sync_geometry_cache.py
```

완료되지 않은 추출은 건너뛰며, 이미 검증된 동일 파일은 다시 다운로드하지 않는다. 모델 추론이나 학습을 시작하지 않는다. 사설망 SSH 접속은 이 동기화에만 필요하다.

저장된 결과만으로 다시 분석할 때:

```powershell
python -X utf8 artifacts/geometry_experiments/analyze_geometry_cache.py
```

이 명령은 서버 연결·GPU 없이 캐시를 읽어 native 점수와 저장된 probe 지표를 재계산한다. 기존 집계와의 수치 일치, 파일 해시, z/h 행 정렬, 차원, 유한값, native score 대응을 확인하고 `geometry_cache/local_analysis/<분석버전>/metrics.csv`와 `verification.json`에 저장한다. 아직 없는 데이터는 누락 사유를 남긴다.

새 분석에서는 다음 API로 기존 벡터를 재사용할 수 있다.

```python
from artifacts.geometry_experiments.geometry_cache import load_embeddings, load_predictions

data = load_embeddings("V")  # "T", "V", "DF_pilot"
z, h, metadata = data.z, data.h, data.rows
direction = data.geometry()["direction"]
predictions = load_predictions("V")
```

`load_embeddings(..., expected_checkpoint_sha256="...")`로 모델 일치를 강제할 수 있고, 서로 다른 모델 캐시를 임의로 섞지 않는다. 다른 모델·입력 변형·crop·추출 방식의 임베딩은 별도 추출과 버전이 필요하다. 현재 캐시는 같은 모델의 분포 분석·거리 변경·채널 선택·CPU probe 등에 재사용할 수 있다. 새 모델 전체 학습을 캐시가 대신하지는 않는다.

로컬 파생 결과 폴더는 `.gitignore`에서 제외했다. 자세한 저장 범위와 검증 결과는 `../../MENTEE_SERVER.md` (원자료 경로; 배포 제외 또는 미확보, 출처 목록 참조)에 이어 기록한다.

검증된 최신 재계산 결과는 `artifacts/geometry_cache/local_analysis/6ac3505da0708324/metrics.csv`, 검증 기록은 같은 폴더의 `verification.json`이다. 서버/GPU 없이43개 지표를 재계산해 서버 집계와1e-9 이내로 일치했다. 추가 동기화에서도38파일을 모두 재사용했고 새 다운로드는0바이트였다.

DF pilot의 native EER는3.35%, C1v14.55%, 방향 추가C2는11.2389%, 잡음 대조Cnoise16.80%, 전체채널 probe13.05%, 상위40채널 probe5.3111%다. 방향 추가가 C1v probe보다 좋아도 native cosine보다 나쁘므로 기존 탐지기를 개선했다고 해석하지 않는다. DF 전체 수치가 아니며 V에서 선택한 설정을 DF에서 다시 고르지 않았다. 상세 결과는 캐시의 `DF_evaluation/summary.json`에 있다.

### 캐시에서 추가로 확인한 해석

사용자의 결과 설명 요청에 따라 [해석 코드](../../../workspace/artifacts/geometry_experiments/explain_df_results.py)를 로컬에서 실행했다. `geometry_cache/local_analysis/df_interpretation_v1/`에 `interpretation.json`, `metric_contrasts.csv`, `df_results.png`를 저장했다. 원래 고정 registry와 예측은 변경하지 않았고 GPU·서버 접속 없이 수행했다.

비등방 대각 거리의 후보 EER는3.05~3.1778%, 전체 공분산 후보는3.0833~3.15%다. 대각gamma0.01의 native 대비 delta0.30pp, source×speaker93그룹 paired bootstrap2,000회의95% CI[0.0680,0.3862]를 얻었다. 다만 DF 결과를 관찰한 뒤 추가 해석한 여섯 후보의 보정 없는 탐색 CI이며 source-family 파생 관계도 미해결이다. V에서 사전 선택된gamma1의 주 결과는 여전히cosine과 같다.3.05%를 독립 확인 성능으로 바꾸어 보고하지 않는다.

C의 cosine-only spline probe C0조차 DF EER17.10%로 native3.35%보다 나빴다. 실제 DF score를 정렬했을 때 변환 확률이 감소하는 인접 구간을 확인해, 단조 제약 없는 spline이 native 점수 순서를 보존하지 않음을 검증했다. 따라서 C2의 C1v 대비 개선만으로 방향의 고유 판별력이나 새 loss 효과를 확정할 수 없다. 다음 진단은 native score를 보존하는 대조군이 필요하다.

상위40채널5.3111%는 무작위40채널20개의 중앙값12.10%보다 낮지만 최저 무작위 subset3.4111%보다 높다. 전체160채널 probe 대비 개선이 attention 또는 native detector 대비 개선의 증거는 아니다. V 정상 오거부율0.962%에서 고정한 native threshold0.9883067은 DF에서 정상39%를 거부했다. EER의 DF threshold sweep과 이 V-fixed 운용 오류를 구분해야 한다.

### 코사인 경로를 유지한 재검증

[검증 코드](../../../workspace/artifacts/geometry_experiments/validate_native_preserving.py)를 같은 캐시에서 실행했다. 결과 폴더는 `geometry_cache/local_analysis/native_preserving_v1/`이고 `summary.json`, `metrics.csv`, `paired_intervals.csv`, `loss_geometry_audit.json`, `validation.png`를 보관했다. 원래 결과를 덮어쓰지 않은 새 탐색 버전이다.

T에서 fit한 양의 기울기의 scalar logistic calibration은 DF EER3.35%를 정확히 유지했다. 이어 native s의 계수를1로 고정한 `S=s+epsilon*tanh(f(u)/2)`를 fit했다. C와epsilon은 T/V만으로 선택했고 보정 크기는0.1 이하다. 방향 보정도 DF EER3.35%, 개선95% CI[-0.0603,0.0954]pp로 이득을 확인하지 못했다. baseline calibrated log-loss0.56368은 방향 보정0.62813보다 낮았다. 잡음 보정은 V에서epsilon0을 선택해 baseline과 같았다. 순서 보존은 fixed u에서 s의 경로와 calibration에 대한 것이며, 모든 샘플의 상대 순서를 보존한다는 뜻은 아니다.

가중 거리3.05%는 재현했지만 이번9개 비교를 보정한 대각gamma.01 개선 구간은[0.0000,0.4641]pp로0을 포함했다. 새 검증에10000회 source×speaker paired bootstrap을 적용했고, DF의 이전 관찰·탐색 이력과 원발화 파생 의존성까지 해소한 것은 아니다. 가중 거리의 주 V 선택은 여전히identity다.

원래 OC는같은cosine에서 tangent 방위가 바뀌어도 값이 같음을 가상 임베딩 수치 실험으로 확인했다. OC loss 최대 차이는5.83e-16이었다. BPL은 정상 사이 방향 관계에는 관여하나 spoof 방위는 직접 입력받지 않는다. 모델에는 이미 학습 가능한 채널 변환 및 레이어/시간 attention이 있으므로 '방향이나 채널을 전혀 사용하지 않는다'고 해석하지 않는다. 여기서 확인한 것은 스푸프 잔여 방위가 기존 목적에서 별도로 식별되지 않는다는 점이다. 실제 유용 정보 손실이나 후처리 실패의 원인으로 확정하려면 새 목적의 공동 학습 비교가 필요하다. 이번 분석은 기존 표현을 고정했으며 새 EBM 학습이 아니다.

### 실제 임베딩과 설명용 기하 시각화

[그림 생성 코드](../../../workspace/artifacts/geometry_experiments/visualize_embedding_geometry.py)를 캐시에서 실행했다. 결과는 `artifacts/geometry_cache/visualizations/embedding_geometry_v1/`에 보관했다. 원래 모델의 현재 표현을 그린 것이며 새 attention/loss 모델의 결과가 아니다.

- `01_objective_geometry.png`: 같은cosine·다른잔여방위의 가상3차원 설명도. 실제 데이터가 아니며 arbitrary metric으로 점수 차이를 설명한다.
- `02_score_distributions.png`: T/V/DF 전체 native 점수를 클래스별100%로 정규화한 공통80bin 히스토그램. 로그축을 명시하고0인 bin은 생략했다. V 문턱값0.9883067을 함께 표시했다.
- `03_embedding_geometry.png`: T 클래스별2,500개만으로 fit한 잔여 단위방향u의 PCA를 T/V/DF에 같은 축으로 적용했다. 위 행은cosine s 대u-PCA1, 아래는u-PCA1 대u-PCA2다. 그림에는 각 dataset·class별1,000개를 고정 표본으로 표시하고 지표에는 전체를 사용했다. 두 축의 T 설명분산은36.83%와18.03%다. PCA component와c의 최대 내적은1.64e-15로 잔여평면과 일치한다.
- `04_channel_structure.png`: T 정상 분산 기반 대각 거리계수의 상위24개와, 기존T Fisher순서를 고정한 클래스 평균차이 heatmap이다. 거리계수와 Fisher는 서로 다른 지표이며 attention 학습 결과나 인과적 중요도가 아니다. 거리계수 전체 범위는0.332~1.874(평균1)다.

분홍 삼각형은 **V에서 고정한 문턱값으로 거부된 DF 정상**이다. DF EER threshold에서의 오류 표시가 아니다. DF정상39%가 해당하며, 투영 그림에서는 T의 정상 군집을 벗어난 정상 일부가 보인다.2차원 투영의 겹침/분리로 원래160차원의 판별력이나 정보 부재를 확정하지 않는다.

재사용용 PCA model과 fit indices는 `T_direction_pca.npz`, 그림용6,000행은 `scatter_points.parquet`, 채널 수치는 `channel_statistics.csv`, 근거와 설정은 `visualization_stats.json`에 저장했다. 모든 그림을 렌더링해 확인했고, glyph 문제를 고친 뒤 로그축 표시도 재확인했다. 별도 `visual_validation.json`에 T-only balanced fit, 좌표 직교성, 행 수와 클래스별 표시 수 검증을 기록했다. GPU·서버 접속·Q·신규 학습은 사용하지 않았다.
