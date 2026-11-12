"""Document archived norm-related evidence; no model fitting or audio inference."""
import csv
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
REPORT = ROOT / 'docs/experiments/ebm_norm_information_research_20261111.md'
OUT.mkdir(exist_ok=True)

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

if REPORT.exists() and not (OUT / 'plan.json').exists():
    raise RuntimeError('An existing report must not be overwritten by this side conversation.')
preserve_paths = [
    'docs/experiments/ebm_five_research_themes_20261111.md',
    'docs/experiments/ebm_completed_evidence_interpretation_20261111.md',
    'docs/experiments/ebm_duration_bias_research_20261111.md',
]
if not (OUT / 'plan.json').exists():
    save('plan.json', {
        'objective': 'Create a separate Korean Markdown report using completed controls that support the bounded role of pre-normalization magnitude statistics and direction-scale interaction.',
        'status': 'in_progress', 'spending_ceiling_usd': 0,
        'new_training': False, 'new_audio_inference': False, 'new_readout_fits': 0,
        'remote_queries': False, 'parent_thread_changes': False,
        'preserved_documents': {p: sha(ROOT / p) for p in preserve_paths},
        'cleanup': 'Synchronous local documentation and saved-prediction arithmetic only; no background or paid resource.',
        'resources_left_running': 0,
        'selection_rule': 'Keep completed controlled comparisons, use the metric actually supported, retain contrary EER outcomes, and do not equate joint mean/scale evidence with norm-only efficacy.'
    })

sources = {}
def source(p):
    q = ROOT / p
    sources[p] = {'path': p, 'sha256': sha(q)}
    return q

def read(p):
    return json.loads(source(p).read_text(encoding='utf-8-sig'))

def csv_rows(p):
    with source(p).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def link(p, name):
    source(p)
    return f'[{name}](../../{p})'

def table(headers, rows):
    return '\n'.join(['|' + '|'.join(headers) + '|', '|' + '|'.join(['---'] * len(headers)) + '|'] + ['|' + '|'.join(map(str, row)) + '|' for row in rows])

BASE = 'artifacts/geometry_cache/local_analysis/frozen_all_v1/P3_P4/B'
HYP = 'artifacts/geometry_cache/local_analysis/cache_hypotheses_20261106_v2'
checks = []
def check(name, ok, detail=None):
    checks.append({'name': name, 'passed': bool(ok), 'detail': detail})
    if not ok:
        raise AssertionError((name, detail))

def eer(y, scores):
    order = np.argsort(-scores, kind='stable')
    y, scores = y[order], scores[order]
    ends = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(scores)-1]
    fp = np.r_[0., np.cumsum(y == 0)[ends] / (y == 0).sum()]
    fn = 1. - np.r_[0., np.cumsum(y == 1)[ends] / (y == 1).sum()]
    gap = fp - fn
    i = np.flatnonzero(gap >= 0)[0]
    if gap[i] == 0:
        return float(fp[i] * 100)
    mix = -gap[i-1] / (gap[i] - gap[i-1])
    return float(100 * (fp[i-1] + mix * (fp[i] - fp[i-1])))

def ll(y, logits):
    # Match the archived probability-clipping policy without using its metric helper.
    p = np.exp(-np.logaddexp(0., -logits))
    p = np.clip(p, 1e-7, 1-1e-7)
    return float(-.5*np.log(p[y == 1]).mean() - .5*np.log1p(-p[y == 0]).mean())

variants = ['direction_linear', 'direction_hstat', 'direction_noise2', 'direction_h', 'direction_noise160']
metrics = {}
effects = {}
means = {}
max_eer_error = max_ll_error = 0.
for ds in ['DF_pilot', '21LA_pilot']:
    complete = read(f'{BASE}/{ds}/complete.json')
    check('completed_' + ds, complete['status'] == 'complete' and complete['outer_folds'] == 5 and complete['inner_folds'] == 3 and complete['seeds'] == [101, 102, 103])
    metrics[ds] = csv_rows(f'{BASE}/{ds}/metrics.csv')
    effects[ds] = read(f'{BASE}/{ds}/paired_effects.json')
    columns = ['utterance_id', 'y', 'dependency_group'] + [f'{v}_{s}' for v in variants for s in [101, 102, 103]]
    frame = pd.read_parquet(source(f'{BASE}/{ds}/predictions.parquet'), columns=columns)
    y = frame.y.to_numpy(dtype=int)
    check('prediction_population_' + ds, len(frame) == 20000 and frame.utterance_id.nunique() == 20000 and (y == 1).sum() == 2000 and (y == 0).sum() == 18000)
    check('speaker_group_count_' + ds, frame.dependency_group.nunique() == (93 if ds == 'DF_pilot' else 67))
    means[ds] = {}
    for v in variants:
        records = [r for r in metrics[ds] if r['variant'] == v]
        check('three_saved_seeds_' + ds + '_' + v, sorted(int(r['seed']) for r in records) == [101, 102, 103])
        for r in records:
            scores = frame[f"{v}_{r['seed']}"].to_numpy(dtype=float)
            e = abs(eer(y, scores) - float(r['eer_percent']))
            l = abs(ll(y, scores) - float(r['balanced_logloss']))
            max_eer_error = max(max_eer_error, e)
            max_ll_error = max(max_ll_error, l)
            check(f"independent_metrics_{ds}_{v}_{r['seed']}", e < 1e-9 and l < 1e-9)
        means[ds][v] = {k: float(np.mean([float(r[k]) for r in records])) for k in ['eer_percent', 'balanced_logloss']}

def contrast(ds, baseline, candidate):
    return next(r for r in effects[ds] if r['baseline'] == baseline and r['candidate'] == candidate)

p4verify = read('artifacts/geometry_cache/local_analysis/frozen_all_v1/P3_P4/verification.json')
check('archived_fold_controls_verified', p4verify['status'] == 'pass' and p4verify['all_scalers_train_fold_only'] and p4verify['outer_test_not_used_for_ridge_epoch'])
reconstruction = read(f'{HYP}/H5_LN_reconstruction.json')
check('reconstruction_completed', reconstruction['status'] == 'complete' and len(reconstruction['checks']) == 4 and all(r['h_reconstruction_max_abs_error'] < 1.3e-7 for r in reconstruction['checks']))
decomposition = {ds: read(f'{HYP}/H5_{ds}_saved_readout_decomposition.json') for ds in ['DF_pilot', '21LA_pilot']}
independent = read(f'{HYP}/independent_verification.json')
check('archived_decomposition_independent_verification', independent['status'] == 'pass' and independent['independent_exact_h_decomposition_max_error'] < 1e-12)

stat_rows = []
for variant, name in [('direction_linear', '코사인·최종 norm·방향'), ('direction_noise2', '같은 입력 + 무작위 숫자 2개'), ('direction_hstat', '같은 입력 + 정규화 전 평균·크기 통계')]:
    r = means['21LA_pilot'][variant]
    stat_rows.append([name, f"{r['balanced_logloss']:.6f}", f"{r['eer_percent']:.6f}%"])
stat_table = table(['21LA 내부 평가의 분류기 입력', '클래스 균형 로그 손실 ↓', 'EER ↓'], stat_rows)
effect_rows = []
for v, name in [('direction_linear', '코사인·최종 norm·방향 기준'), ('direction_noise2', '무작위 숫자 2개 대조')]:
    r = contrast('21LA_pilot', v, 'direction_hstat')
    check('positive_stat_LL_interval_' + v, r['LL_CI95'][0] > 0 and r['draws'] == 10000)
    effect_rows.append([name, f"{r['mean_balanced_LL_reduction']:.6f}", f"[{r['LL_CI95'][0]:.6f}, {r['LL_CI95'][1]:.6f}]"])
effect_table = table(['평균·크기 통계 추가를 비교한 대상', '로그 손실 감소량', '개별 95% 구간'], effect_rows)

vector_rows = []
for ds, name in [('DF_pilot', 'DF'), ('21LA_pilot', '21LA')]:
    for variant, label in [('direction_linear', '기본 코사인·norm·방향'), ('direction_noise160', '기본 + 무작위 숫자 160개'), ('direction_h', '기본 + 정규화 전 벡터 160좌표')]:
        r = means[ds][variant]
        vector_rows.append([name, label, f"{r['balanced_logloss']:.6f}", f"{r['eer_percent']:.6f}%"])
    for baseline in ['direction_linear', 'direction_noise160']:
        check('full_vector_positive_LL_interval_' + ds + '_' + baseline, contrast(ds, baseline, 'direction_h')['LL_CI95'][0] > 0)
vector_table = table(['자료', '분류기 입력', '클래스 균형 로그 손실 ↓', 'EER ↓'], vector_rows)

scale_rows = []
for ds, name in [('DF_pilot', 'DF'), ('21LA_pilot', '21LA')]:
    r = decomposition[ds]
    v = {x['case']: x for x in r['metrics']}
    check('saved_head_reproduced_' + ds, r['maximum_saved_native_logit_error'] == 0 and r['h_decomposition_logit_max_error'] < 2e-6)
    check('scale_fixed_LL_worse_' + ds, v['scale_fixed_to_outer_train_mean']['balanced_logloss'] > v['saved_h']['balanced_logloss'])
    check('mean_fixed_much_smaller_LL_change_' + ds, abs(v['mean_fixed_to_outer_train_mean']['balanced_logloss'] - v['saved_h']['balanced_logloss']) < .0001)
    for case, label in [('saved_h', '원래 평균·크기 유지'), ('scale_fixed_to_outer_train_mean', '발화별 크기만 학습 부분 평균으로 고정'), ('mean_fixed_to_outer_train_mean', '발화별 평균만 학습 부분 평균으로 고정')]:
        a = v[case]
        scale_rows.append([name, label, f"{a['balanced_logloss']:.6f}", f"{a['eer_percent']:.6f}%"])
scale_table = table(['자료', '저장된 동일 분류기에서 바꾼 조건', '클래스 균형 로그 손실 ↓', 'EER ↓'], scale_rows)

def ci(ds, a, b):
    r = contrast(ds, a, b)
    return f"{r['mean_balanced_LL_reduction']:.6f} [{r['LL_CI95'][0]:.6f}, {r['LL_CI95'][1]:.6f}]"

body = rf'''# 코사인 유사도 외에 벡터 크기도 정보인가: 정규화 전 통계와 방향·크기 결합의 실험 근거

**기존 완료 실험은 코사인만으로 읽지 않는 크기 관련 정보가 일부 추가 분류기에서 활용됨을 보여 준다. 가장 분명한 근거는 정규화 전 평균·크기 통계를 추가한 21LA 내부 평가의 예측 품질 개선과, 같은 분류기에서 크기만 고정했을 때의 예측 품질 저하다.**

이 문서가 검증하는 가설은 **“특징 벡터의 크기 관련 통계와 방향의 결합은 코사인 한 값으로 요약되지 않는 예측 정보를 제공할 수 있다”**다. 여기서 크기 정보는 어느 층에서 계산했는지 구분한다. **norm 한 값만 추가하면 기존 EBM의 EER이 개선된다는 가설까지 입증한 것은 아니다.**

## 1. 코사인과 벡터 크기는 무엇이 다른가

벡터의 norm은 좌표들의 제곱합에 제곱근을 취한 크기다. 최종 벡터가 z, 정상 기준 벡터가 c이면 기존 점수는 다음과 같다.

~~~text
cosine(z, c) = (z · c) / (‖z‖ × ‖c‖)
~~~

양의 수 a를 곱해 z를 az로 바꿔도 코사인 점수는 같다. 따라서 코사인 한 값은 벡터가 어느 방향인지를 읽지만, 그 방향을 따라 얼마나 긴지는 직접 읽지 않는다. **이는 수학적 성질이며, 버려진 크기가 실제 정답 예측에 유용하다는 실험 증명은 별도로 필요하다.**

이 연구의 고정 EBM은 마지막 선형 변환 출력 h를 LayerNorm으로 정규화해 z를 만들고, z와 정상 기준의 코사인을 사용한다. 두 벡터는 모두 160차원이다.

|구분|의미|이 문서에서의 역할|
|---|---|---|
|최종 벡터 크기 ‖z‖|LayerNorm 뒤, 코사인 단위화 전의 크기|추가 분류기의 기본 입력에 이미 포함됨|
|정규화 전 크기 ‖h‖|마지막 선형 변환 직후의 크기|최종 크기와 구분해야 하는 값|
|좌표 평균 μ_h, 표준편차 σ_h|한 발화 벡터의 160좌표에서 계산한 통계|크기와 관련된 두 통계의 추가 효과를 검사|
|단위 벡터 x = z/‖z‖|크기를 1로 만든 최종 벡터|이 벡터 자체의 norm은 항상 1이므로 새 정보가 아님|

분산의 분모를 160으로 잡으면 다음 관계가 정확히 성립한다.

~~~text
‖h‖² = 160 × (μ_h² + σ_h²)
‖h − μ_h × 1‖ = √160 × σ_h
~~~

따라서 표준편차는 중심을 뺀 벡터 크기에 해당한다. 그러나 **평균과 표준편차 두 값을 추가한 실험을 원래 norm 한 값만 추가한 실험이라고 바꿔 부르지는 않는다.** 이 통계는 원음의 음량이나 RMS도 아니다.

## 2. 비교 조건: 같은 음성 표현에서 정보만 추가했다

핵심 실험은 **배치 크기 160·50epoch 기존 EBM**에서 저장한 특징을 사용했다. 최근 배치 240이나 B/C 재학습 모델의 실험과 섞지 않는다. EBM과 정상 기준 벡터를 고정하고, 저장된 특징 위에 작은 추가 분류기를 학습했던 완료 결과를 비교했다.

- 자료: ASVspoof 2021 DF와 2021 LA에서 각각 고정 표본 20,000개, 정상 2,000개·위조 18,000개.
- 평가: 각 자료 안에서 별도로 바깥 5분할·안쪽 3분할을 수행했다. 같은 출처·같은 화자 그룹은 학습과 평가에 겹치지 않게 했다. DF 93그룹, 21LA 67그룹이다.
- 선택: 규제 강도 0.0001·0.01와 학습 횟수는 안쪽 검증에서만 선택했다. 표준화 통계도 해당 학습 부분에서만 계산했다.
- 반복: 추가 분류기 초기값 101·102·103의 세 결과를 사용한다. 표의 수치는 세 초기값의 **지표 평균**이며 예측을 평균한 앙상블 성능이 아니다.

기본 입력은 **기존 코사인 s, log‖z‖, 정상 기준에 수직인 단위 방향 160좌표, 방향 유효 여부**다. 수직 방향은 x-sc를 크기 1로 만든 벡터다. 이 연구에서는 방향 정보가 이미 주어진 상태에서도 크기 관련 통계의 추가 이득이 남는지 본다.

주 지표인 **클래스 균형 로그 손실**은 정상·위조에 각각 같은 총 가중치를 주고 예측 확률의 정확성과 확신을 평가한다. 틀린 예측을 강하게 확신할수록 손실이 커진다. EER은 두 클래스를 나누는 점수의 순위와 임계값에 관한 별도 지표다. **로그 손실 개선을 EER 개선이라고 쓰지 않는다.**

이 결과는 평가 자료 내부에서 화자를 나누어 확인한 진단이다. 2019 LA에서 추가 분류기를 학습한 뒤 미지의 2021 자료에 전이한 결과나 공식 평가 전체의 성능은 아니다. DF의 일부 출처·클래스는 화자 그룹 수가 적어 모든 분할에 포함되지 못한 제한도 유지한다.

근거: {link('artifacts/geometry_cache/local_analysis/frozen_all_v1/source_snapshot/frozen_probes.py', '실제 입력·분류기·분할·선택 구현')}, {link('artifacts/geometry_cache/local_analysis/frozen_all_v1/P3_P4/verification.json', '학습 부분 통계와 선택 누수 검산')}, {link('docs/experiments/ebm_condition_audit_20261105.md', 'DF 분할의 출처·클래스 제한')}.

## 3. 실험 1: 방향과 최종 norm을 이미 줘도 정규화 전 통계가 도움이 되는가

### 방법

동일한 기본 입력을 받는 세 선형 추가 분류기를 비교했다.

1. 기본 코사인·최종 norm·방향 정보만 제공.
2. 기본 입력에 발화 ID로 고정한 무작위 숫자 2개 추가.
3. 기본 입력에 정규화 전 벡터의 좌표 평균과 log(σ_h + 10⁻¹²) 추가.

무작위 숫자 대조는 입력 수가 단순히 두 개 늘어난 효과와 실제 통계의 효과를 구분하기 위한 것이다. 같은 분할·초기값·선택 규칙을 사용했다. 평균과 크기 통계는 함께 추가했으므로 두 값의 개별 기여를 분리한 실험은 아니다.

### 21LA 결과

{stat_table}

{effect_table}

구간은 동일한 출처·화자 그룹을 함께 재표집한 10,000회의 저장 결과다. 추가 분류기를 매번 다시 학습하지 않았으며, 전체 다중 비교를 보정한 구간은 아니다.

**실제 평균·크기 통계는 방향까지 포함한 기준과 무작위 숫자 대조보다 로그 손실이 낮았고, 두 감소량의 개별 95% 구간도 모두 0보다 컸다.** 이 21LA 진단에서는 방향과 최종 norm만으로 선형 분류기가 활용하지 못한 예측 정보가 정규화 전 통계에 남아 있었다는 근거다.

EER은 세 조건 모두 **0.800000%**였다. 이 실험이 직접 지지하는 것은 예측 확률의 품질 개선이며, EER 개선은 아니다. 또한 log‖z‖는 기본 입력에 이미 있으므로, 그 한 값을 새로 추가해서 얻은 이득으로 해석하지 않는다.

같은 통계 추가의 DF 로그 손실 감소량은 {ci('DF_pilot', 'direction_linear', 'direction_hstat')}로 구간이 0을 포함했다. 따라서 통계 두 값의 이득이 두 자료 모두에서 확립됐다고 확대하지 않는다.

근거: {link(f'{BASE}/21LA_pilot/metrics.csv', '21LA의 초기값별 실제 지표')}, {link(f'{BASE}/21LA_pilot/paired_effects.json', '21LA의 짝지은 효과와 구간')}, {link(f'{BASE}/DF_pilot/paired_effects.json', 'DF에서의 증거 범위')}.

## 4. 실험 2: 전체 정규화 전 벡터를 주면 같은 방향 기준보다 더 읽을 수 있는가

### 방법

앞의 기본 입력에 h의 160좌표를 추가했다. 비교군에는 같은 수의 고정 무작위 숫자 160개를 추가했다. 이 비교는 정규화 전 벡터 전체의 추가 예측 정보를 확인한다. **전체 벡터에는 방향·평균·크기가 함께 있으므로, 이 실험만으로 이득 전체가 norm 때문이라고 귀속할 수는 없다.**

### 결과

{vector_table}

기본 입력 대비 h 추가의 로그 손실 감소량과 개별 95% 구간은 다음과 같다.

- DF: **{ci('DF_pilot', 'direction_linear', 'direction_h')}**.
- 21LA: **{ci('21LA_pilot', 'direction_linear', 'direction_h')}**.

두 자료 모두 구간이 0보다 컸고, 같은 수의 무작위 숫자를 준 대조와 비교해도 로그 손실 감소 구간이 0보다 컸다. **정규화 전 벡터의 실제 값이 추가 분류기에 유용했고, 단순히 입력 수가 늘어난 것으로 설명되지는 않았다.**

다만 기본 입력 대비 EER 차이의 구간은 두 자료 모두 0을 포함했다. 21LA의 평균 EER은 오히려 0.800000%→0.803704%로 조금 높았다. 그러므로 이 실험 역시 주 근거를 로그 손실에 두며, 일관된 EER 개선을 입증했다고 쓰지 않는다.

근거: {link(f'{BASE}/DF_pilot/metrics.csv', 'DF의 초기값별 지표')}, {link(f'{BASE}/DF_pilot/paired_effects.json', 'DF의 짝지은 구간')}, {link(f'{BASE}/21LA_pilot/paired_effects.json', '21LA의 짝지은 구간')}.

## 5. 실험 3: 같은 분류기에서 크기 변화만 없애면 어떻게 되는가

앞의 두 실험은 정규화 전 정보의 활용 가능성을 보여 주지만, 크기의 역할을 직접 분리하지는 않는다. 그래서 이미 학습된 분류기를 그대로 두고, 그 함수가 발화별 크기에 실제로 의존하는지 확인한 완료 대조를 사용했다.

### 5.1 벡터를 평균·크기·방향으로 나눠 원래 예측을 복원했다

LayerNorm의 좌표별 배율과 편향을 각각 γ와 β라 하면 다음과 같이 쓸 수 있다.

~~~text
q_j = (z_j − β_j) / γ_j
a_h = √(σ_h² + ε)
h_j = μ_h + a_h × q_j
~~~

q는 좌표 평균을 뺀 뒤 스케일로 나눈 패턴이다. 위 식은 z 하나로 h의 원래 크기를 알아냈다는 뜻이 아니다. **복원에는 발화별 μ_h와 a_h가 추가로 필요하다.**

학습 자료 25,380개의 h와 z만으로 정답 없이 고정 LayerNorm 계수를 복원했다. 이후 학습·개발·DF·21LA 총 80,363개의 h를 재구성했고 최대 좌표 오차는 **0.00000013 미만**이었다. 저장 분류기의 점수 분해 오차도 DF **0.00000190**, 21LA **0.00000177** 미만이었다.

분류기의 h 관련 선형 항은 다음처럼 나뉜다. w는 입력 표준화까지 반영한 계수다.

~~~text
w · h = μ_h × Σ_j w_j + a_h × (w · q)
~~~

즉 **평균 항 + 크기×좌표 패턴의 결합**이다. 따라서 평균과 크기 두 숫자를 따로 더하는 것과, 크기를 전체 좌표 패턴에 곱해 읽는 것은 같은 함수가 아니다.

### 5.2 분류기 계수는 유지하고 크기 또는 평균만 고정했다

저장된 초기값 101의 추가 분류기에서 다음 세 조건을 비교했다. 새 분류기를 학습하지 않았다.

1. 원래 발화별 평균·크기 유지.
2. 평균과 q는 유지하고, 크기 a_h만 해당 바깥 학습 부분의 평균으로 교체.
3. 크기와 q는 유지하고, 평균 μ_h만 해당 바깥 학습 부분의 평균으로 교체.

기본 입력의 코사인·log‖z‖·방향 정보도 유지했다. 이 대조는 최종 z의 norm을 바꾼 것이 아니라, 추가 분류기가 받는 정규화 전 h 항의 크기만 바꾼 것이다.

### 결과

{scale_table}

**발화별 크기를 고정하면 두 자료 모두 로그 손실이 커졌다. 평균만 고정한 변화는 훨씬 작았다.** 이 저장 분류기의 함수에서는 정규화 전 크기와 좌표 패턴의 결합이 예측에 실제로 쓰였다.

그러나 EER은 DF에서 1.950000%→2.050000%로 높아졌고, 21LA에서는 0.800000%→0.711111%로 낮아졌다. 따라서 크기를 유지하면 항상 EER이 좋아진다는 결과는 아니다.

또한 크기를 학습 부분 평균으로 교체한 벡터가 실제 음성에서 나올 수 있는지는 보장하지 않는다. 크기를 제거한 조건에 맞춰 분류기를 다시 학습한 실험도 아니다. **이 실험은 해당 저장 함수의 크기 의존성을 확인하며, 최적으로 재학습한 크기 없는 분류기보다 우수하다는 것까지 증명하지 않는다.**

근거: {link(f'{HYP}/H5_LN_reconstruction.json', '벡터 복원의 실제 오차')}, {link(f'{HYP}/H5_DF_pilot_saved_readout_decomposition.json', 'DF의 크기·평균 고정 대조')}, {link(f'{HYP}/H5_21LA_pilot_saved_readout_decomposition.json', '21LA의 크기·평균 고정 대조')}, {link(f'{HYP}/independent_verification.json', '별도 수식과 ROC 검산')}.

## 6. 가설에 대한 결론

근거는 다음 세 단계로 이어진다.

1. **추가 예측 정보:** 21LA에서 방향·최종 norm까지 제공한 상태에서도 정규화 전 평균·크기 통계가 무작위 숫자 대조보다 예측 품질을 높였다.
2. **전체 벡터의 활용성:** DF와 21LA에서 정규화 전 벡터 전체를 추가하면 동일 계열 분류기의 로그 손실이 낮아졌다. 다만 이 이득 전체를 norm 하나로 설명할 수는 없다.
3. **크기의 함수상 역할:** 저장된 분류기의 평균·좌표 패턴은 유지하고 크기만 고정하면 두 자료에서 로그 손실이 커졌다. 크기와 패턴이 결합해 활용되는 경로가 확인됐다.

따라서 현재 실험으로 지지하는 결론은 다음과 같다.

> **코사인 점수만으로 표현을 평가하면 정규화 전 크기 관련 통계와 방향·크기 결합의 활용 가능성을 놓칠 수 있다. 기존 저장 표현에는 이런 정보를 읽어 예측 품질을 높일 수 있는 근거가 있다. 다만 norm 한 값의 독립적인 기여, 기존 EBM의 일관된 EER 개선, 미지 자료로의 전이 개선까지 입증한 것은 아니다.**

이 결과를 새 학습 설계로 옮길 때는 기존 코사인을 기준으로 유지하고, ① 정규화 전 norm 한 값, ② 평균·표준편차, ③ 방향×크기 결합을 각각 분리해 비교하는 것이 맞다. 이는 후속 검증 제안이며 이번 문서에서 학습하거나 효과를 확인한 방법은 아니다.

## 7. 핵심 근거에서 제외한 실험

완료 실험 중에서도 아래 결과는 가설의 성공 근거로 사용하지 않았다.

- 개발 자료에서 모든 분류기가 EER 0%였던 비교: norm의 EER 증분을 구분하지 못한다.
- 초기 학습→DF 전이의 코사인 변환 분류기 17.10%→norm 추가 14.55%: 같은 표본의 원래 코사인 3.35%보다 나빴으므로 기존 탐지기 개선을 증명하지 못한다.
- 정규화 전 h 전체와 정규화 후 z 전체의 차이만으로 norm이 원인이라고 한 해석: 방향·평균·크기·분류기 접근성이 함께 달라져 norm 단독 효과를 분리하지 못한다.
- 잔여 벡터 크기 √(1−s²): 원래 코사인 s로 결정되므로 독립적인 새 norm 정보의 근거가 아니다.
- 미완료 학습, 새 손실의 미실행 계획: 효과를 확인한 결과가 아니다.

불리한 지표를 숨기지는 않았다. 본문의 EER 무변화·역전과 DF 통계 추가의 불확실성을 함께 남겼다. 이 문서는 유리한 행을 골라 가설을 확정하는 대신, **근거가 직접 지지하는 예측 품질과 함수 의존성에 맞춰 가설의 범위를 정했다.**

## 근거와 검산 범위

이 문서 작성에서는 새 음성 추론·EBM 학습·추가 분류기 학습·서버 조회를 하지 않았다. 기존 저장 예측으로 두 자료의 다섯 비교군·세 초기값, 총 30개 지표 쌍의 EER·로그 손실을 별도 계산해 원자료와 대조했다. 기존 신뢰구간은 저장 결과를 사용했으며 새로 재표집하거나 전체 학습 불확실성을 평가하지 않았다.

- [전체 연구 묶음의 norm 절](ebm_five_research_themes_20261111.md#theme4)
- [선별 기준·지출 상한·보존 범위](../../artifacts/norm_information_research_20261111/plan.json)
- [사용한 근거 파일과 해시](../../artifacts/norm_information_research_20261111/sources.json)
- [수치·링크·보존 검산](../../artifacts/norm_information_research_20261111/validation.json)
'''

REPORT.write_text(body, encoding='utf-8')
save('sources.json', list(sources.values()))
save('selection.json', {
    'core_claim': 'Pre-LayerNorm magnitude-related statistics and scale-pattern interaction can contribute to prediction quality in fixed-feature internal diagnostics.',
    'not_claimed': ['Norm alone consistently improves EER.', 'LayerNorm causes existing detection errors.', 'Removing normalization improves transfer.', 'An h-vector effect is solely attributable to its norm.'],
    'retained': ['21LA preLN-statistics vs common baseline and noise2', 'DF/21LA full-h vs common baseline and noise160', 'Saved-head scale/mean interventions with exact reconstruction'],
    'not_used_as_positive_evidence': ['Zero-EER development comparison', 'Poorer-than-native early transfer readouts', 'Uncontrolled h-vs-z attribution', 'Residual radius determined by cosine', 'Unfinished or unexecuted proposals'],
    'contrary_results_preserved': ['Statistics-only DF confidence interval includes zero', '21LA statistics do not reduce EER', 'Full-h EER improvement intervals include zero', 'Scale fixing improves21LA EER while worsening logloss']
})
plan = json.loads((OUT / 'plan.json').read_text(encoding='utf-8'))
for p, h in plan['preserved_documents'].items():
    check('preserved_' + Path(p).name, sha(ROOT / p) == h)
for s in sources.values():
    check('source_preserved_' + s['path'], sha(ROOT / s['path']) == s['sha256'])
text = REPORT.read_text(encoding='utf-8')
table_count = 0
columns = None
for line in text.splitlines():
    if line.startswith('|'):
        n = len(re.findall(r'(?<!\\)\|', line)) - 1
        if columns is None:
            columns = n
            table_count += 1
        check('markdown_table_row_' + str(len(checks)), n == columns)
    else:
        columns = None
check('five_tables_well_formed', table_count == 5)
check('no_control_characters', not any(ord(ch) < 32 and ch != '\n' for ch in text))
for ref in re.findall(r'\]\(([^)]+)\)', text):
    p = (REPORT.parent / ref.split('#')[0]).resolve()
    check('link_' + ref, p.exists() or p == OUT / 'validation.json')
check('core_tables_written', all(t in text for t in [stat_table, effect_table, vector_table, scale_table]))
check('plain_utf8_no_private_ids', '\ufffd' not in text and not re.search(r'\bE\d{3}\b', text))
check('norm_not_confounded_in_conclusion', all(x in text for x in ['norm 한 값만 추가하면 기존 EBM의 EER이 개선된다는 가설까지 입증한 것은 아니다', '평균과 표준편차 두 값을 추가한 실험', '불리한 지표를 숨기지는 않았다']))
check('fixed_head_limits_explicit', all(x in text for x in ['실제 음성에서 나올 수 있는지는 보장하지 않는다', '해당 저장 함수의 크기 의존성을 확인', '새 분류기를 학습하지 않았다']))
save('validation.json', {
    'status': 'passed', 'checked_at': datetime.now().astimezone().isoformat(),
    'report_sha256': sha(REPORT), 'checks': checks, 'check_count': len(checks),
    'independently_recomputed_metric_pairs': 30,
    'maximum_EER_difference_pp': max_eer_error, 'maximum_logloss_difference': max_ll_error,
    'source_count': len(sources), 'new_audio_inference': 0, 'new_model_fits': 0,
    'remote_queries': 0, 'spending_ceiling_usd': 0, 'resources_left_running': 0
})
plan.update(status='complete', report=str(REPORT.relative_to(ROOT)), validation_checks=len(checks))
save('plan.json', plan)
print(json.dumps({'status': 'passed', 'checks': len(checks), 'independent_metric_pairs': 30, 'max_EER_error_pp': max_eer_error, 'max_LL_error': max_ll_error, 'report_bytes': REPORT.stat().st_size}, ensure_ascii=False))
