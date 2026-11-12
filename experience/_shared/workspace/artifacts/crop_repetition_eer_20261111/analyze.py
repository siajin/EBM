"""Compare shortening and repetition using archived scores only; no model/audio IO."""
import bisect
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/repetition_shortcut_20261111'
REPORT = ROOT / 'docs/experiments/ebm_shortening_vs_repetition_eer_20261111.md'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def save(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def read_csv(p):
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def write_csv(name, rows):
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

save('plan.json', {
    'status': 'in_progress',
    'objective': 'Compare EER degradation under shortening in B/native and original/repeated, then add within-model native/repeated controls to distinguish their interpretation.',
    'population': 'Previously selected336 long-original21LA files,128bonafide+208spoof; no new selection/inference.',
    'primary': 'Nested head4.0375s ->2s ->1s, fixed50epoch models, no score averaging.',
    'robustness': 'Tail-only1/2s and separate head/tail score-ensemble sensitivity; tails are not nested in the head4s baseline.',
    'uncertainty': '2000 paired speaker-cluster bootstrap draws, individual exploratory95% intervals; not population confirmatory testing.',
    'spending_ceiling_usd': 0, 'new_training': False, 'new_audio_inference': False,
    'remote_queries': False, 'active_training_changes': False,
    'cleanup': 'Synchronous local CPU analysis only; no persistent process or paid resource.'
})

checks = []
def check(name, value, detail=None):
    checks.append({'name': name, 'passed': bool(value), 'detail': detail})
    if not value:
        raise AssertionError((name, detail))

prior = json.loads((BASE / 'reviewed_analysis/validation.json').read_text(encoding='utf-8-sig'))
sources = [BASE / f'new_results/probe_rank{r}.csv' for r in [0, 1]]
for p in sources:
    check('archived_score_hash_' + p.name, sha(p) == prior['sources'][p.name])
raw = [row for p in sources for row in read_csv(p)]
selected_path = BASE / 'probe_selected.csv'
selected = read_csv(selected_path)
protocol_path = BASE / 'protocol.json'
protocol = json.loads(protocol_path.read_text(encoding='utf-8-sig'))
check('selection_hash', sha(selected_path) == protocol['selection_hashes']['probe_selected.csv'])
check('raw_row_count', len(raw) == 336 * 31)
group = defaultdict(dict)
for r in raw:
    check_key = (r['uid'], r['condition'])
    if r['condition'] in group[r['uid']]:
        raise AssertionError(('duplicate condition', check_key))
    group[r['uid']][r['condition']] = r
check('source_count', len(group) == len(selected) == 336)
check('all_conditions_complete', all(len(g) == 31 for g in group.values()))
check('UID_selection_exact', set(group) == {r['uid'] for r in selected})
uids = sorted(group)
for r in selected:
    check('metadata_' + r['uid'], all(group[r['uid']]['natural4'][k] == r[k] for k in ['phase', 'index', 'speaker', 'label', 'original_samples', 'attack']))
labels = np.array([int(group[u]['natural4']['label']) for u in uids])
speakers = np.array([group[u]['natural4']['speaker'] for u in uids])
check('class_denominators', Counter(labels) == {1: 128, 0: 208})
check('all_originals_long', all(int(group[u]['natural4']['original_samples']) > 64600 for u in uids))
attack_counts = Counter(group[u]['natural4']['attack'] for u in uids if group[u]['natural4']['label'] == '0')
check('spoof_attack_balance', len(attack_counts) == 13 and set(attack_counts.values()) == {16})
check('no_op_wave_and_score', all(group[u][f'{side}_2s_native']['wave_sha256'] == group[u][f'{side}_2s_repeat2']['wave_sha256'] and all(group[u][f'{side}_2s_native'][f'score_{m}'] == group[u][f'{side}_2s_repeat2'][f'score_{m}'] for m in ['legacy240', 'B']) for u in uids for side in ['head', 'tail']))
check('finite_scores', all(np.isfinite(float(r[f'score_{m}'])) for r in raw for m in ['legacy240', 'B']))
check('used_sample_lengths', all(int(r['used_samples']) == (64600 if r['condition'] == 'natural4' or r['condition'].endswith('_repeat4') else 16000 if '_1s_native' in r['condition'] else 32000) for r in raw if r['condition'] == 'natural4' or r['condition'].endswith('_repeat4') or r['condition'].endswith('_native')))

def eer(y, scores, weights=None):
    y, scores = np.asarray(y), np.asarray(scores)
    w = np.ones(len(y), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    keep = w > 0
    y, scores, w = y[keep], scores[keep], w[keep]
    order = np.argsort(-scores, kind='stable')
    y, scores, w = y[order], scores[order], w[order]
    ends = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(scores) - 1]
    fp = np.r_[0, np.cumsum(w * (y == 0))[ends]] / np.sum(w * (y == 0))
    fn = 1 - np.r_[0, np.cumsum(w * (y == 1))[ends]] / np.sum(w * (y == 1))
    difference = fp - fn
    hi = int(np.flatnonzero(difference >= 0)[0])
    if difference[hi] == 0:
        return float(fp[hi] * 100)
    lo = hi - 1
    t = -difference[lo] / (difference[hi] - difference[lo])
    return float(100 * (fp[lo] + t * (fp[hi] - fp[lo])))

def independent_eer(y, scores):
    """Independent threshold counting via sorted class lists, without cumulative ROC."""
    bona = sorted(float(s) for a, s in zip(y, scores) if a == 1)
    spoof = sorted(float(s) for a, s in zip(y, scores) if a == 0)
    previous = (0., 1.)
    for threshold in sorted(set(float(s) for s in scores), reverse=True):
        fp = (len(spoof) - bisect.bisect_left(spoof, threshold)) / len(spoof)
        fn = bisect.bisect_left(bona, threshold) / len(bona)
        if fp >= fn:
            old_fp, old_fn = previous
            gap = (fp - fn) - (old_fp - old_fn)
            t = -(old_fp - old_fn) / gap if gap else 0.
            return (old_fp + t * (fp - old_fp)) * 100
        previous = (fp, fn)
    raise AssertionError('ROC crossing not found')

for y, s, expected in [([1, 0], [1., 0.], 0.), ([1, 0], [0., 1.], 100.), ([1, 0], [0., 0.], 50.)]:
    check('ROC_control_' + str(expected), abs(eer(y, s) - expected) < 1e-10 and abs(independent_eer(y, s) - expected) < 1e-10)

scores = {}
metrics = []
for side in ['head', 'tail', 'head_tail_score_mean']:
    for model in ['legacy240', 'B']:
        for seconds in [4.0375, 2, 1]:
            for policy in ['native', 'repeat4']:
                if seconds == 4.0375:
                    value = np.array([float(group[u]['natural4'][f'score_{model}']) for u in uids])
                else:
                    positions = ['head', 'tail'] if side == 'head_tail_score_mean' else [side]
                    value = np.array([np.mean([float(group[u][f'{p}_{seconds}s_{policy}'][f'score_{model}']) for p in positions]) for u in uids])
                key = (side, model, seconds, policy)
                scores[key] = value
                e, independent = eer(labels, value), independent_eer(labels, value)
                check('independent_ROC_' + '_'.join(map(str, key)), abs(e - independent) < 1e-10)
                metrics.append(dict(position=side, model=model, content_seconds=seconds, input_policy=policy, n_bona=128, n_spoof=208, EER_percent=e))
write_csv('metrics.csv', metrics)

def value(side, model, seconds, policy):
    return eer(labels, scores[(side, model, seconds, policy)])

main = []
for seconds in [4.0375, 2, 1]:
    b = value('head', 'B', seconds, 'native')
    original = value('head', 'legacy240', seconds, 'repeat4')
    main.append(dict(content_seconds=seconds, B_native_EER_percent=b,
                     B_change_from4_pp=b-value('head', 'B', 4.0375, 'native'),
                     original_repeat4_EER_percent=original,
                     original_change_from4_pp=original-value('head', 'legacy240', 4.0375, 'repeat4')))
write_csv('requested_comparison.csv', main)

contrasts = []
for side in ['head', 'tail', 'head_tail_score_mean']:
    for model in ['legacy240', 'B']:
        baseline = value(side, model, 4.0375, 'native')
        for seconds in [2, 1]:
            native, repeat = value(side, model, seconds, 'native'), value(side, model, seconds, 'repeat4')
            contrasts.append(dict(position=side, model=model, content_seconds=seconds,
                                  natural4_EER_percent=baseline, native_EER_percent=native,
                                  repeated_EER_percent=repeat,
                                  shortening_change_pp=native-baseline,
                                  repeat_minus_native_pp=repeat-native,
                                  total_shortened_repeated_change_pp=repeat-baseline))
            check('algebra_' + '_'.join(map(str, (side, model, seconds))), abs((native-baseline)+(repeat-native)-(repeat-baseline)) < 1e-10)
write_csv('within_model_contrasts.csv', contrasts)

dose_rows = []
for model in ['legacy240', 'B']:
    for sec in [2, 1]:
        row = dict(model=model, content_seconds=sec)
        for policy in ['native', 'repeat2', 'repeat3', 'repeat4']:
            s = np.array([float(group[u][f'head_{sec}s_{policy}'][f'score_{model}']) for u in uids])
            e = eer(labels, s)
            check(f'fixed_content_dose_ROC_{model}_{sec}_{policy}', abs(e-independent_eer(labels,s)) < 1e-10)
            row[policy+'_EER_percent'] = e
        dose_rows.append(row)
write_csv('fixed_content_repeat_dose.csv', dose_rows)

# Paired cluster bootstrap: resample speakers, keep all of their files and all conditions.
unique, cluster_index = np.unique(speakers, return_inverse=True)
head_keys = [(model, sec, policy) for model in ['legacy240', 'B'] for sec, policy in [(4.0375, 'native'), (2, 'native'), (1, 'native'), (2, 'repeat4'), (1, 'repeat4')]]
head_scores = [scores[('head',) + k] for k in head_keys]
index = {k: n for n, k in enumerate(head_keys)}
def contrasts_from(v):
    get = lambda m, s, p: v[index[(m, s, p)]]
    result = {}
    for m in ['legacy240', 'B']:
        for sec in [2, 1]:
            result[f'{m}_{sec}s_native_minus4'] = get(m, sec, 'native') - get(m, 4.0375, 'native')
            result[f'{m}_{sec}s_repeat_minus_native'] = get(m, sec, 'repeat4') - get(m, sec, 'native')
            result[f'{m}_{sec}s_repeat_minus4'] = get(m, sec, 'repeat4') - get(m, 4.0375, 'native')
    for sec in [2, 1]:
        result[f'cross_model_degradation_gap_{sec}s'] = result[f'legacy240_{sec}s_repeat_minus4'] - result[f'B_{sec}s_native_minus4']
    return result

point = contrasts_from([eer(labels, s) for s in head_scores])
rng = np.random.default_rng(20261111)
draws = defaultdict(list)
for _ in range(2000):
    multiplicity = np.bincount(rng.integers(len(unique), size=len(unique)), minlength=len(unique))
    w = multiplicity[cluster_index]
    if not (np.sum(w[labels == 0]) and np.sum(w[labels == 1])):
        continue
    for name, delta in contrasts_from([eer(labels, s, w) for s in head_scores]).items():
        draws[name].append(delta)
intervals = []
for name, p in point.items():
    lo, hi = np.quantile(draws[name], [.025, .975])
    intervals.append(dict(contrast=name, estimate_pp=p, exploratory95_low_pp=float(lo), exploratory95_high_pp=float(hi), valid_draws=len(draws[name]), speaker_clusters=len(unique)))
write_csv('speaker_cluster_intervals.csv', intervals)
save('results.json', {'requested': main, 'within_model': contrasts, 'intervals': intervals,
                      'checkpoint_policy': 'Both frozen50epoch; no per-length checkpoint selection.',
                      'primary_position': 'head', 'available_content_seconds': [4.0375, 2, 1],
                      'missing_content_seconds': [3], 'speaker_clusters': len(unique)})

def f(x):
    return f'{x:.4f}'

def sign(x):
    return f'{x:+.4f}'

def table(headers, rows):
    return '\n'.join(['|'+'|'.join(headers)+'|', '|'+'|'.join(['---']*len(headers))+'|'] + ['|'+'|'.join(map(str, row))+'|' for row in rows])

requested_table = table(['남긴 음성 내용', 'B·비반복 EER', '약 4초 대비 변화', '기존·약 4초 반복 EER', '약 4초 대비 변화'], [[str(r['content_seconds'])+'초', f(r['B_native_EER_percent'])+'%', sign(r['B_change_from4_pp'])+'%p', f(r['original_repeat4_EER_percent'])+'%', sign(r['original_change_from4_pp'])+'%p'] for r in main])
control_table = table(['모델', '남긴 내용', '그대로 넣기 EER', '약 4초로 반복 EER', '반복으로 생긴 변화'], [[('기존' if r['model']=='legacy240' else 'B'), str(r['content_seconds'])+'초', f(r['native_EER_percent'])+'%', f(r['repeated_EER_percent'])+'%', sign(r['repeat_minus_native_pp'])+'%p'] for r in contrasts if r['position']=='head'])
tail_table = table(['모델', '뒷부분 길이', '그대로 넣기 EER', '약 4초로 반복 EER', '반복으로 생긴 변화'], [[('기존' if r['model']=='legacy240' else 'B'), str(r['content_seconds'])+'초', f(r['native_EER_percent'])+'%', f(r['repeated_EER_percent'])+'%', sign(r['repeat_minus_native_pp'])+'%p'] for r in contrasts if r['position']=='tail'])
ci_names = ['B_2s_native_minus4', 'B_1s_native_minus4', 'legacy240_2s_repeat_minus4', 'legacy240_1s_repeat_minus4', 'legacy240_2s_repeat_minus_native', 'legacy240_1s_repeat_minus_native', 'cross_model_degradation_gap_2s', 'cross_model_degradation_gap_1s']
ci_labels = ['B 비반복: 약4→2초', 'B 비반복: 약4→1초', '기존 반복: 내용 약4→2초', '기존 반복: 내용 약4→1초', '기존 2초: 반복 추가', '기존 1초: 반복 추가', '2초의 모델 간 악화량 차이', '1초의 모델 간 악화량 차이']
ci_table = table(['대조', 'EER 변화', '탐색적 95% 구간'], [[label, sign(next(r for r in intervals if r['contrast']==name)['estimate_pp'])+'%p', '['+sign(next(r for r in intervals if r['contrast']==name)['exploratory95_low_pp'])+', '+sign(next(r for r in intervals if r['contrast']==name)['exploratory95_high_pp'])+']%p'] for name, label in zip(ci_names, ci_labels)])
dose_table = table(['모델','고정한 내용','실제 길이 그대로','입력 2초로 반복','입력 3초로 반복','입력 4.0375초로 반복'], [[('기존' if r['model']=='legacy240' else 'B'),str(r['content_seconds'])+'초']+[f(r[p+'_EER_percent'])+'%' for p in ['native','repeat2','repeat3','repeat4']] for r in dose_rows])

text = f'''# 짧은 음성 내용과 반복 처리의 EER 영향을 나누어 비교

**두 모델 모두 남긴 음성 내용이 짧아지면 EER이 높아졌다. 기존 모델에서는 같은 짧은 구간을 반복하면 반복하지 않을 때보다 EER이 낮았다. 따라서 이 자료에서 짧아질 때의 성능 저하를 반복 자체가 만든 손해로 설명할 수는 없다. 반복을 제거해도 짧은 내용의 어려움이 남았으며, 기존 모델에서는 반복이 그 손해를 일부 줄였다.**

## 1. 비교 방법

이미 저장된 21LA 긴 원음 336개, 정상 128개·위조 208개의 점수만 다시 계산했다. 위조는 13개 공격마다 16개이며 원음은 모두 4.0375초보다 길다. 전체 21LA의 대표 비율로 뽑은 표본은 아니므로 아래 EER을 공식 전체 성능으로 쓰지 않는다.

기존 모델과 반복 제거 학습 B의 **50epoch 저장 모델을 고정**했다. 입력 길이마다 다른 epoch을 고르면 처리 효과와 모델 선택이 섞이므로, 이 대조에서는 조건별 최소 epoch을 새로 고르지 않았다. 추가 추론 잡음은 0이었다.

주 비교는 같은 원음의 **앞 4.0375초→앞 2초→앞 1초**다. 포함 범위가 점점 줄어들도록 했다. 각 구간을 실제 길이 그대로 넣거나, 같은 내용을 처음부터 순환 반복해 4.0375초로 만들었다. 예를 들어 1초 반복 입력은 약 4초 길이지만 고유한 내용은 1초다.

앞부분 한 구간의 점수만으로 EER을 계산했다. 앞·뒤 점수를 평균해 두 구간의 정보를 합치는 처리는 주 결과에 사용하지 않았다. 3초 내용을 남긴 점수는 이 진단에 없어서 보간하거나 새 추론을 하지 않았다.

**EER은 낮을수록 좋다. 아래 변화량은 해당 모델의 약 4초 조건을 뺀 값이다. 양수는 EER 증가, 즉 성능 저하다.**

## 2. 요청한 비교: B는 실제 길이, 기존 모델은 짧은 내용을 반복

{requested_table}

B는 약 4초에서 2초로 줄이면 **+5.2885%p**, 1초로 줄이면 **+7.2115%p** 악화됐다. 반복이 없어도 짧게 만들수록 분리가 어려워졌다.

기존 모델은 입력 길이를 반복으로 항상 약 4초에 맞췄는데도, 고유 내용이 2초가 되면 **+4.5673%p**, 1초가 되면 **+12.2596%p** 악화됐다. 입력의 물리적 길이를 유지하는 것만으로 짧은 내용의 손해가 사라지지는 않았다.

다만 **두 곡선의 악화량 차이를 반복의 순수 효과로 부르면 안 된다.** 모델의 학습 결과가 서로 다르기 때문이다. 특히 1초에서 기존 반복의 악화량이 더 크다는 결과만으로 반복이 추가 손해를 줬다고 해석할 수 없다. 이를 확인하려면 같은 모델 안에서 같은 구간의 반복 여부를 비교해야 한다.

## 3. 원인을 나누는 핵심 대조: 같은 모델·같은 짧은 내용에서 반복만 추가

{control_table}

기존 모델에서 2초 내용을 그대로 넣으면 EER이 **13.4615%**, 반복하면 **7.6923%**였다. 반복으로 **5.7692%p 개선**됐다. 1초도 **29.3269%→15.3846%**, 즉 **13.9423%p 개선**됐다. 반복의 길이·문맥 변화까지 포함한 전체 처리 효과다.

B에서는 2초 반복이 **+3.6058%p**, 1초 반복이 **+7.6923%p** 악화를 만들었다. 반복 효과는 모델에 따라 방향이 달랐다.

따라서 앞의 두 곡선에서 기존 모델이 1초에 더 많이 악화된 것을 반복 탓으로 돌리는 해석은 같은 모델 대조와 맞지 않는다. **기존 모델 안에서는 반복이 오히려 EER을 낮췄다.**

내용을 고정하고 반복 목표만 2초·3초·약 4초로 바꾼 결과도 확인했다. 여기서 3초는 같은 1초·2초 내용을 반복해 만든 입력 길이이며, 고유 내용 3초를 새로 평가한 결과가 아니다.

{dose_table}

기존 모델은 반복 횟수가 늘 때 EER이 계속 나빠지는 형태가 아니었다. 1초 내용은 비반복 29.3269%에서 반복 조건들 14.9038~15.8654%로 낮아졌다. B의 앞 1초에서는 반대로 반복 목표가 늘면서 EER이 높아졌다. 이는 이 표본에서의 점 추정이며 길이별 작은 차이의 보편적 순서까지 주장하지 않는다.

기존 모델의 결과는 다음처럼 정확히 나눌 수 있다. 이는 EER 변화의 산술 분해이며 서로 독립적인 생물학적·음향학적 원인의 기여율은 아니다.

- 2초: 짧게 만든 비반복 변화 **+10.3365%p** + 같은 2초의 반복 효과 **−5.7692%p** = 최종 변화 **+4.5673%p**.
- 1초: 짧게 만든 비반복 변화 **+26.2019%p** + 같은 1초의 반복 효과 **−13.9423%p** = 최종 변화 **+12.2596%p**.

## 4. 뒷부분에서도 반복 효과의 방향이 같은가

원음의 마지막 2초·1초를 사용해 같은 모델의 반복 여부를 다시 비교했다. 이는 앞부분 결과의 위치 의존성을 점검하는 보조 대조다. 기준 약 4초가 원음 앞부분이므로 뒷부분과의 차이를 순수 길이 감소로 해석하지 않는다.

{tail_table}

기존 모델은 뒷부분에서도 2초·1초 모두 반복 후 EER이 낮았다. B는 2초에서는 반복 후 낮아졌지만 1초에서는 높아졌다. 따라서 **기존 모델의 반복이 EER에 유리했던 방향은 앞·뒤 모두 같았고, B의 반복 반응은 위치·길이에 따라 달랐다.**

## 5. 표본 불확실성

같은 화자의 모든 파일과 모든 조건을 함께 유지해 화자 {len(unique)}개를 2,000회 재표집했다. 조건 간 차이도 동일한 재표집 안에서 계산했다. 아래는 개별 탐색적 95% 구간이며 여러 비교의 동시 검정 보정은 하지 않았다. 표본의 출처·공격 구성 편향을 해소하는 구간도 아니다.

{ci_table}

특히 모델 간 악화량 차이의 구간이 0을 포함하면, 그 차이의 크기·순서가 안정적으로 재현된다고 주장하지 않는다. 주된 해석은 두 모델의 곡선 차이 하나가 아니라, 같은 모델의 반복 여부 대조와 앞·뒤 결과를 함께 보고 내린다.

## 6. 연구 문서에 넣을 결론

> 반복을 제거해 학습한 B에서도 같은 원음을 2초·1초로 줄이면 EER이 높아졌다. 기존 모델도 짧은 내용을 약 4초로 반복해 입력 길이를 유지했지만, 고유 내용이 줄수록 EER이 높아졌다. 그러나 기존 모델에서 같은 짧은 구간을 반복한 결과는 비반복보다 좋았다. 따라서 짧은 음성의 분리력 저하를 반복 자체의 손해로만 설명할 수 없다. 현재 자료는 짧아진 내용·문맥의 손해와 학습 방식에 따른 반복 반응 차이를 함께 보여준다.

여기서 짧아졌다는 것은 같은 긴 원음의 일부를 잘라 사용했다는 뜻이다. 음소 내용·발화량·문맥도 줄어들므로 시간 길이라는 숫자 하나의 인과 효과로 단정하지 않는다. 자연적으로 짧은 원음 전체의 결과로 확대하지도 않는다.

이 결과는 **기존 모델의 반복 전처리가 짧은 정상의 오류를 늘릴 수 있다는 관찰을 부정하지 않는다.** 정상 점수 이동과 정상·위조 전체의 분리 성능은 다른 결과다. 반복을 제거할 때는 정상 일부의 회복뿐 아니라 위조 분리력 손실까지 함께 확인해야 한다.

## 근거와 재현 자료

- [요청한 두 곡선과 변화량](../../artifacts/crop_repetition_eer_20261111/requested_comparison.csv)
- [동일 모델의 짧게 만들기·반복 처리 분해](../../artifacts/crop_repetition_eer_20261111/within_model_contrasts.csv)
- [앞·뒤·점수 평균 방식 전체 지표](../../artifacts/crop_repetition_eer_20261111/metrics.csv)
- [화자 재표집 구간](../../artifacts/crop_repetition_eer_20261111/speaker_cluster_intervals.csv)
- [같은 내용을 고정한 반복량별 EER](../../artifacts/crop_repetition_eer_20261111/fixed_content_repeat_dose.csv)
- [원본 점수 첫 번째 묶음](../../artifacts/repetition_shortcut_20261111/new_results/probe_rank0.csv), [두 번째 묶음](../../artifacts/repetition_shortcut_20261111/new_results/probe_rank1.csv)
- [음성 구간 생성 방법](../../EBM/tools/probe_repetition_shortcut.py)
- [수치·원본 해시·독립 EER 검산](../../artifacts/crop_repetition_eer_20261111/validation.json)

이번 작업은 기존 점수의 CPU 재분석이며 새 학습·음성 추론·서버 변경·유료 자원 생성은 없었다.
'''
REPORT.write_text(text, encoding='utf-8')
check('authored_tables_match_calculated_values', all(t in text for t in [requested_table,control_table,tail_table,ci_table,dose_table]))
for r in main:
    for key in ['B_native_EER_percent','original_repeat4_EER_percent']:
        check('report_primary_value_'+str(r['content_seconds'])+'_'+key, f(r[key])+'%' in text)
columns = None
table_count = 0
for line in text.splitlines():
    if line.startswith('|'):
        n = line.count('|')-1
        if columns is None:
            columns = n
            table_count += 1
        check('report_table_columns_'+str(len(checks)), n==columns)
    else:
        columns = None
for ref in re.findall(r'\]\(([^)]+)\)',text):
    target = (REPORT.parent/ref).resolve()
    check('report_link_'+ref,target.exists() or target==OUT/'validation.json')
check('sample_and_checkpoint_limits_retained', all(s in text for s in ['50epoch 저장 모델을 고정','공식 전체 성능으로 쓰지 않는다','3초 내용을 남긴 점수는','음소 내용·발화량·문맥도 줄어들므로','두 곡선의 악화량 차이를 반복의 순수 효과로 부르면 안 된다']))
save('validation.json', {'status': 'passed', 'checked_at': datetime.now().astimezone().isoformat(),
                         'checks': checks, 'check_count': len(checks),
                         'sources': [{'path': str(p.relative_to(ROOT)), 'sha256': sha(p)} for p in sources+[selected_path, protocol_path,ROOT/'EBM/tools/probe_repetition_shortcut.py']],
                         'report_sha256': sha(REPORT), 'new_training': False, 'new_audio_inference': False,
                         'spending_ceiling_usd': 0, 'resources_left_running': 0})
plan = json.loads((OUT/'plan.json').read_text(encoding='utf-8'))
plan.update(status='complete', validation_check_count=len(checks), report=str(REPORT.relative_to(ROOT)))
save('plan.json', plan)
print(json.dumps({'status': 'passed', 'checks': len(checks), 'primary_comparison': main, 'intervals': intervals}, ensure_ascii=False))
