"""Independent audit and Korean result record for the locked duration study."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from geometry_cache import resolve_run
from audit_cached_core_results import independent_eer, independent_auc

ROOT = Path(__file__).resolve().parents[2]
METHODS = ['N', 'G', 'D', 'N-OFF', 'G-OFF', 'A', 'S']
LABELS = {'N': '기존 EBM', 'G': '정상 전체 평균 중심', 'D': '길이별 평균 중심(주실험)',
          'N-OFF': '기존 점수 길이 보정', 'G-OFF': '전체 평균 점수 길이 보정',
          'A': '기존 중심 보존·방향 보정', 'S': '길이별 수축 평균 중심'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')


def audit(out):
    assert (out / 'complete').read_text().strip() == '0'
    lock = json.loads((out / 'protocol_lock.json').read_text())
    assert sha(out / 'plan_at_lock.md') == lock['plan_sha256']
    assert sha(out / 'runner_at_lock.py') == lock['runner_sha256']
    thresholds = json.loads((out / 'V_thresholds.json').read_text())['methods']
    raw = np.load(out / 'centroids.npz')
    model = {k: raw[k] for k in raw.files}
    all_metrics = pd.read_csv(out / 'metrics_all.csv')
    checks, auxiliary = [], []
    maxima = dict(score_error=0., eer_error=0., auc_error=0., random_error=0.)
    dataset_names = ['V', 'DF_pilot', 'DF_full', '19LA_eval', '21LA_eval', 'Q']
    for name in dataset_names:
        frame = pd.read_parquet(out / f'scores_{name}.parquet')
        assert frame.utterance_id.is_unique and frame.utterance_id.is_monotonic_increasing
        aliases = ['DF_pilot', 'DF_remaining'] if name == 'DF_full' else [name]
        original = []
        for alias in aliases:
            folder, _, split = resolve_run(alias)
            f = pd.read_parquet(folder / 'extract/scores.parquet')
            f = f[f.split == split].copy()
            f['_folder'] = str(folder)
            original.append(f)
        original = pd.concat(original).set_index('utterance_id').loc[frame.utterance_id]
        assert np.array_equal(original.y, frame.y)
        assert np.array_equal(original.duration, frame.duration)
        assert np.max(np.abs(original.cosine_s.to_numpy() - frame.N.to_numpy())) < 1e-10
        y = frame.y.to_numpy()
        records = all_metrics[(all_metrics.dataset == name) & (all_metrics.dimension == 'all')].set_index('method')
        for m in METHODS:
            values = frame[m].to_numpy()
            e, a = independent_eer(y, values), independent_auc(y, values)
            maxima['eer_error'] = max(maxima['eer_error'], abs(e - records.loc[m, 'eer_percent']))
            maxima['auc_error'] = max(maxima['auc_error'], abs(a - records.loc[m, 'auroc']))
            t = thresholds[m]['threshold']
            assert int(np.sum((y == 1) & (values < t))) == records.loc[m, 'false_rejected_bona']
            assert int(np.sum((y == 0) & (values >= t))) == records.loc[m, 'false_accepted_spoof']
            if name == 'V': assert np.mean(values[y == 0] >= t) <= .01
        # Directly read raw rank arrays by UID, independently of the runner's Panel.
        chosen = np.random.default_rng(4810).choice(len(frame), min(512, len(frame)), replace=False)
        sampled = original.iloc[chosen]
        u = np.empty((len(chosen), 160))
        for (folder, rank), part in sampled.groupby(['_folder', 'rank'], sort=False):
            z = np.load(Path(folder) / f'extract/rank{int(rank)}_z.npy', mmap_mode='r')
            positions = sampled.index.get_indexer(part.index)
            values = z[part.embedding_row.to_numpy()].astype(float)
            u[positions] = values / np.linalg.norm(values, axis=1, keepdims=True)
        b = np.searchsorted([2., 3., 4.0375], frame.duration.to_numpy()[chosen], side='right')
        expected = {'N': u @ model['c0'], 'G': u @ model['g']}
        for key, centers in [('D', 'g_k'), ('A', 'A_k'), ('S', 'S_k')]:
            expected[key] = np.sum(u * model[centers][b], axis=1)
        expected['N-OFF'] = expected['N'] - model['off_N'][b]
        expected['G-OFF'] = expected['G'] - model['off_G'][b]
        for m in METHODS:
            maxima['score_error'] = max(maxima['score_error'], float(np.max(np.abs(expected[m] - frame[m].to_numpy()[chosen]))))
        random_scores = np.load(out / f'random_scores_{name}.npy', mmap_mode='r')
        stream = dataset_names.index(name) + 1
        for r in [0, 99, 198]:
            routing = np.random.default_rng(np.random.SeedSequence([202611060100 + r, stream])).permutation(frame.duration_bin.to_numpy())
            expected_random = np.sum(u * model['R_k'][r, routing[chosen]], axis=1)
            maxima['random_error'] = max(maxima['random_error'], float(np.max(np.abs(expected_random - random_scores[r, chosen]))))
        for k in range(4):
            mask = frame.duration_bin.to_numpy() == k
            if len(np.unique(y[mask])) < 2: continue
            for base, shifted in [('N', 'N-OFF'), ('G', 'G-OFF')]:
                assert abs(roc_auc_score(y[mask], frame[base].to_numpy()[mask]) - roc_auc_score(y[mask], frame[shifted].to_numpy()[mask])) < 1e-12
        checks.append(dict(dataset=name, n=len(frame), methods_verified=len(METHODS), raw_vector_samples=len(chosen), random_repeats_checked=[0, 99, 198]))
        del frame, original, random_scores
    assert maxima['eer_error'] < 1e-8 and maxima['auc_error'] < 1e-10
    assert maxima['score_error'] < 1e-12 and maxima['random_error'] < 1e-12
    rframe = pd.read_csv(out / 'random_controls.csv')
    assert rframe.groupby('dataset').size().eq(199).all() and len(rframe) == 6 * 199
    for name in ['DF_full', '19LA_eval', '21LA_eval']:
        values = np.load(out / f'bootstrap_draws_{name}.npz')
        for m in METHODS:
            delta = values['N'] - values[m]
            valid = np.isfinite(delta)
            auxiliary.append(dict(dataset=name, method=m, exploratory=True,
                                  native_minus_candidate_CI95=np.quantile(delta[valid], [.025, .975]).tolist(),
                                  valid_draws=int(valid.sum())))
        assert values['group_weights'].shape[0] == 10000
    dump(out / 'auxiliary_native_comparisons.json', auxiliary)
    report = dict(status='pass', verifier_sha256=sha(__file__), checks=checks, maximum_errors=maxima,
                  random_controls=6 * 199, bootstrap_draws_per_dataset=10000,
                  independent_EER_method='DET crossing', independent_AUROC_method='rank sum',
                  raw_rank_embeddings_reconstructed_by_UID=True)
    dump(out / 'independent_verification.json', report)
    return report


def make_report(out):
    out = out.resolve()
    overall = pd.read_csv(out / 'metrics_overall.csv')
    all_metrics = pd.read_csv(out / 'metrics_all.csv')
    intervals = json.loads((out / 'paired_intervals.json').read_text())
    conclusion = json.loads((out / 'conclusions.json').read_text())
    fit = json.loads((out / 'fit_summary.json').read_text())
    rand = json.loads((out / 'random_summary.json').read_text())
    lines = ['# 원음 길이별 정상 센트로이드: 동결 EBM 실험 결과', '',
             '작성일: 2026-11-06 KST. 승인한 설계의 전체 계산과 독립 검수를 완료했다.', '',
             f'**주실험 D의 DF 개선 판정: {"지지됨" if conclusion["H1_supported"] else "지지되지 않음"}.** '
             f'중심 재추정·스칼라 보정 이상의 이득(H2): {"지지됨" if conclusion["H2_supported"] else "지지되지 않음"}.', '',
             '기존 EBM 가중치·음성 입력·캐시는 변경하지 않았다. T 정상만으로 중심을 추정했고, V는 spoof FAR 1% 운영 threshold에만 사용했다. '
             'DF 내부 fit·교차검증·새 음성 forward·optimizer 학습은 없었다.', '',
             '## 1. 전체 EER', '',
             '| 방법 | DF 전체 | 19LA 평가 | 21LA 평가 | DF pilot |',
             '| --- | ---: | ---: | ---: | ---: |']
    for m in METHODS:
        vals = [float(overall[(overall.dataset == name) & (overall.method == m)].eer_percent.iloc[0]) for name in ['DF_full', '19LA_eval', '21LA_eval', 'DF_pilot']]
        lines.append('| ' + LABELS[m] + ' | ' + ' | '.join(f'{v:.6f}%' for v in vals) + ' |')
    lines += ['', 'DF 전체는 정상 14,869개·스푸프 519,059개, 합계 533,928개다. pilot 20,000개와 remaining 513,928개는 서로소 합집합으로 구성했다. '
              '표의 19LA/21LA와 DF는 독립 반복으로 합치지 않았다.', '',
              '## 2. 사전 고정한 주 비교', '',
              '양수는 D의 EER 감소, 음수는 D의 악화다. source별 화자 군집을 동일하게 재표집한 10,000회 조건부 구간이다.', '',
              '| 대조군 | 대조군 EER − D EER (pp) | 95% 구간 | 98.75% 보정 구간 |',
              '| --- | ---: | --- | --- |']
    for r in intervals[0]['paired']:
        if r['primary']:
            lines.append(f'| {LABELS[r["baseline"]]} | {r["improvement_pp"]:+.6f} | [{r["CI95"][0]:+.6f}, {r["CI95"][1]:+.6f}] | [{r["CI98p75"][0]:+.6f}, {r["CI98p75"][1]:+.6f}] |')
    lines += ['', '중심·V threshold 추정의 불확실성과 미해결 원천 발화 family 의존성은 위 구간에 포함하지 않는다. '
              'A/S 및 추가 native 대비 비교는 탐색적 결과이며 주실험으로 승격하지 않았다.', '',
              '## 3. V에서 threshold를 고정했을 때의 DF 오류', '',
              '모든 방법의 V spoof FAR는 1% 이하로 맞췄다. DF에서 FAR가 같다는 뜻은 아니다. 정상·스푸프 오류를 함께 해석해야 한다.', '',
              '| 방법 | 정상 오탐 FRR | 스푸프 통과 FAR | 정상 오류 수 | 스푸프 오류 수 |',
              '| --- | ---: | ---: | ---: | ---: |']
    for m in METHODS:
        r = overall[(overall.dataset == 'DF_full') & (overall.method == m)].iloc[0]
        lines.append(f'| {LABELS[m]} | {r.frr_percent:.4f}% | {r.far_percent:.4f}% | {int(r.false_rejected_bona):,} | {int(r.false_accepted_spoof):,} |')
    lines += ['', '## 4. 같은 길이 안의 구분 성능', '',
              '| DF 길이 구간 | 정상 / 스푸프 | 기존 EBM AUROC | 전체 평균 G AUROC | 길이별 D AUROC | G EER | D EER |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for k in range(4):
        p = all_metrics[(all_metrics.dataset == 'DF_full') & (all_metrics.dimension == 'duration') & (all_metrics.value == f'B{k}')].set_index('method')
        lines.append(f'| B{k} | {int(p.loc["D","n_bona"]):,} / {int(p.loc["D","n_spoof"]):,} | {p.loc["N","auroc"]:.6f} | {p.loc["G","auroc"]:.6f} | {p.loc["D","auroc"]:.6f} | {p.loc["G","eer_percent"]:.6f}% | {p.loc["D","eer_percent"]:.6f}% |')
    lines += ['', 'B0: 2秒 미만, B1: 2–3초, B2: 3–4.0375초, B3: 4.0375초 이상. '
              'N-OFF/G-OFF는 해당 구간의 N/G 순위를 보존하는 것을 검수했다.', '',
              '### 2초 미만의 출처별 차이', '',
              '| 출처 | 정상 / 스푸프 | G EER | D EER | G AUROC | D AUROC | 작은 표본 |',
              '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for source in ['asvspoof', 'vcc2018', 'vcc2020']:
        p = all_metrics[(all_metrics.dataset == 'DF_full') & (all_metrics.dimension == 'source_duration') & (all_metrics.value == f'{source}::B0')].set_index('method')
        lines.append(f'| {source} | {int(p.loc["D","n_bona"]):,} / {int(p.loc["D","n_spoof"]):,} | {p.loc["G","eer_percent"]:.6f}% | {p.loc["D","eer_percent"]:.6f}% | {p.loc["G","auroc"]:.6f} | {p.loc["D","auroc"]:.6f} | {p.loc["D","small_sample"]} |')
    lines += ['', '각 셀의 전체 N·화자 수·FAR/FRR·tail 및 codec 결과는 상세 CSV에 보존했다.', '',
              '## 5. 199회 무작위 대조와 중심 안정성', '',
              '| 평가셋 | 무작위 EER 2.5% / 중앙값 / 97.5% | D EER | 무작위 EER ≤ D 비중 |',
              '| --- | --- | ---: | ---: |']
    for r in rand:
        if r['dataset'] in ['DF_full', '19LA_eval', '21LA_eval']:
            q = ' / '.join(f'{v:.6f}%' for v in r['EER_quantiles'])
            lines.append(f'| {r["dataset"]} | {q} | {r["D_eer_percent"]:.6f}% | {100*r["random_EER_at_most_D_fraction"]:.2f}% |')
    lines += ['', '위 무작위 분위수는 199개 대조군의 분포이며 D와의 paired confidence interval 또는 인과검정 p-value가 아니다.', '',
              '| 구간 | T 정상 수 | 화자 수 | 정상 평균 norm | 전체 평균 중심과 각도 | 기존 중심과 각도 |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in fit['rows']:
        lines.append(f'| B{r["duration_bin"]} | {r["n"]} | {r["speakers"]} | {r["resultant_norm"]:.6f} | {r["angle_to_global_deg"]:.4f}° | {r["angle_to_native_deg"]:.4f}° |')
    lines += ['', 'T 정상 화자 500회 재표집의 각도 분포·빈 구간 횟수도 저장했다. 각도 안정성은 성능 개선 자체의 증거가 아니다.', '',
              '## 6. 검수·범위와 원자료', '',
              '- 승인한 구간·수식·반복 수·주 방법을 결과에 따라 변경하지 않았다.',
              '- 원본 캐시의 hash를 실행 전후 대조했고 모든 저장 점수를 독립 DET/rank 방식으로 다시 검수했다.',
              '- 각 평가셋에서 512개 원래 rank 임베딩을 UID로 직접 대조하고, 무작위 반복 0/99/198의 점수도 재계산했다.',
              '- 길이와 반복 패딩을 분리한 인과 개입은 하지 않았다. 이번 성공/실패를 EBM 공동 재학습의 성공/실패로 확대하지 않는다.',
              '- DF는 이전 분석에서 관찰한 데이터다. 새 독립 확증 실험이라는 표현을 사용하지 않는다.', '',
              f'- [실험 산출물]({out.as_posix()})',
              f'- [주 비교와 신뢰구간]({(out/"paired_intervals.json").as_posix()})',
              f'- [전체 지표]({(out/"metrics_overall.csv").as_posix()})',
              f'- [출처·길이·codec별 지표]({(out/"metrics_source_duration.csv").as_posix()})',
              f'- [독립 검수]({(out/"independent_verification.json").as_posix()})',
              '- [승인한 설계](ebm_duration_conditioned_centroids_plan_20261106.md)', '']
    path = ROOT / 'docs/experiments/ebm_duration_conditioned_centroids_results_20261106.md'
    assert not path.exists()
    path.write_text('\n'.join(lines).replace('2秒', '2초'), encoding='utf8')
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(2):
        print(json.dumps(audit(args.out), ensure_ascii=True))
        print('REPORT', make_report(args.out))
