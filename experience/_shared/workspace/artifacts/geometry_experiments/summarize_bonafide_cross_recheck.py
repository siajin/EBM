"""Audit a fresh local rerun and add source, codec and duration sensitivity checks.

Reproduce first, using an unused output folder:
  import test_bonafide_domain_shift as e; e.OUT = output; e.main()
  import verify_bonafide_domain_shift as v; v.OUT = output; v.main()
This script fits no model and reads only existing local score caches.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
from pathlib import Path
import argparse
import hashlib
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from test_bonafide_domain_shift import load, mixed, errors
from audit_cached_core_results import independent_eer, independent_auc

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / 'artifacts/geometry_cache/local_analysis/bonafide_domain_shift_v1'


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False) + '\n', encoding='utf8')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weighted_pair(b, s, wb, ws):
    yy = np.r_[np.ones(len(b)), np.zeros(len(s))]
    zz = np.r_[b.cosine_s, s.cosine_s]
    ww = np.r_[wb, ws]
    fa, tp, _ = roc_curve(yy, zz, sample_weight=ww, drop_intermediate=False)
    fr = 1-tp
    d = fa-fr
    hi = np.flatnonzero(d >= 0)[0]
    lo = max(0, hi-1)
    fraction = -d[lo]/(d[hi]-d[lo]) if d[hi] != d[lo] else 0.
    value = 100*(fa[lo]+fraction*(fa[hi]-fa[lo]))
    # Separate ascending-score weighted DET implementation, including ties.
    order = np.argsort(zz, kind='stable')
    z, y, w = zz[order], yy[order], ww[order]
    ends = np.r_[np.flatnonzero(np.diff(z)), len(z)-1]
    reject = np.r_[0., np.cumsum(w*y)[ends]] / sum(w*y)
    accept = 1-np.r_[0., np.cumsum(w*(1-y))[ends]] / sum(w*(1-y))
    diff = reject-accept
    h = np.flatnonzero(diff >= 0)[0]
    l = max(0, h-1)
    f = -diff[l]/(diff[h]-diff[l]) if diff[h] != diff[l] else 0.
    independent = 100*(reject[l]+f*(reject[h]-reject[l]))
    assert abs(value-independent) < 1e-8
    return float(value)


def main(out):
    start = time.monotonic()
    assert (out/'complete').read_text().strip() == '0'
    assert json.loads((out/'independent_verification.json').read_text())['status'] == 'pass'
    assert not (out/'recheck_summary.json').exists(), 'Preserve a completed recheck'
    point = pd.read_csv(out/'class_replacement_eer.csv')
    previous = pd.read_csv(OLD/'class_replacement_eer.csv')
    pd.testing.assert_frame_equal(point, previous, check_exact=True)
    a, c = load('19LA_eval'), load('21LA_eval')
    df = pd.concat([load('DF_pilot'), load('DF_remaining')], ignore_index=True)
    assert not df.utterance_id.duplicated().any()
    panels = {'19LA': a, '21LA': c, 'DF_all': df}
    panels.update({'DF_'+source: g.copy() for source, g in df.groupby('corpus')})
    matrix = []
    for bn, bf in panels.items():
        for sn, sf in panels.items():
            b, s = bf[bf.y == 1], sf[sf.y == 0]
            val = mixed(b, s)
            yy, zz = np.r_[b.y, s.y], np.r_[b.cosine_s, s.cosine_s]
            auc = float(roc_auc_score(yy, zz))
            assert abs(independent_eer(yy, zz)-val) < 1e-8
            assert abs(independent_auc(yy, zz)-auc) < 1e-10
            matrix.append(dict(bona=bn, spoof=sn, n_bona=len(b), n_spoof=len(s),
                               eer_percent=val, auroc=auc))
    pd.DataFrame(matrix).to_csv(out/'source_cross_matrix.csv', index=False)

    boot = np.load(out/'cluster_bootstrap_draws.npz')
    old_boot = np.load(OLD/'cluster_bootstrap_draws.npz')
    assert set(boot.files) == set(old_boot.files)
    assert all(np.array_equal(boot[k], old_boot[k]) for k in boot.files)
    contrasts = []
    for base in ['19LA_eval', '21LA_eval']:
        def get(bn, sn):
            row = point[(point.baseline == base) & (point.bona_dataset == bn) & (point.spoof_dataset == sn)].iloc[0]
            return float(row.eer_percent), boot[f'{bn}_bona__{sn}_spoof']
        A, ab = get(base, base)
        B, bb = get('DF_full', base)
        C, cb = get(base, 'DF_full')
        D, db = get('DF_full', 'DF_full')
        for label, value, draws in [
            ('normal_swap_with_baseline_spoof', B-A, bb-ab),
            ('normal_swap_with_DF_spoof', D-C, db-cb),
            ('spoof_swap_with_baseline_normal', C-A, cb-ab),
            ('spoof_swap_with_DF_normal', D-B, db-bb),
            ('EER_interaction', D-C-B+A, db-cb-bb+ab)]:
            ci = np.quantile(draws, [.025, .975])
            row = dict(baseline=base, contrast=label, delta_pp=value,
                       ci95_lower=float(ci[0]), ci95_upper=float(ci[1]))
            # Four edge comparisons per baseline; interaction remains descriptive.
            if label != 'EER_interaction':
                adjusted = np.quantile(draws, [.00625, .99375])
                row.update(ci98_75_lower=float(adjusted[0]), ci98_75_upper=float(adjusted[1]))
            contrasts.append(row)
    pd.DataFrame(contrasts).to_csv(out/'paired_contrasts.csv', index=False)

    sensitivity = []
    for setting, target in [('all_DF', df), ('DF_nocodec', df[df.codec == 'nocodec'])]:
        for bn, bf in [('19LA', a), ('DF', target)]:
            for sn, sf in [('19LA', a), ('DF', target)]:
                sensitivity.append(dict(setting=setting, bona=bn, spoof=sn,
                                        eer_percent=mixed(bf, sf),
                                        n_bona=int(sum(bf.y == 1)), n_spoof=int(sum(sf.y == 0))))
    pd.DataFrame(sensitivity).to_csv(out/'codec_sensitivity.csv', index=False)

    # Both classes separately standardized to the 19LA duration composition,
    # restricted to bins present in both datasets. A population sensitivity,
    # not same-waveform padding removal or a training intervention.
    edges = np.array([1., 1.5, 2., 3., 4.0375, 6.])
    frames, weights, support = {}, {}, []
    for y in [0, 1]:
        groups = {'19LA': a[a.y == y].copy(), 'DF': df[df.y == y].copy()}
        counts = {n: np.bincount(np.searchsorted(edges, g.duration, side='right'), minlength=7)
                  for n, g in groups.items()}
        common = (counts['19LA'] > 0) & (counts['DF'] > 0)
        mass = counts['19LA'] * common
        mass = mass/mass.sum()
        for n, g in groups.items():
            bins = np.searchsorted(edges, g.duration, side='right')
            w = mass[bins]/counts[n][bins]
            frames[(n, y)], weights[(n, y)] = g, w
            support.append(dict(dataset=n, y=y, bins_counts=counts[n].tolist(),
                                common_bins=common.tolist(), target_mass=mass.tolist(),
                                retained_fraction=float(common[bins].mean())))
    duration = []
    for bn in ['19LA', 'DF']:
        for sn in ['19LA', 'DF']:
            duration.append(dict(bona=bn, spoof=sn, eer_percent=weighted_pair(
                frames[(bn, 1)], frames[(sn, 0)], weights[(bn, 1)], weights[(sn, 0)])))
    save(out/'duration_sensitivity.json', dict(edges_seconds=edges.tolist(),
         support=support, matrix=duration, uncertainty='Point estimates only; no sensitivity CIs',
         scope='19LA class-specific duration prior on common binned support; residual within-bin confounding remains'))

    # A separate deployed-style operating point selected only from V spoof scores.
    v = load('V')
    vs = np.sort(v.loc[v.y == 0, 'cosine_s'].to_numpy())
    allowed = int(np.floor(.01*len(vs)))
    threshold = float(np.nextafter(vs[len(vs)-allowed-1], np.inf))
    assert np.mean(vs >= threshold) <= .01
    fixed = [dict(dataset=n, threshold=threshold, **errors(f, threshold))
             for n, f in panels.items()]
    pd.DataFrame(fixed).to_csv(out/'V_spoof_FAR1_operating.csv', index=False)

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.1), constrained_layout=True)
    for ax, base, label in zip(axes, ['19LA_eval', '21LA_eval'], ['19LA', '21LA']):
        values = np.array([[getattr(point[(point.baseline == base) & (point.bona_dataset == b) &
                         (point.spoof_dataset == s)].iloc[0], 'eer_percent')
                         for s in [base, 'DF_full']] for b in [base, 'DF_full']])
        im = ax.imshow(values, vmin=0, vmax=7, cmap='Blues')
        ax.set_xticks([0, 1], [label, 'DF'])
        ax.set_yticks([0, 1], [label, 'DF'])
        ax.set_xlabel('Spoof scores')
        ax.set_ylabel('Bona fide scores')
        ax.set_title(f'{label} vs DF: EER (%)', pad=13)
        for i in range(2):
            for j in range(2):
                b, s = [base, 'DF_full'][i], [base, 'DF_full'][j]
                row = point[(point.baseline == base) & (point.bona_dataset == b) & (point.spoof_dataset == s)].iloc[0]
                ax.text(j, i, f'{values[i,j]:.3f}%\n[{row.ci95_lower:.3f}, {row.ci95_upper:.3f}]',
                        ha='center', va='center', fontsize=12,
                        color='white' if values[i,j] > 3.5 else '#14283f')
    fig.colorbar(im, ax=axes, shrink=.75, label='EER (%)')
    fig.suptitle('Fixed EBM: class-conditional score replacement', fontsize=15)
    fig.supxlabel('95% source-stratified speaker-cluster intervals; score accounting, not causal attribution', fontsize=10)
    fig.savefig(out/'cross_eer_matrix.png', dpi=180, facecolor='white')
    plt.close(fig)
    summary = dict(status='pass', completed_utc=datetime.now(timezone.utc).isoformat(),
                   rerun_elapsed_seconds=json.loads((out/'summary.json').read_text())['elapsed_seconds'],
                   extension_elapsed_seconds=time.monotonic()-start,
                   exact_previous_point_and_bootstrap_reproduction=True,
                   source_cross_cells_independently_verified=len(matrix),
                   weighted_duration_cells_independently_verified=len(duration),
                   original_score_hash_audit='verification.json',
                   primary_contrasts=contrasts[:5], codec_sensitivity=sensitivity,
                   duration_sensitivity=duration,
                   limits=['Single previously inspected checkpoint and test sets; exploratory.',
                           'Evaluation-set class replacement is not an input or training intervention.',
                           'Spoof replacement changes attack, source, codec and their mixture together.',
                           'Speaker-cluster bootstrap omits unresolved original-utterance families and model-fit uncertainty.',
                           '19LA/21LA/DF share material; two baselines are sensitivity checks, not independent replications.',
                           'No-codec and duration checks are descriptive point estimates with no bootstrap CIs.'])
    save(out/'recheck_summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output.resolve())
