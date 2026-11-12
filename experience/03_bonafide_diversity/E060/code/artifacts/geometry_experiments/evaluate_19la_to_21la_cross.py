"""Local score-only 19LA -> 21LA replacement, with shared speaker bootstrap."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
from test_bonafide_domain_shift import load, GroupedClass, weighted_mixed
from audit_cached_core_results import independent_eer, independent_auc
from geometry_cache import resolve_run

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT/'artifacts/geometry_cache/local_analysis/bonafide_cross_recheck_20261106_v1'
OUT = ROOT/'artifacts/geometry_cache/local_analysis/cross_19la_to_21la_20261106_v1'


def main():
    start = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=False)
    def save(name, obj):
        (OUT/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                                       allow_nan=False)+'\n', encoding='utf8')
    source_audit = []
    data = {}
    for name in ['19LA_eval', '21LA_eval']:
        folder, entry, _ = resolve_run(name)
        target = folder/'extract/scores.parquet'
        sha = hashlib.sha256(target.read_bytes()).hexdigest()
        expected = next(x['sha256'] for x in entry['files'] if x['path'] == 'extract/scores.parquet')
        assert sha == expected
        data[name] = load(name)
        source_audit.append(dict(dataset=name, score_sha256=sha,
                                 checkpoint=entry['identity']['checkpoint_sha256']))
    assert len({x['checkpoint'] for x in source_audit}) == 1
    saved = np.load(PARENT/'cluster_bootstrap_draws.npz')
    groups, weights = saved['groups'].tolist(), saved['weights']
    # Reuse prior draws: 67 shared ASV speaker clusters have identical weights
    # in both datasets. VCC columns have zero membership in this comparison.
    assert weights.shape[0] == 2000
    classes = {(n, y): GroupedClass(f, y, groups) for n, f in data.items() for y in [0, 1]}
    metrics, draws, values = [], {}, {}
    threshold = float(pd.read_csv(PARENT/'V_spoof_FAR1_operating.csv').iloc[0].threshold)
    for tag, bn, sn in [('A','19LA_eval','19LA_eval'), ('B','21LA_eval','19LA_eval'),
                        ('C','19LA_eval','21LA_eval'), ('D','21LA_eval','21LA_eval')]:
        b = data[bn].loc[data[bn].y == 1, 'cosine_s'].to_numpy()
        s = data[sn].loc[data[sn].y == 0, 'cosine_s'].to_numpy()
        y, z = np.r_[np.ones(len(b)),np.zeros(len(s))], np.r_[b,s]
        fa, tp, _ = roc_curve(y, z, drop_intermediate=False)
        fr = 1-tp
        di = fa-fr
        h = np.flatnonzero(di >= 0)[0]
        l = max(0,h-1)
        fraction = -di[l]/(di[h]-di[l]) if di[h] != di[l] else 0.
        point = float(100*(fa[l]+fraction*(fa[h]-fa[l])))
        auc = float(roc_auc_score(y,z))
        assert abs(point-independent_eer(y,z)) < 1e-8
        assert abs(auc-independent_auc(y,z)) < 1e-10
        ones = weighted_mixed(classes[(bn,1)],classes[(sn,0)],np.ones((1,len(groups)),int))[0]
        assert abs(ones-point) < 1e-8
        sample = weighted_mixed(classes[(bn,1)],classes[(sn,0)],weights)
        ci = np.quantile(sample,[.025,.975])
        draws[tag], values[tag] = sample, point
        metrics.append(dict(cell=tag,bona=bn,spoof=sn,n_bona=len(b),n_spoof=len(s),
                            eer_percent=point,auroc=auc,ci95_lower=float(ci[0]),ci95_upper=float(ci[1]),
                            V_threshold=threshold,frr_percent=float(100*np.mean(b<threshold)),
                            far_percent=float(100*np.mean(s>=threshold))))
    contrasts = []
    for label, positive, negative in [('normal_swap_19spoof','B','A'),
                                      ('normal_swap_21spoof','D','C'),
                                      ('spoof_swap_19normal','C','A'),
                                      ('spoof_swap_21normal','D','B'),
                                      ('total','D','A')]:
        change = draws[positive]-draws[negative]
        ci = np.quantile(change,[.025,.975])
        row = dict(contrast=label,delta_pp=values[positive]-values[negative],
                   ci95_lower=float(ci[0]),ci95_upper=float(ci[1]))
        if label != 'total':
            ci4 = np.quantile(change,[.00625,.99375])
            row.update(ci98_75_lower=float(ci4[0]),ci98_75_upper=float(ci4[1]))
        contrasts.append(row)
    pd.DataFrame(metrics).to_csv(OUT/'cross_metrics.csv', index=False)
    pd.DataFrame(contrasts).to_csv(OUT/'paired_contrasts.csv', index=False)
    np.savez_compressed(OUT/'draws.npz',**draws)
    # Independent reconciliation against the already verified 36-cell grid.
    existing = pd.read_csv(PARENT/'source_cross_matrix.csv')
    for row in metrics:
        bn = '19LA' if row['bona']=='19LA_eval' else '21LA'
        sn = '19LA' if row['spoof']=='19LA_eval' else '21LA'
        expected = existing[(existing.bona==bn)&(existing.spoof==sn)].iloc[0]
        assert abs(row['eer_percent']-expected.eer_percent) < 1e-10
    for tag, key in [('A','19LA_eval_bona__19LA_eval_spoof'),
                     ('D','21LA_eval_bona__21LA_eval_spoof')]:
        assert np.array_equal(draws[tag],saved[key])
    result = dict(status='pass', elapsed_seconds=time.monotonic()-start,
                  target='21LA_eval',baseline='19LA_eval',metrics=metrics,contrasts=contrasts,
                  score_files_hash_verified=source_audit,bootstrap_draws=2000,
                  relevant_shared_ASV_clusters=sum(g.startswith('asvspoof::') for g in groups),
                  bootstrap='Source-stratified shared speaker weights from seed 2026110522',
                  verification='Independent DET/rank AUC, weighted ROC point checks, prior source-grid reconciliation',
                  limits=['Post-hoc fixed-model score-distribution accounting; no training intervention.',
                          '19LA and 21LA have shared source material; speaker resampling omits unresolved original-utterance families.',
                          '21LA spoof replacement changes channel and transmission conditions; it is not a pure attack-method intervention.',
                          'Intervals condition on this checkpoint; they exclude training/model-selection uncertainty.',
                          '98.75% intervals correct the four edge contrasts only, not all analyses in this project.'])
    save('summary.json',result)
    (OUT/'complete').write_text('0\n')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
