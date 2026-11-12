"""Independent weighted-ROC and result reconciliation for the DF diagnosis."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from test_bonafide_domain_shift import OUT, GroupedClass, weighted_mixed, load, errors


def main():
    rng=np.random.default_rng(73);checks=0
    for ties in [False,True]:
        for _ in range(20):
            n=120;y=rng.integers(0,2,n);s=rng.normal(size=n)
            if ties:s=np.round(s)
            c=rng.integers(0,5,n)
            f=pd.DataFrame(dict(y=y,cosine_s=s,cluster=['g'+str(x) for x in c]))
            groups=['g'+str(i) for i in range(5)];w=rng.integers(1,5,(8,5))
            actual=weighted_mixed(GroupedClass(f,1,groups),GroupedClass(f,0,groups),w)
            for j in range(len(w)):
                fa,tp,_=roc_curve(y,s,sample_weight=w[j,c],drop_intermediate=False)
                fr=1-tp;d=fa-fr;u=np.flatnonzero(d>=0)[0];l=max(0,u-1)
                a=-d[l]/(d[u]-d[l]) if d[u]!=d[l] else 0
                expected=100*(fa[l]+a*(fa[u]-fa[l]))
                assert abs(actual[j]-expected)<1e-10
                checks+=1
    summary=json.loads((OUT/'summary.json').read_text(encoding='utf8'))
    f=pd.concat([load('DF_pilot'),load('DF_remaining')],ignore_index=True)
    th=summary['thresholds']['DF_near_EER'];bad=f[(f.y==1)&(f.cosine_s<th)]
    assert len(bad)==482 and bad.corpus.value_counts().to_dict()=={'vcc2018':451,'vcc2020':31}
    points=pd.read_csv(OUT/'class_replacement_eer.csv');raw=np.load(OUT/'cluster_bootstrap_draws.npz')
    for row in points.itertuples():
        if pd.isna(row.ci95_lower):continue
        key=f'{row.bona_dataset}_bona__{row.spoof_dataset}_spoof'
        assert np.max(np.abs(np.quantile(raw[key],[.025,.975])-[row.ci95_lower,row.ci95_upper]))<1e-12
    source=pd.read_csv(OUT/'source_errors.csv')
    for label,threshold in summary['thresholds'].items():
        aggregate=source[source.threshold_name==label]
        totals=errors(f,threshold)
        for field in ['n_bona','n_spoof','bona_false_rejects','spoof_false_accepts']:
            assert aggregate[field].sum()==totals[field]
    attacks=pd.read_csv(OUT/'shared_attack_metrics.csv').pivot(index='attack',columns='dataset',values='eer_percent')
    worse=int(sum(attacks.DF_asvspoof>attacks['19LA_eval']))
    assert worse==4
    result=dict(status='pass',independent_random_weighted_ROC_checks=checks,
        tied_scores_included=True,all_bootstrap_intervals_recomputed=True,
        source_counts_reconciled_at_all_thresholds=True,
        normal_errors_at_DF_nearest_EER={'total':len(bad),'by_corpus':bad.corpus.value_counts().to_dict()},
        shared_attack_EER_worsened=4,shared_attacks_total=13,
        caveat='Aggregate spoof shift is favorable; this does not imply every individual attack improves.')
    (OUT/'independent_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
