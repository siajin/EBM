"""Independent result reconciliation for the detailed normal-domain audit."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from audit_cached_core_results import independent_eer,independent_auc
from geometry_cache import load_predictions

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/geometry_cache/local_analysis/normal_cause_audit_v1'
TH=.8619597055730815


def main():
    checks=[]
    a=pd.read_parquet(OUT/'analysis_rows.parquet');features=pd.read_csv(OUT/'features.csv')
    completed=json.loads((OUT/'complete.json').read_text())
    assert hashlib.sha256((OUT/'features.csv').read_bytes()).hexdigest()==completed['features_sha256']
    d=a.merge(features,on=['utterance_id','case_id'],validate='one_to_one');assert len(d)==3419
    assert np.max(np.abs(d.source_frames/d.sample_rate-d.duration))<1e-9
    checks.append('All3419 acoustic records match original cached durations and unique IDs; SHA256 matches server completion')
    n=pd.read_parquet(OUT/'normal_nocodec_acoustics.parquet')
    assert len(n)==1156 and (n.y==1).all() and (n.codec=='nocodec').all()
    assert set(n.sample_rate)=={16000} and set(n.channels)=={1} and set(n.subtype)=={'PCM_16'}
    assert n.source_corpus.value_counts().to_dict()=={'vcc2018':420,'asvspoof':386,'vcc2020':350}
    assert ((n.cosine_s<TH)==n.normal_error).all()
    checks.append('All no-codec normal analysis membership and labels verified; input format identical')
    p=pd.read_parquet(OUT/'normal_reference_scores.parquet');records=json.loads((OUT/'normal_reference_metrics.json').read_text())
    for r in records:
        z=p[r['method']].to_numpy();y=p.y.to_numpy()
        assert abs(independent_eer(y,z)-r['eer_percent'])<1e-8
        assert abs(independent_auc(y,z)-r['auroc'])<1e-10
        r['interpretation']='Fixed original EBM' if r['method']=='native' else 'T-only normal prototype fit' if r['method']=='T_three_prototypes' else 'DF internal normal-only5-fold diagnostic'
    (OUT/'normal_reference_metrics.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf8')
    checks.append('All7 normal-reference EER and AUROC values match independent DET/rank implementations')
    q=pd.read_parquet(OUT/'neighborhood_scores.parquet');rec=json.loads((OUT/'neighborhood_complete.json').read_text())
    for r in rec['metrics']:
        label='T' if r['reference']=='T_reference' else 'DF_OOF';s=q[label+'_'+r['rule']].to_numpy();y=q.y.to_numpy()
        assert abs(independent_eer(y,s)-r['eer_percent'])<1e-8
        assert abs(independent_auc(y,s)-r['auroc'])<1e-10
    checks.append('All4 neighborhood EER/AUROC values independently agree')
    f=pd.read_csv(OUT/'normal_neighborhood_groups.csv')
    for r in f.itertuples():
        label='T' if r.reference=='T_reference' else 'DF_OOF'
        ix=(q.y==1)&(q.source_corpus==r.source)&((q.cosine_s<TH)==r.native_rejected)
        assert sum(ix)==r.n
        assert abs(100*float((q.loc[ix,label+'_nearest1']>0).mean())-r.nearest_normal_percent)<1e-10
    checks.append('All nearest-neighbor normal subgroup counts and rates reconcile')
    paired=pd.read_parquet(OUT/'paired_input_scores.parquet');pr=pd.read_csv(OUT/'paired_input_effects.csv')
    for r in pr.itertuples():
        g=paired[(paired.condition==r.condition)&(paired.source_corpus==r.source)&(paired.y==r.y)]
        b=g.cosine_s_native<TH if r.y==1 else g.cosine_s_native>=TH
        c=g.cosine_s_changed<TH if r.y==1 else g.cosine_s_changed>=TH
        assert len(g)==r.n
        assert int(sum(b&~c))==r.corrected and int(sum(~b&c))==r.new_errors
        assert abs((r.new_errors-r.corrected)/r.n*100-r.change_pp)<1e-10
    checks.append('All18 source/class/input-treatment corrected and new-error counts reconcile exactly')
    o=pd.read_parquet(OUT/'acoustic_error_OOF.parquet');yy=o.normal_error.to_numpy();assert len(o)==1156
    effects=json.loads((OUT/'acoustic_error_model_effects.json').read_text())
    def loss(s):
        s=np.clip(s,1e-7,1-1e-7);return -.5*(np.log(s[yy==1]).mean()+np.log(1-s[yy==0]).mean())
    for r in effects:
        assert abs(roc_auc_score(yy,o[r['case']])-r['OOF_AUROC'])<1e-10
        assert abs(loss(o.metadata_only.to_numpy())-loss(o[r['case']].to_numpy())-r['balanced_LL_reduction'])<1e-12
    checks.append('All7 grouped-acoustic model effect point estimates independently reconcile from saved OOF probabilities')
    folds=json.loads((ROOT/'artifacts/geometry_cache/local_analysis/frozen_all_v1/P3_P4/B/DF_pilot/fold_manifest.json').read_text())
    assert folds['UID_order']==q.utterance_id.tolist()
    for fold in folds['folds']:
        train=q.iloc[fold['train_rows']];test=q.iloc[fold['test_rows']]
        assert not set(train.dependency_group)&set(test.dependency_group)
    checks.append('All reused5-fold reference splits keep source x speaker groups disjoint')
    # Conditional paired cluster uncertainty of the neighborhood change.
    effects=[]
    for source in ['vcc2018','vcc2020']:
        z=q[(q.y==1)&(q.source_corpus==source)&(q.cosine_s<TH)].copy()
        z['delta']=(z.DF_OOF_nearest1>0).astype(int)-(z.T_nearest1>0).astype(int)
        sums=z.groupby('dependency_group').delta.agg(['sum','count'])
        w=np.random.default_rng(5088).multinomial(len(sums),np.full(len(sums),1/len(sums)),size=2000)
        values=100*(w@sums['sum'].to_numpy())/(w@sums['count'].to_numpy())
        effects.append(dict(source=source,n=len(z),normal_neighbor_recovery_change_pp=float(100*z.delta.mean()),
            conditional_cluster_CI95=np.quantile(values,[.025,.975]).tolist()))
    (OUT/'neighborhood_conditional_effects.json').write_text(json.dumps(effects,indent=2),encoding='utf8')
    result=dict(status='pass',checks=checks,normal_neighborhood_effects=effects,
        original_family_map='Official public mapping retrieval stalled; no original mapping/audio was downloaded to local disk. Family dependence remains unresolved.',
        source='Server originals and completed derived caches; no external analytical dataset',
        newly_extracted_rows=3419,new_EBM_training=False,new_GPU_inference=False)
    (OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
