"""Normal-only reference interventions and existing attention ablations.

Diagnostic DF-internal folds; no EBM or spoof-supervised fitting is done here.
All stated methods are reported, including failures. No test-selected method.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
from geometry_cache import load_embeddings
from frozen_r23_cache import load as load_r23
from fast_analyze import eer
from sklearn.metrics import roc_auc_score
from frozen_common import weighted_eers

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/geometry_cache/local_analysis/normal_cause_audit_v1'
FOLDS=ROOT/'artifacts/geometry_cache/local_analysis/frozen_all_v1/P3_P4/B/DF_pilot/fold_manifest.json'
TH=.8619597055730815


def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,
        default=lambda a:a.item() if isinstance(a,np.generic) else a.tolist(),allow_nan=False)+'\n',encoding='utf8')


def unit(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)


def main():
    assert not (OUT/'normal_reference_complete.json').exists()
    names=['native','T_three_prototypes','DF_one_global_normal_center','DF_source_normal_center',
           'DF_source_translation','DF_source_normal_score_standardization','DF_three_normal_prototypes']
    save('normal_reference_lock.json',dict(methods=names,selection=False,fit_uses_only_normal_labels=True,
        folds='identical saved DF 5-fold source x speaker folds',
        source_aware_methods_require_source_metadata=True,
        explanation='Test sufficiency of normal-reference coverage, source mean shift, and normal scalar calibration. Not a causal test of acoustic recording origin.'))
    data=load_embeddings('DF_pilot');t=load_embeddings('T');f=data.rows;y=f.y.to_numpy();src=f.source_corpus.to_numpy();u=unit(data.z.astype(float));c=data.centroid
    folds=json.loads(FOLDS.read_text());assert folds['UID_order']==f.utterance_id.tolist()
    train_unit=unit(t.z[t.rows.y.to_numpy()==1].astype(float));target=train_unit.mean(0)
    km=KMeans(3,random_state=11,n_init=10).fit(train_unit);tc=unit(km.cluster_centers_)
    scores={'native':f.cosine_s.to_numpy(),'T_three_prototypes':np.max(u@tc.T,axis=1)}
    for name in names[2:]:scores[name]=np.full(len(f),np.nan)
    fold_records=[]
    for fold in folds['folds']:
        train=np.asarray(fold['train_rows']);test=np.asarray(fold['test_rows']);normal=train[y[train]==1]
        assert not set(f.iloc[train].dependency_group)&set(f.iloc[test].dependency_group)
        center=unit(u[normal].mean(0));scores['DF_one_global_normal_center'][test]=u[test]@center
        km=KMeans(3,random_state=11,n_init=10).fit(u[normal]);prototypes=unit(km.cluster_centers_)
        scores['DF_three_normal_prototypes'][test]=np.max(u[test]@prototypes.T,axis=1)
        for source in sorted(set(src)):
            ni=normal[src[normal]==source];ti=test[src[test]==source];mu=u[ni].mean(0)
            scores['DF_source_normal_center'][ti]=u[ti]@unit(mu)
            scores['DF_source_translation'][ti]=unit(u[ti]-mu+target)@c
            # Unbounded monotone transform within a source, with normal-only
            # mean/std estimated in the outer training fold.
            a=np.arctanh(np.clip(scores['native'][ni],-1+1e-7,1-1e-7))
            b=np.arctanh(np.clip(scores['native'][ti],-1+1e-7,1-1e-7))
            scores['DF_source_normal_score_standardization'][ti]=(b-a.mean())/max(a.std(),1e-6)
            fold_records.append(dict(fold=fold['fold'],source=source,normal_train=len(ni),test=len(ti)))
    result=[]
    unique,codes=np.unique(f.dependency_group.to_numpy(),return_inverse=True)
    draws=np.random.default_rng(5519).multinomial(len(unique),np.full(len(unique),1/len(unique)),size=2000)
    bootstrap={}
    for name,score in scores.items():
        assert np.isfinite(score).all()
        bootstrap[name]=weighted_eers(y,score,codes,draws,chunk=32)
        result.append(dict(method=name,eer_percent=eer(y,score),auroc=roc_auc_score(y,score),
            native_minus_candidate_EER_pp=eer(y,scores['native'])-eer(y,score),
            paired_improvement_CI95=np.quantile(bootstrap['native']-bootstrap[name],[.025,.975]).tolist(),
            interpretation='DF internal normal-only diagnostic; source methods are metadata-aware'))
        print('normal reference',name,result[-1]['eer_percent'],flush=True)
    predictions=f[['utterance_id','y','dependency_group']].copy()
    for name,score in scores.items():predictions[name]=score
    predictions.to_parquet(OUT/'normal_reference_scores.parquet',index=False)
    save('normal_reference_metrics.json',result);save('normal_reference_folds.json',fold_records)
    # Attention ablations are actual functional changes of the archived model,
    # evaluated with identical audio/SSL and its controlled RNG path.
    attention=[]
    for cond in ['R2-00','R2-10','R2-01','R2-11']:
        g=load_r23('DF_pilot',cond).rows
        for source,p in [('all',g)]+list(g.groupby('source_corpus')):
            b=p[p.y==1];s=p[p.y==0]
            attention.append(dict(condition=cond,source=source,eer_percent=eer(p.y,p.cosine_s),
                normal_false_reject_percent=100*float((b.cosine_s<TH).mean()),
                spoof_false_accept_percent=100*float((s.cosine_s>=TH).mean()),n_bona=len(b),n_spoof=len(s)))
    pd.DataFrame(attention).to_csv(OUT/'attention_ablation_by_source.csv',index=False)
    save('normal_reference_complete.json',dict(status='complete_exploratory',methods=result,attention=attention,
        no_EBM_training=True,no_spoof_labels_used_in_reference_fitting=True,
        source_metadata_required_for_source_methods=True,source_family_dependence_unresolved=True,
        causal_interpretation='Only interventions on the stated scoring/reference or attention paths, not proof of an acoustic mechanism.'))


if __name__=='__main__':
    with threadpool_limits(2):main()
