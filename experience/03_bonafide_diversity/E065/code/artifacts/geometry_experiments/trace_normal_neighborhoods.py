"""Trace native-rejected normals against balanced T and held-out DF neighbors."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from geometry_cache import load_embeddings
from fast_analyze import eer

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/geometry_cache/local_analysis/normal_cause_audit_v1'
TH=.8619597055730815


def unit(x):return x/np.linalg.norm(x,axis=1,keepdims=True)


def select(rows,ix,n,tag):
    return np.asarray(sorted(ix,key=lambda i:hashlib.sha256((tag+rows.iloc[i].utterance_id).encode()).digest())[:n],int)


def score_neighbors(reference,rr,query,tag):
    # Equal numbers per class remove the reference class-count advantage.
    ix0=np.flatnonzero(rr.y.to_numpy()==0);ix1=np.flatnonzero(rr.y.to_numpy()==1);n=min(len(ix0),len(ix1))
    ix0=select(rr,ix0,n,tag+'spoof|');ix1=select(rr,ix1,n,tag+'bona|')
    dist={}
    for y,ix in [(0,ix0),(1,ix1)]:
        model=NearestNeighbors(n_neighbors=5,metric='cosine',algorithm='brute').fit(reference[ix])
        d,_=model.kneighbors(query);dist[y]=d
    return {'nearest1':dist[0][:,0]-dist[1][:,0],
            'mean5':dist[0].mean(1)-dist[1].mean(1),
            'd_bona1':dist[1][:,0],'d_spoof1':dist[0][:,0]},n


def main():
    assert not (OUT/'neighborhood_complete.json').exists()
    lock=dict(scope='exploratory geometry, not acoustic causality',metric='cosine distance of complete160-dimensional z',
        reference_balancing='same number of T/DF training examples per class, metadata-only SHA256 order',
        variants=['nearest1','mean5'],new_EBM_fit=False,
        DF_fold_scope='same saved source x speaker5 folds; target-domain labeled training examples are used only within their training fold')
    (OUT/'neighborhood_lock.json').write_text(json.dumps(lock,indent=2),encoding='utf8')
    t=load_embeddings('T');df=load_embeddings('DF_pilot');ut=unit(t.z.astype(float));u=unit(df.z.astype(float));f=df.rows.copy();y=f.y.to_numpy()
    assert np.max(np.abs(u@df.centroid-f.cosine_s.to_numpy()))<1e-10
    trained,nT=score_neighbors(ut,t.rows,u,'T-neighborhood|')
    foldroot=ROOT/'artifacts/geometry_cache/local_analysis/frozen_all_v1/P3_P4/B/DF_pilot/fold_manifest.json'
    folds=json.loads(foldroot.read_text());assert folds['UID_order']==f.utterance_id.tolist()
    target={k:np.full(len(f),np.nan) for k in trained};balance=[]
    for fold in folds['folds']:
        train=np.asarray(fold['train_rows']);test=np.asarray(fold['test_rows'])
        assert not set(f.iloc[train].dependency_group)&set(f.iloc[test].dependency_group)
        score,n=score_neighbors(u[train],f.iloc[train].reset_index(drop=True),u[test],'DF-neighborhood-fold'+str(fold['fold'])+'|')
        for name,a in score.items():target[name][test]=a
        balance.append(n)
    metrics=[];normals=[]
    for label,preds in [('T_reference',trained),('DF_internal_OOF_reference',target)]:
        for name in ['nearest1','mean5']:
            a=preds[name];assert np.isfinite(a).all()
            metrics.append(dict(reference=label,rule=name,eer_percent=eer(y,a),auroc=roc_auc_score(y,a)))
        for source,g in f[y==1].groupby('source_corpus'):
            for native_bad in [False,True]:
                ix=g.index[(g.cosine_s<TH)==native_bad].to_numpy()
                if not len(ix):continue
                normals.append(dict(reference=label,source=source,native_rejected=native_bad,n=len(ix),
                    nearest_normal_percent=100*float(np.mean(preds['nearest1'][ix]>0)),
                    mean5_normal_percent=100*float(np.mean(preds['mean5'][ix]>0)),
                    median_nearest_bona_distance=float(np.median(preds['d_bona1'][ix])),
                    median_nearest_spoof_distance=float(np.median(preds['d_spoof1'][ix]))))
    predictions=f[['utterance_id','y','source_corpus','dependency_group','cosine_s']].copy()
    for label,preds in [('T',trained),('DF_OOF',target)]:
        for name,a in preds.items():predictions[label+'_'+name]=a
    predictions.to_parquet(OUT/'neighborhood_scores.parquet',index=False)
    pd.DataFrame(normals).to_csv(OUT/'normal_neighborhood_groups.csv',index=False)
    result=dict(status='complete_exploratory',balanced_T_per_class=nT,balanced_DF_per_class_by_fold=balance,
        metrics=metrics,normal_groups=normals,
        limitations=['Nearest-neighbor rule is a diagnostic with fixed k=1/5, not a selected detector.',
            'T support and DF-internal support use different labeled domains and cannot be read as a deployment gain.',
            'These backend z vectors do not isolate the pretrained encoder or any physical acoustic factor.'])
    (OUT/'neighborhood_complete.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    with threadpool_limits(2):main()
