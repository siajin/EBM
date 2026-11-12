"""Shared, offline-only contracts for frozen-model experiments v1.2."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from fast_analyze import eer, balanced_ll
from geometry_cache import load_embeddings, CachedEmbeddings, DEFAULT_ROOT

CHECKPOINT = '104873703a57cad98b686d770e0860e031a2cf9abef79b72668780c1660e0b24'
CAL_A, CAL_B = 8.710543812499347, -0.7151751091389169
SEEDS = [101, 102, 103]
LOCK = DEFAULT_ROOT / 'analysis_manifests/frozen_data_sizes_v1'

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def convert(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, Path): return str(x)
        raise TypeError(type(x).__name__)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=convert, allow_nan=False)+'\n', encoding='utf8')
    tmp.replace(path)

def panel(name):
    raw = {'21LA_pilot':'21LA_eval', '21LA_small':'21LA_eval', 'DF_small':'DF_pilot'}.get(name, name)
    data = load_embeddings(raw, expected_checkpoint_sha256=CHECKPOINT)
    if raw == name: return data
    lock = json.loads((LOCK/'lock.json').read_text())
    entry = lock['views'][name]
    assert data.provenance['cache_fingerprint'] == entry['source_cache_fingerprint']
    # The lock is metadata-only. Verify the file hash recorded anywhere in it,
    # then bind rows and the exact original array indices independently.
    path = LOCK/(name+'.parquet')
    assert digest(path) == entry['manifest_sha256']
    ids = pd.read_parquet(path)
    ix = ids.cache_row.to_numpy()
    assert data.rows.iloc[ix].utterance_id.tolist() == ids.utterance_id.tolist()
    assert len(ix) == len(set(ix))
    provenance = dict(data.provenance, analysis_manifest_sha256=digest(path), analysis_alias=name)
    return CachedEmbeddings(data.z[ix], data.h[ix], data.rows.iloc[ix].reset_index(drop=True), data.centroid, provenance)

def threshold(y, score):
    bs = np.sort(np.asarray(score)[np.asarray(y)==1])
    th = float(bs[int(np.floor(.01*len(bs)))])
    return {'threshold':th, 'V_bona_false_reject':float(np.mean(bs<th)),
            'V_spoof_false_accept':float(np.mean(np.asarray(score)[np.asarray(y)==0]>=th))}

def metric(y, score, th=None, logit=False):
    y, score = np.asarray(y), np.asarray(score)
    assert np.isfinite(score).all()
    out = {'n':len(y), 'n_bona':int(sum(y==1)), 'n_spoof':int(sum(y==0))}
    if len(np.unique(y)) == 2:
        out.update(eer_percent=eer(y,score), auroc=float(roc_auc_score(y,score)),
                   balanced_logloss=balanced_ll(y,expit(score if logit else CAL_A*score+CAL_B)))
    else:
        out.update(eer_percent=None, auroc=None, balanced_logloss=None)
    if th is not None:
        out['bona_false_reject'] = float(np.mean(score[y==1]<th)) if np.any(y==1) else None
        out['spoof_false_accept'] = float(np.mean(score[y==0]>=th)) if np.any(y==0) else None
    return out

def stable_indices(rows, prefix, n):
    order = sorted(range(len(rows)), key=lambda i:(hashlib.sha256((prefix+str(rows.iloc[i].utterance_id)).encode()).hexdigest(), str(rows.iloc[i].utterance_id)))
    return np.asarray(order[:n], dtype=int)

def weighted_eers(y, score, codes, draws, chunk=64):
    """Full tied ROC interpolation, vectorized over bootstrap draws."""
    order = np.argsort(-score, kind='stable')
    ys, ss, cc = y[order], score[order], codes[order]
    ends = np.r_[np.flatnonzero(ss[:-1]!=ss[1:]),len(ss)-1]
    answer=[]
    for start in range(0,len(draws),chunk):
        w=draws[start:start+chunk,cc].astype(float)
        pos=w*ys; neg=w*(1-ys)
        total_b, total_s=pos.sum(1),neg.sum(1)
        good=(total_b>0)&(total_s>0)
        fnr=np.c_[np.ones(len(w)),1-np.cumsum(pos,axis=1)[:,ends]/np.maximum(total_b[:,None],1)]
        fpr=np.c_[np.zeros(len(w)),np.cumsum(neg,axis=1)[:,ends]/np.maximum(total_s[:,None],1)]
        diff=fpr-fnr
        upper=np.argmax(diff>=0,axis=1); lower=np.maximum(0,upper-1); r=np.arange(len(w))
        denom=diff[r,upper]-diff[r,lower]
        frac=np.divide(-diff[r,lower],denom,out=np.zeros(len(w)),where=denom!=0)
        value=100*(fpr[r,lower]+frac*(fpr[r,upper]-fpr[r,lower]));value[~good]=np.nan
        answer.extend(value)
    return np.asarray(answer)

def bootstrap_scores(y, scores, groups, draws=10000):
    groups_unique,codes=np.unique(groups,return_inverse=True)
    counts=np.random.default_rng(7301).multinomial(len(groups_unique),np.full(len(groups_unique),1/len(groups_unique)),size=draws)
    values={}
    for name,s in scores.items():
        s=np.asarray(s)
        assert abs(weighted_eers(y,s,codes,np.ones((1,len(groups_unique)),int))[0]-eer(y,s))<1e-9
        values[name]=weighted_eers(y,s,codes,counts)
    return values, {'draws':draws,'groups':len(groups_unique),'class_groups':[len(set(groups[y==v])) for v in [0,1]],'seed':7301,'unresolved_source_families':True,'fit_selection_uncertainty_included':False}

def interval(values, level=.95):
    a=(1-level)/2
    return np.nanquantile(values,[a,1-a]).tolist()

def noise(rows,c,dim=160,unit=False,prefix='frozen-noise-v1|'):
    values=[]
    for uid in rows.utterance_id:
        seed=int.from_bytes(hashlib.sha256((prefix+uid).encode()).digest()[:8],'big')
        v=np.random.default_rng(seed).normal(size=dim)
        if unit:
            assert dim==160
            v-=np.dot(v,c)*c;v/=np.linalg.norm(v)
        values.append(v)
    return np.asarray(values)
