"""Independent row constraints, response arithmetic and solver fixtures."""
from pathlib import Path
import sys
import json
import hashlib
import itertools
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]/'geometry_cache'
OUT=ROOT/'local_analysis/full_score_overlap_20261106_v2'
summary=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
v1=json.loads((ROOT/'local_analysis/full_score_overlap_20261106_v1/summary.json').read_text(encoding='utf-8'))
cat=json.loads((ROOT/'catalog.json').read_text(encoding='utf-8'))
r23=json.loads((ROOT/'r23_catalog.json').read_text(encoding='utf-8'))


def load(key):return pd.read_parquet(ROOT/cat['runs'][key]['local_folder']/'extract/scores.parquet')


native={'DF_full':pd.concat([load('21df_pilot'),load('frozen_DF_remaining')],ignore_index=True).set_index('utterance_id'),
        '21LA_full':load('21la_eval').set_index('utterance_id')}
raw={}
for ds,rname in [('DF_full','DF_pilot'),('21LA_full','21LA_pilot')]:
    for cond in ['R2-00','R2-10','R2-01','R2-11','R3-noise20','R3-noise10','R3-Opus32k_recompression']:
        e=r23['entries'][rname+'::'+cond]
        raw[ds,cond]=pd.read_parquet(ROOT/e['local_folder']/rname/cond/'scores.parquet').set_index('utterance_id')

checks=0;pair_total=0;max_response_error=0.;stratified=[];missing=[]
for row in summary['overlap']:
    if row['matching']=='score_only_uncontrolled':continue
    p=pd.read_parquet(OUT/row['pair_file']);f=native[row['dataset']]
    assert len(p)==row['n_pairs'] and p.bona_uid.is_unique and p.spoof_uid.is_unique
    b=f.loc[p.bona_uid].reset_index();s=f.loc[p.spoof_uid].reset_index()
    assert (b.y==1).all() and (s.y==0).all()
    assert abs(b.cosine_s-s.cosine_s).le(row['caliper']).all() and abs(b.duration-s.duration).le(.25).all()
    cols=['source_corpus','codec']+(['transmission'] if 'transmission' in f else [])+(['speaker_group'] if row['matching']=='strict' else [])
    for col in cols:assert b[col].tolist()==s[col].tolist()
    bins=[0,2,3,4.0375,6,np.inf]
    assert np.array_equal(np.digitize(b.duration,bins),np.digitize(s.duration,bins))
    old=next(x for x in v1['overlap'] if all(x.get(k)==row.get(k) for k in ['dataset','matching','caliper','availability']))
    assert row['n_pairs']==old['n_pairs']
    if len(p):assert row['mean_score_gap']<=old['mean_score_gap']+1e-12
    assert row['certificate_pass'] and row['n_pairs']==row['minimum_vertex_cover_size']
    checks+=1;pair_total+=len(p)
    available=set(raw[row['dataset'],'R2-00'].index)
    if row['availability']=='existing_R23':
        assert set(p.bona_uid).issubset(available) and set(p.spoof_uid).issubset(available)
        for r in summary['responses']:
            if any(r[k]!=row[k] for k in ['dataset','matching','caliper']):continue
            a=raw[row['dataset'],r['condition']]
            db=a.loc[p.bona_uid,'cosine_s'].to_numpy()-b.cosine_s.to_numpy()
            ds=a.loc[p.spoof_uid,'cosine_s'].to_numpy()-s.cosine_s.to_numpy()
            for key,expected in [('mean_bona_delta',np.mean(db)),('mean_abs_bona_delta',np.mean(abs(db))),('mean_spoof_delta',np.mean(ds)),('mean_D',np.mean(ds-db))]:
                max_response_error=max(max_response_error,abs(float(expected)-r[key]))
            mid=(p.bona_s.to_numpy()+p.spoof_s.to_numpy())/2
            for lo,hi in [(-1.00001,-.9),(-.9,0),(0,.9),(.9,1.00001)]:
                ix=(mid>=lo)&(mid<hi)
                if not ix.any():continue
                stratified.append(dict(dataset=row['dataset'],matching=row['matching'],caliper=row['caliper'],condition=r['condition'],
                    native_midpoint_lo=lo,native_midpoint_hi=hi,n_pairs=int(ix.sum()),mean_bona_delta=float(db[ix].mean()),
                    mean_spoof_delta=float(ds[ix].mean()),mean_D=float(np.mean(ds[ix]-db[ix])),descriptive_only=True))
    elif row['matching']=='strict' and row['caliper']==.02:
        for u in pd.concat([p.bona_uid,p.spoof_uid]):
            if u not in available:
                rr=f.loc[u]
                missing.append(dict(dataset=row['dataset'],utterance_id=u,y=int(rr.y),source_corpus=rr.source_corpus,speaker_group=rr.speaker_group,
                    codec=rr.codec,native_score=float(rr.cosine_s),duration=float(rr.duration)))

assert max_response_error<1e-10
pd.DataFrame(stratified).to_csv(OUT/'existing_response_by_native_score.csv',index=False)
pd.DataFrame(missing).to_parquet(OUT/'missing_endpoints_for_locked_strict_pairs.parquet',index=False)

# Exhaustive enumeration verifies the two objectives on small synthetic graphs.
from check_full_score_overlap_20261106 import certified_match
rng=np.random.default_rng(42)
for trial in range(50):
    nb=int(rng.integers(1,6));ns=int(rng.integers(1,6));bs=rng.normal(size=nb);ss=rng.normal(size=ns)
    allowed=rng.random((nb,ns))<.45;graph=csr_matrix(allowed.astype(np.int8));ib,js,cert=certified_match(graph,bs,ss)
    best_n=-1;best_cost=np.inf
    def search(i,used,n,cost):
        global best_n,best_cost
        if i==nb:
            if n>best_n or (n==best_n and cost<best_cost):best_n,best_cost=n,cost
            return
        search(i+1,used,n,cost)
        for j in np.flatnonzero(allowed[i]):
            if j not in used:search(i+1,used|{int(j)},n+1,cost+abs(bs[i]-ss[j]))
    search(0,set(),0,0.)
    assert len(ib)==best_n and abs(np.sum(abs(bs[ib]-ss[js]))-best_cost)<1e-10

v=json.loads((OUT/'verification.json').read_text(encoding='utf-8'))
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in v['inputs_sha256'].items())
result=dict(status='pass',controlled_pair_files_checked=checks,total_pair_rows_checked_across_nonindependent_settings=pair_total,
    max_response_arithmetic_error=max_response_error,maximum_counts_agree_with_unweighted_v1=True,
    all_v2_native_score_gap_objectives_no_worse_than_v1=True,exhaustive_small_solver_cases=50,
    inputs_unchanged=True,new_forward=False,new_model_fit=False,
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    missing_forward_endpoints_for_selected_strict_pairs=len(missing),
    limits=['no uncertainty intervals or response-effect significance claimed','original-family mapping unresolved','missing endpoint count is for fixed pair lists, not a minimum-inference-cost optimization'])
(OUT/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result))
