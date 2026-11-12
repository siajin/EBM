"""Independent raw-row arithmetic and saved-matching verification, CPU only."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]/'geometry_cache'
OUT=ROOT/'local_analysis/perturbation_candidate_audit_20261106_v1'
summary=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
catalog=json.loads((ROOT/'r23_catalog.json').read_text(encoding='utf-8'))
original=json.loads((OUT/'verification.json').read_text(encoding='utf-8'))
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in original['inputs_sha256'].items())
max_error=0.;checks=0
for ds in ['DF_pilot','21LA_pilot']:
    e=catalog['entries'][ds+'::R2-00']
    native=pd.read_parquet(ROOT/e['local_folder']/ds/'R2-00/scores.parquet').set_index('utterance_id')
    for r in summary['class_response']:
        if r['dataset']!=ds:continue
        raw=pd.read_parquet(ROOT/'local_analysis/frozen_all_v1/R2_R3'/ds/(r['condition']+'_paired.parquet')).set_index('utterance_id')
        ids=raw.index[raw.y==r['y']]
        d=np.asarray([float(raw.loc[u,'score'])-float(native.loc[u,'cosine_s']) for u in ids])
        assert len(d)==r['n']
        for key,value in [('mean_delta',sum(d)/len(d)),('mean_abs_delta',sum(abs(d))/len(d)),('median_delta',np.median(d))]:
            max_error=max(max_error,abs(float(value)-r[key]));checks+=1
    for kind in ['strict_same_speaker','relaxed_cross_speaker']:
        p=pd.read_parquet(OUT/f'{ds}_{kind}_pairs.parquet')
        assert p.bona_uid.is_unique and p.spoof_uid.is_unique
        b=native.loc[p.bona_uid].reset_index();s=native.loc[p.spoof_uid].reset_index()
        assert (b.y==1).all() and (s.y==0).all()
        assert (abs(b.cosine_s-s.cosine_s)<=.02).all() and (abs(b.duration-s.duration)<=.25).all()
        for c in ['source_corpus','codec']+(['transmission'] if 'transmission' in native else [])+(['speaker_group'] if kind=='strict_same_speaker' else []):
            assert b[c].fillna('<missing>').tolist()==s[c].fillna('<missing>').tolist()
        for r in summary['native_score_matching']:
            if r['dataset']!=ds or r['matching']!=kind:continue
            raw=pd.read_parquet(ROOT/'local_analysis/frozen_all_v1/R2_R3'/ds/(r['condition']+'_paired.parquet')).set_index('utterance_id')
            bd=raw.loc[p.bona_uid,'score'].to_numpy()-b.cosine_s.to_numpy()
            sd=raw.loc[p.spoof_uid,'score'].to_numpy()-s.cosine_s.to_numpy()
            max_error=max(max_error,abs(float(np.mean(sd-bd))-r['mean_spoof_minus_bona_delta']));checks+=1

raw=pd.read_parquet(ROOT/'local_analysis/padding_causal_20261106_v1/paired_derived.parquet')
base=raw[raw.condition=='native_wrap'].set_index('utterance_id')
for r in summary['class_response']:
    if r['dataset']!='selected_DF471':continue
    cond=r['condition'];values=[]
    if cond.startswith('native_'):
        for row in raw[(raw.condition==cond)&(raw.y==r['y'])].itertuples():
            values.append(row.cosine_s-base.loc[row.utterance_id,'cosine_s'])
    else:
        sec=cond.split('_')[0].removeprefix('crop')
        same_content='same_content' in cond
        kind='original' if same_content else cond.rsplit('_',1)[-1]
        ids=base.index[(base.y==r['y'])&(base.duration_bin=='long')]
        for uid in ids:
            g=raw[raw.utterance_id==uid].set_index('condition').cosine_s
            after=(g[f'crop{sec}_head_{kind}']+g[f'crop{sec}_tail_{kind}'])/2
            before=(g[f'crop{sec}_head_wrap']+g[f'crop{sec}_tail_wrap'])/2 if same_content else g['native_wrap']
            values.append(after-before)
    assert len(values)==r['n']
    for key,value in [('mean_delta',np.mean(values)),('mean_abs_delta',np.mean(np.abs(values))),('median_delta',np.median(values))]:
        max_error=max(max_error,abs(float(value)-r[key]));checks+=1
assert max_error<1e-10
# Central difference identity: final spherical head perturbation contains no
# independent information when the tangent displacement is symmetric.
rng=np.random.default_rng(6)
for _ in range(100):
    u=rng.normal(size=160);u/=np.linalg.norm(u);c=rng.normal(size=160);c/=np.linalg.norm(c)
    v=rng.normal(size=160);v-=v@u*u;v/=np.linalg.norm(v);eps=.13
    up=np.cos(eps)*u+np.sin(eps)*v;um=np.cos(eps)*u-np.sin(eps)*v
    assert abs(c@u-(c@up+c@um)/2-(1-np.cos(eps))*(c@u))<1e-13
result=dict(status='pass',independently_recomputed_scalar_checks=checks,max_error=max_error,
            matching_constraints_and_unique_IDs_checked=True,all_input_hashes_unchanged=True,
            symmetric_spherical_score_identity_cases=100,
            not_verified=['bootstrap intervals not independently rerun','source-family mapping unresolved','no response classifier transfer tested'])
(OUT/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result))
