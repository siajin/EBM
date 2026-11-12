"""Narrow the normal-domain gap using existing scores and backend h/z.

No silence hypothesis, waveform transfer, new inference, or model fitting.
Restrictions and standardization are observational, not audio interventions.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import hashlib
import numpy as np
import pandas as pd
from geometry_cache import resolve_run
from audit_cached_core_results import independent_eer

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/geometry_cache/local_analysis/normal_source_mechanisms_v1'
PREV=ROOT/'artifacts/geometry_cache/local_analysis/bonafide_domain_shift_v1'
CHECKPOINT='104873703a57cad98b686d770e0860e031a2cf9abef79b72668780c1660e0b24'


def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,
        default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist(),allow_nan=False)+'\n',encoding='utf8')


def get_normal(alias,arrays=False):
    folder,entry,split=resolve_run(alias)
    assert entry['identity']['checkpoint_sha256']==CHECKPOINT
    f=pd.read_parquet(folder/'extract/scores.parquet')
    f=f[(f.split==split)&(f.y==1)].copy().reset_index(drop=True)
    if 'source_corpus' not in f:f['source_corpus']='asvspoof'
    f['analysis_dataset']=alias
    vectors={}
    if arrays:
        for kind in ['h','z']:
            out=np.empty((len(f),160),float)
            for rank in [0,1]:
                pos=np.flatnonzero(f['rank'].to_numpy()==rank)
                ix=f.iloc[pos].embedding_row.to_numpy(int)
                arr=np.load(folder/f'extract/rank{rank}_{kind}.npy',mmap_mode='r')
                ids=pd.read_parquet(folder/f'extract/rank{rank}_rows.parquet',columns=['utterance_id'])
                assert ids.iloc[ix].utterance_id.tolist()==f.iloc[pos].utterance_id.tolist()
                out[pos]=arr[ix]
            assert np.isfinite(out).all()
            vectors[kind]=out
        c=np.load(folder/'extract/rank0_centroid.npy')
        uz=vectors['z']/np.linalg.norm(vectors['z'],axis=1,keepdims=True)
        assert np.max(np.abs(uz@c-f.cosine_s.to_numpy()))<1e-10
    return f,vectors


def stats(f,threshold):
    a=f.cosine_s.to_numpy();n=len(a);k=int(sum(a<threshold))
    return dict(n=n,normal_false_rejects=k,normal_false_reject_percent=100*k/n,
        median_score=float(np.median(a)),median_duration=float(f.duration.median()),
        n_speakers=f.speaker_group.nunique())


def eta(x,labels):
    mu=x.mean(0);total=float(np.sum((x-mu)**2));between=0.
    for label in np.unique(labels):
        q=x[labels==label];between+=len(q)*float(np.sum((q.mean(0)-mu)**2))
    return between/total


def main():
    if (OUT/'complete').exists():raise FileExistsError('Completed output already exists')
    recovery=OUT.exists()
    OUT.mkdir(parents=True,exist_ok=True)
    if recovery:save('recovery.json',dict(reason='OOF files contain scores/IDs only; bind labels by unique ID to the original cache',new_fits=0,prior_results_changed=False))
    previous=json.loads((PREV/'summary.json').read_text(encoding='utf8'))
    threshold=previous['thresholds']['19LA_near_EER']
    save('protocol.json',dict(status='posthoc_exploratory',
        timestamp=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),checkpoint=CHECKPOINT,
        excludes_silence_as_explanation=True,threshold=threshold,
        primary_contrast='DF normals: corpus gap after nocodec/no-repeat-padding restrictions',
        no_new_model_or_fit=True,
        geometry_scope='h is final backend linear output before LayerNorm, z after LayerNorm; neither is raw SSL encoder output',
        warning='Restricted groups differ in speakers/content/duration; this is not a causal removal effect.'))
    parts=[get_normal(n,True) for n in ['DF_pilot','DF_remaining']]
    f=pd.concat([p[0] for p in parts],ignore_index=True)
    assert len(f)==14869 and not f.utterance_id.duplicated().any()
    vec={k:np.concatenate([p[1][k] for p in parts]) for k in ['h','z']}
    f['bad']=f.cosine_s<threshold
    f['duration_bin']=pd.cut(f.duration,[0,2,3,4.0375,5,6,8,np.inf],right=False).astype(str)
    restrictions={
        'all':np.ones(len(f),bool),
        'nocodec':(f.codec=='nocodec').to_numpy(),
        'no_repeat_padding':(f.padding_fraction==0).to_numpy(),
        'nocodec_no_repeat_padding':((f.codec=='nocodec')&(f.padding_fraction==0)).to_numpy(),
        'duration_4p0375_to_6':((f.duration>=4.0375)&(f.duration<6)).to_numpy(),
        'nocodec_duration_4p0375_to_6':((f.codec=='nocodec')&(f.duration>=4.0375)&(f.duration<6)).to_numpy(),
    }
    results=[]
    rng=np.random.default_rng(2026110523)
    for restriction,mask in restrictions.items():
        for corpus,g in f[mask].groupby('source_corpus'):
            r=dict(restriction=restriction,source=corpus,**stats(g,threshold))
            # Conditional percentile intervals over speakers; no family map.
            counts=g.groupby('speaker_group').bad.agg(['sum','count'])
            w=rng.multinomial(len(counts),np.full(len(counts),1/len(counts)),size=2000)
            values=100*(w@counts['sum'].to_numpy())/(w@counts['count'].to_numpy())
            r['cluster95_low'],r['cluster95_high']=np.quantile(values,[.025,.975])
            results.append(r)
    pd.DataFrame(results).to_csv(OUT/'restriction_errors.csv',index=False)
    cell=[]
    for (corpus,codec,dur),g in f.groupby(['source_corpus','codec','duration_bin'],observed=True):
        cell.append(dict(source=corpus,codec=codec,duration_bin=dur,**stats(g,threshold)))
    pd.DataFrame(cell).to_csv(OUT/'codec_duration_cells.csv',index=False)
    speakers=f.groupby(['source_corpus','speaker_group']).bad.agg(['sum','count','mean']).reset_index()
    speakers.to_csv(OUT/'speaker_errors.csv',index=False)
    speaker_ranges=[]
    for corpus,g in speakers.groupby('source_corpus'):
        speaker_ranges.append(dict(source=corpus,n_speakers=len(g),
            minimum_percent=100*g['mean'].min(),median_percent=100*g['mean'].median(),maximum_percent=100*g['mean'].max()))
    # Direct standardization: same empirical ASVspoof codec x duration-bin mix.
    f['cell']=f.codec+'|'+f.duration_bin
    p=f.groupby(['source_corpus','cell']).bad.agg(['sum','count','mean'])
    shared=set.intersection(*(set(p.loc[c].index) for c in ['asvspoof','vcc2018','vcc2020']))
    target=f[(f.source_corpus=='asvspoof')&f.cell.isin(shared)].cell.value_counts(normalize=True)
    standardized=[]
    for corpus in ['asvspoof','vcc2018','vcc2020']:
        estimates=p.loc[corpus].loc[target.index,'mean']
        standardized.append(dict(source=corpus,standardized_FR_percent=100*float(np.dot(target,estimates)),
            included_normals=int(sum((f.source_corpus==corpus)&f.cell.isin(shared))),common_cells=len(shared)))
    save('codec_duration_standardization.json',standardized)
    # The corpus shift is already present before the final LayerNorm, without
    # conflating that layer's input with the frontend representation.
    unitz=vec['z']/np.linalg.norm(vec['z'],axis=1,keepdims=True)
    centeredh=vec['h']-vec['h'].mean(1,keepdims=True)
    centeredh/=np.linalg.norm(centeredh,axis=1,keepdims=True)
    geometry={}
    for name,x in [('backend_h',vec['h']),('centered_unit_backend_h',centeredh),('unit_z',unitz)]:
        geometry[name]={'source_eta_squared':eta(x,f.source_corpus.to_numpy()),
            'interpretation':'fraction of vector variance between source means; not causal contribution or classifier accuracy'}
    save('backend_geometry.json',geometry)
    # Recheck prior completed in-domain diagnostic predictions, without refit.
    bdir=ROOT/'artifacts/geometry_cache/local_analysis/frozen_all_v1/P3_P4/B/DF_pilot'
    assert json.loads((bdir/'complete.json').read_text())['status']=='complete'
    stored=pd.read_csv(bdir/'metrics.csv');verified=[]
    cachefolder,_,split=resolve_run('DF_pilot')
    original=pd.read_parquet(cachefolder/'extract/scores.parquet',columns=['utterance_id','split','y'])
    original=original.loc[original.split==split,['utterance_id','y']]
    for variant in ['direction_linear','direction_mlp','direction_h']:
        for seed in [101,102,103]:
            q=pd.read_parquet(bdir/f'{variant}_{seed}_OOF.parquet')
            q=q.merge(original,on='utterance_id',how='left',validate='one_to_one')
            assert q.y.notna().all() and len(q)==20000
            # Record actual schema, then use the persisted continuous score.
            scorecol=next((c for c in ['score','logit','prediction'] if c in q),None)
            if scorecol is None:raise ValueError('OOF score schema: '+str(q.columns.tolist()))
            value=independent_eer(q.y.to_numpy(),q[scorecol].to_numpy())
            ref=stored[(stored.variant==variant)&(stored.seed==seed)].iloc[0].eer_percent
            assert abs(value-ref)<1e-8
            verified.append(dict(variant=variant,seed=seed,eer_percent=value))
    save('prior_probe_recheck.json',dict(results=verified,
        scope='DF pilot internal source x speaker nested CV, 5 outer x 3 inner; not T-to-DF transfer',
        interpretation='Some discrimination remains in the full frozen representation; not proof that single-centroid scoring caused all errors.'))
    save('summary.json',dict(status='complete_exploratory',normal_count=len(f),threshold=threshold,
        restrictions=results,speaker_ranges=speaker_ranges,standardized=standardized,
        geometry=geometry,prior_probe_recheck=verified,
        limitations=['No microphone, spectral envelope, pitch, accent, or resampling effect was causally isolated.',
            'Backend h is not the SSL frontend; frontend-vs-backend localization remains unresolved.',
            'Duration conditioning and no-padding restriction alter utterance and speaker mix.',
            'Restricted subgroup bootstrap is conditional on observed speakers and unresolved families.',
            'Probe improvements use DF-internal training and are not deployment generalization.']))
    save('verification.json',dict(status='pass',row_alignment=True,normals=14869,backend_score_reconstruction=True,
        prior_OOF_metrics_verified=len(verified),all_source_totals=sum(r['n'] for r in results if r['restriction']=='all')))
    (OUT/'complete').write_text('0\n')
    print(json.dumps(dict(restrictions=results,speaker_ranges=speaker_ranges,standardized=standardized,geometry=geometry),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
