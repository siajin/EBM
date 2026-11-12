"""Retrospective CPU audit of existing paired interventions; no SSH/model fit.

All candidate comparisons are exploratory: outcomes were inspected previously.
Matching is defined from native scores/metadata once, never from intervention
outcomes. No canceled DF internal cross-validation is run.
"""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1] / 'geometry_cache'
OUT = ROOT / 'local_analysis/perturbation_candidate_audit_20261106_v1'
CONDITIONS = ['R2-10','R2-01','R2-11','R3-noise20','R3-noise10','R3-Opus32k_recompression']
INPUTS = {}


def read(p):
    p = Path(p)
    INPUTS[str(p.resolve())] = hashlib.sha256(p.read_bytes()).hexdigest()
    return pd.read_parquet(p) if p.suffix == '.parquet' else json.loads(p.read_text(encoding='utf-8'))


def save(name, obj):
    def default(x):
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, np.ndarray): return x.tolist()
        raise TypeError(type(x).__name__)
    (OUT/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=default, allow_nan=False)+'\n', encoding='utf-8')


def cluster_draws(frame):
    # Fixed source composition; same speaker multiplicity across classes/views.
    keys = (frame.source_corpus.astype(str)+'|'+frame.speaker_group.astype(str)).to_numpy()
    groups, codes = np.unique(keys, return_inverse=True)
    sources = np.array([x.split('|')[0] for x in groups])
    rng = np.random.default_rng(20261106)
    draws = np.zeros((5000, len(groups)), dtype=np.int32)
    for source in np.unique(sources):
        ix = np.flatnonzero(sources == source)
        draws[:, ix] = rng.multinomial(len(ix), np.full(len(ix), 1/len(ix)), size=5000)
    return groups, codes, draws


def summarize(frame, name):
    groups, codes, draws = cluster_draws(frame)
    records = []
    boots = {}
    for y in [0,1]:
        ix = frame.y.to_numpy() == y
        delta = frame.delta.to_numpy()[ix]
        n = np.bincount(codes[ix], minlength=len(groups))
        row = dict(condition=name, y=y, n=int(ix.sum()), speaker_clusters=int((n>0).sum()),
                   native_mean=float(frame.loc[ix,'s0'].mean()), transformed_mean=float(frame.loc[ix,'s1'].mean()),
                   median_delta=float(np.median(delta)), q90_abs_delta=float(np.quantile(abs(delta), .9)),
                   fraction_abs_delta_le_002=float(np.mean(abs(delta)<=.02)),
                   fraction_delta_gt_005=float(np.mean(delta>.05)),
                   fraction_delta_lt_minus005=float(np.mean(delta<-.05)))
        for measure, values in [('mean_delta',delta),('mean_abs_delta',abs(delta))]:
            sums = np.bincount(codes[ix], weights=values, minlength=len(groups))
            denom = draws @ n
            b = np.divide(draws@sums, denom, out=np.full(5000,np.nan), where=denom>0)
            row[measure]=float(np.mean(values))
            row[measure+'_pointwise95']=np.nanquantile(b,[.025,.975]).tolist()
            boots[y,measure]=b
        records.append(row)
    g = frame.groupby('y').delta.mean()
    gap = dict(condition=name, spoof_minus_bona_mean_delta=float(g[0]-g[1]),
               pointwise95=np.nanquantile(boots[0,'mean_delta']-boots[1,'mean_delta'],[.025,.975]).tolist())
    # Delta-only ranking is a diagnostic, not independent incremental information.
    gap['delta_only_spoof_AUC']=float(roc_auc_score(1-frame.y,frame.delta))
    gap['abs_delta_only_spoof_AUC']=float(roc_auc_score(1-frame.y,abs(frame.delta)))
    gap['delta_vs_native_spearman']=float(spearmanr(frame.delta,frame.s0).statistic)
    X=np.column_stack([np.ones(len(frame)),frame.s0.to_numpy()])
    pred=X@np.linalg.lstsq(X,frame.s1.to_numpy(),rcond=None)[0]
    gap['in_sample_label_blind_affine_rmse']=float(np.sqrt(np.mean((frame.s1-pred)**2)))
    gap['no_incremental_prediction_claim']=True
    return records, gap


def match_native(frame, strict):
    """Deterministic greedy no-replacement matching, not maximum matching.

    Source/codec/transmission exact; duration bins exact, absolute duration gap
    <=.25s; native score gap <=.02. Strict also shares speaker group. No norm
    matching: question is incremental response beyond the deployed scalar.
    """
    f=frame.copy()
    f['length_bin']=pd.cut(f.duration,[0,2,3,4.0375,6,np.inf],right=False).astype(str)
    cols=['source_corpus','codec','length_bin']
    if 'transmission' in f: cols += ['transmission']
    if strict: cols += ['speaker_group']
    f['_stratum']=f[cols].fillna('<missing>').astype(str).agg('|'.join,axis=1)
    pools={k:g.index.to_numpy() for k,g in f[f.y==0].groupby('_stratum')}
    used=set(); pairs=[]
    bona=sorted(f.index[f.y==1],key=lambda i:hashlib.sha256(str(f.loc[i,'utterance_id']).encode()).hexdigest())
    for i in bona:
        ix=pools.get(f.loc[i,'_stratum'],np.array([],dtype=int))
        if len(ix)==0: continue
        ds=abs(f.loc[ix,'s0'].to_numpy()-f.loc[i,'s0'])
        dt=abs(f.loc[ix,'duration'].to_numpy()-f.loc[i,'duration'])
        ok=(ds<=.02)&(dt<=.25)&np.array([j not in used for j in ix])
        if not ok.any(): continue
        elig=ix[ok]; order=np.lexsort((f.loc[elig,'utterance_id'].astype(str).to_numpy(),dt[ok],ds[ok]))
        j=elig[order[0]]; used.add(j)
        pairs.append(dict(bona_uid=f.loc[i,'utterance_id'],spoof_uid=f.loc[j,'utterance_id'],
                          bona_index=int(i),spoof_index=int(j),score_gap=float(abs(f.loc[i,'s0']-f.loc[j,'s0'])),
                          duration_gap=float(abs(f.loc[i,'duration']-f.loc[j,'duration'])),
                          source=f.loc[i,'source_corpus'],speaker=f.loc[i,'speaker_group']))
    return pd.DataFrame(pairs)


def matching_summary(frame,pairs):
    if len(pairs)==0: return dict(n_pairs=0, interpretation='no_overlap_under_fixed_matching_rule')
    b=frame.loc[pairs.bona_index,'delta'].to_numpy(); s=frame.loc[pairs.spoof_index,'delta'].to_numpy()
    return dict(n_pairs=len(pairs),bona_coverage=len(pairs)/int((frame.y==1).sum()),
                mean_score_gap=float(pairs.score_gap.mean()),max_score_gap=float(pairs.score_gap.max()),
                mean_duration_gap=float(pairs.duration_gap.mean()),mean_bona_delta=float(b.mean()),
                mean_bona_abs_delta=float(abs(b).mean()),mean_spoof_delta=float(s.mean()),
                mean_spoof_minus_bona_delta=float((s-b).mean()),
                median_paired_difference=float(np.median(s-b)),
                shared_bona_speaker_clusters=int(pairs[['source','speaker']].drop_duplicates().shape[0]),
                interpretation='exploratory_overlap_subset_only_not_conditional_independence_proof')


def main():
    start=time.monotonic(); OUT.mkdir(parents=True,exist_ok=False)
    save('audit_scope.json',dict(status='retrospective_exploratory', datasets=['DF_pilot','21LA_pilot','selected_DF471'],
        no_ssh=True,no_model_training=True,no_new_audio_forward=True,canceled_DF4x3_not_run=True,
        estimands=['classwise signed delta','absolute delta (stability)','native-score/metadata overlap'],
        CI='5000 source-stratified speaker bootstrap; pointwise95, no multiplicity correction; hypotheses selected after prior outcome inspection',
        known_limits=['unresolved original-family dependencies','matching is approximate and may have low support','single EBM checkpoint','no held-out response-based readout exists']))
    cat=read(ROOT/'r23_catalog.json'); summaries=[]; gaps=[]; matches=[]; strata=[]
    for ds in ['DF_pilot','21LA_pilot']:
        entry=cat['entries'][ds+'::R2-00']
        m=read(ROOT/entry['local_folder']/ds/'R2-00/scores.parquet').reset_index(drop=True)
        m=m.rename(columns={'cosine_s':'s0'})
        assert len(m)==20000 and not m.utterance_id.duplicated().any()
        assert m.y.value_counts().to_dict()=={0:18000,1:2000}
        assert m[['source_corpus','speaker_group','codec','duration']].notna().all().all()
        pairsets={}
        for strict in [True,False]:
            label='strict_same_speaker' if strict else 'relaxed_cross_speaker'
            p=match_native(m,strict);pairsets[label]=p;p.to_parquet(OUT/f'{ds}_{label}_pairs.parquet',index=False)
        for cond in CONDITIONS:
            p=read(ROOT/'local_analysis/frozen_all_v1/R2_R3'/ds/f'{cond}_paired.parquet')
            assert p.utterance_id.tolist()==m.utterance_id.tolist() and np.array_equal(p.y,m.y)
            f=m.copy();f['s1']=p.score.to_numpy();f['delta']=f.s1-f.s0
            assert np.max(abs(f.delta-p.score_delta))<1e-10 and np.isfinite(f[['s0','s1','delta']]).all().all()
            rows,gap=summarize(f,cond)
            summaries.extend(dict(dataset=ds,**r) for r in rows);gaps.append(dict(dataset=ds,**gap))
            for label,pairs in pairsets.items():matches.append(dict(dataset=ds,condition=cond,matching=label,**matching_summary(f,pairs)))
            f['length_bin']=pd.cut(f.duration,[0,2,3,4.0375,6,np.inf],right=False).astype(str)
            for keys,g in f.groupby(['source_corpus','y','length_bin'],observed=True):
                strata.append(dict(dataset=ds,condition=cond,source=keys[0],y=int(keys[1]),length_bin=keys[2],n=len(g),mean_delta=float(g.delta.mean()),mean_abs_delta=float(abs(g.delta).mean()),median_delta=float(g.delta.median())))
    raw=read(ROOT/'local_analysis/padding_causal_20261106_v1/paired_derived.parquet')
    base=raw[raw.condition=='native_wrap'].set_index('utterance_id')
    assert len(base)==471 and base.index.is_unique
    m=base.reset_index().rename(columns={'cosine_s':'s0'}); m['length_bin']=m.duration_bin
    padding_frames={}
    for cond in ['native_original','native_reflect']:
        a=raw[raw.condition==cond].set_index('utterance_id').loc[m.utterance_id]
        f=m.copy();f['s1']=a.cosine_s.to_numpy();f['delta']=f.s1-f.s0;padding_frames[cond]=f
    for sec in ['1.5','2']:
        for kind in ['original','wrap','reflect']:
            a=raw[raw.condition.isin([f'crop{sec}_head_{kind}',f'crop{sec}_tail_{kind}'])]
            assert a.groupby('utterance_id').size().eq(2).all()
            means=a.groupby('utterance_id').cosine_s.mean()
            f=m[m.utterance_id.isin(means.index)].copy().reset_index(drop=True)
            f['s1']=means.loc[f.utterance_id].to_numpy();f['delta']=f.s1-f.s0
            padding_frames[f'crop{sec}_headtail_mean_{kind}']=f
        # Same short segment, original-vs-wrap contrast isolates padding/context
        # better than original-vs-long, though total input length still changes.
        orig=padding_frames[f'crop{sec}_headtail_mean_original'].copy()
        wrap=padding_frames[f'crop{sec}_headtail_mean_wrap']
        assert orig.utterance_id.tolist()==wrap.utterance_id.tolist()
        orig['s0']=wrap.s1.to_numpy();orig['delta']=orig.s1-orig.s0
        padding_frames[f'crop{sec}_same_content_original_minus_wrap']=orig
    for cond,f in padding_frames.items():
        rows,gap=summarize(f,cond);summaries.extend(dict(dataset='selected_DF471',**r) for r in rows);gaps.append(dict(dataset='selected_DF471',**gap))
        for (source,y),g in f.groupby(['source_corpus','y']):strata.append(dict(dataset='selected_DF471',condition=cond,source=source,y=int(y),length_bin='selected_for_condition',n=len(g),mean_delta=float(g.delta.mean()),mean_abs_delta=float(abs(g.delta).mean()),median_delta=float(g.delta.median())))
        for strict in [True,False]:
            label='strict_same_speaker' if strict else 'relaxed_cross_speaker'
            pairs=match_native(f,strict)
            matches.append(dict(dataset='selected_DF471',condition=cond,matching=label,**matching_summary(f,pairs)))
    pd.DataFrame(summaries).to_csv(OUT/'class_response_summary.csv',index=False)
    pd.DataFrame(gaps).to_csv(OUT/'response_gaps.csv',index=False)
    pd.DataFrame(matches).to_csv(OUT/'native_score_matching.csv',index=False)
    pd.DataFrame(strata).to_csv(OUT/'source_length_response.csv',index=False)
    save('summary.json',dict(class_response=summaries,response_gaps=gaps,native_score_matching=matches))
    # Recheck immutable inputs after all reads/calculations.
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in INPUTS.items())
    save('verification.json',dict(status='pass',inputs_sha256=INPUTS,input_hashes_unchanged=True,
        unique_ID_and_class_counts_checked=True,score_delta_identity_max_error_bound=1e-10,
        outputs_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file()},
        elapsed_seconds=time.monotonic()-start,scope='data correspondence and arithmetic; no response-readout transfer or scientific validation claimed'))
    print(str(OUT));print(pd.DataFrame(summaries)[['dataset','condition','y','n','mean_delta','mean_abs_delta','median_delta']].to_string(index=False))
    print(pd.DataFrame(matches)[['dataset','condition','matching','n_pairs','mean_spoof_minus_bona_delta']].to_string(index=False))


if __name__=='__main__':
    with threadpool_limits(limits=2): main()
