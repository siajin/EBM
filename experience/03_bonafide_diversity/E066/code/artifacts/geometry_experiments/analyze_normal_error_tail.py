"""Separate actual EER-tail errors from reference-threshold operating errors."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from scipy.stats import rankdata,spearmanr
from sklearn.metrics import roc_curve
from threadpoolctl import threadpool_limits
from geometry_cache import resolve_run
from fast_analyze import eer
from frozen_common import weighted_eers
import analyze_normal_cause_audit as acoustics

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/geometry_cache/local_analysis/normal_cause_audit_v1'
BINS=[0,1,1.5,2,3,4.0375,6,np.inf]


def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,
        default=lambda a:a.item() if isinstance(a,np.generic) else a.tolist(),allow_nan=False)+'\n',encoding='utf8')


def load(n):
    p,e,s=resolve_run(n);f=pd.read_parquet(p/'extract/scores.parquet');return f[f.split==s].reset_index(drop=True)


def vertex(f):
    fa,tp,th=roc_curve(f.y,f.cosine_s,drop_intermediate=False);ix=np.argmin(abs(fa-(1-tp)))
    return float(th[ix]),100*float(fa[ix]),100*float(1-tp[ix])


def weighted_eer(y,s,w):
    fa,tp,_=roc_curve(y,s,sample_weight=w,drop_intermediate=False);d=fa-(1-tp)
    hi=np.flatnonzero(d>=0)[0];lo=max(0,hi-1);t=-d[lo]/(d[hi]-d[lo]) if d[hi]!=d[lo] else 0
    return 100*(fa[lo]+t*(fa[hi]-fa[lo]))


def main():
    assert not (OUT/'EER_tail_complete.json').exists()
    df=pd.concat([load('DF_pilot'),load('DF_remaining')],ignore_index=True)
    cut,fa,fr=vertex(df);b=df[df.y==1].copy();b['wrong_EER']=b.cosine_s<cut;b['duration_bin']=pd.cut(b.duration,BINS,right=False).astype(str)
    errors=b[b.wrong_EER];assert len(errors)==482 and errors.source_corpus.value_counts().to_dict()=={'vcc2018':451,'vcc2020':31}
    cells=b.groupby(['source_corpus','duration_bin'],observed=True).wrong_EER.agg(['sum','count','mean']).reset_index()
    cells['false_reject_percent']=100*cells['mean'];cells.to_csv(OUT/'EER_tail_source_duration.csv',index=False)
    train=load('T');coverage=[]
    for name in ['T','V','Q','19LA_eval']:
        f=train if name=='T' else load(name)
        for y,g in f.groupby('y'):
            coverage.append(dict(dataset=name,y=int(y),n=len(g),under1p5=int(sum(g.duration<1.5)),
                under1p5_percent=100*float((g.duration<1.5).mean()),under2=int(sum(g.duration<2)),
                under2_percent=100*float((g.duration<2).mean()),minimum_duration=float(g.duration.min())))
    pd.DataFrame(coverage).to_csv(OUT/'training_duration_coverage.csv',index=False)
    # Descriptive reweighting of durations while retaining DF source weights.
    tr=train[train.y==1];td=pd.cut(tr.duration,BINS,right=False).astype(str).value_counts(normalize=True)
    weights=np.ones(len(df));notes=[]
    for source,g in b.groupby('source_corpus'):
        counts=g.duration_bin.value_counts();target=td.reindex(counts.index).fillna(0)
        missing=set(td[td>0].index)-set(counts.index);target/=target.sum()
        for group,ix in g.groupby('duration_bin').groups.items():weights[np.asarray(ix)]=len(g)*target[group]/len(ix)
        notes.append(dict(source=source,target_bins_missing=list(missing)))
    reweighted=weighted_eer(df.y.to_numpy(),df.cosine_s.to_numpy(),weights)
    filtering=[]
    for name,keep in [('native_all',np.ones(len(df),bool)),('normals_atleast2s',((df.y==0)|(df.duration>=2)).to_numpy()),
                      ('normals_without_repeat_padding',((df.y==0)|(df.padding_fraction==0)).to_numpy())]:
        q=df[keep];filtering.append(dict(subset=name,n_bona=int(sum(q.y==1)),n_spoof=int(sum(q.y==0)),eer_percent=eer(q.y,q.cosine_s)))
    # Within-source, speaker-adjusted duration/rank-score associations.
    associations=[]
    for source,g in b.groupby('source_corpus'):
        x=rankdata(g.duration);y=rankdata(g.cosine_s);groups=g.speaker_group.to_numpy()
        for speaker in np.unique(groups):
            ii=groups==speaker;x[ii]-=x[ii].mean();y[ii]-=y[ii].mean()
        associations.append(dict(source=source,within_speaker_rank_correlation=float(np.corrcoef(x,y)[0,1]),
            raw_spearman=float(spearmanr(g.duration,g.cosine_s).statistic)))
    save('EER_tail_duration_associations.json',associations)
    # Actual EER-tail geometry, distinct from the earlier19LA fixed threshold.
    q=pd.read_parquet(OUT/'neighborhood_scores.parquet');pcut,pfa,pfr=vertex(q)
    q=q[(q.y==1)&(q.cosine_s<pcut)]
    geom=[]
    for source,g in [('all',q)]+list(q.groupby('source_corpus')):
        geom.append(dict(source=source,n=len(g),T_nearest_normal_percent=100*float((g.T_nearest1>0).mean()),
            DF_nearest_normal_percent=100*float((g.DF_OOF_nearest1>0).mean()),
            DF_mean5_normal_percent=100*float((g.DF_OOF_mean5>0).mean())))
    save('actual_EER_tail_neighborhoods.json',dict(threshold=pcut,normal_false_reject_percent=pfr,spoof_false_accept_percent=pfa,groups=geom))
    # Paired input-treatment EERs, instead of only fixed-threshold errors.
    pair=pd.read_parquet(OUT/'paired_input_scores.parquet');native=pair[pair.condition==pair.condition.iloc[0]].copy()
    groups,codes=np.unique(native.source_corpus+'::'+native.speaker_group,return_inverse=True)
    draws=np.random.default_rng(5508).multinomial(len(groups),np.full(len(groups),1/len(groups)),size=2000)
    y=native.y.to_numpy();baseline=weighted_eers(y,native.cosine_s_native.to_numpy(),codes,draws,chunk=32)
    treat=[dict(condition='native',eer_percent=eer(y,native.cosine_s_native),change_pp=0.,CI95=[0.,0.])]
    for condition,g in pair.groupby('condition',sort=False):
        assert g.utterance_id.tolist()==native.utterance_id.tolist()
        val=weighted_eers(y,g.cosine_s_changed.to_numpy(),codes,draws,chunk=32)
        treat.append(dict(condition=condition,eer_percent=eer(y,g.cosine_s_changed),
            change_pp=eer(y,g.cosine_s_changed)-eer(y,native.cosine_s_native),CI95=np.quantile(val-baseline,[.025,.975]).tolist()))
    save('paired_treatment_EER.json',treat)
    # Boundary discontinuity was extracted separately before its outcomes.
    pad=pd.read_csv(OUT/'padding_features.csv');f=pd.read_parquet(OUT/'normal_nocodec_acoustics.parquet').merge(pad,on=['case_id','utterance_id'],validate='one_to_one')
    assert len(f)==1156
    columns=['join_jump_to_rms','join_jump_to_diff99','join_HF_fraction','join_HF_excess_over_internal_median_db','internal_HF_max_fraction']
    for c in columns:f['actual_'+c]=f[c]*(f.wrap_joins>0)
    block=acoustics.blocks(f);block['wrap_seam']=['actual_'+c for c in columns]
    # Same predictive sensitivity diagnostic, now with actual full-DF EER labels.
    acoustics.OUT=OUT/'EER_tail_models';acoustics.OUT.mkdir(exist_ok=False);acoustics.TH=cut
    metric,effects=acoustics.effects_models(f.copy(),block)
    save('EER_tail_complete.json',dict(status='complete_exploratory',full_DF_EER=eer(df.y,df.cosine_s),threshold=cut,
        false_rejected_normals=len(errors),padded_false_rejected_normals=int(sum(errors.padding_fraction>0)),
        under2s_false_rejected_normals=int(sum(errors.duration<2)),under3s_false_rejected_normals=int(sum(errors.duration<3)),
        error_duration_quantiles=errors.duration.quantile([0,.25,.5,.75,1]).to_dict(),
        normal_without_padding_n=int(sum(b.padding_fraction==0)),normal_without_padding_errors=int(sum((b.padding_fraction==0)&b.wrong_EER)),
        filtering_diagnostics=filtering,T_duration_reweighted_EER=reweighted,reweight_support_notes=notes,
        within_speaker_duration=associations,actual_EER_neighborhoods=geom,
        paired_treatment_EERs=treat,acoustic_tail_model_effects=effects,
        limitations=['Removing short normals changes the test population; it is not padding removal or a model improvement.',
            'Native duration and wrap repetition are linked; their separate causal effects require a same-waveform input intervention.',
            'T has short-view training as well as original recordings. Original-duration counts do not measure all effective augmented training exposure.',
            'Only36 of1156 nocodec normal waveforms are rejected at the full-DF EER threshold; tail-model feature tests have limited power.',
            'The EER threshold is retrospective and must not be presented as a deployment threshold.']))
    print(json.dumps(dict(normal_errors=len(errors),padded_errors=int(sum(errors.padding_fraction>0)),
        filtering=filtering,reweighted=reweighted,duration=associations,treatment=treat),ensure_ascii=False),flush=True)


if __name__=='__main__':
    with threadpool_limits(2):main()
