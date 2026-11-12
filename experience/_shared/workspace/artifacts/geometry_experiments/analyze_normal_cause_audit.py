"""Acoustic associations, grouped diagnostic models, and paired input effects.

All raw descriptors were computed on the server. This file only uses derived
features and the completed frozen EBM caches. No EBM retraining or inference.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
from pathlib import Path
import json
import hashlib
import time
import warnings
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.covariance import LedoitWolf
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler,SplineTransformer,FunctionTransformer
from threadpoolctl import threadpool_limits
from geometry_cache import resolve_run,load_embeddings
from fast_analyze import eer
from frozen_r23_cache import load as load_r23

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/geometry_cache/local_analysis/normal_cause_audit_v1'
TH=.8619597055730815
SEEDS=[11,22,33]


def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,
        default=lambda a:a.item() if isinstance(a,np.generic) else a.tolist(),allow_nan=False)+'\n',encoding='utf8')


def bal_ll(y,p):
    p=np.clip(p,1e-7,1-1e-7)
    return float(-.5*(np.log(p[y==1]).mean()+np.log1p(-p[y==0]).mean()))


def blocks(frame):
    return {
        'level':['rms_active_db','crest_active_db','clipping_fraction','dc_to_rms'],
        'spectrum':['zcr_active','spectral_centroid_hz','spectral_rolloff85_hz','spectral_rolloff95_hz','spectral_flatness_db','spectral_slope_db_per_octave']+[c for c in frame if c.startswith('band_')],
        'cepstrum':[c for c in frame if c.startswith('mfcc_')],
        'pitch_periodicity':['pitch_estimate_hz','pitch_range_semitones','periodicity_score','pitch_available'],
        'time_variation':['rms_frame_range_db','spectral_flux_active'],
    }


def basic_pipeline(columns):
    duration=make_pipeline(FunctionTransformer(np.log1p),SplineTransformer(n_knots=4,degree=3,include_bias=False),StandardScaler())
    transforms=[('source',OneHotEncoder(handle_unknown='ignore'),['source_corpus']),('duration',duration,['duration'])]
    if columns:transforms.append(('audio',make_pipeline(SimpleImputer(strategy='median'),StandardScaler()),columns))
    return make_pipeline(ColumnTransformer(transforms),LogisticRegression(C=1.,max_iter=3000,tol=1e-7))


def boot_weights(frame,draws=2000):
    keys=frame.source_corpus+'::'+frame.speaker_group
    groups,codes=np.unique(keys,return_inverse=True);rng=np.random.default_rng(2026110551)
    w=np.zeros((draws,len(groups)),int)
    for source in sorted(frame.source_corpus.unique()):
        ix=np.flatnonzero(np.char.startswith(groups.astype(str),source+'::'))
        w[:,ix]=rng.multinomial(len(ix),np.full(len(ix),1/len(ix)),size=draws)
    return w,codes,groups


def effects_models(f,block):
    y=(f.cosine_s.to_numpy()<TH).astype(int);groups=(f.source_corpus+'::'+f.speaker_group).to_numpy();strata=f.source_corpus.to_numpy()
    allcols=sum(block.values(),[])
    cases={'metadata_only':[],**block,'all_acoustics':allcols}
    for j in range(len(allcols)):
        f['noise_%02d'%j]=[np.random.default_rng(int.from_bytes(hashlib.sha256(('cause-noise|'+uid+'|'+str(j)).encode()).digest()[:8],'big')).normal() for uid in f.utterance_id]
    cases['noise_control']=['noise_%02d'%j for j in range(len(allcols))]
    save('diagnostic_model_lock.json',dict(outcome='native normal false rejection at frozen19LA threshold',
        predictors=cases,seeds=SEEDS,source_and_duration_in_every_case=True,outer_group='source x speaker',
        C=1,hyperparameter_selection=False,no_EBM_training=True,
        interpretation='DF internal prediction of errors; not causal feature intervention or external validation'))
    arrays={};metric=[]
    for case,cols in cases.items():
        predictions=[]
        for seed in SEEDS:
            p=np.full(len(f),np.nan)
            folds=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed)
            for train,test in folds.split(f,strata,groups):
                assert not set(groups[train])&set(groups[test])
                m=basic_pipeline(cols);m.fit(f.iloc[train],y[train]);p[test]=m.predict_proba(f.iloc[test])[:,1]
            assert np.isfinite(p).all();predictions.append(p)
            metric.append(dict(case=case,seed=seed,auroc=roc_auc_score(y,p),balanced_logloss=bal_ll(y,p),logloss=log_loss(y,p)))
        arrays[case]=np.mean(predictions,axis=0)
        print('diagnostic OOF complete',case,flush=True)
    pd.DataFrame(metric).to_csv(OUT/'acoustic_error_model_metrics.csv',index=False)
    frame=f[['utterance_id','source_corpus','speaker_group','cosine_s']].copy();frame['normal_error']=y
    for name,p in arrays.items():frame[name]=p
    frame.to_parquet(OUT/'acoustic_error_OOF.parquet',index=False)
    w,codes,unique=boot_weights(f);comparisons=[]
    def weighted_ll(p):
        losses=-(y*np.log(np.clip(p,1e-7,1-1e-7))+(1-y)*np.log(np.clip(1-p,1e-7,1-1e-7)));vals=np.zeros(len(w))
        for label in [0,1]:
            ok=y==label;num=np.bincount(codes[ok],weights=losses[ok],minlength=len(unique));den=np.bincount(codes[ok],minlength=len(unique));vals+=.5*(w@num)/(w@den)
        return vals
    reference=weighted_ll(arrays['metadata_only'])
    family=len(arrays)-1;alpha=.05/family
    for name,p in arrays.items():
        if name=='metadata_only':continue
        delta=reference-weighted_ll(p)
        comparisons.append(dict(case=name,OOF_AUROC=roc_auc_score(y,p),
            balanced_LL_reduction=bal_ll(y,arrays['metadata_only'])-bal_ll(y,p),
            CI95=np.quantile(delta,[.025,.975]).tolist(),
            family95_interval=np.quantile(delta,[alpha/2,1-alpha/2]).tolist(),
            comparisons_in_family=family,bootstrap_draws=len(w),speaker_groups=len(unique),
            limitation='Conditional on OOF predictions; no refit uncertainty or source-family dependence included.'))
    save('acoustic_error_model_effects.json',comparisons)
    return metric,comparisons


def acoustic_associations(f,block):
    columns=sum(block.values(),[]);records=[]
    for source,g in f.groupby('source_corpus'):
        for feature in columns:
            ok=g[feature].notna();rho=spearmanr(g.loc[ok,feature],g.loc[ok,'cosine_s']).statistic if g.loc[ok,feature].nunique()>1 else 0.
            records.append(dict(source=source,feature=feature,rho=float(rho),n=int(ok.sum())))
    pd.DataFrame(records).to_csv(OUT/'within_source_spearman.csv',index=False)
    # Partial rank association controls speaker fixed effects and duration.
    source_speaker=f.source_corpus+'::'+f.speaker_group
    dummy=pd.get_dummies(source_speaker,dtype=float).to_numpy()
    duration=SplineTransformer(n_knots=4,degree=3,include_bias=False).fit_transform(np.log1p(f[['duration']]))
    controls=np.c_[dummy,duration]
    ranked=np.c_[rankdata(f.cosine_s),*[rankdata(f[c].fillna(f[c].median())) for c in columns]]
    residual=ranked-controls@np.linalg.lstsq(controls,ranked,rcond=None)[0]
    rho=np.array([np.corrcoef(residual[:,0],residual[:,i+1])[0,1] if np.std(residual[:,i+1])>1e-10 else 0. for i in range(len(columns))])
    pd.DataFrame(dict(feature=columns,partial_rank_rho=rho)).sort_values('partial_rank_rho').to_csv(OUT/'speaker_duration_partial_rank.csv',index=False)


def distance_to_T(data,f,block):
    train=data[(data.audio_dataset=='T')&(data.y==1)].copy();rows=[]
    for name,cols in block.items():
        imp=SimpleImputer(strategy='median');xt=imp.fit_transform(train[cols]);sc=StandardScaler();xt=sc.fit_transform(xt)
        # Fit only normal T; LOSO T values provide a descriptive reference.
        reference=np.empty(len(train));sp=train.speaker_group.to_numpy()
        for speaker in np.unique(sp):
            test=sp==speaker;ref=LedoitWolf().fit(xt[~test]);reference[test]=ref.mahalanobis(xt[test])
        model=LedoitWolf().fit(xt);distance=model.mahalanobis(sc.transform(imp.transform(f[cols])))
        q95=np.quantile(reference,.95);median=np.median(reference)
        for source,g in f.assign(distance=distance).groupby('source_corpus'):
            association=spearmanr(g.distance,g.cosine_s).statistic
            rows.append(dict(block=name,source=source,n=len(g),median_squared_distance=float(g.distance.median()),
                T_LOSO_median=float(median),T_LOSO_q95=float(q95),fraction_above_T95=float((g.distance>q95).mean()),
                distance_score_spearman=float(association)))
    pd.DataFrame(rows).to_csv(OUT/'distance_to_training_normal.csv',index=False)
    return rows


def paired_input_effects():
    baseline=load_r23('DF_pilot','R2-00').rows
    rows=[];pairs=[]
    for condition in ['R3-noise20','R3-noise10','R3-Opus32k_recompression']:
        changed=load_r23('DF_pilot',condition).rows
        joined=baseline[['utterance_id','y','source_corpus','speaker_group','codec','cosine_s']].merge(changed[['utterance_id','y','cosine_s']],on=['utterance_id','y'],validate='one_to_one',suffixes=('_native','_changed'))
        assert len(joined)==20000
        for (source,y),g in joined.groupby(['source_corpus','y']):
            a=g.cosine_s_native.to_numpy();b=g.cosine_s_changed.to_numpy()
            bad_a=a<TH if y==1 else a>=TH;bad_b=b<TH if y==1 else b>=TH
            d=bad_b.astype(float)-bad_a.astype(float)
            group,codes=np.unique(g.speaker_group,return_inverse=True);rng=np.random.default_rng(882)
            w=rng.multinomial(len(group),np.full(len(group),1/len(group)),size=2000)
            num=np.bincount(codes,weights=d);den=np.bincount(codes);effect=100*(w@num)/(w@den)
            rows.append(dict(condition=condition,source=source,y=int(y),n=len(g),
                native_error_percent=100*bad_a.mean(),changed_error_percent=100*bad_b.mean(),
                change_pp=100*d.mean(),change_CI95=np.quantile(effect,[.025,.975]).tolist(),
                corrected=int(sum(bad_a&~bad_b)),new_errors=int(sum(~bad_a&bad_b)),median_score_change=float(np.median(b-a))))
        joined['condition']=condition;pairs.append(joined)
    pd.DataFrame(rows).to_csv(OUT/'paired_input_effects.csv',index=False)
    pd.concat(pairs).to_parquet(OUT/'paired_input_scores.parquet',index=False)
    return rows


def main():
    start=time.monotonic();assert not (OUT/'analysis_complete.json').exists()
    end=json.loads((OUT/'complete.json').read_text());assert end['status']=='pass'
    assert hashlib.sha256((OUT/'features.csv').read_bytes()).hexdigest()==end['features_sha256']
    a=pd.read_parquet(OUT/'analysis_rows.parquet');features=pd.read_csv(OUT/'features.csv')
    d=a.merge(features,on=['case_id','utterance_id'],validate='one_to_one');assert len(d)==3419
    assert set(d.sample_rate)=={16000} and set(d.channels)=={1} and set(d.subtype)=={'PCM_16'}
    f=d[(d.audio_dataset=='DF')&(d.y==1)&(d.codec=='nocodec')].copy().reset_index(drop=True)
    assert len(f)==1156 and f.source_corpus.value_counts().to_dict()=={'vcc2018':420,'asvspoof':386,'vcc2020':350}
    block=blocks(f);cols=sum(block.values(),[]);f['normal_error']=f.cosine_s<TH
    summary=[]
    for source,g in f.groupby('source_corpus'):
        for feature in cols:
            summary.append(dict(source=source,feature=feature,n=int(g[feature].notna().sum()),
                median=float(g[feature].median()),q25=float(g[feature].quantile(.25)),q75=float(g[feature].quantile(.75))))
    pd.DataFrame(summary).to_csv(OUT/'normal_acoustic_distributions.csv',index=False)
    f.to_parquet(OUT/'normal_nocodec_acoustics.parquet',index=False)
    acoustic_associations(f,block)
    distances=distance_to_T(d,f,block)
    metric,comparisons=effects_models(f.copy(),block)
    paired=paired_input_effects()
    save('analysis_complete.json',dict(status='complete_exploratory',elapsed_seconds=time.monotonic()-start,
        normal_nocodec=1156,derived_feature_rows=3419,blocks=block,OOF_metrics=metric,OOF_effects=comparisons,
        training_normal_distances=distances,paired_input_effects=paired,
        original_audio_exported=False,new_GPU_work=False,
        limitations=['Acoustic associations and error prediction do not prove causal physical origin.',
            'Pitch/periodicity are approximate signal descriptors, not verified F0/voice quality labels.',
            'Only existing paired noise and Opus transformations have an input-level intervention interpretation.',
            'Raw microphone, phonetic content, speaker traits and corpus origins remain confounded.',
            'No silence-distribution hypotheses or silence-ratio predictors were used.']))
    print(json.dumps(dict(comparisons=comparisons,paired=paired),ensure_ascii=False),flush=True)


if __name__=='__main__':
    with threadpool_limits(2):main()
