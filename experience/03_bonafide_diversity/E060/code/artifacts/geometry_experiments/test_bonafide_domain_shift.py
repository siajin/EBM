"""Post-hoc diagnosis of DF degradation, using completed local caches only.

Class-conditional score replacement is a descriptive decomposition, not a
training intervention. No EBM fitting, test-set model selection, or inference.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.stats import ks_2samp
from geometry_cache import resolve_run, validate_run
from fast_analyze import eer
from audit_cached_core_results import independent_eer, independent_auc

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'artifacts/geometry_cache/local_analysis/bonafide_domain_shift_v1'
CHECKPOINT = '104873703a57cad98b686d770e0860e031a2cf9abef79b72668780c1660e0b24'
SEED, DRAWS = 2026110522, 2000


def save(name, value):
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2,
        default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist(),
        allow_nan=False)+'\n', encoding='utf8')


def load(name):
    folder, entry, split = resolve_run(name)
    assert entry['identity']['checkpoint_sha256'] == CHECKPOINT
    rows = pd.read_parquet(folder/'extract/scores.parquet')
    rows = rows.loc[rows.split == split].reset_index(drop=True)
    assert not rows.utterance_id.duplicated().any()
    assert np.isfinite(rows.cosine_s).all()
    rows['corpus'] = rows['source_corpus'] if 'source_corpus' in rows else 'asvspoof'
    rows['cluster'] = [f'{c}::{str(s).split("::")[-1]}' for c,s in zip(rows.corpus, rows.speaker_group)]
    return rows


def mixed(bona, spoof):
    b=bona.loc[bona.y==1,'cosine_s'].to_numpy()
    s=spoof.loc[spoof.y==0,'cosine_s'].to_numpy()
    y=np.r_[np.ones(len(b),int),np.zeros(len(s),int)]
    z=np.r_[b,s]
    result=eer(y,z)
    assert abs(independent_eer(y,z)-result)<1e-8
    return result


class GroupedClass:
    def __init__(self, frame, y, groups):
        f=frame.loc[frame.y==y]
        self.scores=np.sort(f.cosine_s.to_numpy())
        self.arrays={i:np.sort(g.cosine_s.to_numpy()) for key,g in f.groupby('cluster')
                     for i in [groups.index(key)]}
        self.n=np.zeros(len(groups),int)
        for i,a in self.arrays.items():self.n[i]=len(a)

    def below(self, thresholds, weights):
        out=np.zeros(len(thresholds))
        for i,a in self.arrays.items():
            out+=weights[:,i]*np.searchsorted(a, thresholds, side='left')
        return out/(weights@self.n)


def weighted_mixed(bona, spoof, weights):
    # Each threshold is an exact ROC vertex; interpolate the crossing segment.
    knots=np.r_[-np.inf,np.unique(np.r_[bona.scores,spoof.scores]),np.inf]
    n=len(weights);lo=np.zeros(n,int);hi=np.full(n,len(knots)-1,int)
    for _ in range(int(np.ceil(np.log2(len(knots))))+1):
        mid=(lo+hi)//2;t=knots[mid]
        diff=1-spoof.below(t,weights)-bona.below(t,weights)
        move=diff>=0
        lo=np.where(move,mid,lo);hi=np.where(move,hi,mid)
    assert np.all(hi-lo<=1)
    falo=1-spoof.below(knots[lo],weights);frlo=bona.below(knots[lo],weights)
    fahi=1-spoof.below(knots[hi],weights);frhi=bona.below(knots[hi],weights)
    dl,dh=falo-frlo,fahi-frhi
    weight=np.divide(dl,dl-dh,out=np.zeros_like(dl),where=(dl-dh)!=0)
    return 100*(falo+weight*(fahi-falo))


def near_eer_threshold(frame):
    fa,tp,thresholds=roc_curve(frame.y,frame.cosine_s,drop_intermediate=False)
    i=int(np.argmin(np.abs(fa-(1-tp))))
    return {'threshold':float(thresholds[i]),'bona_false_reject_percent':100*float(1-tp[i]),
            'spoof_false_accept_percent':100*float(fa[i]),
            'description':'nearest actual ROC vertex, not interpolated EER threshold'}


def errors(frame, th):
    b=frame.loc[frame.y==1,'cosine_s'];s=frame.loc[frame.y==0,'cosine_s']
    nb,ns=len(b),len(s);eb,es=int(sum(b<th)),int(sum(s>=th))
    return dict(n_bona=nb,n_spoof=ns,bona_false_rejects=eb,spoof_false_accepts=es,
        bona_false_reject_percent=100*eb/nb if nb else None,
        spoof_false_accept_percent=100*es/ns if ns else None)


def ci(a):return np.quantile(a,[.025,.975]).tolist()


def main():
    start=time.monotonic();OUT.mkdir(parents=True,exist_ok=False)
    save('protocol.json',dict(status='exploratory_posthoc',checkpoint=CHECKPOINT,
        question='Does DF EER degradation come primarily from bona fide score shift?',
        primary_baseline='19LA_eval',secondary_baseline='21LA_eval',
        target='DF_full=DF_pilot+DF_remaining',score='native 160-dimensional cosine, higher=bona fide',
        bootstrap=dict(draws=DRAWS,seed=SEED,cluster='source corpus x speaker',
            shared_ASV_speaker_weights_across_datasets=True,stratify_by_source=True,
            unresolved_source_families=True),
        frozen_thresholds=['V bona 1st percentile','19LA nearest EER vertex (descriptive)'],
        no_new_fit=True,not_preregistered=True,not_a_training_causal_intervention=True,
        timestamp=datetime.now(ZoneInfo('Asia/Seoul')).isoformat()))
    names=['T','V','Q','19LA_eval','21LA_eval','DF_pilot','DF_remaining']
    audits=[]
    for name in names:
        audits.append(validate_run(name,verify_hashes=True))
        print('cache audit',name,'pass',flush=True)
    data={name:load(name) for name in names}
    data['DF_full']=pd.concat([data['DF_pilot'],data['DF_remaining']],ignore_index=True)
    df=data['DF_full'];assert len(df)==533928 and df.y.value_counts().to_dict()=={0:519059,1:14869}
    assert not df.utterance_id.duplicated().any()
    assert set(df.source_corpus)=={'asvspoof','vcc2018','vcc2020'}
    v=data['V'];b=np.sort(v.loc[v.y==1,'cosine_s'])
    thresholds={'V_bona_1pct':float(b[int(np.floor(.01*len(b)))]),
                '19LA_near_EER':near_eer_threshold(data['19LA_eval'])['threshold'],
                '21LA_near_EER':near_eer_threshold(data['21LA_eval'])['threshold'],
                'DF_near_EER':near_eer_threshold(df)['threshold']}
    metrics=[];operating=[];quantiles=[];source=[]
    for name,f in data.items():
        val=eer(f.y,f.cosine_s);auc=roc_auc_score(f.y,f.cosine_s)
        assert abs(independent_eer(f.y.to_numpy(),f.cosine_s.to_numpy())-val)<1e-8
        assert abs(independent_auc(f.y.to_numpy(),f.cosine_s.to_numpy())-auc)<1e-10
        metrics.append(dict(dataset=name,eer_percent=val,auroc=auc,n=len(f),
                            n_bona=int(sum(f.y==1)),n_spoof=int(sum(f.y==0))))
        for label,th in thresholds.items():operating.append(dict(dataset=name,threshold_name=label,threshold=th,**errors(f,th)))
        for y,g in f.groupby('y'):
            quantiles.append(dict(dataset=name,y=int(y),mean=float(g.cosine_s.mean()),
                **{f'q{q:g}':float(np.quantile(g.cosine_s,q)) for q in [0,.01,.05,.5,.95,.99,1]}))
    for (corpus,codec),f in df.groupby(['corpus','codec']):
        for label,th in thresholds.items():source.append(dict(source_corpus=corpus,codec=codec,threshold_name=label,**errors(f,th)))
    source_aggregate=[]
    for corpus,f in df.groupby('corpus'):
        for label,th in thresholds.items():source_aggregate.append(dict(source_corpus=corpus,threshold_name=label,
            source_eer_percent=eer(f.y,f.cosine_s),speakers=f.cluster.nunique(),**errors(f,th)))
    for name,values in [('dataset_metrics',metrics),('fixed_threshold_errors',operating),
                         ('score_quantiles',quantiles),('source_codec_errors',source),('source_errors',source_aggregate)]:
        pd.DataFrame(values).to_csv(OUT/(name+'.csv'),index=False)
    swaps=[];decomposition=[];boots={};bootstrap_summary={}
    rng=np.random.default_rng(SEED)
    # Same draws across 19LA,21LA,DF; corpus strata have fixed cluster counts.
    groups=sorted(set().union(*(set(data[n].cluster) for n in ['19LA_eval','21LA_eval','DF_full'])))
    weights=np.zeros((DRAWS,len(groups)),int)
    for corpus in ['asvspoof','vcc2018','vcc2020']:
        ix=[i for i,g in enumerate(groups) if g.startswith(corpus+'::')]
        weights[:,ix]=rng.multinomial(len(ix),np.full(len(ix),1/len(ix)),size=DRAWS)
    classes={(name,y):GroupedClass(data[name],y,groups) for name in ['19LA_eval','21LA_eval','DF_full'] for y in [0,1]}
    ones=np.ones((1,len(groups)),int)
    for base in ['19LA_eval','21LA_eval','V']:
        matrix={}
        for bn,sn in [(base,base),('DF_full',base),(base,'DF_full'),('DF_full','DF_full')]:
            point=mixed(data[bn],data[sn]);key=f'{bn}_bona__{sn}_spoof'
            row=dict(bona_dataset=bn,spoof_dataset=sn,baseline=base,eer_percent=point)
            if base!='V':
                bc,sc=classes[(bn,1)],classes[(sn,0)]
                assert abs(weighted_mixed(bc,sc,ones)[0]-point)<1e-8
                if key not in boots:
                    boots[key]=weighted_mixed(bc,sc,weights)
                    print('bootstrap',key,'complete',flush=True)
                row['ci95_lower'],row['ci95_upper']=ci(boots[key])
            swaps.append(row);matrix[(bn,sn)]=point
        bb=matrix[(base,base)];db=matrix[('DF_full',base)]
        bd=matrix[(base,'DF_full')];dd=matrix[('DF_full','DF_full')]
        bona=.5*((db-bb)+(dd-bd));spoof=.5*((bd-bb)+(dd-db))
        assert abs(bona+spoof-(dd-bb))<1e-12
        item=dict(baseline=base,total_change_pp=dd-bb,bona_contribution_pp=bona,
            spoof_contribution_pp=spoof,bona_share_of_net_percent=100*bona/(dd-bb),
            normal_replacement_reduction_percent=100*(dd-bd)/dd,
            interpretation='symmetric Shapley score-distribution accounting; not causal training attribution')
        if base!='V':
            bbv=boots[f'{base}_bona__{base}_spoof'];dbv=boots[f'DF_full_bona__{base}_spoof']
            bdv=boots[f'{base}_bona__DF_full_spoof'];ddv=boots['DF_full_bona__DF_full_spoof']
            estimates={'total_change_pp':ddv-bbv,'bona_contribution_pp':.5*((dbv-bbv)+(ddv-bdv)),
                       'spoof_contribution_pp':.5*((bdv-bbv)+(ddv-dbv)),
                       'normal_replacement_change_pp':bdv-ddv,
                       'spoof_replacement_only_change_pp':bdv-bbv}
            item['bootstrap_ci95']={k:ci(a) for k,a in estimates.items()}
            bootstrap_summary[base]=item['bootstrap_ci95']
        decomposition.append(item)
    pd.DataFrame(swaps).to_csv(OUT/'class_replacement_eer.csv',index=False)
    save('decomposition.json',decomposition)
    np.savez_compressed(OUT/'cluster_bootstrap_draws.npz',**boots,weights=weights,groups=np.asarray(groups))
    # Source- and attack-restricted checks, including identical 13 attack IDs.
    a=data['19LA_eval'];asv=df[df.corpus=='asvspoof'];shared=set(a.attack_id[a.y==0])&set(asv.attack_id[asv.y==0])
    assert len(shared)==13
    attack_rows=[]
    for attack in sorted(shared):
        for sn,f in [('19LA_eval',a),('DF_asvspoof',asv)]:
            g=f[(f.y==1)|(f.attack_id==attack)]
            attack_rows.append(dict(dataset=sn,attack=attack,eer_percent=eer(g.y,g.cosine_s),**errors(g,thresholds['19LA_near_EER'])))
    pd.DataFrame(attack_rows).to_csv(OUT/'shared_attack_metrics.csv',index=False)
    restricted=[]
    for bn,bf in [('19LA_eval',a),('DF_asvspoof',asv)]:
        for sn,sf in [('19LA_eval',a),('DF_asvspoof',asv)]:restricted.append(dict(bona_dataset=bn,spoof_dataset=sn,eer_percent=mixed(bf,sf)))
    # Reweight each source of DF spoof to the same 19LA attack prior.
    from sklearn.metrics import roc_curve as roc
    for sn,sf in [('19LA_eval',a),('DF_asvspoof',asv)]:
        s=sf[sf.y==0];b=a[a.y==1];mass=a[a.y==0].attack_id.value_counts(normalize=True)
        w=s.attack_id.map(mass)/s.attack_id.map(s.attack_id.value_counts())
        yy=np.r_[np.ones(len(b)),np.zeros(len(s))];ss=np.r_[b.cosine_s,s.cosine_s]
        ww=np.r_[np.full(len(b),1/len(b)),w.to_numpy()]
        fa,tp,_=roc(yy,ss,sample_weight=ww,drop_intermediate=False);fr=1-tp;di=fa-fr
        k=np.flatnonzero(di>=0)[0];j=max(0,k-1);lam=-di[j]/(di[k]-di[j]) if di[k]!=di[j] else 0
        restricted.append(dict(bona_dataset='19LA_eval',spoof_dataset=sn+'_19LA_attack_prior',eer_percent=100*(fa[j]+lam*(fa[k]-fa[j]))))
    pd.DataFrame(restricted).to_csv(OUT/'same_source_attack_control.csv',index=False)
    distributions=[]
    for base in ['19LA_eval','21LA_eval']:
        for y in [0,1]:
            b=data[base].loc[data[base].y==y,'cosine_s'];s=df.loc[df.y==y,'cosine_s']
            test=ks_2samp(b,s,method='asymp')
            distributions.append(dict(baseline=base,y=y,ks_statistic=float(test.statistic),
                ks_location=float(test.statistic_location),ks_sign=int(test.statistic_sign),
                median_change=float(s.median()-b.median()),
                p_value_omitted='utterance independence invalid; use cluster intervals instead'))
    save('score_distribution_shift.json',distributions)
    # Export aggregate ECDF points for plot, not utterance-level data.
    curve=[]
    for name in ['19LA_eval','21LA_eval','DF_full']:
        for y in [0,1]:
            s=np.sort(data[name].loc[data[name].y==y,'cosine_s'].to_numpy())
            for q in np.unique(np.r_[np.linspace(0,1,151),.001,.005,.01,.99,.995,.999]):
                curve.append(dict(dataset=name,y=y,quantile=float(q),score=float(np.quantile(s,q))))
    pd.DataFrame(curve).to_csv(OUT/'score_ecdf.csv',index=False)
    # No unseen-speaker explanation: 19LA evaluation speakers are also disjoint from T.
    tgroups=set(data['T'].cluster);overlaps={n:len(tgroups&set(data[n].cluster)) for n in ['19LA_eval','21LA_eval','DF_full']}
    source_train=dict(training_corpus='ASVspoof2019_LA',training_bona=2580,
        T_evaluation_speaker_overlap=overlaps,
        warning='unseen speaker alone differs from unseen recording/source domain; embeddings do not prove the learning mechanism')
    save('training_coverage.json',source_train)
    # Check existing independent full-evaluation record and alias lineage.
    existing=pd.read_csv(ROOT/'artifacts/geometry_cache/local_analysis/frozen_all_v1/R1/DF_full/metrics.csv')
    native=existing.loc[existing.model=='native'].iloc[0]
    actual=next(r for r in metrics if r['dataset']=='DF_full')
    assert abs(native.eer_percent-actual['eer_percent'])<1e-8
    save('verification.json',dict(status='pass',cache_hash_audits=audits,
        checkpoint=CHECKPOINT,full_df_union_disjoint=True,df_expected_rows=533928,
        metric_checks='sklearn ROC interpolation, independent DET and rank AUC; exact weighted ROC matches unweighted',
        existing_full_df_native_eer_matches=True,shared_attacks=sorted(shared),
        bootstrap_clusters=len(groups),strata={c:sum(g.startswith(c+'::') for g in groups) for c in ['asvspoof','vcc2018','vcc2020']},
        unresolved_source_family=True,bootstrap_does_not_cover_training_seed_or_model_selection=True))
    summary=dict(status='complete_exploratory',elapsed_seconds=time.monotonic()-start,
        metrics=metrics,thresholds=thresholds,decomposition=decomposition,source_errors=source_aggregate,
        limitations=['Class replacement is score-level accounting, not retraining causality.',
            'DF new attacks exist; their aggregate score shift does not explain positive EER change.',
            'Official base-utterance/source-family mappings remain unresolved.',
            'Single fixed checkpoint; analysis is post-hoc and not a new independent benchmark.',
            'DF full includes the previously inspected pilot; full size adds precision, not a blind confirmation.'])
    save('summary.json',summary);(OUT/'complete').write_text('0\n')
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
