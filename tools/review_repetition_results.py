"""Independent CPU-only analysis of completed, derived repetition scores."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

BASE=Path(__file__).resolve().parents[2]/'artifacts/repetition_shortcut_20261111'
RAW=BASE/'new_results'
OUT=BASE/'reviewed_analysis'
BINS=[0,16000,24000,32000,40000,48000,56000,64600,np.inf]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name,data):
    (OUT/name).write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def roc(y,s,w=None):
    """Tie-block ROC independent of sklearn/scipy and existing experiment helpers."""
    y=np.asarray(y,dtype=np.int64);s=np.asarray(s,dtype=np.float64)
    w=np.ones(len(y)) if w is None else np.asarray(w,dtype=np.float64)
    assert set(y)=={0,1} and np.isfinite(s).all() and np.isfinite(w).all() and (w>0).all()
    order=np.argsort(-s,kind='stable');y,s,w=y[order],s[order],w[order]
    end=np.r_[np.flatnonzero(s[1:]!=s[:-1]),len(s)-1]
    tp=np.r_[0,np.cumsum(w*y)[end]/np.sum(w*y)]
    fp=np.r_[0,np.cumsum(w*(1-y))[end]/np.sum(w*(1-y))]
    return fp,tp,np.r_[np.inf,s[end]]


def eer(y,s,w=None):
    fp,tp,_=roc(y,s,w);difference=fp+tp-1
    hi=int(np.flatnonzero(difference>=0)[0]);lo=max(0,hi-1)
    a=-difference[lo]/(difference[hi]-difference[lo]) if hi!=lo else 0.
    return float(100*(fp[lo]+a*(fp[hi]-fp[lo])))


def threshold(y,s):
    fp,tp,t=roc(y,s)
    return float(t[np.argmin(abs(fp+tp-1))])


def errors(g,col,t):
    return np.where(g.label.to_numpy()==1,g[col].to_numpy()<t,g[col].to_numpy()>=t)


def ipf_weights(g,target):
    """Match duration while preserving spoof attack x codec shares on support."""
    result=np.zeros(len(g));max_residual=0.
    for label in [0,1]:
        loc=np.flatnonzero(g.label.to_numpy()==label);a=g.iloc[loc]
        key=a.attack.astype(str)+'|'+a.codec.astype(str)
        tab=pd.crosstab(a.bin,key).reindex(index=target.index,fill_value=0)
        initial=tab.to_numpy(dtype=float);q=initial/initial.sum()
        col_target=q.sum(0);row_target=target.to_numpy()
        for iteration in range(10000):
            q*= (row_target/q.sum(1))[:,None]
            q*= (col_target/q.sum(0))[None,:]
            residual=max(abs(q.sum(1)-row_target).max(),abs(q.sum(0)-col_target).max())
            if residual<1e-12:break
        else:raise AssertionError('IPF did not converge')
        max_residual=max(max_residual,float(residual))
        row_ids=pd.Index(tab.index).get_indexer(a.bin)
        col_ids=pd.Index(tab.columns).get_indexer(key)
        result[loc]=q[row_ids,col_ids]/initial[row_ids,col_ids]
    return result,max_residual


def main():
    OUT.mkdir(exist_ok=True)
    protocol=json.loads((BASE/'protocol.json').read_text())
    published=json.loads((RAW/'analysis_complete.json').read_text())
    parts={}
    for kind,conditions,n in [('full',2,173020),('probe',31,336)]:
        frames=[]
        for rank in [0,1]:
            path=RAW/f'{kind}_rank{rank}.csv'
            done=json.loads((RAW/f'{kind}_done_rank{rank}.json').read_text())
            assert digest(path)==done['sha256']
            frame=pd.read_csv(path,float_precision='round_trip')
            assert len(frame)==done['rows'];frames.append(frame)
        f=pd.concat(frames,ignore_index=True)
        assert len(f)==n*conditions and not f.duplicated(['phase','uid','condition']).any()
        assert f.groupby('condition').size().eq(n).all()
        assert not f.isna().any().any() and np.isfinite(f[['score_legacy240','score_B']]).all().all()
        selected=pd.read_csv(BASE/('selected.csv' if kind=='full' else 'probe_selected.csv'))
        assert digest(BASE/('selected.csv' if kind=='full' else 'probe_selected.csv'))==protocol['selection_hashes']['selected.csv' if kind=='full' else 'probe_selected.csv']
        assert len(selected)==n
        for condition,a in f.groupby('condition'):
            joined=a.merge(selected,on=['phase','uid'],suffixes=('', '_selected'),validate='one_to_one')
            for field in ['index','speaker','label','original_samples','attack']:
                assert joined[field].equals(joined[field+'_selected']),field
        parts[kind]=f
    full=parts['full'];probe=parts['probe']
    full['bin']=pd.cut(full.original_samples,BINS,labels=False)
    meta=pd.read_csv(OUT/'codec_metadata.csv')
    full=full.merge(meta[['uid','codec']],on='uid',how='left',validate='many_to_one')
    assert full.loc[full.phase=='21LA','codec'].notna().all()
    full['codec']=full.codec.fillna('dev')
    native=full.query("condition=='native'").set_index(['phase','uid']).sort_index()
    wrap=full.query("condition=='wrap4'").set_index(['phase','uid']).sort_index()
    assert native.index.equals(wrap.index)
    noop=native.original_samples>=64600
    assert native.loc[noop,'wave_sha256'].equals(wrap.loc[noop,'wave_sha256'])
    for model in ['legacy240','B']:
        assert np.array_equal(native.loc[noop,'score_'+model],wrap.loc[noop,'score_'+model])

    validation=[];operating=[];distribution=[];weighted=[];attacks=[];thresholds={};length_eer=[];repeat_groups=[]
    independent={}
    max_published_error=0.
    for model in ['legacy240','B']:
        col='score_'+model
        for condition in ['native','wrap4']:
            dev=full[(full.phase=='dev')&(full.condition==condition)]
            t=threshold(dev.label,dev[col]);thresholds[model+'_'+condition]=t
            assert abs(t-published['thresholds'][model+'_'+condition])<1e-12
            for phase,g in full[full.condition==condition].groupby('phase'):
                metric=eer(g.label,g[col]);independent[(model,phase,condition)]=metric
                ref=next(x['eer'] for x in published['full_metrics'] if x['model']==model and x['phase']==phase and x['condition']==condition and x['weighting']=='observed')
                max_published_error=max(max_published_error,abs(metric-ref));assert abs(metric-ref)<1e-9
                validation.append(dict(model=model,phase=phase,condition=condition,eer=metric,n=len(g)))
            la=full[(full.phase=='21LA')&(full.condition==condition)].copy()
            for group,mask in [('le2',la.original_samples<=32000),('gt2',la.original_samples>32000)]:
                a=la.loc[mask]
                length_eer.append(dict(model=model,condition=condition,group=group,n=len(a),eer=eer(a.label,a[col])))
            for policy,t_used in [('fixed_native_dev',thresholds.get(model+'_native',t)),('condition_matched_dev',t)]:
                for group,mask in [('all',np.ones(len(la),dtype=bool)),('le2',la.original_samples<=32000),('gt2',la.original_samples>32000)]:
                    for label,g in la.loc[mask].groupby('label'):
                        e=errors(g,col,t_used)
                        operating.append(dict(model=model,condition=condition,threshold_policy=policy,group=group,label=int(label),n=len(g),errors=int(e.sum()),error_percent=float(100*e.mean()),threshold=t_used))
                for repeats,g in la[la.label==1].groupby(la.original_samples<64600):
                    e=errors(g,col,t_used)
                    repeat_groups.append(dict(model=model,condition=condition,threshold_policy=policy,needs_repeat=bool(repeats),n=len(g),errors=int(e.sum()),error_percent=float(e.mean()*100)))
            counts=pd.crosstab(la.bin,la.label).reindex(columns=[0,1],fill_value=0)
            common=counts.index[(counts>0).all(axis=1)]
            support=la[la.bin.isin(common)].copy()
            target=counts.loc[common,1]/counts.loc[common,1].sum()
            simple=np.array([target.loc[b]/counts.loc[b,y] for b,y in zip(support.bin,support.label)])
            ipf,residual=ipf_weights(support,target)
            for name,data,w in [('observed',la,None),('common_support_observed',support,None),('duration_matched',support,simple),('duration_attack_codec_preserved',support,ipf)]:
                weighted.append(dict(model=model,condition=condition,policy=name,eer=eer(data.label,data[col],w),n=len(data),ipf_residual=residual if 'preserved' in name else 0))
            for attack,g in la[la.label==0].groupby('attack'):
                e=errors(g,col,thresholds[model+'_native'])
                attacks.append(dict(model=model,condition=condition,attack=attack,n=len(g),mean_score=float(g[col].mean()),errors=int(e.sum()),error_percent=float(100*e.mean()),le2_percent=float((g.original_samples<=32000).mean()*100)))
    for phase,g in native.reset_index().groupby('phase'):
        for label,a in g.groupby('label'):
            distribution.append(dict(phase=phase,label=int(label),n=len(a),le2=int((a.original_samples<=32000).sum()),le2_percent=float((a.original_samples<=32000).mean()*100)))

    # Head/tail are paired views, averaged per source before speaker resampling.
    dose=[];probe_errors=[];contexts=[]
    for model in ['legacy240','B']:
        col='score_'+model
        wide=probe.pivot(index=['uid','speaker','label','attack'],columns='condition',values=col)
        assert len(wide)==336
        for seconds in [1,2]:
            for treatment,control in [('repeat2','native'),('repeat3','native'),('repeat4','native'),('smooth_seams4','repeat4'),('smooth_seams4','smooth_sham4'),('reflect4','repeat4')]:
                d=sum(wide[f'{end}_{seconds}s_{treatment}']-wide[f'{end}_{seconds}s_{control}'] for end in ['head','tail'])/2
                for label,g in d.rename('delta').reset_index().groupby('label'):
                    mean=float(g.delta.mean())
                    ref=next(x for x in published['dose_controls'] if x['model']==model and x['label']==label and x['crop_seconds']==seconds and x['treatment']==treatment and x['control']==control)
                    assert abs(mean-ref['mean_delta'])<1e-12
                    dose.append(dict(model=model,label=int(label),crop_seconds=seconds,treatment=treatment,control=control,n=len(g),mean_delta=mean,published_ci_lo=ref['ci_lo'],published_ci_hi=ref['ci_hi']))
            for condition in ['native','repeat4','smooth_seams4','smooth_sham4','reflect4']:
                for label in [0,1]:
                    values=np.stack([wide.loc[wide.index.get_level_values('label')==label,f'{end}_{seconds}s_{condition}'].to_numpy() for end in ['head','tail']])
                    for policy,t in [('fixed_native_dev',thresholds[model+'_native']),('wrapped_dev',thresholds[model+'_wrap4'])]:
                        e=values<t if label else values>=t
                        probe_errors.append(dict(model=model,label=label,crop_seconds=seconds,condition=condition,threshold_policy=policy,n=values.shape[1],errors_average_head_tail=float(e.sum(1).mean()),error_percent=float(e.mean()*100),mean_score=float(values.mean())))
        for condition in ['natural4','natural4_smooth_1s_positions','natural4_smooth_2s_positions']:
            for label in [0,1]:
                values=wide.loc[wide.index.get_level_values('label')==label,condition]
                contexts.append(dict(model=model,label=label,condition=condition,n=len(values),mean_score=float(values.mean()),mean_delta_from_natural=float((values-wide.loc[values.index,'natural4']).mean())))
    # Pairwise flips in native-short originals, using both fixed and retuned dev thresholds.
    transitions=[]
    for model in ['legacy240','B']:
        col='score_'+model
        for phase in ['21LA','dev']:
            for group in ['all','le2']:
                a=native.reset_index();a=a[(a.phase==phase)&((a.original_samples<=32000) if group=='le2' else True)]
                b=wrap.reset_index().set_index(['phase','uid']).loc[pd.MultiIndex.from_frame(a[['phase','uid']])].reset_index()
                for label in [0,1]:
                    use=a.label.to_numpy()==label;x=a.loc[use];z=b.loc[use]
                    e0=errors(x,col,thresholds[model+'_native']);e1=errors(z,col,thresholds[model+'_native'])
                    transitions.append(dict(model=model,phase=phase,group=group,label=label,n=len(x),native_errors=int(e0.sum()),wrap_errors=int(e1.sum()),correct_to_error=int((~e0&e1).sum()),error_to_correct=int((e0&~e1).sum()),delta=float((z[col].to_numpy()-x[col].to_numpy()).mean())))

    pd.DataFrame(validation).to_csv(OUT/'independent_full_eer.csv',index=False)
    pd.DataFrame(operating).to_csv(OUT/'operating_points_21LA.csv',index=False)
    pd.DataFrame(weighted).to_csv(OUT/'weighted_eer_21LA.csv',index=False)
    pd.DataFrame(attacks).to_csv(OUT/'attacks_21LA.csv',index=False)
    pd.DataFrame(dose).to_csv(OUT/'dose_effects.csv',index=False)
    pd.DataFrame(probe_errors).to_csv(OUT/'probe_errors.csv',index=False)
    pd.DataFrame(contexts).to_csv(OUT/'natural_smoothing_controls.csv',index=False)
    pd.DataFrame(transitions).to_csv(OUT/'paired_transitions.csv',index=False)
    pd.DataFrame(length_eer).to_csv(OUT/'length_specific_eer.csv',index=False)
    pd.DataFrame(repeat_groups).to_csv(OUT/'normal_repeat_groups.csv',index=False)
    save('validation.json',dict(status='passed',full_rows=len(full),probe_rows=len(probe),
        no_op_utterances=int(noop.sum()),independent_eer_max_difference_pp=max_published_error,
        sources={p.name:digest(p) for p in RAW.glob('*.csv')},
        independent_ROC='numpy tie-block sort/cumulative class normalization; no sklearn/scipy/old metric helpers',
        scope='Frozen original24050 and B50, noise0, dev thresholds diagnostic only, no GPU/training/audio/model access'))
    save('summary.json',dict(thresholds=thresholds,full_eer=validation,operating=operating,weighted=weighted,
        distribution=distribution,dose=dose,probe_errors=probe_errors,contexts=contexts,transitions=transitions,
        length_eer=length_eer,repeat_groups=repeat_groups))
    print('EER\n'+pd.DataFrame(validation).query("phase=='21LA'").to_string(index=False))
    print('OPERATING <=2s\n'+pd.DataFrame(operating).query("group=='le2'").to_string(index=False))
    print('WEIGHTS\n'+pd.DataFrame(weighted).to_string(index=False))
    print('NATURAL\n'+pd.DataFrame(contexts).to_string(index=False))


if __name__=='__main__':main()
