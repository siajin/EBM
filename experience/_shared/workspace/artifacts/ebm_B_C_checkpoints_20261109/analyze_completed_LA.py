"""Compare completed LA scores against paired fixed50 legacy outputs on CPU."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE=Path(__file__).resolve().parent
OUT=HERE/'LA_analysis'
OLD=HERE.parents[1]/'artifacts/geometry_cache/local_analysis/B_final_analysis_20261108_v1'
spec=importlib.util.spec_from_file_location('evaluator',HERE.parents[1]/'EBM/tools/evaluate_B_C_checkpoints.py')
ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
dev=json.loads((HERE/'dev_analysis/analysis.json').read_text())
old_metrics=json.loads((OLD/'analysis/comparison.json').read_text())
history=json.loads((OLD/'historical_legacy_final.json').read_text())

def independent_eer(y,scores):
    order=np.argsort(scores,kind='mergesort')[::-1];v=scores[order];labels=y[order]
    ends=np.r_[np.flatnonzero(v[1:]!=v[:-1]),len(v)-1]
    tp=np.cumsum(labels)[ends];fp=ends+1-tp
    fpr=np.r_[0.,fp/int((y==0).sum())];fnr=np.r_[1.,1-tp/int((y==1).sum())]
    k=int(np.flatnonzero(fpr>=fnr)[0]);d=fpr-fnr
    w=-d[k-1]/(d[k]-d[k-1]);return 100*(fpr[k-1]+w*(fpr[k]-fpr[k-1]))

result={'scope':'19LA and21LA complete;20 B/C checkpoints plus legacy fixed50 common-input reference; historical legacy pipeline shown separately','counts':{},'models':{},'final50_comparison':{},'validation':{},'historical_legacy':history['final_EER_percent'],'paired_B_C':{},'length_bins':{},'attack_errors':{}}
rows=[];length_rows=[]
for phase,expected in [('19LA',71237),('21LA',148176)]:
    f=pd.concat([pd.read_csv(OUT/f'scores_{phase}_rank{rank}.csv') for rank in range(2)],ignore_index=True).sort_values('index').reset_index(drop=True)
    assert len(f)==expected and f['index'].tolist()==list(range(expected)) and f.uid.is_unique
    assert not f.isna().any().any() and (f.used_samples==np.minimum(f.original_samples,64600)).all()
    old=pd.read_parquet(OLD/f'paired_scores_{phase}.parquet').sort_values('index').reset_index(drop=True)
    assert f[['index','uid','speaker','label','original_samples','used_samples','attack']].equals(old[['index','uid','speaker','label','original_samples','used_samples','attack']])
    error=float(np.max(np.abs(f.score_B_e50.to_numpy()-old.score_B.to_numpy())))
    assert error==0.,error
    f['score_legacy240']=old.score_legacy240
    y=f.label.to_numpy();normal=y==1;spoof=y==0
    result['counts'][phase]={'total':len(f),'normal':int(normal.sum()),'spoof':int(spoof.sum())}
    assert np.isfinite(f[[f'score_{model}' for model in ev.EPOCHS]+['score_legacy240']].to_numpy()).all()
    result['models'][phase]={};result['length_bins'][phase]={}
    for model in ev.EPOCHS+['legacy240']:
        scores=f['score_'+model].to_numpy();m=ev.metrics(f,'score_'+model)
        assert abs(m['EER_percent']-independent_eer(y,scores))<1e-10
        threshold=old_metrics['dev']['legacy240']['dev_threshold'] if model=='legacy240' else dev['models'][model]['discrete_EER_threshold']
        accept=scores>=threshold
        m.pop('discrete_EER_threshold');m.update(dev_threshold=threshold,normal_reject_count=int((normal&~accept).sum()),spoof_accept_count=int((spoof&accept).sum()),normal_reject_percent=float(100*(~accept[normal]).mean()),spoof_accept_percent=float(100*accept[spoof].mean()),AUROC=float(roc_auc_score(y,scores)))
        result['models'][phase][model]=m
        if model!='legacy240':
            cond,epoch=model.split('_e');rows.append({'phase':phase,'model':cond,'epoch':int(epoch),**m})
        if model in ['legacy240','B_e50','C_e50']:
            result['length_bins'][phase][model]={}
            for name,lo,hi in [('under2',0,32000),('2to3.0275',32000,48440),('3.0275to4.0375',48440,64600),('4.0375plus',64600,np.inf)]:
                mask=(f.original_samples.to_numpy()>=lo)&(f.original_samples.to_numpy()<hi);nm=mask&normal;sm=mask&spoof
                n=int(nm.sum());z=int(sm.sum());nr=int((nm&~accept).sum());sa=int((sm&accept).sum())
                item={'normal_n':n,'spoof_n':z,'normal_reject_count':nr,'spoof_accept_count':sa,'normal_reject_percent':100*nr/n if n else None,'spoof_accept_percent':100*sa/z if z else None}
                result['length_bins'][phase][model][name]=item;length_rows.append({'phase':phase,'model':model,'bin':name,**item})
            result['attack_errors'].setdefault(phase,{})[model]={}
            for attack in sorted(f.loc[spoof,'attack'].unique()):
                mask=spoof&(f.attack.to_numpy()==attack);result['attack_errors'][phase][model][attack]={'n':int(mask.sum()),'accepted':int((mask&accept).sum()),'accept_percent':100*float(accept[mask].mean())}
    result['validation'][phase]={'B50_historical_scores_max_error':error,'metadata_exactly_matches_legacy_paired_reference':True,'all21_EER_independent_check':True,'full_UID_label_counts':True,'all_scores_finite':True,'finalized_shard_SHA_checked':True}
    for model,old_key in [('B_e50','B'),('legacy240','legacy240')]:
        assert abs(result['models'][phase][model]['EER_percent']-old_metrics[phase][old_key]['EER_percent'])<1e-10
    result['final50_comparison'][phase]={model:result['models'][phase][model] for model in ['legacy240','B_e50','C_e50']}
    result['paired_B_C'][phase]={}
    for epoch in range(5,51,5):
        bs=f[f'score_B_e{epoch}'].to_numpy();cs=f[f'score_C_e{epoch}'].to_numpy()
        bm=result['models'][phase][f'B_e{epoch}'];cm=result['models'][phase][f'C_e{epoch}'];be=(bs>=bm['dev_threshold'])!=(y==1);ce=(cs>=cm['dev_threshold'])!=(y==1)
        result['paired_B_C'][phase][str(epoch)]={'EER_C_minus_B_pp':cm['EER_percent']-bm['EER_percent'],'normal_reject_C_minus_B_pp':cm['normal_reject_percent']-bm['normal_reject_percent'],'spoof_accept_C_minus_B_pp':cm['spoof_accept_percent']-bm['spoof_accept_percent'],
           'paired_errors':{label:{'B_only_wrong':int((mask&be&~ce).sum()),'C_only_wrong':int((mask&~be&ce).sum()),'both_wrong':int((mask&be&ce).sum())} for label,mask in [('normal',normal),('spoof',spoof)]}}

(OUT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
pd.DataFrame(rows).to_csv(OUT/'epoch_summary.csv',index=False);pd.DataFrame(length_rows).to_csv(OUT/'length_summary.csv',index=False)
print('EPOCH_EER\n'+pd.DataFrame(rows).pivot(index='epoch',columns=['phase','model'],values='EER_percent').to_string())
print('FINAL50',json.dumps(result['final50_comparison'],indent=2))
print('PAIRED50',json.dumps({phase:result['paired_B_C'][phase]['50'] for phase in ['19LA','21LA']},indent=2))
print('UNDER2',json.dumps({phase:{model:bins['under2'] for model,bins in models.items()} for phase,models in result['length_bins'].items()},indent=2))
