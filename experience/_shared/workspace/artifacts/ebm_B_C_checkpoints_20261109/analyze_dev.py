"""Analyze completed dev outputs only; does not contact the server or use GPUs."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE=Path(__file__).resolve().parent
OUT=HERE/'dev_analysis'
spec=importlib.util.spec_from_file_location('evaluator',HERE.parents[1]/'EBM/tools/evaluate_B_C_checkpoints.py')
evaluator=importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)

frame=pd.concat([pd.read_csv(OUT/f'scores_dev_rank{rank}.csv') for rank in range(2)],ignore_index=True).sort_values('index').reset_index(drop=True)
assert len(frame)==24844 and frame['index'].tolist()==list(range(24844))
assert frame.uid.is_unique and not frame.isna().any().any()
assert set(frame.label)=={0,1} and int(frame.label.sum())==2548
assert (frame.used_samples==np.minimum(frame.original_samples,64600)).all()
columns=[f'score_{model}' for model in evaluator.EPOCHS]
assert np.isfinite(frame[columns].to_numpy()).all()
assert (np.abs(frame[columns].to_numpy())<1.000001).all()
y=frame.label.to_numpy();normal=y==1;spoof=y==0

def independent_eer(labels,scores):
    groups={}
    for label,score in zip(labels,scores):
        counts=groups.setdefault(float(score),[0,0]);counts[int(label)]+=1
    n1=int(sum(labels));n0=len(labels)-n1;tp=fp=0;last_fpr=0.;last_fnr=1.
    for value in sorted(groups,reverse=True):
        fp+=groups[value][0];tp+=groups[value][1]
        fpr=fp/n0;fnr=1-tp/n1
        if fpr>=fnr:
            fraction=(last_fnr-last_fpr)/((fpr-last_fpr)-(fnr-last_fnr))
            return 100*(last_fpr+fraction*(fpr-last_fpr))
        last_fpr,last_fnr=fpr,fnr
    raise AssertionError('No ROC intersection')

rows=[];all_models={}
for model in evaluator.EPOCHS:
    column=f'score_{model}';scores=frame[column].to_numpy()
    values=evaluator.metrics(frame,column)
    check=independent_eer(y,scores)
    assert abs(check-values['EER_percent'])<1e-10
    threshold=values['discrete_EER_threshold'];accept=scores>=threshold
    fr=int((~accept[normal]).sum());fa=int(accept[spoof].sum())
    values.update(AUROC=float(roc_auc_score(y,scores)),normal_reject_count=fr,normal_reject_percent=100*fr/int(normal.sum()),spoof_accept_count=fa,spoof_accept_percent=100*fa/int(spoof.sum()))
    values['score_quantiles']={name:{str(q):float(np.percentile(scores[mask],q)) for q in [0,1,5,50,95,99,100]} for name,mask in [('normal',normal),('spoof',spoof)]}
    values['normal_p01_minus_spoof_p99']=values['score_quantiles']['normal']['1']-values['score_quantiles']['spoof']['99']
    values['normal_rejected_uids']=frame.loc[normal & ~accept,'uid'].tolist()
    values['worst_normals']=[{'uid':row.uid,'score':float(row[column]),'seconds':float(row.original_samples/16000)} for _,row in frame[normal].nsmallest(5,column).iterrows()]
    values['spoof_max']=float(scores[spoof].max())
    values['normals_below_every_spoof']=frame.loc[normal & (scores<scores[spoof].min()),'uid'].tolist()
    values['fixed_dev_spoof_budgets']={}
    for budget in [0.001,0.01]:
        allowed=int(np.floor(budget*int(spoof.sum())))
        cutoff=float(np.nextafter(np.sort(scores[spoof])[::-1][allowed],np.inf))
        reject=int((scores[normal]<cutoff).sum());false_accept=int((scores[spoof]>=cutoff).sum())
        assert false_accept<=allowed
        values['fixed_dev_spoof_budgets'][str(budget)]={'threshold':cutoff,'normal_reject_count':reject,'normal_reject_percent':100*reject/int(normal.sum()),'spoof_accept_count':false_accept,'spoof_accept_percent':100*false_accept/int(spoof.sum())}
    values['length_bins']={}
    for name,lo,hi in [('under2',0,32000),('2to3.0275',32000,48440),('3.0275to4.0375',48440,64600),('4.0375plus',64600,np.inf)]:
        mask=(frame.original_samples.to_numpy()>=lo)&(frame.original_samples.to_numpy()<hi)
        n=int((mask&normal).sum());f=int((mask&spoof).sum())
        values['length_bins'][name]={'normal_n':n,'spoof_n':f,'normal_reject_count':int((mask&normal&~accept).sum()),'spoof_accept_count':int((mask&spoof&accept).sum())}
    all_models[model]=values
    condition,epoch=model.split('_e')
    rows.append({'model':condition,'epoch':int(epoch),**{k:values[k] for k in ['EER_percent','AUROC','discrete_EER_threshold','normal_reject_count','spoof_accept_count','normal_p01_minus_spoof_p99']}})

paired={}
for epoch in range(5,51,5):
    b=frame[f'score_B_e{epoch}'].to_numpy();c=frame[f'score_C_e{epoch}'].to_numpy()
    bm=all_models[f'B_e{epoch}'];cm=all_models[f'C_e{epoch}']
    berr=(b>=bm['discrete_EER_threshold'])!=(y==1)
    cerr=(c>=cm['discrete_EER_threshold'])!=(y==1)
    paired[str(epoch)]={'EER_C_minus_B_pp':cm['EER_percent']-bm['EER_percent'],'score_spearman':float(pd.Series(b).corr(pd.Series(c),method='spearman')),
         'both_wrong':int((berr&cerr).sum()),'B_only_wrong':int((berr&~cerr).sum()),'C_only_wrong':int((~berr&cerr).sum()),
         'normal_rejected_overlap':sorted(set(bm['normal_rejected_uids'])&set(cm['normal_rejected_uids'])),
         'mean_score_change_C_minus_B':{'normal':float((c-b)[normal].mean()),'spoof':float((c-b)[spoof].mean())}}

result={'scope':'completed dev only; common input/masks/batch8/pooling noise1e-5 and paired RNG; no target metrics used','counts':{'n':len(frame),'normal':int(normal.sum()),'spoof':int(spoof.sum())},'one_normal_error_percent':100/int(normal.sum()),'validation':{'source_shards_SHA_checked':True,'unique_UID_full_index_label_counts':True,'all20_EER_independently_recomputed':True,'source_files':'sources.json'},'models':all_models,'paired':paired}
(OUT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
pd.DataFrame(rows).to_csv(OUT/'epoch_summary.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
print('FINAL50',json.dumps({key:all_models[key] for key in ['B_e50','C_e50']},indent=2))
print('PAIRED50',json.dumps(paired['50'],indent=2))
