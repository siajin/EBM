"""Audit saved fold choices, scaler provenance and OOF probe predictions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from scipy.special import ndtr
from threadpoolctl import threadpool_limits
from frozen_common import *
from frozen_probes import raw_features,inputs,VARIANTS,RIDGES
from audit_cached_core_results import independent_eer,independent_auc

OUT=DEFAULT_ROOT/'local_analysis/frozen_all_v1/P3_P4'

def numpy_score(state,x):
    p={k:v.numpy() for k,v in state.items()};s,n,v=x[:,:3].T
    answer=(np.logaddexp(0,p['a0'])+1e-4)*s+p['intercept']+p['beta']*n+p['eta']*v
    if 'w' in p:answer=answer+x[:,3:]@p['w']
    if 'fc1.weight' in p:
        h=x[:,3:]@p['fc1.weight'].T+p['fc1.bias'];answer=answer+((h*ndtr(h))@p['fc2.weight'].T+p['fc2.bias']).ravel()
    return answer

def audit():
    checks=0;metrics_checked=0;max_score_error=0.
    for name in ['DF_pilot','21LA_pilot']:
        root=OUT/'B'/name;assert (root/'complete.json').is_file();d=panel(name);raw=raw_features(d);manifest=json.loads((root/'fold_manifest.json').read_text());assert manifest['UID_order']==d.rows.utterance_id.tolist()
        pred=pd.read_parquet(root/'predictions.parquet');records=pd.read_csv(root/'metrics.csv');assert pred.utterance_id.tolist()==d.rows.utterance_id.tolist()
        for variant in VARIANTS:
            for seed in SEEDS:
                assembled=np.full(len(d.rows),np.nan)
                for fm in manifest['folds']:
                    folder=root/'models'/variant/str(seed)/f"outer{fm['fold']}";train=np.array(fm['train_rows']);test=np.array(fm['test_rows']);selection=json.loads((folder/'selection.json').read_text());candidates=[]
                    assert not set(d.rows.iloc[train].dependency_group)&set(d.rows.iloc[test].dependency_group)
                    for ridge in RIDGES:
                        histories=[]
                        for j,im in enumerate(fm['inner']):
                            fitfolder=folder/f'ridge{ridge}'/f'inner{j}';record=json.loads((fitfolder/'complete.json').read_text());sig=record['signature']
                            ti=np.asarray(im['train_rows']);vi=np.asarray(im['validation_rows']);assert not set(ti)&set(vi) and set(ti)|set(vi)==set(train)
                            assert sig['train_UID_sha256']==hashlib.sha256('\n'.join(d.rows.iloc[ti].utterance_id).encode()).hexdigest()
                            assert sig['val_UID_sha256']==hashlib.sha256('\n'.join(d.rows.iloc[vi].utterance_id).encode()).hexdigest()
                            assert sig['script_sha256']==digest(Path(__file__).with_name('frozen_probes.py'));histories.append(pd.read_csv(fitfolder/'history.csv').validation_balanced_logloss.to_numpy());checks+=1
                        common=min(map(len,histories));means=np.mean([h[:common] for h in histories],axis=0)
                        candidates.extend((float(v),e,-ridge,ridge) for e,v in enumerate(means))
                    minimum=min(r[0] for r in candidates);choice=min([r for r in candidates if r[0]<=minimum+1e-8],key=lambda r:(r[1],r[2]));assert selection['epochs']==choice[1] and selection['ridge']==choice[3];checks+=1
                    refit=folder/'refit';meta=json.loads((refit/'complete.json').read_text());assert meta['signature']['fixed_epochs']==selection['epochs']
                    hist=pd.read_csv(refit/'history.csv');assert hist.epoch.tolist()==list(range(selection['epochs']+1))
                    obj=torch.load(refit/'model.pt',weights_only=False);scaler=obj['scaler']
                    for feature,sc in scaler.items():
                        assert np.max(np.abs(np.asarray(sc['mean'])-raw[feature][train].mean(0)))<1e-12
                        assert np.max(np.abs(np.asarray(sc['std'])-np.maximum(raw[feature][train].std(0),1e-12)))<1e-12
                    x=inputs(raw,test,variant,scaler);assembled[test]=numpy_score(obj['state_dict'],x)
                actual=pred[f'{variant}_{seed}'].to_numpy();err=float(np.max(np.abs(assembled-actual)));assert err<1e-9;max_score_error=max(err,max_score_error);checks+=1
                rec=records[(records.variant==variant)&(records.seed==seed)].iloc[0];assert abs(independent_eer(d.rows.y.to_numpy(),assembled)-rec.eer_percent)<1e-9;assert abs(independent_auc(d.rows.y.to_numpy(),assembled)-rec.auroc)<1e-12;metrics_checked+=1
        # Report native on the same OOF population without fitting a new model.
        save(root/'native_comparison.json',dict(dataset=name,native=metric(d.rows.y.to_numpy(),raw['s']),OOF_probe_metrics=records.to_dict('records'),interpretation='Dataset-internal diagnostic; not T-to-evaluation transfer or a newly trained EBM'))
    assert (OUT/'A/complete.json').exists()
    for name in ['DF_pilot','21LA_pilot']:
        d=panel(name);pred=pd.read_parquet(OUT/'A'/name/'predictions.parquet');records=pd.read_csv(OUT/'A'/name/'metrics.csv')
        for _,r in records.iterrows():
            s=pred[f'{r.variant}_{int(r.seed)}'].to_numpy();assert abs(independent_eer(d.rows.y.to_numpy(),s)-r.eer_percent)<1e-9;assert abs(independent_auc(d.rows.y.to_numpy(),s)-r.auroc)<1e-12;metrics_checked+=1
    report=dict(status='pass',checks=checks,metrics_checked=metrics_checked,max_independent_OOF_score_error=max_score_error,all_scalers_train_fold_only=True,outer_test_not_used_for_ridge_epoch=True,original_EBM_training=False)
    save(OUT/'verification.json',report);print(json.dumps(report))

if __name__=='__main__':
    with threadpool_limits(2):audit()
