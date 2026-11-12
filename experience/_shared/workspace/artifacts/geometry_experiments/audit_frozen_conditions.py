"""Condition audit of completed/ongoing frozen experiments; no model fitting."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json,hashlib,math
import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from threadpoolctl import threadpool_limits
from frozen_common import DEFAULT_ROOT,panel,save,CAL_A,CAL_B,SEEDS,threshold,metric
from frozen_heads import batches,Head,oc
from frozen_probes import raw_features,inputs
from audit_frozen_probes import numpy_score

ROOT=Path(__file__).resolve().parents[2];OUT=DEFAULT_ROOT/'local_analysis/frozen_all_v1';DEST=OUT/'condition_audit_20261105_v2'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def uid_hash(rows,ix):return hashlib.sha256('\n'.join(rows.iloc[ix].utterance_id).encode()).hexdigest()

def main():
    DEST.mkdir(exist_ok=False);checks=[]
    def check(name,ok,**details):
        assert ok,(name,details)
        checks.append(dict(name=name,status='pass',**details))
    from geometry_cache import resolve_run
    frames={}
    for name in ['T','V','DF_pilot','21LA_pilot']:
        if name=='21LA_pilot':
            folder,entry,split=resolve_run('21LA_eval');rows=pd.read_parquet(folder/'extract/scores.parquet');manifest=pd.read_parquet(DEFAULT_ROOT/'analysis_manifests/frozen_data_sizes_v1/21LA_pilot.parquet');rows=rows.iloc[manifest.cache_row].reset_index(drop=True)
            lock=json.loads((DEFAULT_ROOT/'analysis_manifests/frozen_data_sizes_v1/lock.json').read_text())['views'][name]
            check('LA_view_hash_and_source',sha(DEFAULT_ROOT/'analysis_manifests/frozen_data_sizes_v1/21LA_pilot.parquet')==lock['manifest_sha256'] and entry['cache_fingerprint']==lock['source_cache_fingerprint'])
        else:
            folder,entry,split=resolve_run(name);rows=pd.read_parquet(folder/'extract/scores.parquet');rows=rows[rows.split==split].reset_index(drop=True)
        frames[name]=rows;check('UIDs_unique',not rows.utterance_id.duplicated().any(),dataset=name,n=len(rows))
        check('checkpoint_consistent',entry['identity']['checkpoint_sha256']=='104873703a57cad98b686d770e0860e031a2cf9abef79b72668780c1660e0b24')
    for a in frames:
        for b in frames:
            if a>=b:continue
            check('no_exact_UID_leakage',not set(frames[a].utterance_id)&set(frames[b].utterance_id),pair=[a,b])
    for name in ['DF_pilot','21LA_pilot']:
        check('equal_primary_class_counts',frames[name].y.value_counts().to_dict()=={0:18000,1:2000})
    for script in ['frozen_heads.py','frozen_probes.py','frozen_n0.py','frozen_common.py']:
        check('executed_source_unchanged',sha(Path(__file__).with_name(script))==sha(OUT/'source_snapshot'/script),script=script)
    sampling=[]
    y=frames['T'].y.to_numpy()
    for seed in SEEDS:
        for epoch in [1,2,80]:
            first=list(batches(y,seed,epoch));second=list(batches(y,seed,epoch));check('shared_deterministic_sampler',len(first)==162 and all(np.array_equal(a,b) for a,b in zip(first,second)))
            check('all_batches16B144S',all(np.sum(y[ix]==1)==16 and np.sum(y[ix]==0)==144 for ix in first))
            check('every_training_UID_seen',len(set(np.concatenate(first)))==len(y))
            sampling.append(dict(seed=seed,epoch=epoch,steps=len(first),rows_including_wrap=sum(map(len,first))))
    score=torch.linspace(-1,1,160);labels=torch.cat([torch.ones(16),torch.zeros(144)])
    reference=.5*(torch.nn.functional.softplus(20*(.9-score[:16])).mean()+torch.nn.functional.softplus(20*(score[16:]-.2)).mean())
    check('weighted_OC_exact_balanced_mean',abs(float(oc(score,labels)-reference))<1e-12)
    c=np.load(resolve_run('T')[0]/'extract/rank0_centroid.npy');x=np.random.default_rng(997).normal(size=(17,160));x/=np.linalg.norm(x,axis=1,keepdims=True);xx=torch.as_tensor(x)
    for kind,n in [('static',160),('dynamic',5296),('mlp',5296),('metric',160)]:
        head=Head(kind,c,101);check('head_capacity_and_identity',sum(p.numel() for p in head.parameters())==n and np.max(np.abs(head(xx).detach().numpy()-x@c))<1e-10,kind=kind)
        check('centroid_is_nontrainable_buffer','c' not in dict(head.named_parameters()) and not head.c.requires_grad)
        if kind=='metric':
            with torch.no_grad():head.v.copy_(torch.linspace(-100,100,160))
            w=head.weights().detach().numpy();check('metric_PSD_trace_bound',w.min()>0 and abs(w.sum()-160)<1e-10 and w.max()/w.min()<=np.exp(4)+1e-10)
        else:
            with torch.no_grad():
                if kind=='static':head.v.fill_(100)
                else:head.fc2.bias.copy_(torch.linspace(-100,100,160))
            check('transformation_l2_bound',float(torch.linalg.vector_norm(head.transform(xx)-xx,dim=1).max().detach())<=.5+1e-10)
    cv_records=[];fit_records=0;selection_records=0;scaler_checks=0;coverage_issues=[]
    for name in ['DF_pilot','21LA_pilot']:
        rows=frames[name];base=OUT/'P3_P4/B'/name;manifest=json.loads((base/'fold_manifest.json').read_text());check('CV_UID_order',manifest['UID_order']==rows.utterance_id.tolist())
        groups=rows.dependency_group.to_numpy();sources=rows.source_corpus.astype(str).to_numpy();source_classes=set(zip(sources,rows.y))
        observed=[];lookup={}
        for fold in manifest['folds']:
            k=fold['fold'];tr=np.asarray(fold['train_rows']);te=np.asarray(fold['test_rows']);observed.extend(te);check('outer_group_disjoint',not set(groups[tr])&set(groups[te]))
            for phase,idx in [('outer_train',tr),('outer_test',te)]:
                missing=source_classes-set(zip(sources[idx],rows.iloc[idx].y))
                if missing:coverage_issues.append(dict(dataset=name,outer_fold=k,partition=phase,missing_source_class_cells=sorted(missing)))
                check('outer_binary_class_and_source_presence',set(rows.iloc[idx].y)=={0,1} and set(sources[idx])==set(sources))
            lookup[(k,'refit')]=(tr,None)
            for j,inner in enumerate(fold['inner']):
                ti=np.asarray(inner['train_rows']);vi=np.asarray(inner['validation_rows']);check('inner_group_disjoint',not set(groups[ti])&set(groups[vi]) and set(ti)|set(vi)==set(tr))
                for phase,idx in [('train',ti),('validation',vi)]:
                    missing=source_classes-set(zip(sources[idx],rows.iloc[idx].y))
                    if missing:coverage_issues.append(dict(dataset=name,outer_fold=k,partition='inner'+str(j)+'_'+phase,missing_source_class_cells=sorted(missing)))
                    check('inner_binary_class_and_source_presence',set(rows.iloc[idx].y)=={0,1} and set(sources[idx])==set(sources))
                lookup[(k,'inner'+str(j))]=(ti,vi)
        check('one_outer_holdout_per_UID',sorted(observed)==list(range(len(rows))))
        sampled_scalers=set()
        completed=list((base/'models').rglob('complete.json'))
        for path in completed:
            record=json.loads(path.read_text());sig=record['signature'];relative=path.relative_to(base/'models').parts;variant,seed,outer=relative[:3];k=int(outer.removeprefix('outer'));phase=relative[-2];ti,vi=lookup[(k,phase)]
            check('fit_rows_match_locked_fold',sig['train_UID_sha256']==uid_hash(rows,ti) and sig['val_UID_sha256']==(None if vi is None else uid_hash(rows,vi)))
            check('recorded_hyperparameters',sig['seed'] in SEEDS and sig['ridge'] in [1e-4,1e-2] and sig['script_sha256']==sha(Path(__file__).with_name('frozen_probes.py')))
            history=pd.read_csv(path.with_name('history.csv'));check('epoch_budget',history.epoch.iloc[0]==0 and history.epoch.iloc[-1]<=80)
            if phase=='refit':
                selection=json.loads(path.parent.parent.joinpath('selection.json').read_text());check('refit_exact_selected_epochs',sig['fixed_epochs']==selection['epochs'] and history.epoch.iloc[-1]==selection['epochs']);selection_records+=1
                options=selection['all_candidates'];minimum=min(r['LL'] for r in options);best=min([r for r in options if r['LL']<=minimum+1e-8],key=lambda r:(r['epoch'],-r['ridge']))
                check('inner_only_epoch_ridge_choice',not selection['selection_uses_outer_test'] and best['epoch']==selection['epochs'] and best['ridge']==selection['ridge'])
            sample_key=(variant,int(seed),phase=='refit')
            if sample_key not in sampled_scalers:
                obj=torch.load(path.with_name('model.pt'),weights_only=False);n=np.log(rows.iloc[ti].raw_norm.to_numpy());check('scaler_train_rows_only',abs(obj['scaler']['n']['mean']-n.mean())<1e-12 and abs(obj['scaler']['n']['std']-max(n.std(),1e-12))<1e-12);scaler_checks+=1;sampled_scalers.add(sample_key)
            fit_records+=1
        cv_records.append(dict(dataset=name,outer_folds=5,inner_folds=3,locked_partitions=20,completed_fits_checked=len(completed),source_class_cells=len(source_classes)))
    signals=[]
    for name in ['T','V']:
        rows=frames[name];s=rows.cosine_s.to_numpy();yy=rows.y.to_numpy();loss=np.logaddexp(0,20*np.where(yy==1,.9-s,s-.2))
        lb,ls=float(loss[yy==1].mean()),float(loss[yy==0].mean());floor=.5*(np.logaddexp(0,-2)+np.logaddexp(0,-24))
        signals.append(dict(dataset=name,balanced_OC=.5*(lb+ls),theoretical_infimum_OC=float(floor),bona_loss=lb,spoof_loss=ls,spoof_loss_fraction=ls/(lb+ls),spoofs_above_margin=int(np.sum(s[yy==0]>.2)),spoof_rows=int(sum(yy==0)),V_EER_zero=(s[yy==1].min()>s[yy==0].max()) if name=='V' else None))
    # Supplement the missing common V operating-point report from saved models
    # and saved evaluation predictions. No new fit or audio forward is used.
    vd=panel('V');raw=raw_features(vd);vix=np.arange(len(vd.rows));operating=[]
    native_t=threshold(vd.rows.y.to_numpy(),raw['s'])
    baselines=json.loads((OUT/'native_baselines.json').read_text())['datasets']
    for name in ['DF_pilot','21LA_pilot']:
        operating.append(dict(dataset=name,variant='native',seed=0,score_type='cosine',**native_t,**baselines[name]))
    for variant in ['scalar','direction_linear','noise_linear','direction_mlp','noise_mlp']:
        for seed in SEEDS:
            choice=json.loads((OUT/'P3_P4/A/selection'/f'{variant}_{seed}.json').read_text());path=OUT/'P3_P4/A/models'/variant/str(seed)/str(choice['ridge'])/'model.pt';obj=torch.load(path,weights_only=False)
            val=numpy_score(obj['state_dict'],inputs(raw,vix,variant,obj['scaler']));th=threshold(vd.rows.y.to_numpy(),val)
            p=expit(val);vv=vd.rows.y.to_numpy();vll=-.5*(np.log(np.clip(p[vv==1],1e-7,1-1e-7)).mean()+np.log1p(-np.clip(p[vv==0],1e-7,1-1e-7)).mean())
            check('saved_P3_choice_reproduced',abs(vll-choice['V_LL'])<1e-12)
            for name in ['DF_pilot','21LA_pilot']:
                pred=pd.read_parquet(OUT/'P3_P4/A'/name/'predictions.parquet');operating.append(dict(dataset=name,variant=variant,seed=seed,score_type='unsaturated_logit',**th,**metric(pred.y.to_numpy(),pred[f'{variant}_{seed}'].to_numpy(),th['threshold'],logit=True)))
    pd.DataFrame(operating).to_csv(DEST/'P3_transfer_V_operating_points.csv',index=False)
    record=dict(status='needs_qualification_source_class_coverage' if coverage_issues else 'pass_with_design_limits',KST=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),checks=len(checks),completed_fit_records_checked=fit_records,completed_outer_selections_checked=selection_records,scalers_sampled=scaler_checks,all_CV_partitions_checked=cv_records,coverage_issues=coverage_issues,OC_signal=signals,scope='all existing completed fit signatures and partitions; scaler sampling; static checks for pending work',future_R2_R3_runtime_QC_not_yet_passed=True,no_experiments_stopped_or_restarted=True,no_training_or_audio_inference=True,details=checks)
    save(DEST/'verification.json',record)
    print(json.dumps({k:v for k,v in json.loads((DEST/'verification.json').read_text(encoding='utf8')).items() if k!='details'},ensure_ascii=False))

if __name__=='__main__':
    with threadpool_limits(1):main()
