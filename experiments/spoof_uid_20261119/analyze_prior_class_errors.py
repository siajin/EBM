"""Read completed scores only. No model loading, inference or optimizer calls."""
import csv, hashlib, json, math
from pathlib import Path
import numpy as np

SOURCES={
 'bona_allcross':dict(root='/home/mentee/analysis/ebm_dualbpl_shared_wrap_eval_20261117_v1',result='/home/mentee/results/CM/LA/EBM_e50_bs240_lr1e5_spoof_dualBPL_pair1_pull09_wrapeval_20261116_v1',shared='full'),
 'bona_uid':dict(root='/home/mentee/analysis/ebm_spoof_bonauid_pair1_pull09_e50_20261118_v1',result='/home/mentee/results/CM/LA/EBM_e50_bs240_lr1e5_spoof_bonaUIDBPL_pair1_pull09_wrapeval_20261118_v1',shared='shared_eval/full')}
COUNTS={'19LA':(71237,7355),'21LA':(148176,14816),'21DF':(533928,14869)}

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def load(model,phase,epoch):
    cfg=SOURCES[model]; root=Path(cfg['root']); result=Path(cfg['result'])
    assert (root/'run.exit').read_text().strip()=='0'
    if phase=='19LA':
        paths=[result/'stages'/f'eval_{epoch}_19LA_scores.csv']
        receipt=json.loads((result/'stages'/f'eval_{epoch}_19LA.json').read_text())
        assert receipt['status']=='complete' and receipt['scope']=='full'
        expected=receipt['eer_percent']; column='score_bona'
    else:
        shared=root/cfg['shared']; receipt=json.loads((shared/f'metrics_{phase}.json').read_text())
        assert receipt['status']=='complete' and receipt['scope']=='full'
        expected=receipt['metrics'][f'e{epoch}']['EER_percent']; column=f'e{epoch}'
        paths=[shared/f'{phase}_rank{i}.csv' for i in [0,1]]
        for path in paths: assert sha(path)==receipt['score_files'][path.name]['sha256']
    rows=[]
    for path in paths:
        with path.open() as f:
            rows.extend((int(r['index']),r['uid'],int(r['label']),float(r[column])) for r in csv.DictReader(f))
    rows.sort(); n,nb=COUNTS[phase]
    assert len(rows)==n and [r[0] for r in rows]==list(range(n)) and len({r[1] for r in rows})==n
    y=np.array([r[2] for r in rows]); scores=np.array([r[3] for r in rows])
    assert set(y)=={0,1} and int(y.sum())==nb and np.isfinite(scores).all()
    uid_digest=hashlib.sha256('\n'.join(f'{r[0]}|{r[1]}|{r[2]}' for r in rows).encode()).hexdigest()
    return y,scores,expected,dict(model=model,phase=phase,epoch=epoch,rows=n,bona=nb,uid_label_sha256=uid_digest,files={str(p):sha(p) for p in paths})

def roc(y,s):
    order=np.argsort(-s,kind='stable'); ys=y[order]; ss=s[order]
    ends=np.r_[np.flatnonzero(np.diff(ss)!=0),len(ss)-1]
    tp=np.r_[0,np.cumsum(ys)[ends]]; fp=np.r_[0,ends+1-np.cumsum(ys)[ends]]
    thresholds=np.r_[np.inf,ss[ends]]; fpr=fp/(len(y)-y.sum()); fnr=1-tp/y.sum(); d=fpr-fnr
    j=int(np.flatnonzero(d>=0)[0]); mix=-d[j-1]/(d[j]-d[j-1]); eer=100*(fpr[j-1]+mix*(fpr[j]-fpr[j-1]))
    near=int(np.argmin(abs(d))); matched=int(np.flatnonzero(fpr<=.005)[-1])
    return dict(eer=eer,threshold=float(thresholds[near]),matched_threshold=float(thresholds[matched]),source_matched_far=100*float(fpr[matched]))

def rates(y,s,threshold):
    accept=s>=threshold; nb=int(y.sum()); ns=len(y)-nb
    fn=int(((y==1)&~accept).sum()); fp=int(((y==0)&accept).sum())
    return dict(threshold=threshold,bona_count=nb,spoof_count=ns,bona_rejected=fn,spoof_accepted=fp,bona_rejection_percent=100*fn/nb,spoof_acceptance_percent=100*fp/ns)

def main():
    metrics=[]; errors=[]; transitions=[]; checks=[]
    for epoch in [40,50]:
        source={}; digest={}; predictions={}
        for phase in COUNTS:
            for model in SOURCES:
                y,s,expected,check=load(model,phase,epoch); op=roc(y,s)
                assert abs(op['eer']-expected)<1e-8,(model,phase,epoch,op['eer'],expected)
                check['independent_eer_error_pp']=abs(op['eer']-expected); checks.append(check)
                if phase in digest: assert digest[phase]==check['uid_label_sha256']
                else: digest[phase]=check['uid_label_sha256']
                if phase=='19LA': source[model]=op
                metrics.append(dict(model=model,phase=phase,epoch=epoch,eer_percent=op['eer']))
                for mode,t in [('fixed19LA_EER_nearest',source[model]['threshold']),('fixed19LA_FAR0.5',source[model]['matched_threshold']),('own_dataset_EER_nearest',op['threshold'])]:
                    errors.append(dict(model=model,phase=phase,epoch=epoch,mode=mode,source_matched_far_percent=source[model]['source_matched_far'],**rates(y,s,t)))
                    predictions[model,phase,mode]=(s>=t)!=y
            for mode in ['fixed19LA_EER_nearest','fixed19LA_FAR0.5']:
                a=predictions['bona_allcross',phase,mode]; b=predictions['bona_uid',phase,mode]
                for label in [0,1]:
                    mask=y==label
                    transitions.append(dict(epoch=epoch,phase=phase,mode=mode,label=label,new_errors=int((mask&~a&b).sum()),corrected=int((mask&a&~b).sum()),both_wrong=int((mask&a&b).sum())))
    # Second independent ROC implementation, with ties, checks the vectorized
    # operating-point code without reusing its indexing or interpolation logic.
    from sklearn.metrics import roc_curve
    for epoch in [40,50]:
        for phase in COUNTS:
            for model in SOURCES:
                y,s,expected,_=load(model,phase,epoch); fp,tp,_=roc_curve(y,s,drop_intermediate=False)
                independent=100*float(np.interp(0,fp+tp-1,fp))
                assert abs(independent-expected)<1e-8
    return dict(status='complete_verified',analysis='Completed normal-allcross vs normal-sameUID; spoof BPL all-cross in both',epochs=[40,50],metrics=metrics,class_errors=errors,paired_transitions=transitions,checks=checks,
                threshold_note='Each model calibrated separately on19LA eval, then fixed across21LA/DF; descriptive cross-domain diagnostic, not operational calibration',
                limits=['Single seed, no causal guarantee','EER-nearest uses a realizable discrete threshold; EER itself is linearly interpolated','FAR0.5 uses the most permissive source threshold that does not exceed0.5% spoof acceptance, respecting ties'],new_inference=0,new_training=0,new_spending_usd=0)

if __name__=='__main__': print(json.dumps(main(),allow_nan=False))
