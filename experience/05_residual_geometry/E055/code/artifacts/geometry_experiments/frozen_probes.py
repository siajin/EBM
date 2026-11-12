"""Monotone scalar/residual probes and fold-local pre-LN incremental controls.

Both internal datasets receive independent 5x3 grouped CV. All transforms and
epoch/ridge selection occur inside the appropriate train/inner partitions.
"""
import argparse
import copy
import math
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits
from frozen_common import *
from frozen_heads import batches, ll_tensor

torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
VARIANTS=['scalar','direction_linear','noise_linear','direction_mlp','noise_mlp','direction_hstat','direction_noise2','direction_h','direction_noise160']
RIDGES=[1e-4,1e-2]

def raw_features(data):
    g=data.geometry();unit_noise=noise(data.rows,data.centroid,unit=True);unit_noise[~g['valid']]=0
    h=data.h.astype(float)
    return dict(s=g['cosine'],n=np.log(g['norm']),valid=g['valid'].astype(float),u=g['direction'],noise=unit_noise,h=h,
        hstat=np.c_[h.mean(1),np.log(np.sqrt(h.var(1))+1e-12)],noise2=noise(data.rows,data.centroid,dim=2,prefix='P4-noise2-v1|'),
        noise160=noise(data.rows,data.centroid,dim=160,prefix='P4-noise160-v1|'))

def fit_scaler(raw,ix,variant):
    names=['n']
    extra=variant.removeprefix('direction_')
    if extra in ['hstat','h','noise2','noise160']:names.append(extra)
    return {name:dict(mean=np.mean(raw[name][ix],axis=0),std=np.maximum(np.std(raw[name][ix],axis=0),1e-12)) for name in names}

def inputs(raw,ix,variant,scaler):
    n=(raw['n'][ix]-scaler['n']['mean'])/scaler['n']['std'];base=np.c_[raw['s'][ix],n,raw['valid'][ix]]
    if variant=='scalar':res=np.empty((len(ix),0))
    else:res=raw['noise' if variant.startswith('noise') else 'u'][ix]
    extra=variant.removeprefix('direction_')
    if extra in ['hstat','h','noise2','noise160']:
        res=np.c_[res,(raw[extra][ix]-scaler[extra]['mean'])/scaler[extra]['std']]
    return np.c_[base,res]

class Probe(nn.Module):
    def __init__(self,width,variant,seed):
        super().__init__();torch.manual_seed(seed);self.variant=variant
        self.a0=nn.Parameter(torch.tensor(math.log(math.expm1(CAL_A-1e-4))))
        self.intercept=nn.Parameter(torch.tensor(CAL_B));self.beta=nn.Parameter(torch.tensor(0.));self.eta=nn.Parameter(torch.tensor(0.))
        if width:
            if variant.endswith('_mlp'):
                self.fc1=nn.Linear(width,16);self.fc2=nn.Linear(16,1)
                nn.init.zeros_(self.fc1.bias);nn.init.zeros_(self.fc2.weight);nn.init.zeros_(self.fc2.bias)
            else:self.w=nn.Parameter(torch.zeros(width))
    def forward(self,x):
        s,n,v=x[:,:3].unbind(-1)
        logit=(F.softplus(self.a0)+1e-4)*s+self.intercept+self.beta*n+self.eta*v
        if hasattr(self,'w'):logit=logit+x[:,3:]@self.w
        if hasattr(self,'fc1'):logit=logit+self.fc2(F.gelu(self.fc1(x[:,3:]))).squeeze(-1)
        return logit
    def ridge(self):
        return sum((p*p).sum() for name,p in self.named_parameters() if 'bias' not in name and name!='intercept')

def predict(model,x):
    with torch.no_grad():return np.concatenate([model(torch.as_tensor(a)).numpy() for a in np.array_split(x,max(1,math.ceil(len(x)/4096)))])

def fit(raw,rows,ix,variant,seed,ridge,folder,val_raw=None,val_rows=None,val_ix=None,epochs=None):
    # Existing completed fits can be reused after a parent runner interruption.
    signature=dict(variant=variant,seed=seed,ridge=ridge,train_UID_sha256=hashlib.sha256('\n'.join(rows.iloc[ix].utterance_id).encode()).hexdigest(),
        val_UID_sha256=None if val_ix is None else hashlib.sha256('\n'.join(val_rows.iloc[val_ix].utterance_id).encode()).hexdigest(),fixed_epochs=epochs,script_sha256=digest(__file__))
    if (folder/'complete.json').exists():
        old=json.loads((folder/'complete.json').read_text());assert old['signature']==signature
        state=torch.load(folder/'model.pt',weights_only=False);m=Probe(state['width'],variant,seed);m.load_state_dict(state['state_dict'])
        return m,state['scaler'],pd.read_csv(folder/'history.csv'),old
    folder.mkdir(parents=True,exist_ok=False)
    scaler=fit_scaler(raw,ix,variant);xt=inputs(raw,ix,variant,scaler);yt=rows.iloc[ix].y.to_numpy();x=torch.as_tensor(xt);y=torch.as_tensor(yt,dtype=torch.float64)
    model=Probe(xt.shape[1]-3,variant,seed);assert np.max(np.abs(predict(model,xt)-(CAL_A*xt[:,0]+CAL_B)))<1e-10
    if val_ix is not None:
        xv=inputs(val_raw,val_ix,variant,scaler);yv=val_rows.iloc[val_ix].y.to_numpy()
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=0)
    history=[];best=float('inf');best_epoch=0;best_state=copy.deepcopy(model.state_dict());start=time.monotonic()
    for epoch in range((80 if epochs is None else epochs)+1):
        lossvalue=None
        if epoch:
            model.train();ls=[]
            for bi in batches(yt,seed,epoch):
                optimizer.zero_grad(set_to_none=True);loss=ll_tensor(y[bi],model(x[bi]))+ridge*model.ridge();assert torch.isfinite(loss)
                loss.backward();optimizer.step();ls.append(float(loss.detach()))
            lossvalue=float(np.mean(ls))
        model.eval()
        vl=balanced_ll(yv,expit(predict(model,xv))) if val_ix is not None else None
        history.append(dict(epoch=epoch,train_objective=lossvalue,validation_balanced_logloss=vl))
        if vl is not None and vl<best-1e-8:best=vl;best_epoch=epoch;best_state=copy.deepcopy(model.state_dict())
        if epochs is None and epoch-best_epoch>=10:break
    if epochs is None:model.load_state_dict(best_state)
    else:best_epoch=epochs
    torch.save(dict(width=xt.shape[1]-3,state_dict=model.state_dict(),scaler=scaler,variant=variant,seed=seed),folder/'model.pt')
    hist=pd.DataFrame(history);hist.to_csv(folder/'history.csv',index=False)
    rec=dict(status='complete',signature=signature,best_epoch=best_epoch,selected_validation_LL=best if np.isfinite(best) else None,seconds=time.monotonic()-start)
    save(folder/'complete.json',rec)
    return model,scaler,hist,rec

def fold_lock(rows,folder):
    path=folder/'fold_manifest.json'
    groups=rows.dependency_group.to_numpy();source=rows.source_corpus.astype(str).to_numpy() if 'source_corpus' in rows else np.full(len(rows),'asvspoof')
    strata=np.char.add(np.char.add(source.astype(str),'::'),rows.y.to_numpy().astype(str))
    outer=StratifiedGroupKFold(5,shuffle=True,random_state=11);folds=[];meta=[]
    for k,(train,test) in enumerate(outer.split(np.zeros(len(rows)),strata,groups)):
        assert not set(groups[train])&set(groups[test]);assert set(rows.iloc[train].y)==set(rows.iloc[test].y)=={0,1}
        inner=[];im=[]
        for j,(a,b) in enumerate(StratifiedGroupKFold(3,shuffle=True,random_state=12).split(np.zeros(len(train)),strata[train],groups[train])):
            aa,bb=train[a],train[b];assert not set(groups[aa])&set(groups[bb]);assert set(rows.iloc[aa].y)==set(rows.iloc[bb].y)=={0,1}
            inner.append((aa,bb));im.append(dict(fold=j,train_rows=aa,validation_rows=bb,train_n=len(aa),validation_n=len(bb),train_class_counts=rows.iloc[aa].y.value_counts().to_dict(),validation_class_counts=rows.iloc[bb].y.value_counts().to_dict(),train_sources=sorted(set(source[aa])),validation_sources=sorted(set(source[bb]))))
        folds.append((train,test,inner));meta.append(dict(fold=k,train_rows=train,test_rows=test,train_class_counts=rows.iloc[train].y.value_counts().to_dict(),test_class_counts=rows.iloc[test].y.value_counts().to_dict(),train_groups=len(set(groups[train])),test_groups=len(set(groups[test])),inner=im))
    record=dict(group='source::speaker',stratification='source::class',outer_seed=11,inner_seed=12,UID_order=rows.utterance_id.tolist(),folds=meta)
    if path.exists():assert json.loads(path.read_text())==json.loads(json.dumps(record,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else int(x)))
    else:save(path,record)
    return folds

def evaluate_predictions(rows,predictions,out,route):
    metrics_records=[]
    y=rows.y.to_numpy();table=rows[['utterance_id','y','dependency_group']].copy()
    for (variant,seed),score in predictions.items():
        table[f'{variant}_{seed}']=score;metrics_records.append(dict(route=route,variant=variant,seed=seed,**metric(y,score,logit=True)))
    table.to_parquet(out/'predictions.parquet',index=False);pd.DataFrame(metrics_records).to_csv(out/'metrics.csv',index=False)
    # Shared cluster draws; seed deltas are averaged, never ensembled.
    scores={f'{v}_{s}':a for (v,s),a in predictions.items()};bs,meta=bootstrap_scores(y,scores,rows.dependency_group.to_numpy())
    contrasts=[('scalar','direction_linear'),('noise_linear','direction_linear'),('scalar','direction_mlp'),('noise_mlp','direction_mlp'),('direction_linear','direction_hstat'),('direction_noise2','direction_hstat'),('direction_linear','direction_h'),('direction_noise160','direction_h')]
    effects=[]
    for a,b in contrasts:
        if (a,101) not in predictions or (b,101) not in predictions:continue
        delta=np.mean([bs[f'{a}_{s}']-bs[f'{b}_{s}'] for s in SEEDS],axis=0)
        ll_delta=[balanced_ll(y,expit(predictions[(a,s)]))-balanced_ll(y,expit(predictions[(b,s)])) for s in SEEDS]
        groups,codes=np.unique(rows.dependency_group.to_numpy(),return_inverse=True)
        counts=np.random.default_rng(7301).multinomial(len(groups),np.full(len(groups),1/len(groups)),size=10000)
        losses=[]
        for seed in SEEDS:
            def pointloss(logits):
                p=np.clip(expit(logits),1e-7,1-1e-7);return -y*np.log(p)-(1-y)*np.log1p(-p)
            diff=pointloss(predictions[(a,seed)])-pointloss(predictions[(b,seed)])
            vals=np.zeros(10000)
            for label in [0,1]:
                ii=y==label;num=np.bincount(codes[ii],weights=diff[ii],minlength=len(groups));den=np.bincount(codes[ii],minlength=len(groups));vals+=.5*(counts@num)/(counts@den)
            losses.append(vals)
        effects.append(dict(baseline=a,candidate=b,mean_delta_EER_pp=float(np.mean([eer(y,predictions[(a,s)])-eer(y,predictions[(b,s)]) for s in SEEDS])),EER_CI95=interval(delta),mean_balanced_LL_reduction=float(np.mean(ll_delta)),LL_CI95=interval(np.mean(losses,axis=0)),**meta))
    save(out/'paired_effects.json',effects)

def run_A(out):
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();data={n:panel(n) for n in ['T','V','DF_pilot','21LA_pilot']};raw={n:raw_features(d) for n,d in data.items()};t,v=data['T'],data['V'];ti=np.arange(len(t.rows));vi=np.arange(len(v.rows));predictions={n:{} for n in ['DF_pilot','21LA_pilot']}
    # P4 primary is dataset-internal; transfer P3 uses its five specified cases.
    for variant in VARIANTS[:5]:
        for seed in SEEDS:
            choices=[]
            for ridge in RIDGES:
                folder=out/'models'/variant/str(seed)/str(ridge)
                m,sc,hist,rec=fit(raw['T'],t.rows,ti,variant,seed,ridge,folder,raw['V'],v.rows,vi)
                val=balanced_ll(v.rows.y.to_numpy(),expit(predict(m,inputs(raw['V'],vi,variant,sc))))
                choices.append((val,rec['best_epoch'],-ridge,m,sc,ridge))
            minimum=min(r[0] for r in choices);best=min([r for r in choices if r[0]<=minimum+1e-8],key=lambda r:(r[1],r[2]));_,epoch,_,m,sc,ridge=best
            save(out/'selection'/f'{variant}_{seed}.json',dict(ridge=ridge,epoch=epoch,V_LL=best[0]))
            for name in predictions:
                predictions[name][(variant,seed)]=predict(m,inputs(raw[name],np.arange(len(data[name].rows)),variant,sc))
            print('PROBE A',variant,seed,ridge,epoch,flush=True)
    for name,preds in predictions.items():
        folder=out/name;folder.mkdir();evaluate_predictions(data[name].rows,preds,folder,'A_transfer')
    save(out/'complete.json',dict(status='complete',seconds=time.monotonic()-start))

def run_B(name,out):
    out.mkdir(parents=True,exist_ok=True);start=time.monotonic();d=panel(name);raw=raw_features(d);folds=fold_lock(d.rows,out);predictions={(v,s):np.full(len(d.rows),np.nan) for v in VARIANTS for s in SEEDS}
    for variant in VARIANTS:
        for seed in SEEDS:
            for fold,(train,test,inner) in enumerate(folds):
                root=out/'models'/variant/str(seed)/f'outer{fold}';choices=[]
                for ridge in RIDGES:
                    histories=[]
                    for j,(it,iv) in enumerate(inner):
                        _,_,history,_=fit(raw,d.rows,it,variant,seed,ridge,root/f'ridge{ridge}'/f'inner{j}',raw,d.rows,iv)
                        histories.append(history.validation_balanced_logloss.to_numpy())
                    common=min(map(len,histories));means=np.mean([h[:common] for h in histories],axis=0)
                    for epoch,val in enumerate(means):choices.append((float(val),epoch,-ridge,ridge))
                minimum=min(r[0] for r in choices);best=min([r for r in choices if r[0]<=minimum+1e-8],key=lambda r:(r[1],r[2]));val,epoch,_,ridge=best
                save(root/'selection.json',dict(inner_mean_LL=val,ridge=ridge,epochs=epoch,selection_uses_outer_test=False,all_candidates=[dict(LL=r[0],epoch=r[1],ridge=r[3]) for r in choices]))
                m,sc,_,_=fit(raw,d.rows,train,variant,seed,ridge,root/'refit',epochs=epoch)
                predictions[(variant,seed)][test]=predict(m,inputs(raw,test,variant,sc))
                print('PROBE B',name,variant,seed,fold,'epoch',epoch,flush=True)
            assert np.isfinite(predictions[(variant,seed)]).all()
            pd.DataFrame({'utterance_id':d.rows.utterance_id,'score':predictions[(variant,seed)]}).to_parquet(out/f'{variant}_{seed}_OOF.parquet',index=False)
    evaluate_predictions(d.rows,predictions,out,'B_internal_diagnostic')
    save(out/'complete.json',dict(status='complete',seconds=time.monotonic()-start,outer_folds=5,inner_folds=3,variants=VARIANTS,seeds=SEEDS,new_EBM_training=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--route',choices=['A','B'],required=True);p.add_argument('--dataset',choices=['DF_pilot','21LA_pilot']);p.add_argument('--output',default=str(DEFAULT_ROOT/'local_analysis/frozen_all_v1/P3_P4'));o=p.parse_args()
    with threadpool_limits(2):
        if o.route=='A':run_A(Path(o.output)/'A')
        else:
            assert o.dataset;run_B(o.dataset,Path(o.output)/'B'/o.dataset)
