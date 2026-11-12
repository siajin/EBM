"""Actual T/V fitting of bounded gates and diagonal metrics, EBM held fixed."""
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
from threadpoolctl import threadpool_limits
from frozen_common import *

torch.set_num_threads(2)
torch.set_default_dtype(torch.float64)

def batches(y, seed, epoch):
    pools=[np.flatnonzero(y==i) for i in [1,0]]
    rng=np.random.default_rng(np.random.SeedSequence([seed,epoch,9101]))
    pools=[rng.permutation(p) for p in pools];cursors=[0,0]
    def take(j,n):
        pieces=[]
        while n:
            left=min(n,len(pools[j])-cursors[j]);pieces.append(pools[j][cursors[j]:cursors[j]+left])
            cursors[j]+=left;n-=left
            if cursors[j]==len(pools[j]):pools[j]=rng.permutation(pools[j]);cursors[j]=0
        return np.concatenate(pieces)
    for _ in range(max(math.ceil(len(pools[0])/16),math.ceil(len(pools[1])/144))):
        yield np.r_[take(0,16),take(1,144)]

class Head(nn.Module):
    def __init__(self,kind,c,seed):
        super().__init__();self.kind=kind;self.register_buffer('c',torch.as_tensor(c).clone())
        torch.manual_seed(seed)
        if kind in ['static','metric']:self.v=nn.Parameter(torch.zeros(160))
        elif kind in ['dynamic','mlp']:
            self.fc1=nn.Linear(160,16);self.fc2=nn.Linear(16,160)
            nn.init.zeros_(self.fc1.bias);nn.init.zeros_(self.fc2.weight);nn.init.zeros_(self.fc2.bias)
    def gate(self,x):
        if self.kind=='static':return (1+.5*torch.tanh(self.v)).expand_as(x)
        return 1+.5*torch.tanh(self.fc2(F.gelu(self.fc1(x))))
    def weights(self):
        a=torch.exp(2*torch.tanh(self.v));return 160*a/a.sum()
    def transform(self,x):
        if self.kind in ['static','dynamic']:return self.gate(x)*x
        if self.kind=='mlp':return x+(.5/math.sqrt(160))*torch.tanh(self.fc2(F.gelu(self.fc1(x))))
        return x
    def forward(self,x):
        if self.kind=='metric':return 1-.5*((x-self.c)**2*self.weights()).sum(-1)
        return F.normalize(self.transform(x),dim=-1)@self.c
    def penalty(self,x):
        if self.kind=='metric':return ((self.weights()-1)**2).mean()
        return ((self.transform(x)-x)**2).sum(-1).mean()

def oc(score,y):
    return (F.softplus(20*torch.where(y==1,.9-score,score-.2))*torch.where(y==1,5.,5/9)).mean()

def ll_tensor(y,logits):
    loss=F.binary_cross_entropy_with_logits(logits,y,reduction='none')
    return .5*(loss[y==1].mean()+loss[y==0].mean())

def predict(head,x):
    with torch.no_grad():return np.concatenate([head(torch.as_tensor(a)).numpy() for a in np.array_split(x,max(1,math.ceil(len(x)/4096)))])

def fit_head(kind,seed,xt,yt,xv,yv,c,folder):
    folder.mkdir(parents=True,exist_ok=False)
    head=Head(kind,c,seed);x=torch.as_tensor(xt);y=torch.as_tensor(yt,dtype=torch.float64)
    initial=predict(head,xv);assert np.max(np.abs(initial-xv@c))<1e-10
    assert sum(p.numel() for p in head.parameters())=={'static':160,'dynamic':5296,'mlp':5296,'metric':160}[kind]
    optimizer=torch.optim.Adam(head.parameters(),lr=1e-3,weight_decay=0)
    best=float('inf');best_epoch=0;best_state=None;history=[];gradient_record=[]
    start=time.monotonic()
    for epoch in range(81):
        objective=None
        if epoch:
            objectives=[]
            head.train()
            for step,ix in enumerate(batches(yt,seed,epoch)):
                xx,yy=x[ix],y[ix];optimizer.zero_grad(set_to_none=True)
                score=head(xx);loss=oc(score,yy)+.01*head.penalty(xx)
                assert torch.isfinite(loss)
                if epoch==1 and step<2:
                    balanced=.5*(F.softplus(20*(.9-score[yy==1])).mean()+F.softplus(20*(score[yy==0]-.2)).mean())
                    assert abs(float((oc(score,yy)-balanced).detach()))<1e-10
                loss.backward()
                if epoch==1 and step<2:
                    gradient_record.append({n:float(p.grad.norm()) for n,p in head.named_parameters()})
                optimizer.step();objectives.append(float(loss.detach()))
            objective=float(np.mean(objectives))
        head.eval();s=predict(head,xv);val=balanced_ll(yv,expit(CAL_A*s+CAL_B))
        history.append(dict(epoch=epoch,train_objective=objective,V_balanced_logloss=val,V_EER_percent=eer(yv,s)))
        if val<best-1e-8:best=val;best_epoch=epoch;best_state=copy.deepcopy(head.state_dict())
        if epoch-best_epoch>=10:break
    head.load_state_dict(best_state)
    assert gradient_record and sum(gradient_record[0].values())>0
    torch.save({'kind':kind,'seed':seed,'state_dict':best_state,'checkpoint_sha256':CHECKPOINT},folder/'head_state.pt')
    pd.DataFrame(history).to_csv(folder/'train_history.csv',index=False)
    save(folder/'V_selection.json',dict(kind=kind,seed=seed,selected_epoch=best_epoch,V_balanced_logloss=best,gradient_first_two_steps=gradient_record,fit_seconds=time.monotonic()-start,original_EBM_frozen=True,centroid_frozen=True))
    return head

def gate_scores(head,xt,gt,rows,x):
    with torch.no_grad():
        tgate=np.concatenate([head.gate(torch.as_tensor(a)).numpy() for a in np.array_split(xt,8)])
        gate=np.concatenate([head.gate(torch.as_tensor(a)).numpy() for a in np.array_split(x,8)])
    # Conditioning bins are learned on T only; labels are never read here.
    se=np.quantile(gt['cosine'],np.linspace(0,1,11)[1:-1]);ne=np.quantile(np.log(gt['norm']),np.linspace(0,1,6)[1:-1])
    s=x@head.c.numpy();norm=rows.raw_norm.to_numpy()
    cell=np.digitize(s,se)*5+np.digitize(np.log(norm),ne)
    moved=gate.copy();rng=np.random.default_rng(7302);changed=np.zeros(len(x),bool)
    for val in np.unique(cell):
        ix=np.flatnonzero(cell==val)
        if len(ix)<2:continue
        # A random cyclic shift has no fixed points, including 2-row cells.
        order=rng.permutation(ix);donors=np.roll(order,1);moved[order]=gate[donors];changed[order]=True
    c=head.c.numpy()
    def score(g):
        xx=x*g;return xx/np.linalg.norm(xx,axis=1,keepdims=True)@c
    return {'identity':x@c,'T_mean_gate':score(tgate.mean(0)),'actual_gate':score(gate),'within_T_bins_shuffled_gate':score(moved)},gate,changed

def run(output):
    start=time.monotonic();out=Path(output);(out/'P1').mkdir(parents=True,exist_ok=False);(out/'P2').mkdir(exist_ok=False)
    save(out/'head_protocol_lock.json',dict(spec_version='frozen-model-spec-v1.2-size-matched-21LA',script_sha256=digest(__file__),seeds=SEEDS,fit='T',select='V',evaluate=['DF_pilot','21LA_pilot'],all_original_EBM_parameters_frozen=True,centroid_frozen=True,epoch0_included=True,maximum_epochs=80,patience=10,learning_rate=.001,regularizer=.01,calibration=[CAL_A,CAL_B]))
    datasets={name:panel(name) for name in ['T','V','DF_pilot','21LA_pilot','21LA_eval']}
    geo={name:d.geometry() for name,d in datasets.items()};t,v=datasets['T'],datasets['V'];xt,xv=geo['T']['unit_z'],geo['V']['unit_z'];yt,yv=t.rows.y.to_numpy(),v.rows.y.to_numpy()
    scores={name:{'native':geo[name]['cosine']} for name in datasets if name!='T'};records=[];heads={}
    for seed in SEEDS:
        for kind in ['static','dynamic','mlp','metric']:
            print('FIT',kind,seed,flush=True)
            folder=out/('P2' if kind=='metric' else 'P1')/str(seed)/kind
            head=fit_head(kind,seed,xt,yt,xv,yv,t.centroid,folder);heads[(kind,seed)]=head
            vs=predict(head,xv);th=threshold(yv,vs)
            for name,d in datasets.items():
                if name=='T':continue
                score=predict(head,geo[name]['unit_z']);scores[name][f'{kind}_{seed}']=score
                f=d.rows[['utterance_id','y','dependency_group']].copy();f['score']=score;f.to_parquet(folder/(name+'_scores.parquet'),index=False)
                records.append(dict(model=kind,seed=seed,dataset=name,**metric(d.rows.y.to_numpy(),score,th['threshold'])))
                if kind=='dynamic' and name!='21LA_eval':
                    interventions,g,changed=gate_scores(head,xt,geo['T'],d.rows,geo[name]['unit_z'])
                    gs=d.rows[['utterance_id']].copy();gs['min']=g.min(1);gs['max']=g.max(1);gs['mean']=g.mean(1);gs['std']=g.std(1);gs['L2_shift']=np.linalg.norm((g-1)*geo[name]['unit_z'],axis=1);gs['shuffled']=changed
                    assert gs.L2_shift.max()<=.5+1e-10 and g.min()>=.5 and g.max()<=1.5
                    gs.to_parquet(folder/(name+'_gate_statistics.parquet'),index=False)
                    ir=[]
                    for condition,ss in interventions.items():
                        ir.append(dict(condition=condition,changed_fraction=float(changed.mean()),**metric(d.rows.y.to_numpy(),ss,th['threshold'])))
                    pd.DataFrame(ir).to_csv(folder/(name+'_gate_interventions.csv'),index=False)
            save(folder/'threshold.json',th)
            if kind=='metric':
                a=head.weights().detach().numpy();assert abs(a.sum()-160)<1e-10 and a.max()/a.min()<=np.exp(4)+1e-10
                save(folder/'metric_spectrum.json',dict(weights=a,min=a.min(),max=a.max(),trace=a.sum(),condition=a.max()/a.min(),identity_L2=np.linalg.norm(a-1)))
            print('SELECT',kind,seed,json.loads((folder/'V_selection.json').read_text())['selected_epoch'],flush=True)
    pd.DataFrame(records).to_csv(out/'head_metrics.csv',index=False)
    effects=[]
    for name in ['DF_pilot','21LA_pilot']:
        d=datasets[name];bs,meta=bootstrap_scores(d.rows.y.to_numpy(),scores[name],d.rows.dependency_group.to_numpy())
        contrasts=[('native','dynamic'),('static','dynamic'),('mlp','dynamic'),('native','metric'),('static','metric')]
        for base,cand in contrasts:
            vals=[];pts=[]
            for seed in SEEDS:
                bn=base if base=='native' else f'{base}_{seed}';cn=f'{cand}_{seed}'
                vals.append(bs[bn]-bs[cn]);pts.append(eer(d.rows.y.to_numpy(),scores[name][bn])-eer(d.rows.y.to_numpy(),scores[name][cn]))
            delta=np.mean(vals,axis=0)
            effects.append(dict(dataset=name,baseline=base,candidate=cand,seed_deltas_pp=pts,mean_delta_EER_pp=float(np.mean(pts)),CI95=interval(delta),CI99=interval(delta,.99),**meta))
        np.savez_compressed(out/(name+'_head_bootstrap.npz'),**bs)
    save(out/'head_paired_effects.json',effects)
    save(out/'heads_complete.json',dict(status='complete',seconds=time.monotonic()-start,fits=12,all_predictions_frozen_before_external_expansion=True,new_EBM_training=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(DEFAULT_ROOT/'local_analysis/frozen_all_v1'));o=p.parse_args()
    with threadpool_limits(2):run(o.output)
