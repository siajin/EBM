"""Independent NumPy/SciPy formulas against saved cache-only follow-up outputs."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import torch
from scipy.linalg import lstsq
from scipy.special import erf,expit
from sklearn.metrics import roc_curve,roc_auc_score
from threadpoolctl import threadpool_limits
from frozen_common import panel,DEFAULT_ROOT,save
from frozen_r23_cache import load

OUT=DEFAULT_ROOT/'local_analysis/cache_hypotheses_20261106_v2'
OLD=DEFAULT_ROOT/'local_analysis/frozen_all_v1'
CONDS=['R2-00','R3-noise20','R3-noise10','R3-Opus32k_recompression']
VARIANTS=['scalar','direction_linear','direction_mlp','direction_h']

def geometry(z,c):
    z=z.astype(float);n=np.sqrt(np.sum(z*z,axis=1));x=z/n[:,None];s=np.sum(x*c,axis=1)
    r=x-s[:,None]*c;rho=np.sqrt(np.sum(r*r,axis=1));ok=rho>=1e-4
    u=np.zeros_like(r);u[ok]=r[ok]/rho[ok,None]
    return n,s,u,rho,ok

def forward(state,data,index):
    w={k:v.detach().numpy() for k,v in state['state_dict'].items()};n,s,u,rho,ok=geometry(data.z[index],data.centroid)
    sc=state['scaler'];logit=(np.logaddexp(0,w['a0'])+1e-4)*s+w['intercept']+w['beta']*(np.log(n)-sc['n']['mean'])/sc['n']['std']+w['eta']*ok
    res=u
    if state['variant']=='direction_h':res=np.c_[u,(data.h[index].astype(float)-sc['h']['mean'])/sc['h']['std']]
    if 'w' in w:logit=logit+res@w['w']
    if 'fc1.weight' in w:
        h=res@w['fc1.weight'].T+w['fc1.bias'];h=.5*h*(1+erf(h/np.sqrt(2)))
        logit=logit+(h@w['fc2.weight'].T+w['fc2.bias']).ravel()
    return logit

def eer(y,s):
    fp,tp,_=roc_curve(y,s,drop_intermediate=False);d=fp+tp-1;j=int(np.searchsorted(d,0))
    if d[j]==0:return float(100*fp[j])
    f=-d[j-1]/(d[j]-d[j-1]);return float(100*(fp[j-1]+f*(fp[j]-fp[j-1])))

def run():
    assert json.loads((OUT/'complete.json').read_text())['status']=='complete'
    maximum=0.;points=0;models=0;metric_checks=0;ols_error=0.;decomposition_error=0.
    for ds in ['DF_pilot','21LA_pilot']:
        ref=panel(ds);raw={}
        for cond in CONDS:
            d=load(ds,cond);pos=pd.Index(d.rows.utterance_id).get_indexer(ref.rows.utterance_id)
            assert (pos>=0).all();d.z=d.z[pos];d.h=d.h[pos];d.rows=d.rows.iloc[pos].reset_index(drop=True);raw[cond]=d
        folds=json.loads((OLD/'P3_P4/B'/ds/'fold_manifest.json').read_text())['folds']
        pred=pd.read_parquet(OUT/f'H1_{ds}_saved_readout_predictions.parquet')
        assert pred.utterance_id.tolist()==ref.rows.utterance_id.tolist()
        for fold in folds:
            test=fold['test_rows'];ix=np.asarray(sorted(test,key=lambda i:hashlib.sha256(('independent-check|'+ref.rows.iloc[i].utterance_id).encode()).digest())[:64])
            for variant in VARIANTS:
                for seed in [101,102,103]:
                    st=torch.load(OLD/'P3_P4/B'/ds/'models'/variant/str(seed)/f"outer{fold['fold']}/refit/model.pt",weights_only=False);models+=1
                    for cond in CONDS:
                        v=forward(st,raw[cond],ix);truth=pred[f'{variant}__{seed}__{cond}'].to_numpy()[ix]
                        err=float(np.max(np.abs(v-truth)));assert err<1e-8,(ds,variant,cond,err)
                        maximum=max(maximum,err);points+=len(ix)
                    if variant=='direction_h' and seed==101:
                        d=raw['R2-00'];n,s,u,rho,ok=geometry(d.z[ix],d.centroid);hh=d.h[ix].astype(float)
                        mu=hh.mean(1);sig=np.sqrt(hh.var(1)+1e-12);q=(hh-mu[:,None])/sig[:,None]
                        w={k:v.detach().numpy() for k,v in st['state_dict'].items()};sc=st['scaler']
                        scalar=(np.logaddexp(0,w['a0'])+1e-4)*s+w['intercept']+w['beta']*(np.log(n)-sc['n']['mean'])/sc['n']['std']+w['eta']*ok
                        wh=w['w'][160:]/sc['h']['std']
                        decomp=scalar+u@w['w'][:160]+mu*wh.sum()+sig*(q@wh)-sc['h']['mean']@wh
                        decomposition_error=max(decomposition_error,float(np.max(np.abs(decomp-forward(st,d,ix)))))
        metrics=pd.read_csv(OUT/f'H1_{ds}_readout_metrics.csv');y=ref.rows.y.to_numpy()
        for row in metrics.itertuples():
            v=pred[f'{row.variant}__{row.seed}__{row.condition}'].to_numpy()
            assert abs(eer(y,v)-row.eer_percent)<1e-9
            assert abs(roc_auc_score(y,v)-row.auroc)<1e-12;metric_checks+=2
        h4=json.loads((OUT/f'H4_{ds}_summary.json').read_text())
        for rec in h4['regressions']:
            f=pd.read_parquet(OUT/f"H4_{ds}_{rec['condition']}_rows.parquet")
            m=f.valid.to_numpy()&(f.unit_z_shift.to_numpy()>1e-10)&(f.direction_chord.to_numpy()>1e-10)&(f.native_radius.to_numpy()>1e-6)
            strata=f.source_corpus.astype(str)+'::'+f.y.astype(str)
            X=np.c_[np.log(f.native_radius[m]),np.log(f.unit_z_shift[m]),np.column_stack([(strata[m]==n).to_numpy() for n in rec['source_class_strata']])]
            b=lstsq(X,np.log(f.direction_chord[m]),lapack_driver='gelsy')[0]
            er=abs(float(b[0])-rec['radius_log_coefficient']);assert er<1e-10;ols_error=max(ols_error,er)
    d=panel('T');a=torch.tensor(d.z.astype(float),requires_grad=True);c=torch.tensor(d.centroid);s=torch.nn.functional.normalize(a,dim=-1)@c
    y=d.rows.y.to_numpy();b=y==1;t=y==0
    loss=.5*(torch.nn.functional.softplus(20*(.9-s[b])).mean()+torch.nn.functional.softplus(20*(s[t]-.2)).mean());loss.backward()
    norm=torch.linalg.vector_norm(a.grad,dim=1).detach().numpy();h3=json.loads((OUT/'H3_OC_gradients.json').read_text())
    gradient_error=0.
    for rec in h3['class_summaries']:
        if rec['dataset']!='T':continue
        m=y==rec['y'];value=float(norm[m].mean()*2*m.sum());er=abs(value-rec['raw_z_gradient']['mean']);assert er<1e-11;gradient_error=max(gradient_error,er)
    assert decomposition_error<1e-10
    result=dict(status='pass',models_checked=models,independent_NumPy_logit_points=points,max_logit_error=maximum,
        independent_EER_AUROC_checks=metric_checks,independent_exact_h_decomposition_max_error=decomposition_error,
        independent_QR_radius_coefficients_checked=6,max_radius_coefficient_error=ols_error,
        independent_T_autograd_class_mean_norm_max_error=gradient_error,
        verifier_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Independent formulas and aggregation, no refitting; statistical assumptions and new-method efficacy remain unproven.')
    save(OUT/'independent_verification.json',result);print(json.dumps(result))

if __name__=='__main__':
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    with threadpool_limits(2):run()
