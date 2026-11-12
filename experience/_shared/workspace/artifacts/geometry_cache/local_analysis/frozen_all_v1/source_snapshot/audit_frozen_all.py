"""Independent numeric checks on completed portions of the actual experiment."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import ndtr
from threadpoolctl import threadpool_limits
from frozen_common import *
from audit_cached_core_results import independent_eer, independent_auc

def audit(out):
    results=[];checked_metrics=0;max_error=0.;data={n:panel(n) for n in ['T','V','DF_pilot','21LA_pilot']};geo={n:d.geometry() for n,d in data.items()}
    def check(name,condition,**details):
        assert condition,(name,details)
        results.append(dict(name=name,status='pass',**details))
    def check_metric(y,s,record):
        nonlocal checked_metrics,max_error
        a=independent_eer(y,s);b=independent_auc(y,s);err=max(abs(a-record['eer_percent']),abs(b-record['auroc']))
        assert err<1e-9,(a,b,record);checked_metrics+=1;max_error=max(err,max_error)
    # The bootstrap helper receives independent tied/full-ROC edge checks.
    for seed in range(8):
        rng=np.random.default_rng(seed);y=np.r_[np.ones(13),np.zeros(39)];s=rng.integers(-2,3,len(y)).astype(float);codes=np.arange(len(y))%7;counts=rng.integers(1,5,(4,7))
        got=weighted_eers(y,s,codes,counts)
        for w,value in zip(counts,got):
            yy=np.repeat(y,w[codes]);ss=np.repeat(s,w[codes]);check('weighted_full_tied_ROC',abs(value-independent_eer(yy,ss))<1e-9)
    native=json.loads((out/'native_baselines.json').read_text())
    for name,record in native['datasets'].items():check_metric(data[name].rows.y.to_numpy(),geo[name]['cosine'],record)
    check('default_evaluation_equal_sizes',all(len(data[n].rows)==20000 and sum(data[n].rows.y)==2000 for n in ['DF_pilot','21LA_pilot']))
    if (out/'heads_complete.json').exists():
        import torch
        table=pd.read_csv(out/'head_metrics.csv');executed_hash=json.loads((out/'head_protocol_lock.json').read_text())['script_sha256']
        check('head_source_unchanged',digest(Path(__file__).with_name('frozen_heads.py'))==executed_hash)
        for kind in ['static','dynamic','mlp','metric']:
            for seed in SEEDS:
                folder=out/('P2' if kind=='metric' else 'P1')/str(seed)/kind;obj=torch.load(folder/'head_state.pt',weights_only=True);state={k:v.numpy() for k,v in obj['state_dict'].items()}
                hist=pd.read_csv(folder/'train_history.csv');sel=json.loads((folder/'V_selection.json').read_text());minimum=hist.V_balanced_logloss.min();expected=int(hist[hist.V_balanced_logloss<=minimum+1e-8].iloc[0].epoch)
                check('V_epoch0_and_selection',sel['selected_epoch']==expected and hist.iloc[0].epoch==0,model=kind,seed=seed)
                for name in ['V','DF_pilot','21LA_pilot']:
                    d=data[name];x=geo[name]['unit_z'];c=d.centroid
                    if kind=='metric':
                        a=np.exp(2*np.tanh(state['v']));a*=160/a.sum();s=1-.5*np.sum((x-c)**2*a,axis=1)
                    else:
                        if kind=='static':xx=(1+.5*np.tanh(state['v']))*x
                        else:
                            h=x@state['fc1.weight'].T+state['fc1.bias'];ff=(h*ndtr(h))@state['fc2.weight'].T+state['fc2.bias']
                            xx=(1+.5*np.tanh(ff))*x if kind=='dynamic' else x+.5/np.sqrt(160)*np.tanh(ff)
                        s=xx/np.linalg.norm(xx,axis=1,keepdims=True)@c
                    saved=pd.read_parquet(folder/(name+'_scores.parquet'));check('head_UID_order',saved.utterance_id.tolist()==d.rows.utterance_id.tolist())
                    err=float(np.max(np.abs(s-saved.score.to_numpy())));check('independent_numpy_head_score',err<1e-10,maximum_error=err,model=kind,seed=seed,dataset=name)
                    record=table[(table.model==kind)&(table.seed==seed)&(table.dataset==name)].iloc[0];check_metric(d.rows.y.to_numpy(),s,record)
        effects=json.loads((out/'head_paired_effects.json').read_text());check('P1_P2_five_comparisons_per_dataset',len(effects)==10 and all(e['draws']==10000 and len(e['CI99'])==2 for e in effects))
    if (out/'f2_complete.json').exists():
        summary=json.loads((out/'F2/summary.json').read_text())
        for row in summary:
            name,tag=row['dataset'],row['condition'];m=pd.read_parquet(out/'F2'/(name+'_'+tag+'_matches.parquet'));d=data[name];g=geo[name]
            bi=m.bona_index.to_numpy();si=m.spoof_index.to_numpy();ds,dn=(.02,.002) if tag=='primary' else (.05,.005)
            check('F2_no_replacement',len(set(bi))==len(bi) and len(set(si))==len(si))
            check('F2_class_and_scalar_calipers',np.all(d.rows.y.to_numpy()[bi]==1) and np.all(d.rows.y.to_numpy()[si]==0) and np.all(np.abs(g['cosine'][bi]-g['cosine'][si])<=ds) and np.all(np.abs(np.log(g['norm'][bi]/g['norm'][si]))<=dn))
            columns=['source_corpus','speaker_group','codec'] if name=='DF_pilot' else ['speaker_group','codec','transmission']
            check('F2_matching_cells',all(np.array_equal(d.rows.iloc[bi][c].to_numpy(),d.rows.iloc[si][c].to_numpy()) for c in columns))
            check('F2_low_overlap_not_negative_evidence',(len(m)>=200 and m.dependency_group.nunique()>=10) or row['status']=='inconclusive_low_overlap')
    if (out/'f3_complete.json').exists():
        reg=json.loads((out/'F3/metric_registry.json').read_text());check('F3_V_ties_identity',reg['selected']=={'diagonal':'diagonal_1.0','full':'full_1.0'})
        table=pd.read_csv(out/'F3/condition_effects.csv')
        for name in ['DF_pilot','21LA_pilot']:
            scores=pd.read_parquet(out/'F3'/(name+'_scores.parquet'));d=data[name]
            for c in [c for c in scores if c not in ['utterance_id','y','dependency_group']]:
                rec=table[(table.dataset==name)&(table.model==c)&(table.condition=='all')].iloc[0];check_metric(d.rows.y.to_numpy(),scores[c].to_numpy(),rec)
    if (out/'f4_complete.json').exists():
        table=pd.read_csv(out/'F4/vector_variance.csv')
        for _,r in table.iterrows():
            rows=data[r.dataset].rows.copy();rows['padding_bin']=np.digitize(rows.padding_fraction,[0,.25,.5],right=True);sel=rows.y==r.y
            if r.stratum!='all':
                col,value=r.stratum.split(':',1);sel&=rows[col].astype(str)==value
            ii=rows.index[sel].to_numpy();u=geo[r.dataset]['direction'][ii];labels=rows.iloc[ii][r.factor].astype(str).to_numpy()
            total=float(np.sum(u*u)-len(u)*np.sum(u.mean(0)**2));within=sum(float(np.sum(u[labels==v]**2)-sum(labels==v)*np.sum(u[labels==v].mean(0)**2)) for v in np.unique(labels))
            if total>1e-12:check('F4_ANOVA_decomposition',abs((1-within/total)-r.eta_squared)<1e-9)
    if (out/'N0_complete.json').exists():
        masks=json.loads((out/'F1/masks.json').read_text());table=pd.read_csv(out/'F1/mask_curves.csv')
        for basis in ['original','511','512','513']:
            q=np.load(out/'F1'/(basis+'_orthogonal.npy'));check('orthogonal_basis',np.max(np.abs(q.T@q-np.eye(160)))<1e-10)
            for name in ['V','DF_pilot','21LA_pilot']:
                d=data[name];z=d.z.astype(float)@q;c=d.centroid@q;frame=pd.read_parquet(out/'F1'/(basis+'_'+name+'_scores.parquet'))
                check('F1_complete_panel_count',len(masks[basis])==(169 if basis=='original' else 43))
                for model,ix in masks[basis].items():
                    # Independent full-size zero-mask implementation.
                    mask=np.zeros(160);mask[ix]=1;zz=z*mask;cc=c*mask;s=(zz@cc)/(np.linalg.norm(zz,axis=1)*np.linalg.norm(cc))
                    check('F1_zero_mask_score',np.max(np.abs(s-frame[model]))<1e-10)
                    rec=table[(table.basis.astype(str)==basis)&(table.dataset==name)&(table['mask']==model)].iloc[0];check_metric(d.rows.y.to_numpy(),s,rec)
        effects=json.loads((out/'F1/paired_effects.json').read_text());check('F1_all_basis_primary_intervals',len(effects)==16 and all(e['draws']==10000 and e['mask_seed_resampling'] for e in effects))
    report=dict(status='pass_within_completed_scope',checked_metric_rows=checked_metrics,maximum_independent_metric_error=max_error,checks=len(results),completed_markers=[p.name for p in out.glob('*complete.json')],details=results,remaining_work_is_not_claimed_complete=True)
    save(out/'verification.json',report);print(json.dumps({k:v for k,v in report.items() if k!='details'}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(DEFAULT_ROOT/'local_analysis/frozen_all_v1'));o=p.parse_args()
    with threadpool_limits(2):audit(Path(o.output))
