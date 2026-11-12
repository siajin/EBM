"""Paired frozen-model attention/audio robustness analysis, no fitting."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from frozen_common import *
from frozen_r23_cache import load,CATALOG
from frozen_heads import Head,predict
from frozen_fast_bootstrap import fast_eers,verify
from audit_cached_core_results import independent_eer,independent_auc

OUT=DEFAULT_ROOT/'local_analysis/frozen_all_v1'
CONDITIONS=['R2-00','R2-10','R2-01','R2-11','R3-noise20','R3-noise10','R3-Opus32k_recompression']

def run(name):
    folder=OUT/'R2_R3'/name;folder.mkdir(parents=True,exist_ok=False);baseline=load(name,'R2-00');g0=baseline.geometry();y=baseline.rows.y.to_numpy();records=[];effects=[];paired=[];scores={};gate_changes={}
    v=panel('V');th=threshold(v.rows.y.to_numpy(),v.geometry()['cosine'])['threshold'];heads={}
    for seed in SEEDS:
        head=Head('dynamic',baseline.centroid,seed);obj=torch.load(OUT/'P1'/str(seed)/'dynamic/head_state.pt',weights_only=True);head.load_state_dict(obj['state_dict']);head.eval();heads[seed]=head
    native_gates={}
    with torch.no_grad():
        for seed,head in heads.items():native_gates[seed]=head.gate(torch.as_tensor(g0['unit_z'])).numpy()
    for cond in CONDITIONS:
        d=load(name,cond);assert d.rows.utterance_id.tolist()==baseline.rows.utterance_id.tolist();g=d.geometry();scores[cond]=g['cosine'];valid=g['valid']&g0['valid'];angles=np.full(len(y),np.nan);angles[valid]=np.degrees(np.arccos(np.clip(np.sum(g['direction'][valid]*g0['direction'][valid],axis=1),-1,1)))
        frame=d.rows[['utterance_id','y','dependency_group','small']].copy();frame['score']=g['cosine'];frame['score_delta']=g['cosine']-g0['cosine'];frame['direction_angle_degrees']=angles;frame['unit_z_L2_shift']=np.linalg.norm(g['unit_z']-g0['unit_z'],axis=1);frame['raw_norm_delta']=g['norm']-g0['norm']
        gate_changes[cond]=[]
        for seed,head in heads.items():
            with torch.no_grad():gate=head.gate(torch.as_tensor(g['unit_z'])).numpy()
            gd=np.linalg.norm(gate-native_gates[seed],axis=1);gate_changes[cond].append(gd);frame[f'gate_L2_change_{seed}']=gd;scores[f'{cond}_dynamic_{seed}']=predict(head,g['unit_z']);frame[f'dynamic_score_{seed}']=scores[f'{cond}_dynamic_{seed}']
        frame.to_parquet(folder/(cond+'_paired.parquet'),index=False)
    metric_checks=0
    for subset in ['small2000','pilot20000']:
        ix=np.flatnonzero(baseline.rows.small.to_numpy()) if subset=='small2000' else np.arange(len(y));yy=y[ix];groups=baseline.rows.iloc[ix].dependency_group.to_numpy();ug,codes=np.unique(groups,return_inverse=True)
        counts=np.random.default_rng(7301).multinomial(len(ug),np.full(len(ug),1/len(ug)),size=10000);bs={}
        for model,ss in scores.items():
            threshold_used=th
            if '_dynamic_' in model:
                seed=int(model.rsplit('_',1)[-1]);threshold_used=json.loads((OUT/'P1'/str(seed)/'dynamic/threshold.json').read_text())['threshold']
            ss=ss[ix];record=dict(dataset=name,subset=subset,condition=model,**metric(yy,ss,threshold_used));records.append(record)
            assert abs(independent_eer(yy,ss)-record['eer_percent'])<1e-9 and abs(independent_auc(yy,ss)-record['auroc'])<1e-12;metric_checks+=1
            bs[model]=fast_eers(yy,ss,codes,counts)
        for cond in CONDITIONS[1:]:
            delta=bs['R2-00']-bs[cond];effects.append(dict(dataset=name,subset=subset,baseline='R2-00',condition=cond,delta_EER_pp=eer(yy,scores['R2-00'][ix])-eer(yy,scores[cond][ix]),CI95=interval(delta),CI_family95=interval(delta,1-.05/6),draws=10000,groups=len(ug),fitted_on_interventions=False))
            if cond.startswith('R3'):
                deltas=np.mean([bs[f'R2-00_dynamic_{seed}']-bs[f'{cond}_dynamic_{seed}'] for seed in SEEDS],axis=0)
                effects.append(dict(dataset=name,subset=subset,baseline='R2-00_dynamic',condition=cond+'_dynamic',delta_EER_pp=float(np.mean([eer(yy,scores[f'R2-00_dynamic_{seed}'][ix])-eer(yy,scores[f'{cond}_dynamic_{seed}'][ix]) for seed in SEEDS])),CI95=interval(deltas),draws=10000,groups=len(ug),seed_average_not_ensemble=True))
        for cond in CONDITIONS:
            frame=pd.read_parquet(folder/(cond+'_paired.parquet')).iloc[ix]
            for measure in ['score_delta','direction_angle_degrees','unit_z_L2_shift','raw_norm_delta']:
                values=frame[measure].to_numpy();ok=np.isfinite(values)
                if not ok.any():continue
                total=np.bincount(codes[ok],weights=values[ok],minlength=len(ug));ns=np.bincount(codes[ok],minlength=len(ug));draw_values=(counts@total)/(counts@ns)
                paired.append(dict(dataset=name,subset=subset,condition=cond,measure=measure,mean=float(np.mean(values[ok])),median=float(np.median(values[ok])),CI95_mean=interval(draw_values),valid_rows=int(ok.sum())))
            gd=np.mean([a[ix] for a in gate_changes[cond]],axis=0);total=np.bincount(codes,weights=gd,minlength=len(ug));ns=np.bincount(codes,minlength=len(ug));paired.append(dict(dataset=name,subset=subset,condition=cond,measure='seed_mean_gate_L2_change',mean=float(gd.mean()),median=float(np.median(gd)),CI95_mean=interval((counts@total)/(counts@ns)),valid_rows=len(gd)))
    pd.DataFrame(records).to_csv(folder/'metrics.csv',index=False);save(folder/'paired_effects.json',effects);save(folder/'paired_geometry_gate_summary.json',paired)
    save(folder/'complete.json',dict(status='complete',dataset=name,conditions=CONDITIONS,small_n=2000,full_n=20000,paired_unique_utterances=20000,independent_metric_checks=metric_checks,bootstrap_algorithm_check=verify(),all_methods_and_thresholds_T_V_frozen=True,no_new_EBM_training=True,limits=['attention ablation may produce OOD hidden states','codec is additional compression of possibly already coded audio','small subset is nested, not independent replication','original source families unresolved']))
    print('R2/R3 analyzed',name,flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['DF_pilot','21LA_pilot'],required=True);o=p.parse_args()
    with threadpool_limits(2):run(o.dataset)
