"""Independent audit of completed P3 transfer fits, before internal CV ends."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from frozen_common import *
from frozen_probes import raw_features,inputs,RIDGES,VARIANTS
from audit_frozen_probes import numpy_score
from audit_cached_core_results import independent_eer,independent_auc

def main():
    out=DEFAULT_ROOT/'local_analysis/frozen_all_v1/P3_P4/A';assert (out/'complete.json').exists()
    data={n:panel(n) for n in ['T','V','DF_pilot','21LA_pilot']};raw={n:raw_features(d) for n,d in data.items()};checks=0;maximum=0.;metrics_checked=0
    for variant in VARIANTS[:5]:
        for seed in SEEDS:
            selection=json.loads((out/'selection'/f'{variant}_{seed}.json').read_text());choices=[];models={}
            for ridge in RIDGES:
                folder=out/'models'/variant/str(seed)/str(ridge);rec=json.loads((folder/'complete.json').read_text());hist=pd.read_csv(folder/'history.csv');obj=torch.load(folder/'model.pt',weights_only=False)
                best=int(hist[hist.validation_balanced_logloss<=hist.validation_balanced_logloss.min()+1e-8].iloc[0].epoch);assert rec['best_epoch']==best
                scaler=obj['scaler'];assert abs(scaler['n']['mean']-raw['T']['n'].mean())<1e-12 and abs(scaler['n']['std']-raw['T']['n'].std())<1e-12
                x=inputs(raw['V'],np.arange(len(data['V'].rows)),variant,scaler);s=numpy_score(obj['state_dict'],x);vl=balanced_ll(data['V'].rows.y.to_numpy(),expit(s));choices.append((vl,best,-ridge,ridge));models[ridge]=obj;checks+=1
            minimum=min(r[0] for r in choices);selected=min([r for r in choices if r[0]<=minimum+1e-8],key=lambda r:(r[1],r[2]));assert selection['ridge']==selected[3] and selection['epoch']==selected[1]
            obj=models[selected[3]]
            for name in ['DF_pilot','21LA_pilot']:
                pred=pd.read_parquet(out/name/'predictions.parquet');metrics=pd.read_csv(out/name/'metrics.csv');d=data[name];assert pred.utterance_id.tolist()==d.rows.utterance_id.tolist()
                x=inputs(raw[name],np.arange(len(d.rows)),variant,obj['scaler']);s=numpy_score(obj['state_dict'],x);err=float(np.max(np.abs(s-pred[f'{variant}_{seed}'])));assert err<1e-9;maximum=max(maximum,err)
                row=metrics[(metrics.variant==variant)&(metrics.seed==seed)].iloc[0];assert abs(independent_eer(d.rows.y.to_numpy(),s)-row.eer_percent)<1e-9 and abs(independent_auc(d.rows.y.to_numpy(),s)-row.auroc)<1e-12;metrics_checked+=1
    save(out/'verification.json',dict(status='pass',T_fit_V_selection_verified=True,metrics_checked=metrics_checked,candidate_fit_checks=checks,max_independent_numpy_score_error=maximum,EBM_retraining=False));print('P3 transfer audit pass',metrics_checked,checks,maximum)

if __name__=='__main__':
    with threadpool_limits(2):main()
