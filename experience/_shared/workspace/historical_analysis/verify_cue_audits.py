from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata
assert Path('/.dockerenv').exists()
root=Path('/results/analysis'); checks={}
for folder,file in [('EBM80_21DF_pooling_audit','pooling_scores.npz'),('EBM80_21DF_quiet_texture','quiet_texture_scores.npz')]:
    d=np.load(root/folder/file); q=json.loads((root/folder/'summary.json').read_text()); y=d['labels']
    assert len(y)==2000 and y.sum()==1000 and np.isfinite(d['scores']).all()
    checks[folder]={}
    for j,name in enumerate(d['names']):
        score=d['scores'][:,j]; sorted_scores=np.sort(score)
        assert sorted_scores[999]<sorted_scores[1000]
        threshold=(float(sorted_scores[999])+float(sorted_scores[1000]))/2
        fn=int(((y==1)&(score<threshold)).sum()); fp=int(((y==0)&(score>=threshold)).sum())
        assert fn==fp and abs(fn/10-q['overall'][name]['sample_eer_pct'])<1e-6
        checks[folder][str(name)]={'independent_eer_pct':fn/10,'equal_errors_per_class':fn}
p=root/'EBM80_21DF_cue_audit'; d=np.load(p/'cue_predictions.npz',allow_pickle=True); q=json.loads((p/'summary.json').read_text())
sources=d['sources']; y=d['labels']; errors=[]
for name,record in q['models'].items():
    for scope,mask in [('21DF_all',np.ones(len(y),dtype=bool))]+[('21DF_'+s,sources==s) for s in np.unique(sources)]:
        yy=y[mask]; ranks=rankdata(d[name][mask]); n=int(yy.sum()); m=len(yy)-n
        auc=float((ranks[yy==1].sum()-n*(n+1)/2)/(n*m))
        errors.append(abs(auc-record[scope]['auc']))
assert max(errors)<1e-12
checks['auc_independent_rank_formula_max_error']=max(errors)
checks['auc_comparisons']=len(errors)
(p/'verification.json').write_text(json.dumps(checks,indent=2))
print(json.dumps(checks,indent=2))
