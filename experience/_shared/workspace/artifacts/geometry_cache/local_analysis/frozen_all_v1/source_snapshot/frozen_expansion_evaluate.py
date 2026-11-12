"""Apply only V-frozen heads to completed expansion caches; no new fitting."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from frozen_common import *
from frozen_heads import Head,predict
from geometry_cache import validate_run,resolve_run
from audit_cached_core_results import independent_eer,independent_auc

OUT=DEFAULT_ROOT/'local_analysis/frozen_all_v1'

def run(name):
    assert (OUT/'heads_complete.json').is_file()
    folder=OUT/'R1'/name;folder.mkdir(parents=True,exist_ok=False)
    parts=['DF_pilot','DF_remaining'] if name=='DF_full' else [name]
    audits=[validate_run(n) for n in parts];shards=[];frames=[];centroid=None
    # Stream only z. Earlier h/noise/direction vectors are unnecessary here;
    # this also bounds memory on the user's16-GB laptop for full DF.
    for alias in parts:
        source,entry,split=resolve_run(alias);assert entry['identity']['checkpoint_sha256']==CHECKPOINT
        for rank in [0,1]:
            rows=pd.read_parquet(source/f'extract/rank{rank}_rows.parquet');keep=np.flatnonzero(rows.split.to_numpy()==split)
            rows=rows.iloc[keep].reset_index(drop=True);z=np.load(source/f'extract/rank{rank}_z.npy',mmap_mode='r');c=np.load(source/f'extract/rank{rank}_centroid.npy')
            if centroid is None:centroid=c
            assert np.max(np.abs(c-centroid))<1e-12
            frames.append(rows);shards.append((z,keep))
    rows=pd.concat(frames,ignore_index=True);assert not rows.utterance_id.duplicated().any()
    if name=='DF_full':assert len(rows)==533928 and rows.y.value_counts().to_dict()=={0:519059,1:14869}
    y=rows.y.to_numpy();frame=rows[['utterance_id','y','dependency_group']].copy();records=[];heads={}
    for kind in ['static','dynamic','mlp','metric']:
        for seed in SEEDS:
            source=OUT/('P2' if kind=='metric' else 'P1')/str(seed)/kind
            obj=torch.load(source/'head_state.pt',weights_only=True);head=Head(kind,centroid,seed);head.load_state_dict(obj['state_dict']);head.eval();heads[(kind,seed)]=head
    arrays={'native':np.empty(len(rows))}
    arrays.update({f'{kind}_{seed}':np.empty(len(rows)) for kind,seed in heads})
    cursor=0
    for z,indices in shards:
        for start in range(0,len(indices),4096):
            ix=indices[start:start+4096];unit=z[ix].astype(float);unit/=np.linalg.norm(unit,axis=1,keepdims=True);end=cursor+len(ix)
            arrays['native'][cursor:end]=unit@centroid
            for (kind,seed),head in heads.items():arrays[f'{kind}_{seed}'][cursor:end]=predict(head,unit)
            cursor=end
    assert cursor==len(rows)
    v=panel('V');th=threshold(v.rows.y.to_numpy(),v.geometry()['cosine'])['threshold'];records.append(dict(model='native',seed=0,**metric(y,arrays['native'],th)))
    for (kind,seed),head in heads.items():
        source=OUT/('P2' if kind=='metric' else 'P1')/str(seed)/kind;th=json.loads((source/'threshold.json').read_text())['threshold'];records.append(dict(model=kind,seed=seed,**metric(y,arrays[f'{kind}_{seed}'],th)))
    for key,values in arrays.items():frame[key]=values
    frame.to_parquet(folder/'scores.parquet',index=False);pd.DataFrame(records).to_csv(folder/'metrics.csv',index=False)
    for rec in records:
        score=frame['native' if rec['model']=='native' else f"{rec['model']}_{rec['seed']}"].to_numpy()
        assert abs(independent_eer(y,score)-rec['eer_percent'])<1e-9 and abs(independent_auc(y,score)-rec['auroc'])<1e-12
    save(folder/'complete.json',dict(status='complete',dataset=name,rows=len(frame),metrics_verified=len(records),cache_audits=audits,T_V_heads_frozen=True,no_fit_on_expansion=True,independent_blind_test=False))
    print(name,len(frame),'complete',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['Q','19LA_eval','DF_full'],required=True);o=p.parse_args()
    with threadpool_limits(2):run(o.dataset)
