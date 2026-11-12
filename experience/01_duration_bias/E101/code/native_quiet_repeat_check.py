from pathlib import Path
import json,sys,os,time
import numpy as np
sys.path.insert(0,'/home/mentee/analysis')
from native_quiet_strict import worker

def main():
    assert Path('/.dockerenv').exists()
    import torch
    from huggingface_hub import snapshot_download
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0,1' and torch.cuda.device_count()==2
    root=Path('/results/analysis/EBM80_native_quiet_strict');out=root/'repeat_check';out.mkdir(exist_ok=False)
    alltrials=json.loads((root/'sample_manifest.json').read_text());chosen=[]
    for label in [1,0]:
        for source in ['asvspoof','vcc2018','vcc2020']:
            group=sorted([t for t in alltrials if t['cohort']=='new_files' and t['label']==label and t['source']==source],key=lambda t:t['id'])
            chosen.extend(group[:2])
    assert len(chosen)==12
    trials=[dict(t,replicate=k) for t in chosen for k in range(5)]
    cfg={'out':str(out),'workers':2,'seed':8317,'cpu':False,
       'plm_path':snapshot_download('facebook/wav2vec2-xls-r-300m',local_files_only=True)}
    (out/'plan.json').write_text(json.dumps({'purpose':'Secondary reproducibility check; selection by ID before inspecting scores',
       'n_files':12,'replicates':5,'selection':'first two IDs in each new-file label/source stratum','primary_epsilon_rule_unchanged':True,
       'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())},indent=2))
    (out/'sample_manifest.json').write_text(json.dumps(trials,indent=2))
    torch.multiprocessing.spawn(worker,args=(cfg,trials),nprocs=2,join=True)
    shards=[np.load(out/f'shard_{k}.npz') for k in range(2)];ix=np.concatenate([v['indices'] for v in shards]);order=np.argsort(ix)
    assert sorted(ix.tolist())==list(range(60))
    scores=np.concatenate([v['scores'] for v in shards])[order].reshape(12,5,16)
    names=shards[0]['names'].tolist();ranges=[]
    for col in range(16):
        a=scores[:,:,col];valid=np.isfinite(a).all(1)
        v=np.ptp(a[valid],axis=1)
        ranges.append({'condition':names[col],'files_n':int(valid.sum()),'max_score_range':float(v.max()) if len(v) else None,
             'median_score_range':float(np.median(v)) if len(v) else None})
    contrasts=[]
    for f,b,word in [(10,11,'zero'),(12,13,'own40')]:
        for before,name in [(0,'front_original'),(b,'front_back')]:
            a=scores[:,:,f]-scores[:,:,before];valid=np.isfinite(a).all(1);v=np.ptp(a[valid],axis=1)
            contrasts.append({'family':word,'contrast':name,'files_n':int(valid.sum()),'max_effect_range':float(v.max())})
    result={'all_checks_passed':True,'files_n':12,'replicates':5,'conditions':ranges,'contrasts':contrasts,
      'maximum_score_range':max(x['max_score_range'] or 0 for x in ranges),
      'maximum_primary_effect_range':max(x['max_effect_range'] for x in contrasts),
      'limits':'12 fixed files; does not bound stochastic variability for every possible input. Includes repeated transformed inputs and both GPU workers.'}
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
