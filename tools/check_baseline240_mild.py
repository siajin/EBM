"""Immutable sources, actual cache, original schedule and final240 audits."""
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch

ROOT=Path(sys.argv[1]);sys.path.insert(0,str(ROOT/'code'))
from tools.train_baseline240_mild import save,load_file


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_consumed(epoch):
    ids={0:set(),1:set()};counts={0:0,1:0}
    for rank in [0,1]:
        with (ROOT/f'consumed_epoch{epoch}_rank{rank}.csv').open() as stream:
            rows=list(csv.DictReader(stream))
        assert len(rows)==12906
        for row in rows:
            assert int(row['epoch'])==epoch and int(row['rank'])==rank
            y=int(row['label']);ids[y].add(int(row['index']));counts[y]+=1
        done=json.loads((ROOT/f'epoch{epoch}_rank{rank}.json').read_text())
        assert done['steps']==done['backend_updates']==done['center_updates']==108
        assert done['BPL_pair_counts']==[144]*107+[36] and done['PLM_grads_absent']
    assert {y:len(v) for y,v in ids.items()}=={0:22800,1:2580}
    assert counts=={0:23232,1:2580}
    return {'epoch':epoch,'unique_bonafide':2580,'unique_spoof':22800,'visits_bonafide':2580,
        'visits_spoof':23232,'originals_all_visited':True,'positive_pairs_global':30888}


def main():
    cfg=json.loads((ROOT/'experiment.json').read_text())
    assert Path('/.dockerenv').is_file() and os.environ['CUDA_VISIBLE_DEVICES']=='0,1'
    for tree,key in [('code','source_sha256'),('baseline_code','baseline_source_sha256')]:
        for name,expected in cfg[key].items():assert sha(ROOT/tree/name)==expected,(tree,name)
    assert sha(ROOT/'duration_policy.json')==cfg['duration_policy_sha256']
    assert subprocess.check_output(['git','-C','/home/mentee/code/EBM','rev-parse','HEAD'],text=True).strip()==cfg['live_commit']
    assert not subprocess.check_output(['git','-C','/home/mentee/code/EBM','status','--porcelain'],text=True)
    for c in cfg['protected_checkpoints'].values():
        for path,key in [('path','sha256'),('TrainState_path','TrainState_sha256')]:
            if path in c:assert sha(c[path])==c[key]
    if '--final' in sys.argv:
        assert (ROOT/'train.exit').read_text().strip()=='0'
        result=Path(cfg['result_path']);state=torch.load(result/'model/TrainState_20.pt',map_location='cpu',weights_only=True)
        head=torch.load(result/'model/CLS_20.pt',map_location='cpu',weights_only=True)
        assert state['epoch']==state['args']['epoch']==20 and state['args']['batch_size']==120
        assert state['args']['frontend_resolved_revision']==cfg['frontend_revision']
        assert all(torch.equal(v,state['classifier'][name]) and torch.isfinite(v).all() for name,v in head.items())
        steps=[int(v['step']) for v in state['optimizer']['state'].values()]
        assert min(steps)==max(steps)==2160
        for rank in [0,1]:
            rows=[json.loads(s) for s in (ROOT/f'steps_rank{rank}.jsonl').read_text().splitlines()]
            assert len(rows)==2160
            assert all(np.isfinite([r[k] for k in ['loss1','loss2','oc','bpl']]).all() for r in rows)
            done=json.loads((ROOT/f'train_done_rank{rank}.json').read_text())
            assert done['epoch']==20 and done['frontend_versions_unchanged']
        coverage=[audit_consumed(epoch) for epoch in range(1,21)]
        save(ROOT/'final_training_audit.json',{'status':'passed','epoch':20,'logical_steps':2160,
            'backend_updates_per_rank':2160,'center_updates_per_rank':2160,'classifier_matches_TrainState':True,
            'original_240_core_source_unchanged':True,'original_schedule_and_partial_batches':True,
            'epoch_coverage':coverage,'CLS20_sha256':sha(result/'model/CLS_20.pt'),
            'TrainState20_sha256':sha(result/'model/TrainState_20.pt')})
        return
    import arguments
    from mild_duration import MildTrainSet
    from dataset.ASVspoof2019_la_train import load_train_items_only
    from baseline240_mild import MildOriginal240TrainSet,collate_rows
    args,_,_=arguments.get_args();args.update(rank=0,world_size=2,batch_size=120,rand_seed=1,
        duration_policy_path=str(ROOT/'duration_policy.json'),mild_transform_cache=str(ROOT/'audio_cache'),activity_policy_path=None)
    ds=MildTrainSet(args,load_train_items_only(args['path_19LA']))
    wave_count=0
    for record in ds.records:
        for v in record['variants']:
            wave=np.load(ROOT/'audio_cache'/v['file'],allow_pickle=False)
            assert wave.shape==(v['output_samples'],) and wave.dtype==np.float32 and np.isfinite(wave).all()
            assert hashlib.sha256(wave.tobytes()).hexdigest()==ds.wave_hashes[v['file']]
            wave_count+=1
    assert wave_count==46580 and torch.cuda.device_count()==2
    archived=load_file('archived240_data_check',ROOT/'baseline_code/data_loaders.py')
    adapter=MildOriginal240TrainSet(args,ds.items,archived.TrainSet,ds,ROOT)
    selected=[ds.class_indices[1][0],ds.class_indices[0][0],ds.class_indices[1][-1],ds.class_indices[0][-1]]
    loader=torch.utils.data.DataLoader(adapter,batch_sampler=[selected[:2],selected[2:]],num_workers=2,collate_fn=collate_rows)
    for long,short,labels,metadata in loader:
        assert long.shape==(2,64600) and short.shape==(2,48440)
        assert torch.isfinite(long).all() and torch.isfinite(short).all() and len(metadata)==2
    assert not Path(cfg['result_path']).exists()
    save(ROOT/'preflight.json',{'status':'passed','source_files':len(cfg['source_sha256']),
        'archived_source_files':len(cfg['baseline_source_sha256']),'cached_waveforms_verified':wave_count,
        'worker_processes_checked':2,'training_long_samples':64600,'training_short_samples':48440,
        'repeat_short_audio':True,'source_fingerprint':ds.source_fingerprint,'original_class_loaders':True,
        'num_batches_per_epoch':108,'last_BPL_pairs_per_rank':36,'normal_BPL_pairs_per_rank':144,
        'source_originals':{'bonafide':2580,'spoof':22800},'visits_per_epoch':{'bonafide':2580,'spoof':23232},
        'GPU_training_updates':0,'dev_used':False})


if __name__=='__main__':main()
