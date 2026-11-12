"""Train OC+0.5legacyBPL+0.5same-source conditionalBPL2; nodev, atomic pre-eval saves."""
import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)

def atomic_torch(path,state):
    import torch
    path=Path(path);assert not path.exists(),'Refuse checkpoint overwrite'
    tmp=path.with_suffix('.tmp');torch.save(state,tmp);tmp.replace(path)

def monitor(trainer,root,epoch,smoke=False):
    import torch
    import torch.distributed as dist
    from native_steps import features,unwrap
    from positive_bpl import evaluation_rng
    from tools.evaluate_B_C_checkpoints import make_items,Waves,collate,metrics
    import pandas as pd
    args=trainer.args;rank=args['rank'];items=make_items(args,'19LA');ds=Waves(items)
    indices=list(range(rank,len(ds),2));indices=indices[:32] if smoke else indices
    loader=torch.utils.data.DataLoader(ds,batch_size=8,sampler=indices,num_workers=2,
        collate_fn=collate,multiprocessing_context='spawn',generator=torch.Generator().manual_seed(8800+rank))
    rows=[];model=unwrap(trainer.classifier);was_training=model.training;model.eval()
    with evaluation_rng(args,525252),torch.no_grad():
        for waves,metadata in loader:
            x,mask=features(trainer.plm,waves,args['device']);scores=model(x,mask=mask)[1].flatten().cpu().tolist()
            rows.extend(dict(**m,score=float(s)) for m,s in zip(metadata,scores));del x,mask
    model.train(was_training)
    gathered=[None]*2;dist.all_gather_object(gathered,rows)
    if rank==0:
        frame=pd.DataFrame([r for part in gathered for r in part]).sort_values('index')
        assert not frame.uid.duplicated().any()
        if not smoke:assert frame['index'].tolist()==list(range(71237))
        path=root/f'monitor_19LA_e{epoch}{"_smoke" if smoke else ""}.csv';assert not path.exists();frame.to_csv(path,index=False)
        result=metrics(frame,'score') if frame.label.nunique()==2 else {'n':len(frame)}
        result.update(epoch=epoch,sha256=sha(path),smoke=smoke,dev_evaluated=False)
        save(path.with_suffix('.json'),result)
    dist.barrier()

def worker(rank,root_string,smoke):
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader,DistributedSampler
    root=Path(root_string);sys.path.insert(0,str(root/'code'));cfg=json.loads((root/'experiment.json').read_text())
    import arguments,model,archived240_data_loaders
    from dataset.ASVspoof2019_la_train import load_train_items_only
    from conditional_bpl2 import ConditionalTrainSet,collate_train,merge_train
    from conditional_bpl2_steps import train_step
    from transformers import Wav2Vec2Model
    random.seed(1);np.random.seed(1);torch.manual_seed(1);torch.set_num_threads(2);torch.cuda.set_device(rank)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    os.environ['MASTER_ADDR']='127.0.0.1';os.environ['MASTER_PORT']=str(cfg['smoke_port'] if smoke else cfg['train_port'])
    dist.init_process_group('nccl',rank=rank,world_size=2,timeout=datetime.timedelta(minutes=20))
    args,_,_=arguments.get_args();args.update(rank=rank,world_size=2,device=f'cuda:{rank}',batch_size=120,
        rand_seed=1,epoch=50,num_train_frames=404,num_train_frames_short=303,evaluation_seed=1729,
        short_oc_weight=0.,padding_loss_weight=0.)
    items=load_train_items_only(args['path_19LA']);assert len(items)==25380
    splits={y:[i for i in items if i.label==y] for y in [0,1]};assert len(splits[1])==2580 and len(splits[0])==22800
    datasets={y:ConditionalTrainSet(args,splits[y],archived240_data_loaders.TrainSet) for y in [0,1]}
    samplers={y:DistributedSampler(datasets[y],num_replicas=2,rank=rank,shuffle=True,seed=0,drop_last=False) for y in [0,1]}
    def loader(y,batches=None):
        kw={'batch_sampler':batches} if batches is not None else {'batch_size':12 if y else 108,'sampler':samplers[y]}
        return DataLoader(datasets[y],num_workers=2,collate_fn=collate_train,multiprocessing_context='spawn',
            generator=torch.Generator().manual_seed(1+rank+y*100),**kw)
    plm=Wav2Vec2Model.from_pretrained(args['PLM_name'],local_files_only=True).to(args['device']).eval().requires_grad_(False)
    assert getattr(plm.config,'_commit_hash',None)==cfg['frontend_revision']
    base=model.B_Linear(25,1024,160,128,loss=model.OCSoftmax(160,class_weight=[5/9,5]),loss_bpl=model.BPL(30)).to(args['device'])
    classifier=torch.nn.parallel.DistributedDataParallel(base,device_ids=[rank],find_unused_parameters=True)
    optimizer=torch.optim.Adam(list(plm.parameters())+list(classifier.parameters()),lr=1e-5,weight_decay=args['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=150,T_mult=1,eta_min=1e-5)
    trainer=SimpleNamespace(classifier=classifier,plm=plm,optimizer=optimizer,args=args)
    versions=[p._version for p in plm.parameters()];initial_center=base.loss.weight.detach().clone()
    log=root/f'steps_{"smoke_" if smoke else ""}rank{rank}.jsonl';assert not log.exists()
    total_steps=0;start=time.time()
    for epoch in ([0] if smoke else range(1,51)):
        for y in [0,1]:samplers[y].set_epoch(epoch);datasets[y].epoch=epoch;datasets[y].visits={}
        if smoke:
            # Includes both conditional policies and original ragged batch shapes.
            meta=json.loads((root/'train_length_metadata.json').read_text())
            selected=[]
            for cond in [lambda n:n>80000,lambda n:64600<n<80000,lambda n:n<48440]:
                selected.append(next(i for i,row in enumerate(meta['bona']) if cond(row['samples'])))
            ordered=selected+[i for i in list(samplers[1]) if i not in selected]
            b1=[ordered[:12],ordered[12:24],ordered[24:30]]
            ids=list(samplers[0]);b0=[ids[:108],ids[108:168],ids[168:276]]
            bona_loader=loader(1,b1);spoof_loader=loader(0,b0)
        else:bona_loader=loader(1);spoof_loader=loader(0)
        assert len(bona_loader)>len(spoof_loader) or smoke
        bit=iter(bona_loader);sit=iter(spoof_loader);consumed={0:set(),1:set()};counts={0:0,1:0};pair_counts=[];pair54=0;pair43=0
        with log.open('a') as stream:
            for batch_number,bona in enumerate(bit,1):
                try:spoof=next(sit)
                except StopIteration:sit=iter(spoof_loader);spoof=next(sit)
                batch=merge_train(bona,spoof);classifier.train();stats=train_step(trainer,batch)
                assert np.isfinite(list(stats.values())).all();total_steps+=1
                for row in batch['rows']:
                    y=row['label'];consumed[y].add(row['uid']);counts[y]+=1
                pair_counts.append(stats['legacy_pairs']);pair54+=stats['pair54'];pair43+=stats['pair43']
                row={'epoch':epoch,'batch':batch_number,**stats};stream.write(json.dumps(row)+'\n');stream.flush()
                if batch_number==1 or batch_number%5==0:
                    save(root/f'train_progress_{"smoke_" if smoke else ""}rank{rank}.json',dict(**row,batches=len(bona_loader),elapsed_seconds=time.time()-start,updated_unix=time.time()))
                if batch_number%20==0 and rank==0:print(json.dumps(row),flush=True)
        # Shut down the partially consumed second small-loader cycle explicitly;
        # do not decode another hundred unused batches merely to exhaust it.
        if hasattr(sit,'_shutdown_workers'):sit._shutdown_workers()
        del bit,sit,bona_loader,spoof_loader
        expected=3 if smoke else 108;assert batch_number==expected
        assert pair_counts==([144,144,36] if smoke else [144]*107+[36]),pair_counts
        assert versions==[p._version for p in plm.parameters()]
        scheduler.step()
        report={'epoch':epoch,'steps':batch_number,'backend_updates':batch_number,'center_updates':batch_number,
                'unique':{str(y):len(v) for y,v in consumed.items()},'visits':counts,'legacy_pair_counts':pair_counts,
                'pair54':pair54,'pair43':pair43,'PLM_unchanged':True}
        # UID lists are derived Train metadata, retained on server for coverage audit.
        save(root/f'coverage_e{epoch}_rank{rank}.json',{**report,'uids':{str(y):sorted(v) for y,v in consumed.items()}})
        if smoke:
            assert pair54>0 and pair43>0 and not torch.equal(initial_center,base.loss.weight)
            state=classifier.state_dict();clone=model.B_Linear(25,1024,160,128,loss=model.OCSoftmax(160,class_weight=[5/9,5]),loss_bpl=model.BPL(30)).to(args['device'])
            clone.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True)
            assert all(torch.equal(v,clone.state_dict()[k.removeprefix('module.')]) for k,v in state.items());del clone
            monitor(trainer,root,0,True)
            save(root/f'train_gpu_smoke_rank{rank}.json',dict(status='passed',**report,checkpoint_strict_load=True,spawned_eval_workers_passed=True))
        elif epoch%5==0:
            # Always persist BEFORE invoking any evaluation worker.
            if rank==0:
                result=Path(cfg['result_path']);(result/'model').mkdir(parents=True,exist_ok=True)
                atomic_torch(result/f'model/CLS_{epoch}.pt',classifier.state_dict())
                safe_args={k:v for k,v in args.items() if not any(s in k.lower() for s in ['token','password','secret'])}
                atomic_torch(result/f'model/TrainState_{epoch}.pt',{'epoch':epoch,'classifier':classifier.state_dict(),
                    'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'args':safe_args,'experiment':cfg['method'],
                    'steps':total_steps})
            atomic_torch(root/f'RNG_e{epoch}_rank{rank}.pt',{'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state(rank),
                'python':random.getstate(),'numpy':[np.random.get_state()[0],np.random.get_state()[1].tolist(),*np.random.get_state()[2:]]})
            dist.barrier();monitor(trainer,root,epoch)
        dist.barrier()
    save(root/f'train_done_{"smoke_" if smoke else ""}rank{rank}.json',{'status':'complete','steps':total_steps,'epochs':0 if smoke else 50,'frontend_unchanged':True})
    dist.destroy_process_group()

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    assert Path('/.dockerenv').exists() and os.environ.get('CUDA_VISIBLE_DEVICES')=='0,1'
    cfg=json.loads((a.root/'experiment.json').read_text())
    for name,h in cfg['source_sha256'].items():assert sha(a.root/'code'/name)==h,name
    import torch.multiprocessing as mp
    claim=a.root/('smoke.claim' if a.smoke else 'train.claim');claim.mkdir()
    mp.set_sharing_strategy('file_system');mp.spawn(worker,args=(str(a.root),a.smoke),nprocs=2,join=True)
    save(a.root/('smoke_complete.json' if a.smoke else 'training_complete.json'),{'status':'complete','epoch':0 if a.smoke else 50,'optimizer_updates_per_parameter':3 if a.smoke else 5400})

if __name__=='__main__':main()
