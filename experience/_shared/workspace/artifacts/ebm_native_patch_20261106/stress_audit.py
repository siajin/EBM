"""Bounded codec-off checks and timing/memory stress; never a full training run."""
import argparse
import copy
import datetime
import json
import math
from pathlib import Path
import socket
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd


def save(p,o):p.write_text(json.dumps(o,indent=2)+'\n')


def worker(rank,root_s,port):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    import soundfile as sf
    from transformers import AutoConfig,Wav2Vec2Model
    root=Path(root_s);sys.path.insert(0,'/home/mentee/code/EBM')
    import model,data_loaders,arguments,native_steps
    from dataset import ASVspoof2019_LA_Train
    torch.set_num_threads(2);torch.cuda.set_device(rank);device=f'cuda:{rank}'
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.manual_seed(20261106);torch.cuda.manual_seed(20261106)
    dist.init_process_group('nccl',init_method=f'tcp://127.0.0.1:{port}',rank=rank,world_size=2,timeout=datetime.timedelta(minutes=5))
    args,_,_=arguments.get_args();args.update(rank=rank,world_size=2,batch_size=120,device=device,num_workers=0)
    checks={}
    def ck(k,v):assert v,k;checks[k]=True
    database=ASVspoof2019_LA_Train(args['path_19LA'])
    ck('commented_extra_vocoder_data_stays_disabled',len(database.train_set)==25380 and sum(i.label for i in database.train_set)==2580)
    dataset=data_loaders.TrainSet(args,database.train_set)
    ncodec=0
    def forbidden(*a,**kw):
        nonlocal ncodec
        ncodec+=1
        raise AssertionError('Online codec unexpectedly called')
    probe=pd.read_parquet('/home/mentee/analysis/embedding_geometry/20261106_variable_length_01/manifest.parquet').set_index('selection')
    codec_cases=[]
    for key in ['train_y0_minimum','train_y1_minimum','train_y0_maximum']:
        r=probe.loc[key];item=SimpleNamespace(path=r.path,label=int(r.y),type=r.attack_id)
        d=data_loaders.TrainSet(args,[item]);d.codec_aug.convert=forbidden
        original,sr=sf.read(r.path)
        for aug in range(9):
            np.random.seed(17+aug)
            chooser=lambda a,b,j=aug: j if (a,b)==(0,8) else 0
            with patch.object(data_loaders.random,'random',return_value=0.),patch.object(data_loaders.random,'randint',side_effect=chooser):
                long,short,y,index=d[0]
            ck(f'rawboost_shape_finite_{key}_{aug}',len(long)==len(original) and len(short)==min(len(original),48440) and np.isfinite(long).all() and np.isfinite(short).all())
        codec_cases.append(dict(key=key,raw_samples=len(original),tested_aug_types=9))
    ck('forced_codec_branch_zero_calls',ncodec==0)
    save(root/f'rank{rank}_progress.json',{'stage':'codec_and_augmentation_pass','codec_calls':ncodec})

    meta=pd.read_parquet('/home/mentee/analysis/embedding_geometry/20261105_fast_p0_01/extract/scores.parquet')
    manifest=pd.read_parquet('/home/mentee/analysis/embedding_geometry/20261105_fast_p0_01/manifest.parquet')
    table=manifest[manifest.split=='T'].merge(meta[['utterance_id','duration']],on='utterance_id',validate='one_to_one')
    cohort=[]
    for y,n in [(1,24),(0,216)]:
        q=table[table.y==y].sort_values(['duration','utterance_id'])
        indices=np.linspace(0,len(q)-1,n).round().astype(int)
        cohort.extend([r for _,r in q.iloc[indices].iloc[rank::2].iterrows()])
    def prepare(rows,legacy=False):
        result=[]
        for i,r in enumerate(rows):
            a,sr=sf.read(r.path,dtype='float32');n=min(len(a),48440);start=(len(a)-n)//2;s=a[start:start+n].copy()
            long=a.copy()
            if legacy:
                long=np.pad(long,(0,max(0,64600-len(long))),mode='wrap')[:64600]
                s=np.pad(s,(0,48440-len(s)),mode='wrap')
            result.append((long,s,int(r.y),i))
        return data_loaders.collate_train(result,short_size=48440)
    batches={'fixed_two_views':prepare(cohort,True),'native_two_views':prepare(cohort),'native_three_views':prepare(cohort)}
    maxrows=[]
    for y,n in [(1,12),(0,108)]:
        r=probe.loc['train_y1_maximum' if y==1 else 'train_y0_maximum']
        maxrows.extend([r]*n)
    batches['worst_train_three_views']=prepare(maxrows)
    config=AutoConfig.from_pretrained(args['PLM_name'],local_files_only=True)
    plm=Wav2Vec2Model.from_pretrained(args['PLM_name'],config=config,local_files_only=True).to(device).eval().requires_grad_(False)
    cls=model.B_Linear(args['num_layers'],args['hidden_size'],args['output_size'],args['agg_size'],
        loss=model.OCSoftmax(args['output_size'],r_real=args['r_real'],r_fake=args['r_fake'],alpha=args['alpha'],class_weight=[5/9,5]),loss_bpl=model.BPL(args['alpha_cl'])).to(device)
    checkpoint='/home/mentee/results/CM/LA/EBM_e50_bs160_lr1e5/model/BestCLS_50_21DF.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=True);state={k.removeprefix('module.'):v for k,v in state.items()}
    cls.load_state_dict(state,strict=True);ddp=DDP(cls,device_ids=[rank],find_unused_parameters=True)
    counts={'calls':0,'sample_slots':0,'waveform_samples':0}
    def count(m,i,o):
        counts['calls']+=1;counts['sample_slots']+=len(i[0]);counts['waveform_samples']+=i[0].numel()
    hook=plm.register_forward_hook(count)
    versions=[p._version for p in plm.parameters()]
    profiles=[]
    for name,batch in batches.items():
        args['short_oc_weight']=.25 if 'three' in name else 0.
        args['padding_loss_weight']=.1 if 'three' in name else 0.
        iterations=1 if name.startswith('worst') else 3
        for iteration in range(iterations):
            cls.load_state_dict(state,strict=True);cls.train()
            optimizer=torch.optim.Adam(ddp.parameters(),lr=args['lr'],weight_decay=args['weight_decay'])
            engine=SimpleNamespace(plm=plm,classifier=ddp,args=args,optimizer=optimizer)
            for k in counts:counts[k]=0
            dist.barrier();torch.cuda.synchronize(rank);torch.cuda.reset_peak_memory_stats(rank)
            begin=time.perf_counter();stats=native_steps.train_batch(engine,batch);torch.cuda.synchronize(rank)
            elapsed=time.perf_counter()-begin
            ck(f'finite_update_{name}_{iteration}',all(np.isfinite(v) for v in stats.values()))
            profiles.append(dict(case=name,iteration=iteration,warmup=iterations>1 and iteration==0,seconds=elapsed,
                peak_allocated_MiB=torch.cuda.max_memory_allocated(rank)/2**20,
                local_batch=len(batch['labels']),physical_batch=args['micro_batch_size'],
                max_long_seconds=max(len(w) for w in batch['long'])/16000,
                frontend_counts=dict(counts),active_normal_pairs=stats['active_pairs']))
            save(root/f'rank{rank}_progress.json',profiles[-1])
            del engine,optimizer
    ck('frontend_never_updated',versions==[p._version for p in plm.parameters()] and all(p.grad is None for p in plm.parameters()))
    hook.remove();cls.eval()
    a,sr=sf.read(probe.loc['DF_remaining_maximum'].path,dtype='float32')
    waves=[torch.from_numpy(a.copy()) for _ in range(8)]
    torch.cuda.reset_peak_memory_stats(rank);torch.cuda.synchronize(rank);begin=time.perf_counter()
    with torch.no_grad():
        f,m=native_steps.features(plm,waves,device);z=cls(f,mask=m,return_embedding=True)
    torch.cuda.synchronize(rank)
    ck('eval8_longest29_312_finite',bool(torch.isfinite(z).all()) and abs(len(a)/sr-29.312)<1e-9)
    worst_eval=dict(batch=8,seconds_per_clip=len(a)/sr,wall_seconds=time.perf_counter()-begin,peak_allocated_MiB=torch.cuda.max_memory_allocated(rank)/2**20)
    # Reconcile per-epoch frontend work, respecting actual microbatch padding.
    sampler=data_loaders.BalancedLengthBatchSampler(dataset,args)
    samples=(meta[meta.split=='T'].set_index('utterance_id').duration*16000).round()
    sizes=np.array([int(samples[Path(i.path).stem]) for i in dataset.items])
    total={'legacy_direct_fixed':0,'cached_fixed_two':0,'cached_native_two':0,'cached_native_three':0}
    for ids in sampler:
        n=sizes[ids];short=np.minimum(n,48440);bona=int(sum(dataset.items[i].label for i in ids))
        def padded(v):return sum(len(v[i:i+8])*int(max(v[i:i+8])) for i in range(0,len(v),8))
        total['legacy_direct_fixed']+=len(n)*(64600+48440)
        total['cached_fixed_two']+=2*len(n)*(64600+48440)+bona*64600
        ntwo=2*padded(n)+2*padded(short)+padded(n[:bona])
        total['cached_native_two']+=ntwo
        total['cached_native_three']+=ntwo+2*len(n)*48440
    ratios={k:v/total['legacy_direct_fixed'] for k,v in total.items()}
    result=dict(status='pass',rank=rank,checks=checks,codec_calls=ncodec,codec_cases=codec_cases,
        timing_profiles=profiles,worst_evaluation=worst_eval,epoch_frontend_padded_sample_slots=total,
        frontend_work_ratios_to_original_two_direct_fixed_views=ratios,
        qualifications=['Timing fixed_two uses new gradient cache, not original full-graph training wall time',
            'Three trials per quantile cohort: first warmup, next two measurements; not an epoch throughput estimate',
            'Worst-train stress repeats true-label longest train normal/spoof clips; not an accuracy evaluation',
            'Forward-sample proxy is not FLOPs or runtime; attention additionally depends on squared frame length',
            'No50epoch training or performance-improvement claim'],full_training_started=False,raw_audio_or_SSL_saved=False)
    save(root/f'rank{rank}_stress.json',result);dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
    import torch.multiprocessing as mp
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    assert Path('/.dockerenv').exists()
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    mp.spawn(worker,args=(str(a.root),port),nprocs=2,join=True)
    save(a.root/'stress_complete.json',{'status':'pass','ranks':[0,1]})
