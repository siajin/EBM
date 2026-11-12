"""Bounded real-data and two-GPU C validation; never saves/loads trained weights."""
import ast
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

CODE=Path('/home/mentee/code/EBM')
BASE='2c2c6ff65bbadf33b811983f7c648c97fffb81a1'
sys.path.insert(0,str(CODE))
import arguments
import data_loaders as dl
from dataset import ASVspoof2019_LA_Train

def save(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def git(*args):return subprocess.check_output(['git',*args],cwd=CODE,text=True).strip()
def seed(n):random.seed(n);np.random.seed(n);torch.manual_seed(n)

def metadata(root):
    args,_,_=arguments.get_args()
    assert args['name']=='EBM_e50_bs240_lr1e5_C_nativeLong_short303'
    assert args['native_length'] and not args['cap_long']
    assert args['micro_batch_size']==args['eval_batch_size']==8 and args['batch_size']==240
    assert args['short_oc_weight']==args['padding_loss_weight']==0
    old_source=subprocess.check_output(['git','show',BASE+':data_loaders.py'],cwd=CODE,text=True)
    old={};exec(compile(old_source,'B_data_loaders.py','exec'),old)
    ds=ASVspoof2019_LA_Train(args['path_19LA']);items=ds.train_set
    cargs=dict(args,world_size=2,rank=0,batch_size=120,num_workers=0)
    bargs=dict(cargs,cap_long=True)
    b=old['TrainSet'](bargs,items);c=dl.TrainSet(cargs,items)
    sb=old['BalancedLengthBatchSampler'](b,bargs);sc=dl.BalancedLengthBatchSampler(c,cargs)
    assert len(items)==25380 and np.array_equal(sb.lengths,sc.lengths)
    schedules={}
    for rank in [0,1]:
        sb.rank=sc.rank=rank
        digest=hashlib.sha256()
        for epoch in range(1,51):
            sb.set_epoch(epoch);sc.set_epoch(epoch)
            left=list(sb);right=list(sc)
            assert left==right and len(left)==108
            digest.update(np.asarray(left,dtype=np.int32).tobytes())
            for batch in right:
                assert len(batch)==120 and sum(items[i].label for i in batch)==12
        schedules[str(rank)]=digest.hexdigest()
    y=np.array([i.label for i in items]);lengths=sc.lengths
    chosen=set()
    for label in [1,0]:
        ids=np.flatnonzero(y==label)
        chosen.update([int(ids[np.argmin(lengths[ids])]),int(ids[np.argmax(lengths[ids])])])
        for threshold in [32000,48440,64600]:
            chosen.add(int(ids[np.argmin(abs(lengths[ids]-threshold))]))
    original_randint=random.randint
    def no_wrap(*a,**kw):raise AssertionError('Codec or repeat must not execute')
    original_pad=np.pad
    def guarded_pad(*a,**kw):
        assert kw.get('mode',a[2] if len(a)>2 else 'constant')!='wrap'
        return original_pad(*a,**kw)
    tests=0;python_rng_matches=True;short_noaug_equal=True
    with patch.object(np,'pad',side_effect=guarded_pad),patch.object(c.codec_aug,'convert',side_effect=no_wrap),patch.object(b.codec_aug,'convert',side_effect=no_wrap):
        for idx in sorted(chosen):
            for aug in range(9):
                def select(a,b):return aug if (a,b)==(0,8) else original_randint(a,b)
                with patch.object(random,'randint',side_effect=select):
                    seed(7000+idx);before=b[idx];rngb=random.getstate()
                    seed(7000+idx);after=c[idx];rngc=random.getstate()
                assert len(before[0])==min(lengths[idx],64600)
                assert len(after[0])==lengths[idx]
                assert len(after[1])==min(lengths[idx],48440)
                assert before[2:]==after[2:] and rngb==rngc
                if lengths[idx]<=64600:
                    assert np.array_equal(before[0],after[0]) and np.array_equal(before[1],after[1])
                if aug==0:assert np.array_equal(before[1],after[1])
                assert np.isfinite(after[0]).all() and np.isfinite(after[1]).all()
                tests+=1
    # Verify real loader wiring, not just a manually overridden EnrollmentSet.
    train,sampler,val,targets=dl.get_loaders(cargs,items,ds.dev_set,{})
    assert not train.dataset.cap_long and val.dataset.cap_long and val.dataset.native_length
    row=val.dataset[0];ref=old['EnrollmentSet'](bargs,ds.dev_set)[0]
    assert np.array_equal(row[0],ref[0]) and row[1:]==ref[1:]
    zero=dl.collate_train([c[int(np.argmin(lengths))]],48440,include_wrapped=False)
    assert set(zero)=={'long','short','labels','indices'}
    caps={str(label):{'N':int((y==label).sum()),'longer_than_B_cap':int(((y==label)&(lengths>64600)).sum()),
         'max_samples':int(lengths[y==label].max()),'total_original_samples':int(lengths[y==label].sum()),
         'total_B_used_samples':int(np.minimum(lengths[y==label],64600).sum())} for label in [1,0]}
    longest={str(label):int(np.flatnonzero(y==label)[np.argmax(lengths[y==label])]) for label in [1,0]}
    manifest=hashlib.sha256('\n'.join(f'{i.uid if hasattr(i,"uid") else i.path}|{i.label}|{int(n)}' for i,n in zip(items,lengths)).encode()).hexdigest()
    result={'status':'passed','input_tests':tests,'train_N':len(items),'dev_N':len(ds.dev_set),'class_length_counts':caps,
            'sampler_all50epochs_exactly_matches_B':True,'sampler_hashes':schedules,'logical_steps_per_epoch':108,
            'Python_crop_and_augmentation_choice_RNG_same':True,'noaug_short_exactly_matches_B':True,
            'same_input_under_cap_exact':True,'evaluation_fixed_B_input':True,'codec_and_wrap_forbidden':True,
            'training_manifest_sha256':manifest,'longest_training_indices':longest,
            'RNG_limit':'NumPy RawBoost and CUDA random tensors change with long input shape; policy and seed unchanged'}
    save(root/'metadata_validation.json',result)

def worker(rank,root,port):
    root=Path(root)
    import main
    import model
    from transformers import AutoConfig,Wav2Vec2Model
    import native_steps
    args,_,_=arguments.get_args();args.update(process_id=rank,world_size=2,batch_size=120,port=port,flag_parent=rank==0,num_workers=0)
    main.set_experiment_environment(args);device=args['device']
    frozen=Wav2Vec2Model.from_pretrained(args['PLM_name'],config=AutoConfig.from_pretrained(args['PLM_name'],finetuning_task='audio-classification',revision='main'),revision='main',ignore_mismatched_sizes=False).to(device)
    frozen.eval();frozen.requires_grad_(False)
    cw=[1/(2*.9),1/(2*.1)]
    classifier=model.B_Linear(num_layer=25,hidden_size=1024,output_size=160,agg_size=128,
          loss=model.OCSoftmax(160,r_real=.9,r_fake=.2,alpha=20.,class_weight=cw),loss_bpl=model.BPL(alpha=30.)).to(device)
    classifier=torch.nn.SyncBatchNorm.convert_sync_batchnorm(classifier)
    classifier=torch.nn.parallel.DistributedDataParallel(classifier,device_ids=[device],find_unused_parameters=True)
    optimizer=torch.optim.Adam(list(frozen.parameters())+list(classifier.parameters()),lr=1e-5,weight_decay=.0001)
    original_step=optimizer.step;step_calls=[]
    def counted_step(*a,**kw):
        step_calls.append(1)
        return original_step(*a,**kw)
    optimizer.step=counted_step
    trainer=SimpleNamespace(args=args,plm=frozen,classifier=classifier,optimizer=optimizer)
    meta=json.loads((root/'metadata_validation.json').read_text())
    items=ASVspoof2019_LA_Train(args['path_19LA']).train_set;data=dl.TrainSet(args,items)
    # Upper bound every physical chunk with the actual longest bona/spoof waveform.
    rows=[]
    for label,n in [(1,12),(0,108)]:
        idx=meta['longest_training_indices'][str(label)]
        with patch.object(random,'randint',side_effect=lambda a,b: 0 if (a,b)==(0,8) else a):row=data[idx]
        rows.extend([row]*n)
    batch=dl.collate_train(rows,48440,include_wrapped=False)
    assert [len(w) for w in batch['long'][:12]]==[meta['class_length_counts']['1']['max_samples']]*12
    frontend_before={n:p._version for n,p in frozen.named_parameters()}
    centroid_before=classifier.module.loss.weight.detach().clone()
    theta_before=classifier.module.linear.weight.detach().clone()
    torch.cuda.reset_peak_memory_stats(device)
    classifier.train();stats=native_steps.train_batch(trainer,batch)
    torch.cuda.synchronize(device)
    assert all(np.isfinite(v) for v in stats.values())
    assert abs(stats['loss1']-stats['oc_long']-stats['bpl'])<2e-5
    assert stats['padding']==stats['active_pairs']==0
    assert not torch.equal(theta_before,classifier.module.linear.weight)
    assert not torch.equal(centroid_before,classifier.module.loss.weight)
    assert all(p.grad is None and p._version==frontend_before[n] for n,p in frozen.named_parameters())
    assert len(step_calls)==2 and set(float(s['step']) for s in optimizer.state.values())=={1.}
    # Mask validity and standalone/batch invariance without pooling noise.
    classifier.eval()
    for module in classifier.modules():
        if hasattr(module,'noise_std'):module.noise_std=0.
    short=torch.as_tensor(rows[0][1],dtype=torch.float32)
    waves=[short,batch['long'][-1]]
    with torch.no_grad():
        x,mask=native_steps.features(frozen,waves,device)
        assert mask.sum(1).tolist()==frozen._get_feat_extract_output_lengths(torch.tensor([len(w) for w in waves],device=device)).tolist()
        mixed=classifier.module(x,mask=mask,return_embedding=True)[0];del x,mask
        x,mask=native_steps.features(frozen,[short],device)
        solo=classifier.module(x,mask=mask,return_embedding=True)[0];del x,mask
        error=float((mixed-solo).abs().max());assert error<1e-4,error
    peak=torch.cuda.max_memory_allocated(device)/1024**2
    assert peak<.85*torch.cuda.get_device_properties(device).total_memory/1024**2
    result={'status':'passed','rank':rank,'GPU':torch.cuda.get_device_name(rank),'worst_case_physical_microbatch':8,
            'logical_batch':120,'bonafide':12,'spoof':108,'full_BPL_pairs':144,'stats':stats,
            'two_Adam_updates_verified':True,'frozen_frontend_unchanged':True,'mask_solo_batch_embedding_max_error':error,
            'peak_allocated_MiB':peak,'peak_reserved_MiB':torch.cuda.max_memory_reserved(device)/1024**2,
            'frontend_commit':getattr(frozen.config,'_commit_hash',None),'torch_version':torch.__version__}
    save(root/f'verification_rank{rank}.json',result)
    torch.distributed.barrier();torch.distributed.destroy_process_group()

if __name__=='__main__':
    root=Path(sys.argv[1]);assert Path('/.dockerenv').exists() and os.environ['CUDA_VISIBLE_DEVICES']=='0,1'
    metadata(root)
    torch.multiprocessing.spawn(worker,args=(str(root),'29748'),nprocs=2,join=True)
    save(root/'verification_complete.json',{'status':'passed','metadata':True,'both_GPUs':True,'no_training_result_written':True})
