"""Isolated two-GPU positive-only BPL pilot; never edits the active EBM repo.

Two independent groups of 12 bona fide crops per GPU; all 144 pairs.
Frozen XLS-R, existing backend and BPL; no OC loss/centroid optimizer step,
no codec, no fixed Long/Short roles. Scores use a post-hoc Train prototype.
Only derived scores/metrics and this experiment's backend checkpoints persist.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import numpy as np

CODE=Path('/home/mentee/code/EBM')

def save(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');tmp.replace(path)

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def diagnostics(z):
    z=np.asarray(z,dtype='float64')
    norms=np.linalg.norm(z,axis=1)
    assert np.isfinite(z).all() and (norms>1e-12).all()
    u=z/norms[:,None];center=u.mean(axis=0);centered=u-center
    eigen=np.linalg.svd(centered,compute_uv=False)**2/len(u)
    total=float(eigen.sum())
    prob=eigen[eigen>0]/total if total>1e-18 else np.array([])
    rank=float(np.exp(-np.sum(prob*np.log(prob)))) if len(prob) else 0.
    return {'n':len(u),'raw_norm_mean':float(norms.mean()),'unit_covariance_trace':total,
            'unit_mean_norm':float(np.linalg.norm(center)),'covariance_effective_rank':rank,
            'mean_offdiagonal_cosine':float((np.square(u.sum(axis=0)).sum()-len(u))/(len(u)*(len(u)-1)))}

def eer(y,score):
    from sklearn.metrics import roc_curve,roc_auc_score
    y=np.asarray(y);score=np.asarray(score)
    fpr,tpr,threshold=roc_curve(y,score,pos_label=1,drop_intermediate=False)
    diff=fpr-(1-tpr);k=int(np.flatnonzero(diff>=0)[0])
    value=float(fpr[0]) if k==0 else float(fpr[k-1]+(-diff[k-1]/(diff[k]-diff[k-1]))*(fpr[k]-fpr[k-1]))
    return {'EER_percent':100*value,'AUC':float(roc_auc_score(y,score)),
            'score_min':float(score.min()),'score_max':float(score.max()),
            'mean_bona_minus_spoof_score':float(score[y==1].mean()-score[y==0].mean())}

def add_noise(wave,uid,snr_db):
    seed=int(hashlib.sha256(('bpl-pilot-noise-v1|'+uid).encode()).hexdigest()[:8],16)
    noise=np.random.default_rng(seed).normal(size=len(wave))
    noise-=noise.mean();rms=float(np.sqrt(np.mean(wave.astype('float64')**2)))
    assert rms>0
    noise*=rms/(10**(snr_db/20)*np.sqrt(np.mean(noise**2)))
    return (wave+noise).astype('float32')

def augment(wave,kind,rb):
    x=wave
    if kind in [1,4,5,8]:x=rb.LnL_convolutive_noise(x)
    if kind in [2,4,5,7]:x=rb.ISD_additive_noise(x)
    if kind in [3,4,6,7]:x=rb.SSI_additive_noise(x)
    if kind==8:x=rb.normWav(x+rb.SSI_additive_noise(wave),0)
    return np.asarray(x,dtype='float32')

def worker(rank,root_string,smoke):
    import importlib.util
    import soundfile as sf
    import torch
    import torch.distributed as dist
    from transformers import Wav2Vec2Model
    root=Path(root_string);protocol=json.loads((root/'protocol.json').read_text())
    policy=json.loads((root/'policy.json').read_text())
    assert digest(root/'policy.json')==protocol['policy_sha256']
    assert digest(root/'manifest.json')==protocol['manifest_sha256']
    manifest=json.loads((root/'manifest.json').read_text())
    stage='smoke' if smoke else 'pilot';out=root/stage
    torch.set_num_threads(2);torch.cuda.set_device(rank)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(protocol['seed']);np.random.seed(protocol['seed']+rank);random.seed(protocol['seed']+rank)
    device=f'cuda:{rank}'
    dist.init_process_group('nccl',init_method=f'tcp://127.0.0.1:{protocol["port"]}',rank=rank,world_size=2)
    sys.path.insert(0,str(CODE))
    from model import B_Linear,BPL
    from native_steps import features
    plm=Wav2Vec2Model.from_pretrained('facebook/wav2vec2-xls-r-300m',revision=protocol['frontend_revision'],local_files_only=True).to(device)
    plm.eval();plm.requires_grad_(False)
    torch.manual_seed(protocol['seed'])
    head=B_Linear(num_layer=25,hidden_size=1024,output_size=160,agg_size=128,loss=None,loss_bpl=None).to(device)
    assert not any('loss.weight' in key for key in head.state_dict())
    optimizer=torch.optim.Adam(head.parameters(),lr=protocol['lr'],weight_decay=protocol['weight_decay'])
    criterion=BPL(alpha=protocol['alpha'])
    torch.manual_seed(protocol['seed']+rank*10000)
    data_rng=np.random.default_rng(protocol['seed']+rank*10000)
    spec=importlib.util.spec_from_file_location('pilot_rawboost',CODE/'data_processing/augmentation/raw_boost.py')
    rb=importlib.util.module_from_spec(spec);spec.loader.exec_module(rb)
    profiles={(p['speaker'],p['attack']):np.asarray(p['conditional_probability']) for p in policy['conditional_profiles'] if p['label']==1}
    train=manifest['train_bonafide'];count=len(train)
    relative={p['speaker']:p['original_weight_multiplier'] for p in policy['original_relative_weights'] if p['label']==1}
    probabilities=np.array([relative[p['speaker']] for p in train]);probabilities/=probabilities.sum()
    targets=np.array(policy['target_samples'],dtype='int64')
    train_dir=Path('/data/ASVspoof2019/LA/ASVspoof2019_LA_train/flac')
    dev_dir=Path('/data/ASVspoof2019/LA/ASVspoof2019_LA_dev/flac')
    def read(row,training=False):
        folder=train_dir if row['split']=='T' else dev_dir
        wave,sr=sf.read(folder/(row['uid']+'.flac'),dtype='float32')
        assert sr==16000 and wave.ndim==1 and len(wave)==row['samples']
        return wave
    def draw(row):
        wave=read(row,True);bucket=min(len(wave)//8000,len(targets))
        p=profiles[(row['speaker'],'bonafide')][bucket-1]
        size=int(data_rng.choice(targets,p=p));tile=int(data_rng.integers(len(wave)//size))
        back=bool(data_rng.integers(2));start=len(wave)-(tile+1)*size if back else tile*size
        wave=wave[start:start+size]
        if protocol['rawboost']:
            wave=augment(wave,int(data_rng.integers(9)),rb)
        assert len(wave)==size and np.isfinite(wave).all()
        return wave,size
    def embedding(waves,grad=False):
        values=[]
        for start in range(0,len(waves),protocol['micro_batch_size']):
            x,mask=features(plm,[torch.from_numpy(w) for w in waves[start:start+protocol['micro_batch_size']]],device)
            with torch.set_grad_enabled(grad):
                values.append(head(x,mask=mask,return_embedding=True))
            del x,mask
        return torch.cat(values)
    def evaluate(step):
        head.eval();old_noise=[]
        for module in head.modules():
            if hasattr(module,'noise_std'):
                old_noise.append((module,module.noise_std));module.noise_std=0.
        references=manifest['reference'][:16] if smoke else manifest['reference']
        dev=manifest['dev'][:32] if smoke else manifest['dev']
        noise_ids=set(p['uid'] for p in (manifest['noise'][:8] if smoke else manifest['noise']))
        local=[]
        for source,items in [('reference',references),('dev',dev)]:
            assigned=list(enumerate(items))[rank::2]
            for start in range(0,len(assigned),protocol['micro_batch_size']):
                chunk=assigned[start:start+protocol['micro_batch_size']];waves=[]
                for _,row in chunk:
                    w=read(row)
                    if source=='reference':w=w[row['start']:row['stop']]
                    waves.append(w)
                clean=embedding(waves).detach().cpu().numpy()
                for (i,row),z in zip(chunk,clean):local.append((source,i,row['uid'],row['label'],z.tolist()))
                noisy=[(i,row,w) for (i,row),w in zip(chunk,waves) if source=='dev' and row['uid'] in noise_ids]
                if noisy:
                    z=embedding([add_noise(w,row['uid'],protocol['noise_snr_db']) for i,row,w in noisy]).detach().cpu().numpy()
                    for (i,row,w),v in zip(noisy,z):local.append(('noise',i,row['uid'],row['label'],v.tolist()))
        gathered=[None,None];dist.all_gather_object(gathered,local)
        if rank==0:
            rows=[row for part in gathered for row in part]
            arrays={name:sorted([r for r in rows if r[0]==name],key=lambda r:r[1]) for name in ['reference','dev','noise']}
            ref=np.array([r[4] for r in arrays['reference']]);z=np.array([r[4] for r in arrays['dev']])
            u=ref/np.linalg.norm(ref,axis=1,keepdims=True);prototype=u.mean(axis=0);prototype/=np.linalg.norm(prototype)
            units=z/np.linalg.norm(z,axis=1,keepdims=True);score=units@prototype;y=np.array([r[3] for r in arrays['dev']])
            diag={'step':step,'scope':'locked dev subset; diagnostic, not official full-set EER',
                  'reference':diagnostics(ref),'dev':diagnostics(z),'bona_dev':diagnostics(z[y==1]),'spoof_dev':diagnostics(z[y==0]),
                  'clean_detection':eer(y,score)}
            lookup={row[2]:(v,s) for row,v,s in zip(arrays['dev'],units,score)}
            nz=np.array([r[4] for r in arrays['noise']]);nu=nz/np.linalg.norm(nz,axis=1,keepdims=True)
            clean_noise=np.array([lookup[r[2]][0] for r in arrays['noise']]);cs=np.array([lookup[r[2]][1] for r in arrays['noise']])
            ns=nu@prototype;ny=np.array([r[3] for r in arrays['noise']])
            diag['noise']={'n':len(nu),'snr_db':protocol['noise_snr_db'],'mean_cosine_distance':float(np.mean(1-np.sum(nu*clean_noise,axis=1))),
                           'mean_absolute_score_change':float(np.abs(ns-cs).mean()),'clean_same_subset':eer(ny,cs),'noisy_detection':eer(ny,ns)}
            save(out/f'diagnostics_step{step:04d}.json',diag)
            import csv
            with (out/f'scores_step{step:04d}.csv').open('w',newline='') as stream:
                writer=csv.writer(stream);writer.writerow(['uid','label','clean_score'])
                writer.writerows((row[2],row[3],float(s)) for row,s in zip(arrays['dev'],score))
            torch.save({'step':step,'backend':head.state_dict(),'optimizer':optimizer.state_dict(),'protocol_sha256':digest(root/'protocol.json')},out/f'backend_step{step:04d}.pt')
            print(json.dumps({'event':'diagnostics',**diag}),flush=True)
        for module,value in old_noise:module.noise_std=value
        dist.barrier();head.train()
    evaluate(0)
    steps=2 if smoke else protocol['steps']
    check_steps={steps} if smoke else set(protocol['diagnostic_steps'])
    params=list(head.parameters());hist=[];start_time=time.time()
    for step in range(1,steps+1):
        chosen=data_rng.choice(count,size=24,replace=True,p=probabilities)
        drawn=[draw(train[int(i)]) for i in chosen]
        waves=[w for w,size in drawn];sizes=[int(size) for w,size in drawn]
        optimizer.zero_grad(set_to_none=True)
        z=embedding(waves,grad=True);assert z.shape==(24,160)
        loss=criterion(z[:12],z[12:]);assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
        flat=torch.cat([p.grad.view(-1) for p in params]);dist.all_reduce(flat);flat/=2
        offset=0
        for p in params:p.grad.copy_(flat[offset:offset+p.numel()].view_as(p));offset+=p.numel()
        optimizer.step()
        value=torch.tensor([float(loss.detach())],device=device);dist.all_reduce(value);value/=2
        hist.extend(sizes)
        if rank==0:
            record={'step':step,'loss':float(value.item()),'pairs_per_gpu':144,'elapsed_s':time.time()-start_time,
                    'rank0_distinct_originals':int(len(np.unique(chosen))),
                    'rank0_same_original_cross_pairs':int(np.equal.outer(chosen[:12],chosen[12:]).sum()),
                    'peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20}
            with (out/'training.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
            if step==1 or step%5==0:print(json.dumps({'event':'progress',**record}),flush=True)
        if step in check_steps:evaluate(step)
    sampled=[None,None];dist.all_gather_object(sampled,hist)
    if rank==0:
        values=np.array([x for part in sampled for x in part]);counts={str(t/16000):int((values==t).sum()) for t in targets}
        save(out/'complete.json',{'steps':steps,'waveform_draws':len(values),'target_length_counts':counts,
                                 'frontend_frozen':not any(p.requires_grad for p in plm.parameters()),
                                 'optimizer_contains_frontend':False,'OC_loss_used':False,'centroid_update_used':False,
                                 'codec_calls':0,'positive_pairs_per_gpu':144,'world_size':2,'elapsed_s':time.time()-start_time})
    dist.barrier();dist.destroy_process_group()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('root');parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    root=Path(args.root);out=root/('smoke' if args.smoke else 'pilot');out.mkdir(exist_ok=False)
    assert Path('/.dockerenv').exists() and os.environ.get('CUDA_VISIBLE_DEVICES')=='0,1'
    import torch.multiprocessing as mp
    mp.spawn(worker,args=(str(root),args.smoke),nprocs=2,join=True)

if __name__=='__main__':main()
