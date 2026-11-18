"""Shared frozen frontend evaluation; no optimizer, feature files or model copies."""
import argparse
import csv
import hashlib
import heapq
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from controlled_data_loaders import EnrollmentSet, ExactDistributedEvalSampler, collate_eval
from role_lifecycle import load_scoring_head, calculate_eer, wrapped_features
from role_training import phase_rng
from short32_lifecycle import coordinated, gather, write_json
from tools.train_role_experiment import wait_workers

COUNTS={'19LA':(71237,7355),'21LA':(148176,14816),'21DF':(533928,14869)}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def independent_eer(labels,scores):
    pairs=sorted(zip(scores,labels),reverse=True);P=sum(labels);N=len(labels)-P
    assert P and N
    tp=fp=0;oldx=0.;oldd=-1.;i=0
    while i<len(pairs):
        j=i
        while j<len(pairs) and pairs[j][0]==pairs[i][0]:
            tp+=pairs[j][1];fp+=1-pairs[j][1];j+=1
        x=fp/N;d=x+tp/P-1
        if d>=0:return 100*(oldx-oldd/(d-oldd)*(x-oldx))
        oldx,oldd=x,d;i=j
    raise AssertionError('ROC did not cross')


def probe_steps(trials,rank):
    n=len(range(rank,len(trials),2));steps=(n+7)//8
    chosen={0,1,steps//2,steps-1}
    for y in [0,1]:
        first=next(i for i in range(rank,len(trials),2) if trials[i].label==y)
        chosen.add((first//2)//8)
    return sorted(chosen)


def verify(root,cfg):
    for name,h in json.loads((root/'control_manifest.json').read_text()).items():assert digest(root/name)==h,name
    for name,h in cfg['source'].items():assert digest(Path(cfg['source_root'])/name)==h,name
    for name,h in cfg['controls'].items():assert digest(root/name)==h,name
    for item in cfg['preserved']:
        assert digest(item['record_path'])==item['record_sha256']
        assert digest(item['score_path'])==item['score_sha256']


def reference_rows(cfg,phase,indices):
    out={}
    for item in cfg['preserved']:
        if item['phase']!=phase:continue
        chosen={}
        with Path(item['score_path']).open() as f:
            for row in csv.DictReader(f):
                i=int(row['index'])
                if i in indices:chosen[i]=row
        assert set(chosen)==indices
        out['e'+str(item['epoch'])]=chosen
    return out


def score_heads(heads,hidden,device,rank,step,phase):
    result={}
    for key,head in heads.items():
        with phase_rng(device,1729,0,step,rank,'matched_wrap:'+phase):
            s=head(hidden,phase='score')['score_bona'].flatten()
        assert bool(torch.isfinite(s).all())
        result[key]=s.detach().cpu().tolist()
    return result


def summarize(dest,phase,heads,trials,hashes,smoke,expected_indices):
    paths=[dest/(phase+'_rank%d.csv'%r) for r in [0,1]]
    streams=[p.open() for p in paths]
    try:
        merged=heapq.merge(*(csv.DictReader(f) for f in streams),key=lambda r:int(r['index']))
        labels=[];scores={key:[] for key in heads};uids=set();seen=[]
        for row in merged:
            i=int(row['index']);seen.append(i);uid=row['uid'];y=int(row['label'])
            assert uid==Path(trials[i].path).stem and y==trials[i].label and uid not in uids
            uids.add(uid);labels.append(y)
            for key in heads:scores[key].append(float(row[key]))
        assert seen==expected_indices
        if not smoke:assert (len(seen),sum(labels))==COUNTS[phase]
        metrics={}
        for key in heads:
            eer=calculate_eer(scores[key],labels);check=independent_eer(labels,scores[key])
            assert abs(eer-check)<1e-9
            metrics[key]=dict(EER_percent=eer,independent_error_pp=abs(eer-check),checkpoint_sha256=hashes[key])
        out=dict(status='complete',phase=phase,rows=len(seen),bonafide=sum(labels),scope='selected_full_batches' if smoke else 'full',metrics=metrics,
                 protocol='wrap64600/no-mask/batch8/frozen XLSR/matched_wrap1729/ASP1e-5/higher=bona',
                 score_files={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in paths})
        write_json(dest/('metrics_'+phase+'.json'),out)
        return out
    finally:
        for f in streams:f.close()


def worker(rank,root,mode):
    import datetime
    from dataset import ASVspoof2019_LA_Eval, ASVspoof2021_LA_Eval, ASVspoof2021_DF_Eval
    from transformers import Wav2Vec2Model
    os.setsid();torch.set_num_threads(1);device='cuda:'+str(rank);torch.cuda.set_device(rank)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    root=Path(root);dest=root/mode;cfg=json.loads((root/'manifest.json').read_text())
    dist.init_process_group('nccl',init_method=(dest/'rendezvous').as_uri(),rank=rank,world_size=2,timeout=datetime.timedelta(minutes=20))
    try:
        def setup():
            verify(root,cfg);heads={};hashes={}
            for e in range(5,51,5):
                key='e'+str(e);head,m=load_scoring_head(cfg['result_root'],e,device)
                assert head.role.metadata()==cfg['expected_role']
                assert m['files'][f'CLS_{e}.pt']['sha256']==cfg['checkpoint_sha256'][str(e)]
                heads[key]=head;hashes[key]=m['files'][f'CLS_{e}.pt']['sha256']
            plm=Wav2Vec2Model.from_pretrained('facebook/wav2vec2-xls-r-300m',revision=cfg['frontend_revision'],local_files_only=True).to(device).eval().requires_grad_(False)
            assert plm.config._commit_hash==cfg['frontend_revision']
            assert not plm.training and all(not p.requires_grad for p in plm.parameters())
            db={'19LA':ASVspoof2019_LA_Eval('/data/ASVspoof2019'), '21LA':ASVspoof2021_LA_Eval('/data/ASVspoof2021_LA_eval'),'21DF':ASVspoof2021_DF_Eval('/data/ASVspoof2021_DF')}
            return heads,hashes,plm,db
        heads,hashes,plm,db=coordinated('setup',setup)
        write_json(dest/f'ready_rank{rank}.json',dict(pid=os.getpid(),models=list(heads),hashes=hashes))
        checks=[]
        phases=['19LA','21LA','21DF'] if mode=='smoke' else ['21LA','21DF']
        for phase in phases:
            trials=db[phase].trials;assert (len(trials),sum(t.label for t in trials))==COUNTS[phase]
            active=heads if mode=='smoke' else {k:heads[k] for k in cfg['pending'][phase]}
            if not active:continue
            ds=EnrollmentSet(dict(native_length=False,cap_long=True,num_test_frames=404,num_seg=1),trials)
            sampler=ExactDistributedEvalSampler(ds,dict(rank=rank,world_size=2))
            if mode=='smoke':
                selected=probe_steps(trials,rank);ids=list(iter(sampler));batches=[ids[s*8:s*8+8] for s in selected]
                loader=DataLoader(ds,batch_sampler=batches,num_workers=4,collate_fn=collate_eval,multiprocessing_context='spawn',timeout=120)
                expected_local=[i for batch in batches for i in batch];refs=reference_rows(cfg,phase,set(expected_local))
            else:
                selected=None;refs={};expected_local=list(iter(sampler))
                loader=DataLoader(ds,batch_size=8,sampler=sampler,num_workers=4,collate_fn=collate_eval,multiprocessing_context='spawn',timeout=120)
            expected=sorted(i for part in gather(expected_local) for i in part)
            count=coordinated('loader_length',lambda:len(loader));steps=max(gather(count));rows_done=0;front_calls=0;start=time.time();max_diff=0.;saved_diff=0.;saved_checks=0
            stream=(dest/(phase+f'_rank{rank}.csv')).open('x',newline='')
            iterator=iter(loader)
            try:
                writer=csv.writer(stream);writer.writerow(['index','uid','label']+list(active))
                with torch.no_grad():
                    for localstep in range(steps):
                        def infer():
                            nonlocal rows_done,front_calls,max_diff,saved_diff,saved_checks
                            if localstep>=count:return
                            step=selected[localstep] if selected is not None else localstep
                            batch=next(iterator)
                            hidden,mask=wrapped_features(plm,batch['waves'],device);assert mask is None;front_calls+=1
                            values=score_heads(active,hidden,device,rank,step,phase)
                            if mode=='smoke':
                                separate,_=wrapped_features(plm,batch['waves'],device);front_calls+=1
                                baseline=score_heads(active,separate,device,rank,step,phase)
                                for key in active:
                                    difference=max(abs(a-b) for a,b in zip(values[key],baseline[key]));assert difference<=1e-7;max_diff=max(max_diff,difference)
                                del separate
                            for j,(i,y) in enumerate(zip(batch['indices'],batch['labels'])):
                                i=int(i);y=int(y);uid=Path(trials[i].path).stem
                                writer.writerow([i,uid,y]+[values[key][j] for key in active])
                                for key,table in refs.items():
                                    ref=table[i];assert ref['uid']==uid and int(ref['label'])==y
                                    difference=abs(values[key][j]-float(ref['score_bona']));assert difference<=1e-6,(phase,key,i,difference)
                                    saved_diff=max(saved_diff,difference);saved_checks+=1
                            rows_done+=len(batch['labels']);stream.flush()
                            write_json(dest/(phase+f'_progress_rank{rank}.json'),dict(status='running',phase=phase,rows=rows_done,total=len(sampler),batches=localstep+1,total_batches=count,
                                models=list(active),frontend_calls=front_calls,elapsed_seconds=time.time()-start,rows_per_second=rows_done/max(.001,time.time()-start),updated_unix=time.time()))
                            del hidden,batch,values
                        coordinated(phase+':batch',infer)
                def exhaust():
                    try:next(iterator)
                    except StopIteration:return
                    raise AssertionError('Unexpected extra batch')
                coordinated(phase+':exhaust',exhaust)
            finally:
                stream.close()
                if hasattr(iterator,'_shutdown_workers'):iterator._shutdown_workers()
            check=dict(rank=rank,phase=phase,rows=rows_done,frontend_calls=front_calls,batches=count,reference_score_max_error=max_diff,saved_score_max_error=saved_diff,saved_score_checks=saved_checks,models=len(active),peak_cuda_bytes=torch.cuda.max_memory_allocated(device))
            assert rows_done==len(expected_local) and front_calls==count*(2 if mode=='smoke' else 1)
            checks.extend(gather(check))
            coordinated(phase+':summary',lambda:summarize(dest,phase,active,trials,hashes,mode=='smoke',expected) if rank==0 else None)
        coordinated('final_integrity',lambda:verify(root,cfg))
        if rank==0:
            if mode=='smoke':assert sum(c['saved_score_checks'] for c in checks)>0
            result=dict(status='complete',mode=mode,checks=checks,optimizer_updates=0,feature_files=0,preserved_cells=len(cfg['preserved']))
            if mode=='full':
                cells={}
                for item in cfg['preserved']:
                    m=json.loads(Path(item['record_path']).read_text());assert m['status']=='complete'
                    cells[(item['phase'],item['epoch'])]=dict(phase=item['phase'],epoch=item['epoch'],eer_percent=m['eer_percent'],rows=m['rows'],source='preserved',receipt=item['record_path'])
                for phase in ['21LA','21DF']:
                    if not cfg['pending'][phase]:continue
                    m=json.loads((dest/('metrics_'+phase+'.json')).read_text());assert m['status']=='complete'
                    for key,v in m['metrics'].items():
                        e=int(key[1:]);assert (phase,e) not in cells
                        cells[(phase,e)]=dict(phase=phase,epoch=e,eer_percent=v['EER_percent'],rows=m['rows'],source='shared_frontend',receipt=str(dest/('metrics_'+phase+'.json')))
                assert set(cells)=={(p,e) for p in COUNTS for e in range(5,51,5)}
                result['results']=[cells[k] for k in sorted(cells)]
                result['minimum_eer']={p:min((v for (phase,e),v in cells.items() if phase==p),key=lambda v:(v['eer_percent'],v['epoch'])) for p in COUNTS}
            write_json(dest/'complete.json',result)
    finally:dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('mode',choices=['smoke','full']);p.add_argument('--timeout-hours',type=float,default=24.);a=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0,1' and torch.cuda.device_count()==2
    root=Path(a.root)
    if a.mode=='full':assert json.loads((root/'smoke/complete.json').read_text())['status']=='complete'
    (root/a.mode).mkdir(exist_ok=False)
    assert 0<a.timeout_hours<=168
    ctx=torch.multiprocessing.spawn(worker,args=(str(root),a.mode),nprocs=2,join=False)
    wait_workers(ctx,min(a.timeout_hours*3600,7200 if a.mode=='smoke' else 24*3600))
