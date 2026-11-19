"""Evaluate all committed/preserved heads with one frozen XLSR call per batch."""
import argparse,csv,hashlib,heapq,importlib.util,json,os,time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from controlled_data_loaders import ExactDistributedEvalSampler
from role_model import RoleBLinear
from role_objectives import RoleConfig
from role_lifecycle import load_scoring_head,calculate_eer
from role_training import phase_rng
from short32_lifecycle import coordinated,gather,write_json
from tools.train_role_experiment import wait_workers

REV='1a640f32ac3e39899438a2931f9924c02f080a54'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def wrap(wave):
 a=np.asarray(wave);assert a.ndim==1 and len(a)>0 and np.isfinite(a).all()
 return np.pad(a,(0,64600-len(a)),mode='wrap') if len(a)<64600 else a[:64600]

class Waves(Dataset):
 def __init__(self,trials):self.trials=trials
 def __len__(self):return len(self.trials)
 def __getitem__(self,i):
  r=self.trials[i];wave,sr=sf.read(r['path']);assert sr==16000 and len(wave)==r['frames']
  return torch.tensor(wrap(wave),dtype=torch.float32),i,r['label'],r['uid']

def normalize_state(obj):
 state=obj.get('classifier',obj)
 assert isinstance(state,dict) and all(torch.is_tensor(v) and torch.isfinite(v).all() for v in state.values())
 if all(k.startswith('module.') for k in state):state={k[7:]:v for k,v in state.items()}
 return state

def legacy_head(state,architecture,device):
 with torch.random.fork_rng(devices=[torch.device(device).index] if str(device).startswith('cuda') else []):
  head=RoleBLinear(RoleConfig(compact_label=1),**architecture).to(device)
 head.load_state_dict(normalize_state(state),strict=True)
 return head.eval().requires_grad_(False)

def score(heads,hidden,device,rank,step):
 result={}
 for key,head in heads.items():
  with phase_rng(device,1729,0,step,rank,'matched_wrap:ITW'):
   s=head(hidden,phase='score')['score_bona'].flatten()
  assert torch.isfinite(s).all();result[key]=s.detach().cpu().tolist()
 return result

def independent_eer(labels,scores):
 rows=sorted(zip(scores,labels),reverse=True);P=sum(labels);N=len(labels)-P;assert P and N
 tp=fp=i=0;x0=0.;d0=-1.
 while i<len(rows):
  j=i
  while j<len(rows) and rows[j][0]==rows[i][0]:tp+=rows[j][1];fp+=1-rows[j][1];j+=1
  x=fp/N;d=x+tp/P-1
  if d>=0:return 100*(x0-d0/(d-d0)*(x-x0))
  x0,d0=x,d;i=j
 raise AssertionError('ROC crossing absent')

def verify(root,cfg,full_data=False):
 for n,h in json.loads((root/'control_manifest.json').read_text()).items():assert sha(root/n)==h,n
 for n,h in cfg['source'].items():assert sha(Path(cfg['source_root'])/n)==h,n
 assert sha(root/'dataset_manifest.json')==cfg['dataset_manifest_sha256']
 acceptance=json.loads((root/'dataset_acceptance.json').read_text());assert acceptance['status']=='passed' and acceptance['zip_removed']
 assert not Path(acceptance['zip_removed_path']).exists()
 for r in cfg['models']:assert sha(r['container_path'])==r['sha256'],r['key']
 if full_data:
  data=json.loads((root/'dataset_manifest.json').read_text())
  for r in data['trials']:assert sha(r['path'])==r['sha256'],r['uid']

def summarize(root,dest,heads,trials,expected,smoke,cfg,checks):
 streams=[(dest/f'ITW_rank{r}.csv').open() for r in [0,1]]
 try:
  merged=heapq.merge(*(csv.DictReader(s) for s in streams),key=lambda r:int(r['index']))
  ys=[];scores={k:[] for k in heads};seen=[];uids=set()
  for row in merged:
   i=int(row['index']);r=trials[i];assert row['uid']==r['uid'] and int(row['label'])==r['label'] and row['uid'] not in uids
   uids.add(row['uid']);seen.append(i);ys.append(r['label'])
   for k in heads:scores[k].append(float(row[k]))
  assert seen==expected
  if not smoke:assert len(ys)==31779 and sum(ys)==19963
  records={r['key']:r for r in cfg['models']};metrics={}
  for k in heads:
   assert np.isfinite(scores[k]).all();eer=calculate_eer(scores[k],ys);other=independent_eer(ys,scores[k]);assert abs(eer-other)<1e-9
   metrics[k]=dict(family=records[k]['family'],epoch=records[k]['epoch'],eer_percent=eer,independent_error_pp=abs(eer-other),checkpoint_sha256=records[k]['sha256'])
  result=dict(status='complete',scope='selected_full_batches' if smoke else 'full',rows=len(ys),bonafide=sum(ys),models=len(heads),metrics=metrics,checks=checks,
    protocol='ITW31779/wrap64600/no-mask/batch8/shared frozen XLSR/common matched_wrap1729/ASP1e-5/higher=bona',optimizer_updates=0,feature_files=0,
    frontend_calls=sum(r['frontend_calls'] for r in checks),batches=sum(r['batches'] for r in checks),score_files={f'ITW_rank{r}.csv':sha(dest/f'ITW_rank{r}.csv') for r in [0,1]})
  assert result['frontend_calls']==result['batches']
  if not smoke:
   families={}
   for r in metrics.values():families.setdefault(r['family'],[]).append(r)
   result['families']={name:dict(available_epochs=sorted(r['epoch'] for r in rows),candidates=len(rows),minimum=min(rows,key=lambda r:(r['eer_percent'],r['epoch'])),epoch50=next((r for r in rows if r['epoch']==50),None),common15_50_minimum=min((r for r in rows if 15<=r['epoch']<=50),key=lambda r:(r['eer_percent'],r['epoch']),default=None)) for name,rows in families.items()}
   result['selection_note']='Observed ITW minima are retrospective target selections, not unbiased checkpoint selection. Available epoch counts differ; complete curves and fixed50 are provided.'
  write_json(dest/'complete.json',result)
  return result
 finally:
  for s in streams:s.close()

def worker(rank,root,mode):
 import datetime
 from transformers import Wav2Vec2Model
 os.setsid();torch.set_num_threads(1);device='cuda:'+str(rank);torch.cuda.set_device(rank)
 torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 root=Path(root);dest=root/mode;cfg=json.loads((root/'manifest.json').read_text());trials=json.loads((root/'dataset_manifest.json').read_text())['trials']
 dist.init_process_group('nccl',init_method=(dest/'rendezvous').as_uri(),rank=rank,world_size=2,timeout=datetime.timedelta(minutes=20))
 try:
  def setup():
   verify(root,cfg);heads={}
   for r in cfg['models']:
    if r['kind']=='role':
     h,m=load_scoring_head(Path(r['container_path']).parent.parent,r['epoch'],device)
     assert h.role.metadata()==r['metadata']['role']
    else:h=legacy_head(torch.load(r['container_path'],map_location='cpu',weights_only=True),r['architecture'],device)
    assert h.role.score_sign==r['score_sign'] and h.architecture==r['architecture']
    heads[r['key']]=h
   plm=Wav2Vec2Model.from_pretrained('facebook/wav2vec2-xls-r-300m',revision=REV,local_files_only=True).to(device).eval().requires_grad_(False)
   assert plm.config._commit_hash==REV;return heads,plm
  heads,plm=coordinated('setup',setup)
  write_json(dest/f'ready_rank{rank}.json',dict(pid=os.getpid(),models=len(heads),parameters=sum(p.numel() for h in heads.values() for p in h.parameters())))
  ds=Waves(trials);sampler=ExactDistributedEvalSampler(ds,dict(rank=rank,world_size=2));ids=list(iter(sampler));num=(len(ids)+7)//8
  selected=None
  if mode=='smoke':
   chosen={0,1,num//2,num-1}
   for label in [0,1]:chosen.add(next(j for j,i in enumerate(ids) if trials[i]['label']==label)//8)
   selected=sorted(chosen);batches=[ids[s*8:s*8+8] for s in selected]
   loader=DataLoader(ds,batch_sampler=batches,num_workers=4,multiprocessing_context='spawn',timeout=120)
   expected_local=[i for b in batches for i in b]
  else:
   loader=DataLoader(ds,batch_size=8,sampler=sampler,num_workers=4,multiprocessing_context='spawn',timeout=120);expected_local=ids
  expected=sorted(i for part in gather(expected_local) for i in part);steps=max(gather(len(loader)));rows=front_calls=0;start=time.time();max_error=0.;iterator=iter(loader)
  stream=(dest/f'ITW_rank{rank}.csv').open('x',newline='')
  try:
   writer=csv.writer(stream);writer.writerow(['index','uid','label']+list(heads))
   with torch.no_grad():
    for localstep in range(steps):
     def infer():
      nonlocal rows,front_calls,max_error
      if localstep>=len(loader):return
      step=selected[localstep] if selected is not None else localstep
      waves,indices,labels,uids=next(iterator);assert waves.shape[1]==64600
      hidden=torch.stack(plm(waves.to(device),output_hidden_states=True).hidden_states,dim=1);front_calls+=1
      values=score(heads,hidden,device,rank,step)
      if mode=='smoke':
       # Serial head calls, reordered on the same frozen tensor. No second XLSR call.
       serial=score(dict(reversed(list(heads.items()))),hidden.clone(),device,rank,step)
       for k in heads:
        error=max(abs(a-b) for a,b in zip(values[k],serial[k]));assert error==0;max_error=max(max_error,error)
      for j in range(len(labels)):writer.writerow([int(indices[j]),uids[j],int(labels[j])]+[values[k][j] for k in heads])
      stream.flush();rows+=len(labels)
      write_json(dest/f'progress_rank{rank}.json',dict(stage=mode,rows=rows,total=len(expected_local),batches=localstep+1,total_batches=len(loader),models=len(heads),frontend_calls=front_calls,elapsed_seconds=time.time()-start,updated_unix=time.time()))
      del hidden,values,waves
     coordinated('batch',infer)
   def exhaust():
    try:next(iterator)
    except StopIteration:return
    raise AssertionError('Extra scheduled batch')
   coordinated('exhaust',exhaust)
  finally:
   stream.close()
   if hasattr(iterator,'_shutdown_workers'):iterator._shutdown_workers()
  assert rows==len(expected_local) and front_calls==len(loader)
  assert all(p.grad is None and not p.requires_grad for h in heads.values() for p in h.parameters())
  assert all(p.grad is None and not p.requires_grad for p in plm.parameters())
  checks=gather(dict(rank=rank,rows=rows,batches=len(loader),frontend_calls=front_calls,models=len(heads),serial_shared_max_error=max_error,peak_cuda_bytes=torch.cuda.max_memory_allocated(device)))
  coordinated('integrity',lambda:verify(root,cfg,full_data=rank==0))
  coordinated('summary',lambda:summarize(root,dest,heads,trials,expected,mode=='smoke',cfg,checks) if rank==0 else None)
 finally:dist.destroy_process_group()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('mode',choices=['smoke','full']);p.add_argument('--timeout-hours',type=float,default=24.);a=p.parse_args()
 assert 0<a.timeout_hours<=24
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0,1' and torch.cuda.device_count()==2
 root=Path(a.root)
 if a.mode=='full':assert json.loads((root/'smoke/complete.json').read_text())['status']=='complete'
 (root/a.mode).mkdir(exist_ok=False)
 context=torch.multiprocessing.spawn(worker,args=(str(root),a.mode),nprocs=2,join=False)
 wait_workers(context,min(a.timeout_hours*3600,7200 if a.mode=='smoke' else 24*3600))
