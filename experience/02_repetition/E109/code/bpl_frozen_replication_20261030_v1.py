import os,sys,json,time
from pathlib import Path
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2');sys.dont_write_bytecode=True
import bpl_frozen_audit_20261030_v1 as base
ROOT=base.ROOT/'checkpoint_replication'
PATHS=[Path('/home/mentee/results/CM/LA/EBM/model')/f'BestCLS_{e}_21DF.pt' for e in [65,70,75,80]]+[Path('/home/mentee/results/CM/LA/EBM_e50_bs160_lr1e5/model/BestCLS_50_21DF.pt')]
NAMES=['old65','old70','old75','old80','new50']
def worker(rank,cfg,rows):
 import torch,numpy as np,soundfile as sf
 from transformers import Wav2Vec2Model
 sys.path.insert(0,str(base.SOURCE));from model import B_Linear,BPL,OCSoftmax
 torch.set_num_threads(2);torch.cuda.set_device(rank);device=torch.device(f'cuda:{rank}');nets=[];centers=[]
 for path in PATHS:
  net=B_Linear(num_layer=25,hidden_size=1024,output_size=160,agg_size=128,loss=OCSoftmax(160),loss_bpl=BPL(30));state=torch.load(path,map_location='cpu',weights_only=True);net.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True);net.to(device).eval()
  for module in net.modules():
   if hasattr(module,'noise_std'):module.noise_std=0.
  nets.append(net);centers.append(torch.nn.functional.normalize(net.loss.weight.detach(),dim=1))
 plm=Wav2Vec2Model.from_pretrained(cfg['plm'],local_files_only=True).to(device).eval();ix=[];ss=[];zz=[]
 with torch.inference_mode():
  for k,i in enumerate(range(rank,len(rows),2)):
   row=rows[i];raw,sr=sf.read(row['path'],dtype='float32');assert sr==16000;a=base.crop(raw,64600);b,_=base.shuffle_quiet(a,np.random.default_rng(81203+row['view_index']));h=torch.stack(plm(torch.from_numpy(np.stack([a,b])).to(device),output_hidden_states=True).hidden_states,dim=1);scores=[];zall=[]
   for net,c in zip(nets,centers):
    z=torch.nn.functional.normalize(net.layer_norm(net.linear(net.AHP_ASP(net.bridge_module(h)))),dim=1);scores.append((z@c.T).squeeze(1).cpu().numpy());zall.append(z.cpu().numpy())
   ix.append(i);ss.append(scores);zz.append(zall)
 np.savez_compressed(ROOT/f'shard_{rank}.npz',indices=ix,scores=ss,z=zz)
def main():
 assert Path('/.dockerenv').exists() and os.environ.get('CUDA_VISIBLE_DEVICES')=='0,1'
 import torch
 assert torch.cuda.device_count()==2;ROOT.mkdir(exist_ok=False);rows=json.loads((base.ROOT/'representation_patch/items_private.json').read_text());cfg=json.loads((base.ROOT/'plan.json').read_text());before=[base.digest(p) for p in PATHS]
 (ROOT/'plan.json').write_text(json.dumps({'names':NAMES,'paths':[str(p) for p in PATHS],'checkpoint_sha256':before,'n':len(rows),'note':'Same96randomcases as feature patch, no effect selection. Common frozen XLSR justified byT0=150forbothtrainingruns;noise_std0. Not independenttrainingseeds or BPLablation.'},indent=2));worker_args=(cfg,rows);torch.multiprocessing.spawn(worker,args=worker_args,nprocs=2,join=True)
 assert before==[base.digest(p) for p in PATHS];(ROOT/'COMPLETED.json').write_text(json.dumps({'n':len(rows),'checkpoint_unchanged':True}));print('REPLICATION_COMPLETE',flush=True)
if __name__=='__main__':main()
