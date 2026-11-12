import os,sys,json,random,time
from pathlib import Path
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
sys.dont_write_bytecode=True
import bpl_frozen_audit_20261030_v1 as base
ROOT=base.ROOT/'representation_patch'
def worker(rank,cfg,items):
 import numpy as np,soundfile as sf,torch
 from transformers import Wav2Vec2Model
 sys.path.insert(0,str(base.SOURCE));from model import B_Linear,BPL,OCSoftmax
 torch.set_num_threads(2);torch.cuda.set_device(rank);torch.manual_seed(39402+rank);device=torch.device(f'cuda:{rank}')
 net=B_Linear(num_layer=25,hidden_size=1024,output_size=160,agg_size=128,loss=OCSoftmax(160),loss_bpl=BPL(30))
 state=torch.load(base.CKPT,map_location='cpu',weights_only=True);net.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True);net.to(device).eval()
 for module in net.modules():
  if hasattr(module,'noise_std'):module.noise_std=0.
 plm=Wav2Vec2Model.from_pretrained(cfg['plm'],local_files_only=True).to(device).eval();center=torch.nn.functional.normalize(net.loss.weight.detach(),dim=1)
 def classify(h):
  z=net.layer_norm(net.linear(net.AHP_ASP(net.bridge_module(h))));z=torch.nn.functional.normalize(z,dim=1)
  return float((z@center.T).item()),z.cpu().numpy()[0]
 inds=[];scores=[];emb=[];checks=[];started=time.time()
 with torch.inference_mode():
  for k,idx in enumerate(range(rank,len(items),2)):
   row=items[idx];raw,sr=sf.read(row['path'],dtype='float32');assert sr==16000;a=base.crop(raw,64600);m,l,t=base.regions(a);b,_=base.shuffle_quiet(a,np.random.default_rng(81203+row['view_index']))
   h=torch.stack(plm(torch.from_numpy(np.stack([a,b])).to(device),output_hidden_states=True).hidden_states,dim=1);orig=h[0:1];changed=h[1:2];T=h.shape[2]
   receptive=1;jump=1
   for kernel,stride in zip(plm.config.conv_kernel,plm.config.conv_stride):receptive+=(kernel-1)*jump;jump*=stride
   starts=np.arange(T)*jump;ends=starts+receptive;quiet=(ends<=l)|(starts>=t);qm=torch.from_numpy(quiet).to(device);active=~qm
   assert active.sum()>1
   hybrid_quiet=orig.clone();hybrid_quiet[:,:,qm,:]=changed[:,:,qm,:]
   hybrid_active=orig.clone();hybrid_active[:,:,active,:]=changed[:,:,active,:]
   hs=[orig,changed,hybrid_quiet,hybrid_active,orig[:,:,active,:],changed[:,:,active,:]];s=[];z=[]
   for hv in hs:u,v=classify(hv);s.append(u);z.append(v)
   diff=(changed-orig).square().mean(dim=(0,3)).sqrt().cpu().numpy() # L,T
   cc=[]
   for layer in [0,12,24]:cc.extend([float(diff[layer,~quiet].mean()),float(diff[layer,quiet].mean()) if quiet.any() else 0.])
   # Same post-SSL kept-frame count for original/changed; speech samples identical.
   assert np.array_equal(a[~m],b[~m]);checks.append([int(quiet.sum()),int(active.sum()),int(np.count_nonzero(a!=b)),*cc]);inds.append(idx);scores.append(s);emb.append(z)
   if k==0 or (k+1)%20==0:print('PATCH',rank,k+1,'seconds',round(time.time()-started),flush=True)
 np.savez_compressed(ROOT/f'shard_{rank}.npz',indices=inds,scores=scores,z=emb,checks=checks)
def main():
 assert Path('/.dockerenv').exists() and os.environ.get('CUDA_VISIBLE_DEVICES')=='0,1' and (base.ROOT/'COMPLETED.json').exists()
 import torch
 assert torch.cuda.device_count()==2;ROOT.mkdir(exist_ok=False);items=json.loads((base.ROOT/'items_private.json').read_text());rng=random.Random(20261031);selected=[]
 for ds in ['19LAdev','21DF']:
  for lab in [1,0]:selected+=rng.sample([dict(row,view_index=i) for i,row in enumerate(items) if row['dataset']==ds and row['label']==lab],24)
 cfg=json.loads((base.ROOT/'plan.json').read_text());(ROOT/'items_private.json').write_text(json.dumps(selected,indent=2));(ROOT/'plan.json').write_text(json.dumps({'n':96,'selection':'Random24 per dataset/class, no effect selection.','names':['original','quiet_changed','patch_quiet_positions_only','patch_other_positions_only','original_exclude_quiet_SSL_positions','changed_exclude_quiet_SSL_positions'],'method':'Quiet positions require full conv receptive field inside edge ROI. Patch all25SSLlayers. Noise_std temporarily0; eval; noweightsupdates. Exclusion removes quiet positions from every backendpoolingstage.','limits':'Hybrid feature combinations are mechanistic interventions, not natural audio or additive causal importance. Quiet proxy not verified nonspeech; contextual representations can carry global information.'},indent=2))
 torch.multiprocessing.spawn(worker,args=(cfg,selected),nprocs=2,join=True);(ROOT/'COMPLETED.json').write_text(json.dumps({'n':96,'checkpoint_sha256':base.digest(base.CKPT)}));print('PATCH_COMPLETE',flush=True)
if __name__=='__main__':main()
