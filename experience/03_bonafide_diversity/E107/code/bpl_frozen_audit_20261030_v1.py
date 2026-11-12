import os,sys,json,hashlib,time,random
from pathlib import Path
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2');sys.dont_write_bytecode=True
ROOT=Path('/results/analysis/EBM80_BPL_frozen_audit_20261030_v1');SOURCE=Path('/home/mentee/results/CM/LA/EBM/backup/scripts');CKPT=SOURCE.parent.parent/'model/BestCLS_80_21DF.pt'
DF=Path('/data/ASVspoof2021_DF/ASVspoof2021_DF_eval/flac');BANK=Path('/results/analysis/EBM80_quiet_donor_audit')
def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def crop(a,n,start=0):
 import numpy as np
 return np.pad(a,(0,n-len(a)),'wrap') if len(a)<n else a[start:start+n].copy()
def regions(a):
 import numpy as np
 rr=np.array([np.sqrt(np.mean(a[q:q+320].astype(float)**2)+1e-20) for q in range(0,len(a),320)]);active=np.flatnonzero(rr>=rr.max()*.01)
 m=np.zeros(len(a),bool);l=int(active[0])*320;t=min((int(active[-1])+1)*320,len(a));m[:l]=True;m[t:]=True
 return m,l,t
def shuffle_quiet(a,rng):
 import numpy as np
 m,l,t=regions(a);b=a.copy()
 for q in range(0,len(a),320):
  if m[q]:b[q:q+320]=rng.permutation(a[q:q+320])
 assert np.array_equal(a[~m],b[~m])
 return b,[l/16000,(len(a)-t)/16000,int(np.count_nonzero(a!=b))]
def worker(rank,cfg,items,rec):
 import numpy as np,soundfile as sf,torch
 from types import SimpleNamespace
 from transformers import Wav2Vec2Model
 sys.path.insert(0,str(SOURCE));from model import B_Linear,BPL,OCSoftmax
 from data_loaders import TrainSet
 torch.set_num_threads(2);torch.cuda.set_device(rank);random.seed(74001+rank);np.random.seed(74001+rank);torch.manual_seed(74001+rank);dev=torch.device(f'cuda:{rank}')
 state=torch.load(CKPT,map_location='cpu',weights_only=True);state={k.removeprefix('module.'):v for k,v in state.items()}
 net=B_Linear(num_layer=25,hidden_size=1024,output_size=160,agg_size=128,loss=OCSoftmax(160),loss_bpl=BPL(30));net.load_state_dict(state,strict=True);net.to(dev).eval()
 plm=Wav2Vec2Model.from_pretrained(cfg['plm'],local_files_only=True).to(dev).eval();center=torch.nn.functional.normalize(net.loss.weight.detach(),dim=1)
 def infer(arrays):
  x=torch.from_numpy(np.stack(arrays).astype(np.float32)).to(dev);h=torch.stack(plm(x,output_hidden_states=True).hidden_states,dim=1)
  z=net.layer_norm(net.linear(net.AHP_ASP(net.bridge_module(h))));z=torch.nn.functional.normalize(z,dim=1)
  return z.cpu().numpy(),(z@center.T).squeeze(1).cpu().numpy()
 inds=[];zz=[];ss=[];ft=[];started=time.time();toggle=None
 with torch.inference_mode():
  for k,idx in enumerate(range(rank,len(items),2)):
   row=items[idx];raw,sr=sf.read(row['path'],dtype='float32');assert sr==16000 and raw.ndim==1
   if 'pcm_sha256' in row:assert hashlib.sha256(raw.tobytes()).hexdigest()==row['pcm_sha256']
   rng=np.random.default_rng(81203+idx);random.seed(81203+idx);np.random.seed(81203+idx)
   L=crop(raw,64600);offset=int(rng.integers(max(1,len(raw)-48440+1)));S=crop(raw,48440,offset);Lq,fl=shuffle_quiet(L,rng);Sq,fs=shuffle_quiet(S,rng)
   ds=TrainSet({'num_train_frames':404,'num_train_frames_short':303},[SimpleNamespace(path=row['path'],label=row['label'])]);La,Sa,_=ds[0];La=np.asarray(La,dtype=np.float32);Sa=np.asarray(Sa,dtype=np.float32);assert len(La)==64600 and len(Sa)==48440
   zl,sl=infer([L,Lq,La]);zs,sS=infer([S,Sq,Sa]);repeat=sl[0]
   if k<20:_,r=infer([L]);repeat=r[0]
   if toggle is None:
    pools=[(m,m.noise_std) for m in net.modules() if hasattr(m,'noise_std')]
    for m,_ in pools:m.noise_std=0.
    h=torch.stack(plm(torch.from_numpy(L[None]).to(dev),output_hidden_states=True).hidden_states,dim=1);keep=net.loss_bpl;_,a=net(h);net.loss_bpl=None;_,b=net(h);toggle=float((a-b).abs().max());net.loss_bpl=keep
    for m,v in pools:m.noise_std=v
   inds.append(idx);zz.append(np.stack([zl[0],zs[0],zl[1],zs[1],zl[2],zs[2]]));ss.append([sl[0],sS[0],sl[1],sS[1],sl[2],sS[2],repeat]);ma,la,ta=regions(La);mb,lb,tb=regions(Sa);ft.append(fl+fs+[offset/16000,la/16000,lb/16000,int(ma.sum()),int(mb.sum())])
   if k==0 or (k+1)%50==0:print('VIEWS',rank,k+1,'seconds',round(time.time()-started),flush=True)
  np.savez_compressed(ROOT/f'views_{rank}.npz',indices=inds,z=zz,scores=ss,features=ft,center=center.cpu().numpy(),bpl_toggle_max=toggle)
  bank=np.load(BANK/'donor_frames_private.npz')['frames'];meta=json.loads((BANK/'donor_manifest_private.json').read_text());groups={}
  for row in meta:groups.setdefault((row['source'],row['speaker'],row['codec']),{0:[],1:[]})[row['label']].append(row)
  inds=[];zz=[];ss=[];checks=[];dm=[]
  for k,idx in enumerate(range(rank,len(rec),2)):
   row=rec[idx];raw,sr=sf.read(row['path'],dtype='float32');assert sr==16000 and hashlib.sha256(raw.tobytes()).hexdigest()==row['pcm_sha256'];a=crop(raw,64600);m,l,t=regions(a)
   rng=np.random.default_rng(99431+idx);group=groups[(row['source'],row['speaker'],row['codec'])];views=[a,a.copy()];chosen=[];err=0.;changed=[]
   for draw in range(2):
    for lab in [1,0]:
     donor=group[lab][int(rng.integers(len(group[lab])))];chosen.append(donor['id']);b=a.copy()
     for q in row['eligible_frame_indices']:
      lo=q*320+16;hi=(q+1)*320-16;v=a[lo:hi].astype(float);u=bank[int(rng.choice(donor['frame_ids']))].astype(float);u=(u-u.mean())/u.std();b[lo:hi]=(u*v.std()+v.mean()).astype(np.float32)
      err=max(err,abs(float(b[lo:hi].astype(float).mean())-v.mean()),abs(float(np.mean(b[lo:hi].astype(float)**2))-np.mean(v*v))/max(np.mean(v*v),1e-20))
     assert np.array_equal(a[l:],b[l:])
     for q in range(l//320):assert np.array_equal(a[q*320:q*320+16],b[q*320:q*320+16]) and np.array_equal(a[(q+1)*320-16:(q+1)*320],b[(q+1)*320-16:(q+1)*320])
     views.append(b);changed.append(int(np.count_nonzero(a!=b)))
   b=a.copy()
   for q in row['eligible_frame_indices']:
    lo=q*320+16;hi=(q+1)*320-16;b[lo:hi]=rng.permutation(a[lo:hi])
   views.append(b);z=[];s=[]
   for j in range(0,len(views),4):u,v=infer(views[j:j+4]);z.extend(u);s.extend(v)
   inds.append(idx);zz.append(z);ss.append(s);checks.append([err,l/16000,len(row['eligible_frame_indices']),*changed]);dm.append(chosen)
   if k==0 or (k+1)%50==0:print('DONOR',rank,k+1,'seconds',round(time.time()-started),flush=True)
  np.savez_compressed(ROOT/f'donor_{rank}.npz',indices=inds,z=zz,scores=ss,checks=checks,donor_ids=np.array(dm))
 print('WORKER_DONE',rank,'seconds',round(time.time()-started),flush=True)
def main():
 assert Path('/.dockerenv').exists() and os.environ.get('CUDA_VISIBLE_DEVICES')=='0,1'
 import torch
 from huggingface_hub import snapshot_download
 assert torch.cuda.device_count()==2;ROOT.mkdir(exist_ok=False);rng=random.Random(20261030);items=[]
 tr=json.loads(Path('/results/analysis/EBM80_cue_factor_audit/sample_manifest.json').read_text())
 for lab in [1,0]:items+=rng.sample([dict(t,path=str(DF/(t['id']+'.wav')),dataset='21DF') for t in tr if t['label']==lab],200)
 rows=[];proto=Path('/data/ASVspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt')
 for line in proto.read_text().splitlines():
  f=line.split();rows.append(dict(id=f[1],speaker=f[0],label=int(f[4]=='bonafide'),source='19LAdev',codec='original',dataset='19LAdev',path=str(Path('/data/ASVspoof2019/LA/ASVspoof2019_LA_dev/flac')/(f[1]+'.flac'))))
 for lab in [1,0]:items+=rng.sample([t for t in rows if t['label']==lab],200)
 rng.shuffle(items);meta=json.loads((BANK/'donor_manifest_private.json').read_text());keys={}
 for t in meta:keys.setdefault((t['source'],t['speaker'],t['codec']),set()).add(t['label'])
 rec=[dict(t,path=str(DF/(t['id']+'.wav'))) for t in json.loads((BANK/'recipient_manifest_private.json').read_text()) if t['eligible_frame_indices'] and keys.get((t['source'],t['speaker'],t['codec']))=={0,1}]
 cfg={'plm':snapshot_download('facebook/wav2vec2-xls-r-300m',local_files_only=True),'seed':20261030};plan=dict(cfg,checkpoint_sha256=digest(CKPT),script_sha256=digest(Path(__file__)),views_n=len(items),donor_n=len(rec),purpose='Frozen model diagnostic, not BPL removal or retraining. Actual TrainSet augmentation viewed with eval features; not historical training loss.',view_names=['long','short_random_fullfile_noaug','long_quiet_shuffle','short_quiet_shuffle','actual_train_long','actual_train_short'],donor_names=['baseline','identity','bona_donor_draw1','spoof_donor_draw1','bona_donor_draw2','spoof_donor_draw2','inner_shuffle'],donor_controls='Match source/speaker/codec; preserve recipient total length, all samples outside leading quiet, frame boundary16samples, inner mean and variance;two donor draws.',limitations=['Quiet proxy -40dB20ms not groundtruth nonspeech;genuine synthesis traces may be changed.','Cannot identify causal BPL training effect.','Reused21DF samples;shared donors;no externaldomain validation.'])
 (ROOT/'items_private.json').write_text(json.dumps(items,indent=2));(ROOT/'donor_recipients_private.json').write_text(json.dumps(rec,indent=2));(ROOT/'plan.json').write_text(json.dumps(plan,indent=2));print('PLAN',json.dumps(plan),flush=True)
 torch.multiprocessing.spawn(worker,args=(cfg,items,rec),nprocs=2,join=True)
 assert digest(CKPT)==plan['checkpoint_sha256'];(ROOT/'COMPLETED.json').write_text(json.dumps({'views':len(items),'donors':len(rec),'checkpoint_unchanged':True}));print('AUDIT_COMPLETED',flush=True)
if __name__=='__main__':main()
