"""Fixed201-frame classifier control: repeat before vs after the frontend.

All SSL tensors stay in GPU memory. Layer block swaps are descriptive internal
interventions, not physically realizable audio or a trained replacement model.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
BASE=Path('/home/mentee/analysis/embedding_geometry/20261106_padding_causal_01')
sys.path.insert(0,str(BASE))
import padding_causal_sidecar_v1 as ex
core,ref,REFERENCE=ex.core,ex.ref,ex.REFERENCE


def worker(rank,root_s):
    import torch
    import soundfile as sf
    from transformers import AutoConfig,Wav2Vec2Model
    root=Path(root_s);torch.set_num_threads(2);torch.cuda.set_device(rank);device=torch.device('cuda',rank)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
    archived,args=core.settings(REFERENCE);sys.path.insert(0,str(archived));import model
    config=AutoConfig.from_pretrained(args['PLM_name'],local_files_only=True)
    plm=Wav2Vec2Model.from_pretrained(args['PLM_name'],config=config,local_files_only=True).to(device).eval()
    cls=model.B_Linear(num_layer=args['num_layers'],hidden_size=args['hidden_size'],output_size=args['output_size'],agg_size=args['agg_size'],loss=model.OCSoftmax(args['output_size'],r_real=args['r_real'],r_fake=args['r_fake'],alpha=args['alpha']),loss_bpl=model.BPL(alpha=args['alpha_cl'])).to(device).eval()
    state=torch.load(REFERENCE/'model/BestCLS_50_21DF.pt',map_location='cpu',weights_only=True);cls.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True);del state
    for p in list(plm.parameters())+list(cls.parameters()):p.requires_grad_(False)
    initial=ref.state_digest(cls);c=cls.loss.weight.detach().cpu().numpy().ravel().astype(float);c/=np.linalg.norm(c)
    captured={}
    def hook(m,inp,out):captured['z']=out.detach().cpu().numpy()[0].copy()
    handle=cls.layer_norm.register_forward_hook(hook)
    frame=pd.read_parquet(BASE/'manifest.parquet');part=frame.iloc[rank::2]
    old=pd.read_parquet(BASE/'scores.parquet').set_index(['utterance_id','condition'])
    rows=[];profiles=[];maxnative=0.;maxnoop=0.;start=time.monotonic()
    def seed(row):
        ss=core.utterance_seed(row,20261105);torch.random.default_generator.manual_seed(ss);torch.cuda.manual_seed(ss)
    def backend(features,row):
        seed(row)
        for _ in range(config.num_hidden_layers):torch.rand([])
        cls(features);z=captured['z'].astype(float);return float(z@c/np.linalg.norm(z))
    blocks=[range(0,9),range(9,17),range(17,25)]
    with torch.inference_mode():
        for n,(_,row) in enumerate(part.iterrows()):
            a,sr=sf.read(row.path);assert sr==16000;a=a.astype(np.float32)
            ff={}
            for kind in ['wrap','original']:
                audio=ex.pad(a,kind);seed(row)
                ff[kind]=torch.stack(plm(torch.as_tensor(audio,device=device).unsqueeze(0),output_hidden_states=True).hidden_states,dim=1)
            fw,fo=ff['wrap'],ff['original'];assert fw.shape[2]==201
            t=fo.shape[2];repeats=(201+t-1)//t;target=fo.repeat(1,1,repeats,1)[:,:,:201,:]
            record={k:row[k] for k in ['utterance_id','y','source_corpus','speaker_group','duration_bin']}
            record['original_frames']=t
            for mask in range(8):
                if mask==0:mix=fw
                elif mask==7:mix=target
                else:
                    mix=fw.clone()
                    for b,indices in enumerate(blocks):
                        if mask&(1<<b):mix[:,list(indices)]=target[:,list(indices)]
                ss=backend(mix,row);record[f'mask{mask}_score']=ss
                if mask==0:
                    e=abs(ss-old.loc[(row.utterance_id,'native_wrap')].cosine_s);maxnative=max(maxnative,e);assert e<=1e-5
                if t==201:maxnoop=max(maxnoop,abs(ss-record['mask0_score']));assert abs(ss-record['mask0_score'])<1e-5
            record['original_unpadded_score']=float(old.loc[(row.utterance_id,'native_original')].cosine_s)
            rows.append(record)
            for layer in range(25):
                left=fw[0,layer];right=target[0,layer]
                u,v=left.mean(0),right.mean(0);cos=float(torch.nn.functional.cosine_similarity(u,v,dim=0))
                # First convolution-valid prefix frames carry the same original
                # waveform samples; contextual features may still change.
                common=min(t,201);ll=fw[0,layer,:common];rr=fo[0,layer,:common]
                relative=float(torch.sqrt(torch.mean((ll-rr)**2)/torch.mean(rr**2)))
                profiles.append({'utterance_id':row.utterance_id,'source_corpus':row.source_corpus,'y':row.y,'duration_bin':row.duration_bin,'layer':layer,'mean_feature_cosine':cos,'aligned_prefix_relative_RMS':relative})
            if (n+1)%16==0 or n+1==len(part):
                core.save_json(root/f'rank{rank}_progress.json',{'completed':n+1,'total':len(part),'seconds':time.monotonic()-start,'native_score_max_error':maxnative,'long_noop_max_error':maxnoop})
                print(rank,n+1,'/',len(part),flush=True)
    handle.remove();assert ref.state_digest(cls)==initial
    pd.DataFrame(rows).to_parquet(root/f'rank{rank}_results.parquet',index=False)
    pd.DataFrame(profiles).to_parquet(root/f'rank{rank}_profiles.parquet',index=False)
    core.save_json(root/f'rank{rank}_complete.json',{'status':'pass','N':len(rows),'native_score_max_error':maxnative,'long_noop_max_error':maxnoop,'state_before':initial,'state_after':ref.state_digest(cls),'seconds':time.monotonic()-start})


def main(root):
    import torch
    import torch.multiprocessing as mp
    assert torch.cuda.device_count()==2 and not (root/'rank0_progress.json').exists()
    mp.spawn(worker,args=(str(root),),nprocs=2,join=True)
    for kind in ['results','profiles']:
        f=pd.concat([pd.read_parquet(root/f'rank{r}_{kind}.parquet') for r in [0,1]],ignore_index=True)
        f.to_parquet(root/(kind+'.parquet'),index=False)
    names=['results.parquet','profiles.parquet']+[f'rank{r}_complete.json' for r in [0,1]]
    core.save_json(root/'complete.json',{'status':'pass','N':471,'files':{n:core.digest(root/n) for n in names},'classifier_frames':201,'raw_audio_or_SSL_hidden_saved':False})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();assert Path('/.dockerenv').exists() and os.environ.get('CUDA_VISIBLE_DEVICES')=='0,1';main(a.root)
