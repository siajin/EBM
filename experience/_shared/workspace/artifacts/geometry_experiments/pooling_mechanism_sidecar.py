"""Mechanistic follow-up: identical SSL frame duplication and pooled-stat swaps.

Same471 preselected IDs; no optimizer, waveform export, hidden-state files,
new fitted detector, silence intervention, or original-source modifications.
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
ref,core,REFERENCE=ex.ref,ex.core,ex.REFERENCE


def worker(rank,root_s):
    import torch
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
    pools=list(cls.bridge_module.AHP_ASP)+[cls.AHP_ASP];saved_noise=[p.noise_std for p in pools]
    captured={}
    def hook(module,inp,out):captured['pool']=out.detach().clone()
    hh=cls.AHP_ASP.register_forward_hook(hook)
    frame=pd.read_parquet(BASE/'manifest.parquet');part=frame.iloc[rank::2]
    old=pd.read_parquet(BASE/'scores.parquet').set_index(['utterance_id','condition'])
    results=[];maxnative=0.;maxcorrected=0.;start=time.monotonic()
    import soundfile as sf

    def seed(row):
        s=core.utterance_seed(row,20261105);torch.random.default_generator.manual_seed(s);torch.cuda.manual_seed(s)

    def score_pool(pool):
        z=cls.layer_norm(cls.linear(pool)).detach().cpu().numpy()[0].astype(float)
        return float((z/np.linalg.norm(z))@c)

    def backend(features,row,k=1,fix='none'):
        seed(row)
        for _ in range(config.num_hidden_layers):torch.rand([])
        handles=[];t=features.shape[2]//k
        factor=np.sqrt(k*(k*t-1)/(t-1)) if k>1 else 1.
        def corrected(m,inp,out):
            half=out.shape[1]//2
            return torch.cat([out[:,:half],out[:,half:]*factor],dim=1)
        if fix in ['bridge','both']:
            handles += [p.register_forward_hook(corrected) for p in cls.bridge_module.AHP_ASP]
        if fix in ['final','both']:handles.append(cls.AHP_ASP.register_forward_hook(corrected))
        try:
            cls(features)
            # The capture hook runs before the correction hook: explicitly
            # use returned network score through a temporary LayerNorm hook.
            pool=captured['pool'].clone()
            if fix in ['final','both']:
                half=pool.shape[1]//2;pool=torch.cat([pool[:,:half],pool[:,half:]*factor],dim=1)
            return score_pool(pool),pool
        finally:
            for h in handles:h.remove()

    with torch.inference_mode():
        for n,(_,row) in enumerate(part.iterrows()):
            audio,sr=sf.read(row.path);assert sr==16000;audio=audio.astype(np.float32)
            record={k:row[k] for k in ['utterance_id','y','source_corpus','speaker_group','duration_bin']}
            pp={};original_features=None
            for kind in ['wrap','original']:
                x=ex.pad(audio,kind);seed(row)
                features=torch.stack(plm(torch.as_tensor(x,device=device).unsqueeze(0),output_hidden_states=True).hidden_states,dim=1)
                s,pool=backend(features,row)
                expected=old.loc[(row.utterance_id,'native_'+kind)].cosine_s
                maxnative=max(maxnative,abs(s-expected));assert abs(s-expected)<=1e-5
                pp[kind]=(s,pool.clone())
                if kind=='original':original_features=features
            sw,pw=pp['wrap'];so,po=pp['original'];half=pw.shape[1]//2
            mean_only=score_pool(torch.cat([po[:,:half],pw[:,half:]],dim=1))
            std_only=score_pool(torch.cat([pw[:,:half],po[:,half:]],dim=1))
            std_contribution=.5*((std_only-sw)+(so-mean_only));mean_contribution=.5*((mean_only-sw)+(so-std_only))
            assert abs((so-sw)-(std_contribution+mean_contribution))<1e-12
            record.update(native_wrap_score=sw,native_original_score=so,mean_only_score=mean_only,std_only_score=std_only,
                std_shapley_score_change=std_contribution,mean_shapley_score_change=mean_contribution,
                wrap_mu_norm=float(pw[:,:half].norm()),original_mu_norm=float(po[:,:half].norm()),
                wrap_sigma_norm=float(pw[:,half:].norm()),original_sigma_norm=float(po[:,half:].norm()),
                wrap_frames=201,original_frames=original_features.shape[2])
            # Remove tiny injected noise to test a deterministic algebraic
            # invariant. This is a mechanistic control, not native performance.
            for p in pools:p.noise_std=0.
            base,pool0=backend(original_features,row)
            record['noise0_base_score']=base
            for k in [2,4]:
                doubled=original_features.repeat(1,1,k,1)
                for fix in (['none','bridge','final','both'] if k==2 else ['none','both']):
                    ss,pool=backend(doubled,row,k,fix)
                    record[f'dup{k}_{fix}_score']=ss
                    if fix=='both':maxcorrected=max(maxcorrected,abs(ss-base));assert abs(ss-base)<1e-4
            for p,v in zip(pools,saved_noise):p.noise_std=v
            results.append(record)
            if (n+1)%16==0 or n+1==len(part):
                core.save_json(root/f'rank{rank}_progress.json',{'completed':n+1,'total':len(part),'native_score_max_error':maxnative,'corrected_duplicate_max_error':maxcorrected,'seconds':time.monotonic()-start})
                print(rank,n+1,'/',len(part),flush=True)
    hh.remove();assert ref.state_digest(cls)==initial
    pd.DataFrame(results).to_parquet(root/f'rank{rank}_results.parquet',index=False)
    core.save_json(root/f'rank{rank}_complete.json',{'status':'pass','N':len(results),'native_score_max_error':maxnative,'corrected_duplicate_max_error':maxcorrected,'classifier_state_before':initial,'classifier_state_after':ref.state_digest(cls),'seconds':time.monotonic()-start,'frontend_hidden_files':False})


def main(root):
    import torch
    import torch.multiprocessing as mp
    assert torch.cuda.device_count()==2 and not (root/'rank0_progress.json').exists()
    assert json.loads((BASE/'complete.json').read_text())['status']=='pass'
    mp.spawn(worker,args=(str(root),),nprocs=2,join=True)
    f=pd.concat([pd.read_parquet(root/f'rank{r}_results.parquet') for r in [0,1]],ignore_index=True)
    assert len(f)==471 and not f.utterance_id.duplicated().any()
    f.to_parquet(root/'results.parquet',index=False)
    names=['results.parquet']+[f'rank{r}_{suffix}' for r in [0,1] for suffix in ['results.parquet','complete.json']]
    core.save_json(root/'complete.json',{'status':'pass','N':len(f),'files':{n:core.digest(root/n) for n in names},'optimizer_used':False,'frontend_hidden_saved':False})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    assert Path('/.dockerenv').exists() and os.environ.get('CUDA_VISIBLE_DEVICES')=='0,1'
    main(a.root)
