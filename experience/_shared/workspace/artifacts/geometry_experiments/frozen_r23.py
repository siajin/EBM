"""Frozen EBM attention ablations and paired noise/Opus interventions.

Server-only sidecar. Native single-utterance SSL CUDA graph is shared across
four classifier conditions; transformed audio requires three new SSL forwards.
Functional parameter overrides leave archived sources and model state intact.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
import numpy as np
import pandas as pd

LA=Path('/home/mentee/analysis/embedding_geometry/20261105_21la_full_03')
DF=Path('/home/mentee/analysis/embedding_geometry/20261105_df_fast_01')
TV=Path('/home/mentee/analysis/embedding_geometry/20261105_fast_p0_01')
sys.path[:0]=[str(LA),str(DF)]
import la21_extract_v1 as la
core=la.core
REFERENCE=Path('/home/mentee/results/CM/LA/EBM_e50_bs160_lr1e5')
CONDITIONS=['R2-00','R2-10','R2-01','R2-11','R3-noise20','R3-noise10','R3-Opus32k_recompression']

def opus(audio):
    # Ogg Opus pre-skip/encoder delay are honored by the decoder. Right-edge
    # length correction is explicit; no content-dependent lag optimization.
    enc=['ffmpeg','-v','error','-f','f32le','-ar','16000','-ac','1','-i','pipe:0','-c:a','libopus','-b:a','32k','-vbr','off','-application','voip','-f','ogg','pipe:1']
    dec=['ffmpeg','-v','error','-i','pipe:0','-f','f32le','-ar','16000','-ac','1','pipe:1']
    blob=subprocess.run(enc,input=np.asarray(audio,dtype='<f4').tobytes(),capture_output=True,check=True).stdout
    result=subprocess.run(dec,input=blob,capture_output=True,check=True).stdout
    decoded=np.frombuffer(result,dtype='<f4').copy();difference=len(decoded)-len(audio)
    if len(decoded)<len(audio):decoded=np.pad(decoded,(0,len(audio)-len(decoded)))
    decoded=decoded[:len(audio)]
    assert np.isfinite(decoded).all()
    return decoded,difference

def audio_views(row):
    import soundfile as sf
    audio,sr=sf.read(row.path);assert sr==16000 and audio.ndim==1 and len(audio)>0
    audio=audio.astype('float32');rms=float(np.sqrt(np.mean(audio.astype(float)**2)))
    seed=int.from_bytes(hashlib.sha256(('R3-base-noise-v1|'+row.utterance_id).encode()).digest()[:8],'big')
    base=np.random.default_rng(seed).normal(size=len(audio));base/=np.sqrt(np.mean(base**2))
    noisy={};snrs={}
    for snr in [20,10]:
        delta=base*rms/(10**(snr/20));noisy[f'R3-noise{snr}']=(audio+delta).astype('float32')
        achieved=10*np.log10(np.mean(audio.astype(float)**2)/np.mean((noisy[f'R3-noise{snr}'].astype(float)-audio)**2)) if rms>0 else None
        assert achieved is None or abs(achieved-snr)<1e-3
        snrs[str(snr)]=None if achieved is None else float(achieved)
    decoded,length_delta=opus(audio)
    return dict(native=audio,**noisy,**{'R3-Opus32k_recompression':decoded}),dict(source_frames=len(audio),source_rms=rms,codec_length_delta_before_right_edge_correction=length_delta,achieved_SNR_dB=snrs)

def crop(audio):
    return np.pad(audio,(0,64600-len(audio)),mode='wrap') if len(audio)<64600 else audio[:64600]

def prepare(root):
    archived,args=core.settings(REFERENCE);assert core.digest(REFERENCE/'model/BestCLS_50_21DF.pt')==la.EXPECTED_HASH
    inputs=root/'inputs';records={}
    for name,base,view in [('DF_pilot',DF,'DF_small'),('21LA_pilot',LA,'21LA_small')]:
        ids=pd.read_parquet(inputs/(name+'.parquet'));small=set(pd.read_parquet(inputs/(view+'.parquet')).utterance_id)
        native=pd.read_parquet(base/'manifest.parquet');frame=native.merge(ids[['utterance_id']],on='utterance_id',how='inner',validate='one_to_one')
        assert len(frame)==20000 and frame.y.value_counts().to_dict()=={0:18000,1:2000}
        frame['small']=frame.utterance_id.isin(small);assert frame.small.sum()==2000
        frame=frame.sort_values(['small','utterance_id'],ascending=[False,True],kind='stable').reset_index(drop=True)
        target=root/name;target.mkdir(exist_ok=False);frame.to_parquet(target/'manifest.parquet',index=False)
        records[name]=dict(manifest_sha256=core.digest(target/'manifest.parquet'),source_cache=str(base),source_manifest_sha256=core.digest(base/'manifest.parquet'),counts=frame.y.value_counts().to_dict(),small_rows=2000)
    v=pd.read_parquet(TV/'manifest.parquet');ids=pd.read_parquet(inputs/'V_QC.parquet');v=v.merge(ids[['utterance_id']],on='utterance_id',validate='one_to_one');assert len(v)==128 and v.y.value_counts().to_dict()=={0:96,1:32}
    v.to_parquet(root/'V_QC_manifest.parquet',index=False)
    assert shutil.which('ffmpeg');version=subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0]
    qc=[]
    from scipy.signal import correlate, correlation_lags
    for _,row in v.iterrows():
        views,info=audio_views(row);native=views['native'];decoded=views['R3-Opus32k_recompression']
        n=min(len(native),64600);corr=correlate(decoded[:n],native[:n],mode='full',method='fft');lags=correlation_lags(n,n);ok=np.abs(lags)<=320
        lag=int(lags[ok][np.argmax(corr[ok])]);qc.append(dict(utterance_id=row.utterance_id,peak_correlation_lag_samples=lag,**info))
    # The decoder must preserve timestamps to within a 1-ms diagnostic tolerance.
    # If that fails, preserve this failed version instead of changing test inputs.
    assert max(abs(x['peak_correlation_lag_samples']) for x in qc)<=16
    core.save_json(root/'preprocessing_QC.json',dict(status='pass',ffmpeg_version=version,codec='libopus 32kbps CBR voip, Ogg, decode to16k mono',delay_policy='ffmpeg automatic Opus pre-skip; no adaptive lag alignment',right_edge='truncate or zero-pad to original full audio length, then original first64600/wrap padding',noise='same UID-fixed full-utterance Gaussian waveform rescaled at20/10dB; no clipping or peak normalization',rows=qc,V_QC_manifest_sha256=core.digest(root/'V_QC_manifest.parquet')))
    core.save_json(root/'protocol_lock.json',dict(status='pass',datasets=records,script_sha256=core.digest(__file__),checkpoint_sha256=la.EXPECTED_HASH,conditions=CONDITIONS,small_first_then_full_20000=True,selection_on_interventions=False,original_EBM_frozen=True,optimizer_used=False,archive_code_sha256={str(p.relative_to(archived)):core.digest(p) for p in archived.rglob('*.py')}))

def state_digest(module):
    h=hashlib.sha256()
    for name,value in module.state_dict().items():h.update(name.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def worker(rank,root_s,name):
    import torch
    from transformers import AutoConfig,Wav2Vec2Model
    root=Path(root_s);folder=root/name;frame=pd.read_parquet(folder/'manifest.parquet');part=frame.iloc[rank::2].reset_index(drop=True)
    torch.set_num_threads(2);torch.cuda.set_device(rank);device=torch.device('cuda',rank)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
    archived,args=core.settings(REFERENCE);sys.path.insert(0,str(archived));import model
    config=AutoConfig.from_pretrained(args['PLM_name'],local_files_only=True);plm=Wav2Vec2Model.from_pretrained(args['PLM_name'],config=config,local_files_only=True).to(device).eval()
    cls=model.B_Linear(num_layer=args['num_layers'],hidden_size=args['hidden_size'],output_size=args['output_size'],agg_size=args['agg_size'],loss=model.OCSoftmax(args['output_size'],r_real=args['r_real'],r_fake=args['r_fake'],alpha=args['alpha']),loss_bpl=model.BPL(alpha=args['alpha_cl'])).to(device).eval()
    state=torch.load(REFERENCE/'model/BestCLS_50_21DF.pt',map_location='cpu',weights_only=True);cls.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True)
    for param in list(plm.parameters())+list(cls.parameters()):param.requires_grad_(False)
    before=state_digest(cls);c=cls.loss.weight.detach().cpu().numpy().ravel().astype(float);c/=np.linalg.norm(c);captured={}
    def capture(m,inp,out):captured['h']=inp[0].detach().cpu().numpy()[0].copy();captured['z']=out.detach().cpu().numpy()[0].copy()
    handle=cls.layer_norm.register_forward_hook(capture)
    with torch.random.fork_rng(devices=[rank]),torch.inference_mode():
        gi=torch.zeros((1,64600),device=device);stream=torch.cuda.Stream(device=device);stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(3):plm(gi,output_hidden_states=True)
        torch.cuda.current_stream(device).wait_stream(stream);torch.cuda.synchronize(device);graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):gh=plm(gi,output_hidden_states=True).hidden_states
    outputs={};metadata={cond:[] for cond in CONDITIONS}
    for cond in CONDITIONS:
        dest=folder/cond;dest.mkdir(exist_ok=True)
        outputs[cond]={kind:np.lib.format.open_memmap(dest/f'rank{rank}_{kind}.npy',mode='w+',dtype='float32',shape=(len(part),160)) for kind in ['z','h']}
        np.save(dest/f'rank{rank}_centroid.npy',c)
    source=DF if name=='DF_pilot' else LA;oldrows=pd.read_parquet(source/'extract/scores.parquet').set_index('utterance_id');oldz={r:np.load(source/f'extract/rank{r}_z.npy',mmap_mode='r') for r in [0,1]};oldh={r:np.load(source/f'extract/rank{r}_h.npy',mmap_mode='r') for r in [0,1]}
    overrides={
        'R2-00':{},
        'R2-10':{'bridge_module.AHP_W_att':torch.zeros_like(cls.bridge_module.AHP_W_att)},
        'R2-01':{'AHP_ASP.mm_weights':torch.zeros_like(cls.AHP_ASP.mm_weights)},
        'R2-11':{'bridge_module.AHP_W_att':torch.zeros_like(cls.bridge_module.AHP_W_att),'AHP_ASP.mm_weights':torch.zeros_like(cls.AHP_ASP.mm_weights)}}
    maxerr=dict(z=0.,h=0.,score=0.);start=time.monotonic();small_end=None
    for i,(_,row) in enumerate(part.iterrows()):
        views,ainfo=audio_views(row);seed=core.utterance_seed(row,20261105)
        with torch.inference_mode():
            for view in ['native','R3-noise20','R3-noise10','R3-Opus32k_recompression']:
                gi.copy_(torch.as_tensor(crop(views[view]),device=device).unsqueeze(0));graph.replay();features=torch.stack(gh,dim=1)
                conditions=CONDITIONS[:4] if view=='native' else [view]
                for cond in conditions:
                    torch.random.default_generator.manual_seed(seed);torch.cuda.manual_seed(seed)
                    for _ in range(config.num_hidden_layers):torch.rand([])
                    _,rawscore=torch.func.functional_call(cls,overrides.get(cond,{}),(features,))
                    z,h=captured['z'],captured['h'];assert np.isfinite(z).all() and np.isfinite(h).all()
                    norm=float(np.linalg.norm(z.astype(float)));unit=z.astype(float)/norm;s=float(unit@c);assert abs(s-float(rawscore.item()))<=1e-5
                    if cond=='R2-00':
                        old=oldrows.loc[row.utterance_id];rr,ii=int(old['rank']),int(old.embedding_row)
                        errors=dict(z=float(np.max(np.abs(z-oldz[rr][ii]))),h=float(np.max(np.abs(h-oldh[rr][ii]))),score=abs(s-old.cosine_s))
                        assert max(errors.values())<=1e-5,errors
                        maxerr={k:max(v,errors[k]) for k,v in maxerr.items()}
                    outputs[cond]['z'][i]=z;outputs[cond]['h'][i]=h
                    safe={k:row[k] for k in frame.columns if k!='path'}
                    safe.update(condition_id=cond,rng_seed_condition_id=row.condition_id,extraction_seed=seed,rank=rank,embedding_row=i,cosine_s=s,raw_score=float(rawscore.item()),raw_norm=norm,direction_valid=int(np.sqrt(max(0,1-s*s))>=1e-4),duration=ainfo['source_frames']/16000,crop_start=0,crop_samples=64600,padding_fraction=max(0,64600-ainfo['source_frames'])/64600,codec_length_delta=ainfo['codec_length_delta_before_right_edge_correction'])
                    metadata[cond].append(safe)
        done=i+1
        if done==1000:small_end=time.monotonic()-start
        if done%32==0 or done in [1000,len(part)]:
            for values in outputs.values():
                for arr in values.values():arr.flush()
            core.save_json(folder/f'rank{rank}_progress.json',dict(completed=done,total=len(part),small_complete=done>=1000,elapsed_seconds=time.monotonic()-start,small_seconds=small_end,native_max_errors=maxerr))
            print(name,'rank',rank,done,'/',len(part),flush=True)
    for cond in CONDITIONS:pd.DataFrame(metadata[cond]).to_parquet(folder/cond/f'rank{rank}_rows.parquet',index=False)
    after=state_digest(cls);assert before==after
    core.save_json(folder/f'rank{rank}_complete.json',dict(status='pass',sample_count=len(part),state_before=before,state_after=after,native_max_errors=maxerr,optimizer_used=False,seconds=time.monotonic()-start,small_seconds=small_end))
    handle.remove()

def extract(root,name):
    import torch
    import torch.multiprocessing as mp
    lock=json.loads((root/'protocol_lock.json').read_text());assert lock['script_sha256']==core.digest(__file__)
    assert lock['datasets'][name]['manifest_sha256']==core.digest(root/name/'manifest.parquet');assert json.loads((root/'preprocessing_QC.json').read_text())['status']=='pass'
    assert torch.cuda.device_count()==2 and core.digest(REFERENCE/'model/BestCLS_50_21DF.pt')==la.EXPECTED_HASH
    assert not (root/name/'rank0_progress.json').exists(),'Never overwrite/restart an existing run'
    mp.spawn(worker,args=(str(root),name),nprocs=2,join=True)
    folder=root/name
    for cond in CONDITIONS:
        dest=folder/cond;rows=pd.concat([pd.read_parquet(dest/f'rank{r}_rows.parquet') for r in [0,1]],ignore_index=True)
        assert len(rows)==20000 and not rows.utterance_id.duplicated().any();rows.to_parquet(dest/'scores.parquet',index=False)
        identities=dict(dataset=name,condition=cond,checkpoint_sha256=la.EXPECTED_HASH,protocol_lock_sha256=core.digest(root/'protocol_lock.json'),manifest_sha256=lock['datasets'][name]['manifest_sha256'])
        fingerprint=hashlib.sha256(json.dumps(identities,sort_keys=True).encode()).hexdigest()
        core.save_json(dest/'summary.json',dict(status='pass',rows=len(rows),small_rows=int(rows.small.sum()),fingerprint=fingerprint,identity=identities,all_original_parameters_frozen=True))
        (dest/'complete').write_text('0\n')
    archived,_=core.settings(REFERENCE);assert lock['archive_code_sha256']=={str(p.relative_to(archived)):core.digest(p) for p in archived.rglob('*.py')}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','extract']);p.add_argument('--root',required=True);p.add_argument('--dataset',choices=['DF_pilot','21LA_pilot']);o=p.parse_args();root=Path(o.root)
    assert Path('/.dockerenv').exists() and os.environ.get('CUDA_VISIBLE_DEVICES')=='0,1'
    if o.stage=='prepare':prepare(root)
    else:extract(root,o.dataset)
