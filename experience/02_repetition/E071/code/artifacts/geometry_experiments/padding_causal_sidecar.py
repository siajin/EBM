"""Server-only frozen EBM, paired original-content / padding interventions.

Never writes audio or frontend hidden states. Only final 160D h/z and derived
scores are persisted. Original model/source/data remain untouched.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
import pandas as pd

R23 = Path('/home/mentee/analysis/embedding_geometry/20261105_frozen_all_01/R23')
sys.path.insert(0,str(R23))
import frozen_r23_v1 as ref
core,REFERENCE = ref.core,ref.REFERENCE
CACHE = {'DF_pilot':ref.DF, 'DF_remaining':Path('/home/mentee/analysis/embedding_geometry/20261105_frozen_all_01/DF_remaining')}
L = 64600


def save(path,obj):
    core.save_json(Path(path),obj)


def pad(x,kind):
    x=np.asarray(x,dtype=np.float32)
    if kind=='original':return x[:L].copy()
    if len(x)>=L:return x[:L].copy()
    return np.pad(x,(0,L-len(x)),mode=kind).astype(np.float32)


def views(audio):
    base=np.asarray(audio[:L],dtype=np.float32)
    result={'native_wrap':pad(base,'wrap'),'native_reflect':pad(base,'reflect'),'native_original':pad(base,'original')}
    if len(audio)>=L:
        for seconds in [1.5,2.0]:
            length=int(seconds*16000)
            for location,x in [('head',base[:length]),('tail',base[-length:])]:
                for kind in ['wrap','reflect','original']:
                    result[f'crop{seconds:g}_{location}_{kind}']=pad(x,kind)
    return result


def synthetic_qc():
    x=np.arange(16001,dtype=np.float32)/16001
    v=views(x)
    assert np.array_equal(v['native_wrap'],np.tile(x,5)[:L])
    assert np.array_equal(v['native_original'],x)
    assert np.array_equal(v['native_reflect'][:len(x)],x)
    assert not np.array_equal(v['native_reflect'],v['native_wrap'])
    long=np.arange(L+23,dtype=np.float32)/(L+23)
    v=views(long)
    assert np.array_equal(v['native_wrap'],v['native_original']) and np.array_equal(v['native_reflect'],v['native_wrap'])
    assert len(v)==15
    for k,a in v.items():
        assert a.ndim==1 and np.isfinite(a).all()
        if not k.endswith('original'):assert len(a)==L
    return {'status':'pass','no_audio_normalization':True,'no_silence_predictors_or_trimming':True,'long_noop_identity':True,'wrap_matches_numpy_tile':True}


def prepare(root):
    assert not (root/'prepare_complete.json').exists()
    selected=pd.read_parquet(root/'selected.parquet')
    frames=[]
    for alias,folder in CACHE.items():
        ids=selected[selected.cache_alias==alias]
        if ids.empty:continue
        source=pd.read_parquet(folder/'manifest.parquet')
        source=source[source.utterance_id.isin(ids.utterance_id)].copy()
        assert len(source)==len(ids)
        q=source.merge(ids[['utterance_id','duration_bin','cache_alias','selection_order']],on='utterance_id',validate='one_to_one')
        old=pd.read_parquet(folder/'extract/scores.parquet').set_index('utterance_id')
        q['expected_cosine']=q.utterance_id.map(old.cosine_s)
        q['old_rank']=q.utterance_id.map(old['rank'])
        q['old_embedding_row']=q.utterance_id.map(old.embedding_row)
        for _,r in q.iterrows():
            s=ids.set_index('utterance_id').loc[r.utterance_id]
            assert int(r.y)==int(s.y) and r.source_corpus==s.source_corpus and r.codec=='nocodec'
        frames.append(q)
    frame=pd.concat(frames,ignore_index=True).sort_values('selection_order').reset_index(drop=True)
    assert len(frame)==len(selected) and not frame.utterance_id.duplicated().any()
    frame.to_parquet(root/'manifest.parquet',index=False)
    archived,args=core.settings(REFERENCE)
    assert core.digest(REFERENCE/'model/BestCLS_50_21DF.pt')==ref.la.EXPECTED_HASH
    lock=json.loads((root/'protocol.json').read_text())
    assert lock['selected_sha256']==core.digest(root/'selected.parquet')
    save(root/'prepare_complete.json',{'status':'pass','N':len(frame),'manifest_sha256':core.digest(root/'manifest.parquet'),
        'script_sha256':core.digest(__file__),'synthetic_QC':synthetic_qc(),
        'archive_sha256':{str(p.relative_to(archived)):core.digest(p) for p in archived.rglob('*.py')}})


def worker(rank,root_s,barrier):
    import torch
    import soundfile as sf
    from transformers import AutoConfig,Wav2Vec2Model
    root=Path(root_s)
    torch.set_num_threads(2);torch.cuda.set_device(rank);device=torch.device('cuda',rank)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
    archived,args=core.settings(REFERENCE);sys.path.insert(0,str(archived));import model
    from data_loaders import EnrollmentSet
    config=AutoConfig.from_pretrained(args['PLM_name'],local_files_only=True)
    plm=Wav2Vec2Model.from_pretrained(args['PLM_name'],config=config,local_files_only=True).to(device).eval()
    cls=model.B_Linear(num_layer=args['num_layers'],hidden_size=args['hidden_size'],output_size=args['output_size'],agg_size=args['agg_size'],
        loss=model.OCSoftmax(args['output_size'],r_real=args['r_real'],r_fake=args['r_fake'],alpha=args['alpha']),loss_bpl=model.BPL(alpha=args['alpha_cl'])).to(device).eval()
    state=torch.load(REFERENCE/'model/BestCLS_50_21DF.pt',map_location='cpu',weights_only=True)
    cls.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True)
    del state
    for p in list(plm.parameters())+list(cls.parameters()):p.requires_grad_(False)
    initial=ref.state_digest(cls)
    c=cls.loss.weight.detach().cpu().numpy().ravel().astype(float);c/=np.linalg.norm(c)
    np.save(root/f'rank{rank}_centroid.npy',c)
    captured={}
    def hook(module,inp,out):captured['h']=inp[0].detach().cpu().numpy()[0].copy();captured['z']=out.detach().cpu().numpy()[0].copy()
    handle=cls.layer_norm.register_forward_hook(hook)
    frame=pd.read_parquet(root/'manifest.parquet')
    oldarrays={alias:{k:{r:np.load(folder/f'extract/rank{r}_{k}.npy',mmap_mode='r') for r in [0,1]} for k in ['h','z']} for alias,folder in CACHE.items()}
    errors={'h':0.,'z':0.,'score':0.}

    def read(row):
        a,sr=sf.read(row.path);assert sr==16000 and a.ndim==1 and len(a)>=400
        return a.astype(np.float32)

    def infer(audio,row):
        seed=core.utterance_seed(row,20261105)
        torch.random.default_generator.manual_seed(seed);torch.cuda.manual_seed(seed)
        with torch.inference_mode():
            hiddens=plm(torch.as_tensor(audio,device=device).unsqueeze(0),output_hidden_states=True).hidden_states
            # Derived temporal summaries remain in memory, never written as frontend outputs.
            sslmean=hiddens[-1][0].mean(0).detach().cpu().numpy().astype(float)
            _,score=cls(torch.stack(hiddens,dim=1))
        z,h=captured['z'].copy(),captured['h'].copy()
        s=float((z.astype(float)/np.linalg.norm(z.astype(float)))@c)
        assert np.isfinite(z).all() and np.isfinite(h).all() and abs(s-score.item())<1e-5
        return h,z,s,sslmean

    def check_native(row,h,z,s):
        arr=oldarrays[row.cache_alias];r,i=int(row.old_rank),int(row.old_embedding_row)
        e={'h':float(np.max(np.abs(h-arr['h'][r][i]))),'z':float(np.max(np.abs(z-arr['z'][r][i]))),'score':abs(s-row.expected_cosine)}
        assert max(e.values())<=1e-5,e
        for k,v in e.items():errors[k]=max(errors[k],v)

    shared=frame.groupby(['source_corpus','y','duration_bin'],observed=True,sort=True).head(1)
    qa=[]
    for _,row in shared.iterrows():
        audio=read(row);input0=pad(audio,'wrap')
        item=SimpleNamespace(path=row.path,type=row.attack_id,label=int(row.y))
        original,_,label=EnrollmentSet(args,[item])[0]
        assert np.array_equal(input0,np.asarray(original,dtype=np.float32)) and int(label)==row.y
        h,z,s,ss=infer(input0,row);h2,z2,s2,_=infer(input0,row)
        assert np.array_equal(h,h2) and np.array_equal(z,z2) and s==s2
        check_native(row,h,z,s)
        qa.append({'uid':row.utterance_id,'s':s,'h':h.tolist(),'z':z.tolist()})
    save(root/f'rank{rank}_QC.json',{'status':'pass','rows':qa,'native_max_errors':errors,'repeat_bitwise':True,'EnrollmentSet_input_identity':True})
    barrier.wait(timeout=180)
    other=json.loads((root/f'rank{1-rank}_QC.json').read_text())
    for a,b in zip(qa,other['rows']):
        assert a['uid']==b['uid'] and abs(a['s']-b['s'])<1e-5
        assert np.max(np.abs(np.array(a['h'])-b['h']))<1e-5 and np.max(np.abs(np.array(a['z'])-b['z']))<1e-5
    print('QC PASS',rank,len(shared),'full extraction allowed',flush=True)
    results=[];zs=[];hs=[];start=time.monotonic()
    part=frame.iloc[rank::2]
    for n,(_,row) in enumerate(part.iterrows()):
        audio=read(row);vv=views(audio);base_ssl=None;base_input=None
        outputs={}
        for condition,a in vv.items():
            h,z,s,ssl=infer(a,row)
            if condition=='native_wrap':
                check_native(row,h,z,s);base_ssl=ssl;base_input=a
            # Genuine no-ops are tested with independent forward calls, not copied scores.
            if np.array_equal(a,base_input):assert np.array_equal(z,outputs['native_wrap'][1]) if condition!='native_wrap' else True
            sslcos=float((ssl@base_ssl)/(np.linalg.norm(ssl)*np.linalg.norm(base_ssl)))
            outputs[condition]=(h,z,s)
            record={k:row[k] for k in ['utterance_id','y','source_corpus','speaker_group','duration_bin','codec','cache_alias','selection_order']}
            record.update(condition=condition,input_samples=len(a),original_samples=len(audio),duration=len(audio)/16000,
                cosine_s=s,theta_deg=float(np.degrees(np.arccos(np.clip(s,-1,1)))),ssl_last_mean_cosine_to_native=sslcos,rank=rank,embedding_row=len(results))
            results.append(record);zs.append(z);hs.append(h)
        if (n+1)%8==0 or n+1==len(part):
            save(root/f'rank{rank}_progress.json',{'completed_utterances':n+1,'total_utterances':len(part),'forward_calls':len(results),'seconds':time.monotonic()-start,'native_max_errors':errors})
            print('rank',rank,n+1,'/',len(part),'forwards',len(results),flush=True)
    np.save(root/f'rank{rank}_z.npy',np.array(zs,dtype=np.float32));np.save(root/f'rank{rank}_h.npy',np.array(hs,dtype=np.float32))
    pd.DataFrame(results).to_parquet(root/f'rank{rank}_rows.parquet',index=False)
    assert initial==ref.state_digest(cls)
    handle.remove()
    save(root/f'rank{rank}_complete.json',{'status':'pass','N_utterances':len(part),'N_forwards':len(results),'seconds':time.monotonic()-start,'native_max_errors':errors,'classifier_state_before':initial,'classifier_state_after':ref.state_digest(cls),'optimizer_used':False,'raw_audio_saved':False,'frontend_hidden_saved':False})


def extract(root):
    import torch
    import torch.multiprocessing as mp
    assert torch.cuda.device_count()==2
    lock=json.loads((root/'prepare_complete.json').read_text())
    assert lock['status']=='pass' and lock['script_sha256']==core.digest(__file__) and lock['manifest_sha256']==core.digest(root/'manifest.parquet')
    assert not (root/'rank0_progress.json').exists() and not (root/'rank0_QC.json').exists(),'Never restart/overwrite an existing extraction'
    barrier=mp.get_context('spawn').Barrier(2)
    mp.spawn(worker,args=(str(root),barrier),nprocs=2,join=True)
    rows=pd.concat([pd.read_parquet(root/f'rank{r}_rows.parquet') for r in [0,1]],ignore_index=True)
    assert not rows[['utterance_id','condition']].duplicated().any()
    rows.to_parquet(root/'scores.parquet',index=False)
    archived,_=core.settings(REFERENCE)
    assert lock['archive_sha256']=={str(p.relative_to(archived)):core.digest(p) for p in archived.rglob('*.py')}
    names=['scores.parquet']+[f'rank{r}_{kind}' for r in [0,1] for kind in ['z.npy','h.npy','centroid.npy','rows.parquet','QC.json','complete.json']]
    save(root/'complete.json',{'status':'pass','rows':len(rows),'N_utterances':rows.utterance_id.nunique(),'files':{n:core.digest(root/n) for n in names},'checkpoint_sha256':ref.la.EXPECTED_HASH,'optimizer_used':False,'raw_audio_saved':False,'frontend_hidden_saved':False})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','extract']);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    assert Path('/.dockerenv').is_file() and os.environ.get('CUDA_VISIBLE_DEVICES')=='0,1'
    prepare(a.root) if a.stage=='prepare' else extract(a.root)
