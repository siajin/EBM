"""Server-only bounded variable-length EBM feasibility probe.

Imports archived lab modules in place; never saves audio or SSL features.
Does not edit original sources, run original main/logger, or update weights.
"""
import argparse
import gc
import inspect
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd

R23=Path('/home/mentee/analysis/embedding_geometry/20261105_frozen_all_01/R23')
sys.path.insert(0,str(R23))
import frozen_r23_v1 as ref
core,REFERENCE=ref.core,ref.REFERENCE
CACHE={'T':ref.TV, 'DF_pilot':ref.DF,
       'DF_remaining':Path('/home/mentee/analysis/embedding_geometry/20261105_frozen_all_01/DF_remaining')}


def save(p,x):core.save_json(Path(p),x)


def prepare(root):
    import soundfile as sf
    selected=pd.read_parquet(root/'probe_selection.parquet');rows=[];header_n=0
    for alias,folder in CACHE.items():
        q=selected[selected.cache_alias==alias]
        src=pd.read_parquet(folder/'manifest.parquet')
        if alias=='T':
            expected=pd.read_parquet(folder/'extract/scores.parquet',columns=['utterance_id','duration'])
            train=src[src.split=='T'].merge(expected,on='utterance_id',validate='one_to_one')
            assert len(train)==25380
            for _,r in train.iterrows():
                info=sf.info(r.path)
                assert info.samplerate==16000 and info.channels==1
                assert info.frames==round(r.duration*16000),(r.utterance_id,info.frames,r.duration)
                header_n+=1
        found=src[src.utterance_id.isin(q.utterance_id)].merge(q[['utterance_id','selection','cache_alias','samples']],on='utterance_id',validate='one_to_one')
        assert len(found)==len(q)
        for _,r in found.iterrows():
            info=sf.info(r.path);assert info.samplerate==16000 and info.channels==1 and info.frames==r.samples
        rows.append(found)
    manifest=pd.concat(rows,ignore_index=True)
    assert len(manifest)==len(selected)==10 and not manifest.utterance_id.duplicated().any()
    manifest.to_parquet(root/'manifest.parquet',index=False)
    archive,args=core.settings(REFERENCE)
    names=['data_loaders.py','train.py','main.py','model/b_linear.py','model/loss.py']
    save(root/'prepare_complete.json',dict(status='pass',N=len(manifest),
        actual_train_headers_checked=header_n,all_train_header_samplecounts_match_metadata=True,
        selected_DF_headers_checked=4,mono_16000=True,
        archived_hashes={str(p.relative_to(archive)):core.digest(p) for p in archive.rglob('*.py')},
        current_source_hashes={n:core.digest(Path('/home/mentee/code/EBM')/n) for n in names},
        checkpoint_sha256=core.digest(REFERENCE/'model/BestCLS_50_21DF.pt'),
        effective_args={k:args[k] for k in ['PLM_name','num_layers','hidden_size','output_size','agg_size','num_train_frames','num_train_frames_short','num_test_frames','batch_size','T_0','ratio_bona','ratio_spoof','alpha_cl']}))


def masked_pool(module,x,mask):
    """Keep original sum(alpha*x), sample-std(alpha*x+noise) semantics.

    This intentionally is NOT a conversion to conventional weighted variance.
    Mask every temporal softmax and exclude padding from std's count/mean.
    """
    import torch
    assert module.num_head==1 and mask.shape==x.shape[:2]
    weights=torch.bmm(x,module.mm_weights.T.contiguous().unsqueeze(0).repeat(len(x),1,1))
    alpha=torch.softmax(torch.tanh(weights).masked_fill(~mask.unsqueeze(-1),-torch.inf),dim=1)
    weighted=x*alpha.expand_as(x);mu=weighted.sum(1)
    if not module.asp:return mu
    noisy=weighted+module.noise_std*torch.randn_like(weighted)
    m=mask.unsqueeze(-1);count=m.sum(1)
    assert (count>=2).all()
    mean=(noisy*m).sum(1)/count
    var=(((noisy-mean.unsqueeze(1))**2)*m).sum(1)/(count-1)
    std=var.clamp_min(1e-30).sqrt()
    return torch.cat([mu,std],dim=1)


def encode(cls,features,mask):
    import torch
    bridge=cls.bridge_module;outs=[];stats=[]
    for i in range(bridge.num_layer):
        value=bridge.squeeze_module_norms[i](bridge.squeeze_module_linears[i](features[:,i]))
        outs.append(value);stats.append(masked_pool(bridge.AHP_ASP[i],value,mask))
    output=torch.stack(outs,dim=1);att=torch.stack(stats,dim=1)
    weights=torch.bmm(att,bridge.AHP_W_att.T.contiguous().unsqueeze(0).repeat(len(features),1,1))
    alpha=torch.softmax(torch.tanh(weights),dim=1).unsqueeze(-1)
    x=(output*alpha).sum(1)
    h=cls.linear(masked_pool(cls.AHP_ASP,x,mask))
    return cls.layer_norm(h)


def worker(rank,root_s):
    import torch
    import soundfile as sf
    import transformers
    from transformers import AutoConfig,Wav2Vec2Model
    root=Path(root_s);start=time.monotonic()
    torch.set_num_threads(2);torch.cuda.set_device(rank);device=torch.device('cuda',rank)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
    torch.manual_seed(20261106);torch.cuda.manual_seed(20261106)
    archive,args=core.settings(REFERENCE);sys.path.insert(0,str(archive));import model
    config=AutoConfig.from_pretrained(args['PLM_name'],local_files_only=True)
    plm=Wav2Vec2Model.from_pretrained(args['PLM_name'],config=config,local_files_only=True).to(device).eval()
    cls=model.B_Linear(num_layer=args['num_layers'],hidden_size=args['hidden_size'],output_size=args['output_size'],agg_size=args['agg_size'],
        loss=model.OCSoftmax(args['output_size'],r_real=args['r_real'],r_fake=args['r_fake'],alpha=args['alpha'],class_weight=[5/9,5]),
        loss_bpl=model.BPL(alpha=args['alpha_cl'])).to(device).eval()
    state=torch.load(REFERENCE/'model/BestCLS_50_21DF.pt',map_location='cpu',weights_only=True)
    cls.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True);del state
    for p in plm.parameters():p.requires_grad_(False)
    for module in cls.modules():
        if hasattr(module,'noise_std'):module.noise_std=0.
    state_before=ref.state_digest(cls)
    assert config.feat_extract_norm=='layer'
    manifest=pd.read_parquet(root/'manifest.parquet').set_index('selection')
    audio={}
    for key,row in manifest.iterrows():
        a,sr=sf.read(row.path,dtype='float32');assert sr==16000 and len(a)==row.samples
        audio[key]=a
    groups={
        'similar_medians':['train_y1_median','train_y0_median'],
        'mixed_short_medium':['train_y1_minimum','train_y0_minimum','train_y1_median','train_y0_median'],
        'train_original_maxima':['train_y1_maximum','train_y0_maximum'],
        'DF_shortest_longest':['DF_remaining_minimum','DF_remaining_maximum'],
    }
    profiles=[];singles={}
    captured={}
    handle=cls.layer_norm.register_forward_hook(lambda m,i,o:captured.update(z=o.detach().clone()))
    for key in dict.fromkeys(k for group in groups.values() for k in group):
        t0=time.monotonic();torch.cuda.reset_peak_memory_stats(device)
        x=torch.as_tensor(audio[key],device=device).unsqueeze(0)
        with torch.no_grad():
            hs=plm(x,output_hidden_states=True).hidden_states;features=torch.stack(hs,dim=1)
            mask=torch.ones(features.shape[0],features.shape[2],device=device,dtype=torch.bool)
            cls(features);native=captured['z'].clone();new=encode(cls,features,mask)
            err=float((new-native).abs().max());assert err<5e-5,(key,err)
            singles[key]=dict(z=new.detach().clone(),frontend=hs[-1].detach().clone(),frames=features.shape[2])
        torch.cuda.synchronize(device)
        profiles.append(dict(test='original_single',key=key,batch=1,input_seconds=len(audio[key])/16000,frames=features.shape[2],
            adapter_vs_native_max_z_error=err,seconds=time.monotonic()-t0,peak_allocated_MiB=torch.cuda.max_memory_allocated(device)/2**20))
        del hs,features,x,new,native;gc.collect();torch.cuda.empty_cache()
        save(root/f'rank{rank}_progress.json',dict(stage='single',key=key,seconds=time.monotonic()-start))
    comparisons=[]
    def batch(keys,extra=0,frontend_mask=True,backend_mask=True):
        sizes=torch.tensor([len(audio[k]) for k in keys],device=device)
        x=torch.zeros(len(keys),int(sizes.max())+extra,device=device)
        for i,k in enumerate(keys):x[i,:len(audio[k])]=torch.as_tensor(audio[k],device=device)
        am=torch.arange(x.shape[1],device=device)[None,:]<sizes[:,None]
        hs=plm(x,attention_mask=am.long() if frontend_mask else None,output_hidden_states=True).hidden_states
        f=torch.stack(hs,dim=1)
        fm=plm._get_feature_vector_attention_mask(f.shape[2],am)
        expected=plm._get_feat_extract_output_lengths(sizes)
        assert torch.equal(fm.sum(1),expected)
        assert all(int(n)==singles[k]['frames'] for n,k in zip(expected,keys))
        z=encode(cls,f,fm if backend_mask else torch.ones_like(fm))
        return z,f,fm,hs[-1]
    with torch.no_grad():
        for name,keys in groups.items():
            t0=time.monotonic();torch.cuda.reset_peak_memory_stats(device)
            z,f,fm,last=batch(keys)
            for i,k in enumerate(keys):
                e=float((z[i]-singles[k]['z'][0]).abs().max())
                frontend_e=float((last[i,:singles[k]['frames']]-singles[k]['frontend'][0]).abs().max())
                cosine=cls.loss(z[i:i+1])[1];old=cls.loss(singles[k]['z'])[1]
                score_e=float((cosine-old).abs().max())
                assert e<5e-4 and score_e<5e-5,(name,k,e,score_e)
                comparisons.append(dict(group=name,key=k,max_z_error=e,max_last_layer_error=frontend_e,max_score_error=score_e))
            torch.cuda.synchronize(device)
            profiles.append(dict(test='original_padded_batch',key=name,batch=len(keys),input_seconds=max(len(audio[k]) for k in keys)/16000,
                frames=f.shape[2],seconds=time.monotonic()-t0,peak_allocated_MiB=torch.cuda.max_memory_allocated(device)/2**20))
            if name=='mixed_short_medium':
                zbase=z.clone();fbase=f.clone();mbase=fm.clone()
            del z,f,fm,last;gc.collect();torch.cuda.empty_cache()
            save(root/f'rank{rank}_progress.json',dict(stage='batch',group=name,seconds=time.monotonic()-start))
        keys=groups['mixed_short_medium']
        zextra,fextra,mextra,_=batch(keys,extra=16000)
        extra_error=float((zextra-zbase).abs().max());assert extra_error<5e-4,extra_error
        zrev,*_=batch(keys[::-1]);order_error=float((zrev.flip(0)-zbase).abs().max());assert order_error<5e-4,order_error
        z_nomask,*_=batch(keys,frontend_mask=False,backend_mask=False)
        z_frontonly,*_=batch(keys,backend_mask=False)
        no_mask_error=float((z_nomask-zbase).abs().max());front_only_error=float((z_frontonly-zbase).abs().max())
        # Deliberately corrupt only padding features: all26 temporal pools must
        # ignore these values, independent of frontend implementation details.
        altered=torch.where(mbase[:,None,:,None],fbase,torch.full_like(fbase,37.))
        altered_z=encode(cls,altered,mbase);backend_pad_error=float((altered_z-zbase).abs().max())
        assert backend_pad_error<5e-5,backend_pad_error
        del zextra,fextra,mextra,zrev,z_nomask,z_frontonly,altered,altered_z
    handle.remove()
    # Validate both original optimization stages' gradient routes, without an
    # optimizer step or weight update. The frontend remains frozen as baseline.
    cls.train();labels=torch.tensor([1,1,0,0,0,0],device=device)
    long_f=torch.cat([fbase[0:1],fbase[2:3],fbase[1:2],fbase[3:4],fbase[1:2],fbase[3:4]],dim=0).detach()
    long_m=torch.cat([mbase[0:1],mbase[2:3],mbase[1:2],mbase[3:4],mbase[1:2],mbase[3:4]],dim=0)
    # Different actual short-view frame counts; no invented/repeated frames.
    short_f=long_f[:,:,:max(2,long_f.shape[2]//2)].clone()
    valid_counts=torch.minimum(long_m.sum(1),torch.tensor(short_f.shape[2],device=device))
    short_m=torch.arange(short_f.shape[2],device=device)[None,:]<valid_counts[:,None]
    for name,p in cls.named_parameters():p.requires_grad_(name!='loss.weight')
    cls.zero_grad(set_to_none=True)
    zlong=encode(cls,long_f,long_m);zshort=encode(cls,short_f,short_m)
    oc,_=cls.loss(zlong,labels);bpl=cls.loss_bpl(zlong[:2],zshort[:2]);total=oc+bpl;total.backward()
    grads=[p.grad for p in cls.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    stage1=dict(loss=float(total),oc_loss=float(oc),bpl_loss=float(bpl),finite_gradient_tensors=len(grads),centroid_frozen=cls.loss.weight.grad is None,
        short_view_scope='Truncated SSL features solely for backend gradient-contract test; not an audio-level BPL augmentation or training-quality validation')
    cls.zero_grad(set_to_none=True)
    for name,p in cls.named_parameters():p.requires_grad_(name=='loss.weight')
    zcent=encode(cls,long_f[:2],long_m[:2]);cent_loss,_=cls.loss(zcent,labels[:2]);cent_loss.backward()
    assert cls.loss.weight.grad is not None and torch.isfinite(cls.loss.weight.grad).all()
    assert all(p.grad is None for name,p in cls.named_parameters() if name!='loss.weight')
    stage2=dict(loss=float(cent_loss),centroid_gradient_norm=float(cls.loss.weight.grad.norm()),other_parameters_without_grad=True)
    assert state_before==ref.state_digest(cls)
    props=torch.cuda.get_device_properties(device)
    result=dict(status='pass',rank=rank,seconds=time.monotonic()-start,
        versions=dict(torch=torch.__version__,transformers=transformers.__version__),
        frontend_config={k:getattr(config,k) for k in ['feat_extract_norm','do_stable_layer_norm','num_hidden_layers','hidden_size','conv_kernel','conv_stride']},
        device=dict(name=props.name,total_memory_MiB=props.total_memory/2**20),
        original_single_profiles=profiles,padded_batch_equivalence=comparisons,
        pad_extra_second_max_z_error=extra_error,row_order_max_z_error=order_error,
        arbitrary_backend_padding_max_z_error=backend_pad_error,
        omitted_all_masks_max_z_change=no_mask_error,omitted_backend_masks_max_z_change=front_only_error,
        classifier_backward_stage1=stage1,centroid_backward_stage2=stage2,
        classifier_state_before=state_before,classifier_state_after=ref.state_digest(cls),
        no_optimizer_or_parameter_update=True,noise_std_for_equivalence=0.,
        short_bpl_features_are_gradient_contract_only=True,
        actual_DDP_training_tested=False,baseline_batch80_full_length_capacity_tested=False,
        raw_audio_or_SSL_features_saved=False)
    save(root/f'rank{rank}_results.json',result)


def main(root):
    import torch
    import torch.multiprocessing as mp
    assert torch.cuda.device_count()==2 and not (root/'rank0_results.json').exists()
    mp.spawn(worker,args=(str(root),),nprocs=2,join=True)
    archive,_=core.settings(REFERENCE);before=json.loads((root/'prepare_complete.json').read_text())
    assert before['archived_hashes']=={str(p.relative_to(archive)):core.digest(p) for p in archive.rglob('*.py')}
    assert before['current_source_hashes']=={n:core.digest(Path('/home/mentee/code/EBM')/n) for n in before['current_source_hashes']}
    assert before['checkpoint_sha256']==core.digest(REFERENCE/'model/BestCLS_50_21DF.pt')
    r=[json.loads((root/f'rank{i}_results.json').read_text()) for i in [0,1]]
    assert all(x['status']=='pass' for x in r)
    save(root/'complete.json',dict(status='pass',all_archive_current_source_checkpoint_hashes_unchanged=True,
        files={n:core.digest(root/n) for n in ['prepare_complete.json','rank0_results.json','rank1_results.json']},
        new_model_training=False,production_code_modified=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','probe']);p.add_argument('--root',required=True,type=Path);a=p.parse_args()
    assert Path('/.dockerenv').exists() and os.environ.get('CUDA_VISIBLE_DEVICES')=='0,1'
    prepare(a.root) if a.stage=='prepare' else main(a.root)
