"""Additional server-only audit of codec lengths and true waveform short views.

No parameter update, no source mutation, no saved waveform/SSL features.
"""
import argparse
import os
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd

BASE=Path('/home/mentee/analysis/embedding_geometry/20261106_variable_length_01')
sys.path.insert(0,str(BASE))
import variable_length_sidecar_v1 as probe
core,ref,REFERENCE=probe.core,probe.ref,probe.REFERENCE


def codec_audit(root):
    import soundfile as sf
    archive,args=core.settings(REFERENCE);sys.path.insert(0,str(archive))
    from data_processing import CodecAugmentationLight
    from data_processing.augmentation.raw_boost import LnL_convolutive_noise,ISD_additive_noise,SSI_additive_noise
    aug=CodecAugmentationLight();manifest=pd.read_parquet(BASE/'manifest.parquet').set_index('selection')
    records=[];rawboost=[]
    for key in ['train_y0_minimum','train_y1_median','train_y0_maximum']:
        row=manifest.loc[key];audio,sr=sf.read(row.path)
        for codec,quality in [('mp3',4),('ogg',4),('aac',64),('a-law',8),('u-law',8)]:
            aug.selectCodec=lambda c=codec,q=quality:(c,q)
            out=aug.convert(audio,sr)
            assert out.ndim==1 and np.isfinite(out).all() and len(out)>=720
            records.append(dict(key=key,codec=codec,quality=quality,input_samples=len(audio),output_samples=len(out),delta_samples=len(out)-len(audio)))
        for func in [LnL_convolutive_noise,ISD_additive_noise,SSI_additive_noise]:
            np.random.seed(20261106);out=func(audio)
            assert len(out)==len(audio) and np.isfinite(out).all()
            rawboost.append(dict(key=key,augmentation=func.__name__,samples=len(out),length_preserved=True))
    core.save_json(root/'augmentation_results.json',dict(status='pass',codec_cases=records,rawboost_cases=rawboost,
        codec_arguments_ignored_by_convert=True,selection_method_override_used=True,
        augmentation_length_must_be_measured_after_conversion=True,raw_audio_or_SSL_features_saved=False))


def worker(rank,root_s):
    import torch
    import soundfile as sf
    from transformers import AutoConfig,Wav2Vec2Model
    root=Path(root_s);torch.set_num_threads(2);torch.cuda.set_device(rank);device=torch.device('cuda',rank)
    torch.manual_seed(20261106);torch.cuda.manual_seed(20261106)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    archive,args=core.settings(REFERENCE);sys.path.insert(0,str(archive));import model
    config=AutoConfig.from_pretrained(args['PLM_name'],local_files_only=True)
    plm=Wav2Vec2Model.from_pretrained(args['PLM_name'],config=config,local_files_only=True).to(device).eval()
    cls=model.B_Linear(num_layer=args['num_layers'],hidden_size=args['hidden_size'],output_size=args['output_size'],agg_size=args['agg_size'],
        loss=model.OCSoftmax(args['output_size'],r_real=args['r_real'],r_fake=args['r_fake'],alpha=args['alpha'],class_weight=[5/9,5]),
        loss_bpl=model.BPL(alpha=args['alpha_cl'])).to(device)
    state=torch.load(REFERENCE/'model/BestCLS_50_21DF.pt',map_location='cpu',weights_only=True)
    cls.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True);del state
    for p in plm.parameters():p.requires_grad_(False)
    before=ref.state_digest(cls);manifest=pd.read_parquet(BASE/'manifest.parquet').set_index('selection')
    keys=['train_y1_minimum','train_y1_median','train_y0_minimum','train_y0_median','train_y0_maximum']
    full=[];short=[];lengths=[]
    for key in keys:
        a,sr=sf.read(manifest.loc[key].path,dtype='float32');assert sr==16000
        long=a.copy();n=min(len(a),48440);start=(len(a)-n)//2
        small=a[start:start+n].copy();full.append(long);short.append(small)
        lengths.append(dict(key=key,long_samples=len(long),short_samples=len(small),short_start=start,
            short_is_exact_contiguous_source_view=bool(np.array_equal(small,a[start:start+n]))))
    def features(wavs):
        sizes=torch.tensor([len(a) for a in wavs],device=device)
        x=torch.zeros(len(wavs),int(sizes.max()),device=device)
        for i,a in enumerate(wavs):x[i,:len(a)]=torch.as_tensor(a,device=device)
        am=torch.arange(x.shape[1],device=device)[None,:]<sizes[:,None]
        with torch.no_grad():
            hs=plm(x,attention_mask=am.long(),output_hidden_states=True).hidden_states;f=torch.stack(hs,dim=1)
        return f,plm._get_feature_vector_attention_mask(f.shape[2],am)
    fl,ml=features(full);fs,ms=features(short)
    assert fl.shape[2]!=fs.shape[2]
    labels=torch.tensor([1,1,0,0,0],device=device)
    for name,p in cls.named_parameters():p.requires_grad_(name!='loss.weight')
    cls.zero_grad(set_to_none=True);zl=probe.encode(cls,fl,ml);zs=probe.encode(cls,fs,ms)
    oc,_=cls.loss(zl,labels);bpl=cls.loss_bpl(zl[:2],zs[:2]);loss=oc+bpl;loss.backward()
    grads=[p.grad for p in cls.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert cls.loss.weight.grad is None and ref.state_digest(cls)==before
    # Independent ragged reference for the masked sample-std and its gradient.
    pool=cls.AHP_ASP.double();pool.noise_std=0.;pool.eval()
    v=torch.randn(3,11,128,device=device,dtype=torch.float64,requires_grad=True)
    counts=torch.tensor([2,7,11],device=device);m=torch.arange(11,device=device)[None,:]<counts[:,None]
    got=probe.masked_pool(pool,v,m)
    expected=torch.cat([pool(v[i:i+1,:int(n)]) for i,n in enumerate(counts)],dim=0)
    value_error=float((got-expected).abs().max());assert value_error<1e-12
    g1=torch.autograd.grad(got.sum(),v,retain_graph=True)[0];g2=torch.autograd.grad(expected.sum(),v)[0]
    grad_error=float((g1-g2).abs().max());assert grad_error<1e-12
    assert (g1[~m]==0).all()
    core.save_json(root/f'rank{rank}_audio_backward.json',dict(status='pass',rank=rank,views=lengths,
        long_feature_shape=list(fl.shape),short_feature_shape=list(fs.shape),
        oc_loss=float(oc),bpl_loss=float(bpl),finite_gradient_tensors=len(grads),
        pooling_noise_std_during_real_audio_backward=1e-5,
        ragged_pool_value_error=value_error,ragged_pool_gradient_error=grad_error,padded_feature_gradient_exact_zero=True,
        parameters_unchanged_before_double_pool_QC=True,optimizer_used=False,
        class_ratio='2 bona+3 spoof only for gradient smoke test, not a training configuration',
        original_sources_modified=False,raw_audio_or_SSL_features_saved=False))


def main(root):
    import torch
    import torch.multiprocessing as mp
    assert torch.cuda.device_count()==2 and not (root/'augmentation_results.json').exists()
    codec_audit(root)
    mp.spawn(worker,args=(str(root),),nprocs=2,join=True)
    core.save_json(root/'complete.json',dict(status='pass',files={n:core.digest(root/n) for n in ['augmentation_results.json','rank0_audio_backward.json','rank1_audio_backward.json']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    assert Path('/.dockerenv').exists() and os.environ.get('CUDA_VISIBLE_DEVICES')=='0,1';main(a.root)
