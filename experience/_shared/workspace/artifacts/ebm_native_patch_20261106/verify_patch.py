"""Bounded server/container tests; never run the experiment logger or full training."""
import argparse
import copy
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd


def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


class DummyLogger:
    def __getattr__(self,name):return lambda *a,**k:None


def worker(rank,root_s,port,installed):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    root=Path(root_s);code=Path('/home/mentee/code/EBM') if installed else root/'stage'
    sys.path.insert(0,str(code))
    import model,data_loaders,arguments,train,native_steps
    from transformers import AutoConfig,Wav2Vec2Model
    import soundfile as sf
    torch.set_num_threads(2);torch.cuda.set_device(rank);device=f'cuda:{rank}'
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(20261106);torch.cuda.manual_seed(20261106)
    dist.init_process_group('nccl',init_method=f'tcp://127.0.0.1:{port}',world_size=2,rank=rank,timeout=datetime.timedelta(minutes=4))
    checks={};report={'rank':rank,'installed':installed}
    def ck(name,value):
        assert value,name
        checks[name]=True
    args,_,_=arguments.get_args()
    ck('preserved_global_batch240',args['batch_size']==240)
    ck('short303',args['num_train_frames_short']==303)
    ck('main_import',__import__('main') is not None)
    ck('rank_current_cuda_device',torch.cuda.current_device()==rank and "torch.cuda.set_device(args['process_id'])" in (code/'main.py').read_text())
    if rank==0:
        from logger.local import LocalLogger
        sentinel=root/'guard_test'/'existing';sentinel.mkdir(parents=True,exist_ok=True)
        marker=sentinel/'sentinel.txt';marker.write_text('preserve')
        refused=False
        try:LocalLogger(str(root),'existing','guard_test',[])
        except FileExistsError:refused=True
        ck('logger_refuses_overwrite',refused and marker.read_text()=='preserve')

    # Same parameters/checkpoint format, and ASP mathematical/gradient reference.
    small=lambda:model.B_Linear(3,16,8,4,loss=model.OCSoftmax(8,class_weight=[5/9,5]),loss_bpl=model.BPL(30)).to(device)
    tiny=small().double().eval()
    pool=tiny.AHP_ASP;pool.noise_std=0
    x=torch.randn(3,11,4,device=device,dtype=torch.double,requires_grad=True)
    lengths=torch.tensor([2,7,11],device=device);mask=torch.arange(11,device=device)[None,:]<lengths[:,None]
    actual=pool(x,mask=mask)
    expected=torch.cat([pool(x[i:i+1,:int(n)]) for i,n in enumerate(lengths)])
    ga=torch.autograd.grad(actual.sum(),x,retain_graph=True)[0];ge=torch.autograd.grad(expected.sum(),x)[0]
    ck('ragged_pool_value',float((actual-expected).abs().max())<1e-10)
    ck('ragged_pool_gradient',float((ga-ge).abs().max())<1e-10 and (ga[~mask]==0).all().item())
    ck('original_BPL_local_formula',torch.allclose(tiny.loss_bpl(actual[:2],actual[1:3]),(torch.nn.functional.softplus(30*(1-torch.nn.functional.normalize(actual[:2],dim=1)@torch.nn.functional.normalize(actual[1:3],dim=1).T))-.3).mean()))
    del tiny,pool,x,actual,expected,ga,ge

    class ToyPLM(torch.nn.Module):
        def forward(self,x,attention_mask=None,output_hidden_states=True):
            y=torch.nn.functional.avg_pool1d(x.unsqueeze(1),400,320).transpose(1,2)
            channels=torch.arange(1,17,device=x.device,dtype=x.dtype).reshape(1,1,-1)
            states=tuple(torch.sin(y*channels+i/7) for i in range(3))
            return SimpleNamespace(hidden_states=states)
        def _get_feature_vector_attention_mask(self,t,mask):
            n=(mask.sum(1)-400)//320+1
            return torch.arange(t,device=mask.device)[None,:]<n[:,None]

    def batches(seed):
        rng=np.random.default_rng(seed+rank);rows=[]
        for i in range(12):
            a=rng.normal(0,.1,3000+i*23).astype('float32')
            n=720+i*200 if rank==1 and i<3 and seed!=79 else 2000
            rows.append((a,a[:n].copy(),int(i<6),i))
        return data_loaders.collate_train(rows,short_size=2000)

    cached=DDP(small(),device_ids=[rank],find_unused_parameters=True)
    reference=copy.deepcopy(cached.module)
    cfg={'device':device,'micro_batch_size':2,'short_oc_weight':.25,'padding_loss_weight':.1}
    opt=torch.optim.Adam(cached.parameters(),lr=1e-5,weight_decay=1e-4)
    refopt=torch.optim.Adam(reference.parameters(),lr=1e-5,weight_decay=1e-4)
    toy=ToyPLM().to(device).eval()
    context=SimpleNamespace(classifier=cached,plm=toy,args=cfg,optimizer=opt)
    def average_gradients(parameters):
        for p in parameters:
            if p.grad is not None:
                dist.all_reduce(p.grad);p.grad.div_(2)
    errors=[]
    for iteration in range(3):
        batch=batches(77+iteration)
        reference.train();cached.train()
        cpu_rng=torch.get_rng_state();gpu_rng=torch.cuda.get_rng_state(rank)
        native_steps._freeze(reference,False);refopt.zero_grad(set_to_none=True)
        embeddings={}
        for name in ['long','short','wrapped']:
            parts=[]
            for i in range(0,12,cfg['micro_batch_size']):
                f,m=native_steps.features(toy,batch[name][i:i+cfg['micro_batch_size']],device)
                parts.append(reference(f,mask=m,return_embedding=True))
            embeddings[name]=torch.cat(parts)
        loss,bona,values=native_steps._objective(reference,embeddings,batch,cfg)
        loss.backward();average_gradients(reference.parameters());refopt.step()
        native_steps._freeze(reference,True);refopt.zero_grad(set_to_none=True)
        for i in range(0,bona,cfg['micro_batch_size']):
            stop=min(i+cfg['micro_batch_size'],bona);f,m=native_steps.features(toy,batch['long'][i:stop],device)
            v,_=reference(f,mask=m,label=batch['labels'][i:stop].to(device),logical_batch_size=bona)
            (v*((stop-i)/bona)).backward()
        average_gradients(reference.parameters());refopt.step()
        torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(gpu_rng,rank)
        result=native_steps.train_batch(context,batch)
        error=max(float((p-q).abs().max()) for p,q in zip(reference.parameters(),cached.module.parameters()))
        errors.append(error);ck(f'gradient_cache_two_stage_DDP_step{iteration}',error<2e-5)
        ck(f'global_active_pairs_step{iteration}',result['active_pairs']==(0 if iteration==2 else 3))
        cfg['micro_batch_size']=3
    for label,rho,padding in [('E1',0.,0.),('E2',.25,0.)]:
        cfg.update(short_oc_weight=rho,padding_loss_weight=padding)
        values=native_steps.train_batch(context,batches(80))
        ck('auxiliary_mode_'+label,all(np.isfinite(v) for v in values.values()))
    report['toy_update_max_errors']=errors
    del cached,reference,opt,refopt,context,toy
    torch.cuda.empty_cache()

    # Actual train set sampler and full XLS-R -> classifier -> update -> eval.
    table=pd.read_parquet('/home/mentee/analysis/embedding_geometry/20261105_fast_p0_01/manifest.parquet')
    table=table[table.split=='T'].copy()
    items=[SimpleNamespace(path=r.path,label=int(r.y),type=r.attack_id) for _,r in table.iterrows()]
    actualargs=dict(args,rank=rank,world_size=2,batch_size=120,num_workers=0,device=device,flag_parent=rank==0)
    dataset=data_loaders.TrainSet(actualargs,items)
    sampler=data_loaders.BalancedLengthBatchSampler(dataset,actualargs)
    local=list(sampler);ck('actual_sampler108steps',len(local)==108)
    ck('actual_class12_108',all([items[i].label for i in b]==[1]*12+[0]*108 for b in local))
    all_ids=[None,None];dist.all_gather_object(all_ids,[i for b in local for i in b])
    flat=[i for q in all_ids for i in q]
    ck('actual_sampler_full_coverage',len(flat)==25920 and len(set(flat))==25380)
    # Dataset indexing/collation uses source codec settings untouched; this test
    # selects the existing no-codec/no-RawBoost random branches deterministically.
    with patch.object(data_loaders.random,'random',return_value=1.),patch.object(data_loaders.random,'randint',return_value=0):
        sample=dataset[0]
    original,sr=sf.read(items[0].path)
    ck('actual_loader_long_uncropped',len(sample[0])==len(original))
    ck('actual_loader_short_cap',len(sample[1])==min(len(original),48440))
    prepared=json.loads((root/'prepared.json').read_text())
    original_codec=[v.strip() for v in (root/'before/data_loaders.py').read_text().splitlines() if 'p_codec_aug' in v or '.codec_aug.convert(' in v]
    current_codec=[v.strip() for v in (code/'data_loaders.py').read_text().splitlines() if 'p_codec_aug' in v or '.codec_aug.convert(' in v]
    ck('codec_probability_and_calls_untouched',original_codec==current_codec)
    del sampler,local,flat,all_ids,dataset

    config=AutoConfig.from_pretrained(args['PLM_name'],local_files_only=True)
    plm=Wav2Vec2Model.from_pretrained(args['PLM_name'],config=config,local_files_only=True).to(device).eval().requires_grad_(False)
    cls=model.B_Linear(args['num_layers'],args['hidden_size'],args['output_size'],args['agg_size'],
        loss=model.OCSoftmax(args['output_size'],r_real=args['r_real'],r_fake=args['r_fake'],alpha=args['alpha'],class_weight=[5/9,5]),loss_bpl=model.BPL(args['alpha_cl'])).to(device)
    checkpoint='/home/mentee/results/CM/LA/EBM_e50_bs160_lr1e5/model/BestCLS_50_21DF.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    cls.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True);del state
    baseline_path='/home/mentee/results/CM/LA/EBM_e50_bs160_lr1e5/backup/scripts/model/b_linear.py'
    spec=importlib.util.spec_from_file_location('original_b_linear',baseline_path);baseline_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(baseline_module)
    baseline=baseline_module.B_Linear(args['num_layers'],args['hidden_size'],args['output_size'],args['agg_size'],loss=None).to(device).eval()
    baseline.load_state_dict({k:v for k,v in cls.state_dict().items() if k!='loss.weight'},strict=True)
    for obj in [cls,baseline]:
        obj.eval()
        for m in obj.modules():
            if hasattr(m,'noise_std'):m.noise_std=0.
    probe=pd.read_parquet('/home/mentee/analysis/embedding_geometry/20261106_variable_length_01/manifest.parquet').set_index('selection')
    keys=['train_y1_minimum','train_y0_minimum','train_y1_median','train_y0_median','train_y0_maximum','DF_remaining_maximum']
    waves=[torch.from_numpy(sf.read(probe.loc[k].path,dtype='float32')[0]) for k in keys]
    zlist=[];max_native=0.
    with torch.no_grad():
        for w in waves:
            f,m=native_steps.features(plm,[w],device)
            new=cls(f,mask=m,return_embedding=True);old=baseline(f)
            max_native=max(max_native,float((new-old).abs().max()));zlist.append(new)
        f,m=native_steps.features(plm,waves[:4],device)
        z=cls(f,mask=m,return_embedding=True)
        batch_error=float((z-torch.cat(zlist[:4])).abs().max())
    ck('strict_old_checkpoint_and_native_mask',max_native<1e-5)
    ck('real_batch_single_equivalence',batch_error<1e-4)
    report.update(native_max_z_error=max_native,real_batch_z_error=batch_error,max_input_seconds=max(len(w) for w in waves)/16000)
    del baseline,waves,zlist,z,f,m
    for m in cls.modules():
        if hasattr(m,'noise_std'):m.noise_std=1e-5
    cls.train();ddp=DDP(cls,device_ids=[rank],find_unused_parameters=True)
    optimizer=torch.optim.Adam(ddp.parameters(),lr=args['lr'],weight_decay=args['weight_decay'])
    selected=[]
    for y,quota in [(1,24),(0,216)]:
        q=table[table.y==y].sort_values('utterance_id').head(quota)
        selected.extend([r for _,r in q.iloc[rank::2].iterrows()])
    rows=[]
    for index,r in enumerate(selected):
        a,sr=sf.read(r.path,dtype='float32');n=min(len(a),48440);start=(len(a)-n)//2
        rows.append((a,a[start:start+n].copy(),int(r.y),index))
    batch=data_loaders.collate_train(rows,short_size=48440)
    engine=SimpleNamespace(classifier=ddp,plm=plm,args=actualargs,optimizer=optimizer)
    versions=[p._version for p in plm.parameters()]
    before={k:v.detach().clone() for k,v in cls.state_dict().items()}
    torch.cuda.reset_peak_memory_stats(rank)
    stats=native_steps.train_batch(engine,batch)
    ck('real_240_logical_batch_forward_backward',all(np.isfinite(v) for v in stats.values()))
    ck('real_encoder_updated',not torch.equal(before['linear.weight'],cls.linear.weight))
    ck('real_centroid_updated',not torch.equal(before['loss.weight'],cls.loss.weight))
    ck('real_frontend_frozen',versions==[p._version for p in plm.parameters()] and all(p.grad is None for p in plm.parameters()))
    report.update(real_train_stats=stats,peak_training_allocated_MiB=torch.cuda.max_memory_allocated(rank)/2**20)
    # Every rank must shard the very same evaluation dataset.
    evalitems=[SimpleNamespace(path=probe.loc[k].path,label=int(probe.loc[k].y),type=probe.loc[k].attack_id)
               for k in ['train_y1_minimum','train_y1_median','train_y0_minimum','train_y0_median','train_y0_maximum']]
    evalset=data_loaders.EnrollmentSet(actualargs,evalitems)
    evalloader=torch.utils.data.DataLoader(evalset,batch_size=2,sampler=data_loaders.ExactDistributedEvalSampler(evalset,actualargs),collate_fn=data_loaders.collate_eval)
    trainer=train.ModelTrainer();trainer.args=actualargs;trainer.plm=plm;trainer.classifier=ddp;trainer.val_loader=evalloader;trainer.eval_loader={};trainer.logger=None
    values=trainer.test('V',0)
    ck('uneven_eval_no_deadlock_no_duplicate',len(trainer.last_eval_rows)==5 and np.isfinite(values[0]))
    del ddp,cls,optimizer,engine,trainer,plm,batch,before
    torch.cuda.empty_cache()

    # Exercise main.run wiring without real experiment logger/full training.
    import main
    smallargs=dict(actualargs,batch_size=20,num_layers=3,hidden_size=16,output_size=8,agg_size=4,micro_batch_size=2)
    trainitems=[SimpleNamespace(path=r.path,label=int(r.y),type=r.attack_id) for r in selected[:2]+selected[12:30]]
    ck('main_smoke_class_count',sum(i.label for i in trainitems)==2 and len(trainitems)==20)
    class Builder:
        def __init__(self,*a,**k):pass
        def __getattr__(self,k):return lambda *a,**kw:self
        def build(self):return DummyLogger()
    def init(args):
        args['rank']=rank;args['device']=device
    def bounded_run(self):
        self.train(1)
        self.test('V',0)
        ck('main_run_all_connections',len(self.last_eval_rows)==5)
    with patch.object(main,'set_experiment_environment',side_effect=init),patch.object(main.Logger,'Builder',Builder),\
         patch.object(main,'ASVspoof2019_LA_Train',return_value=SimpleNamespace(train_set=trainitems,dev_set=evalitems)),\
         patch.object(main,'ASVspoof2019_LA_Eval',return_value=SimpleNamespace(trials=evalitems)),\
         patch.object(main,'ASVspoof2021_LA_Eval',return_value=SimpleNamespace(trials=evalitems)),\
         patch.object(main,'ASVspoof2021_DF_Eval',return_value=SimpleNamespace(trials=evalitems)),\
         patch.object(main.AutoConfig,'from_pretrained',return_value=SimpleNamespace()),\
         patch.object(main.Wav2Vec2Model,'from_pretrained',side_effect=lambda *a,**kw:ToyPLM().to(device)),\
         patch.object(train.ModelTrainer,'run',bounded_run),\
         patch.object(data_loaders.random,'random',return_value=1.),patch.object(data_loaders.random,'randint',return_value=0):
        main.run(rank,smallargs,{})
    report.update(status='pass',checks=checks,full_training_launched=False,raw_audio_or_SSL_saved=False)
    save(root/f'{"installed" if installed else "stage"}_rank{rank}.json',report)
    dist.barrier();dist.destroy_process_group()


def main():
    import torch.multiprocessing as mp
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--installed',action='store_true');args=parser.parse_args()
    assert Path('/.dockerenv').exists() and os.environ.get('CUDA_VISIBLE_DEVICES')=='0,1'
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    mp.spawn(worker,args=(str(args.root),port,args.installed),nprocs=2,join=True)
    stage='installed' if args.installed else 'stage'
    reports=[json.loads((args.root/f'{stage}_rank{i}.json').read_text()) for i in [0,1]]
    assert all(r['status']=='pass' for r in reports)
    save(args.root/f'{stage}_complete.json',{'status':'pass','files':{f'{stage}_rank{i}.json':hashlib.sha256((args.root/f'{stage}_rank{i}.json').read_bytes()).hexdigest() for i in [0,1]}})


if __name__=='__main__':main()
