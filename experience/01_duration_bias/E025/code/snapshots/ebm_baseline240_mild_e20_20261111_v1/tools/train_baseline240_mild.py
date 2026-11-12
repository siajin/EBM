"""Archived240 full-graph training, mild-source adapter, controlled monitoring."""
import hashlib
import csv
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import FunctionType

ROOT=Path(sys.argv[1])


def save(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def load_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module);return module


def controlled_test(trainer,root,phase,epoch,limit=None):
    import numpy as np
    import torch
    import torch.distributed as dist
    from positive_bpl import evaluation_rng,oc_evaluation_protocol
    from native_steps import features
    from sklearn.metrics import roc_curve
    cfg=json.loads((root/'experiment.json').read_text())
    trainer.plm.eval();trainer.classifier.eval()
    loader=trainer.eval_loader[phase]
    versions=[p._version for p in trainer.classifier.parameters()]
    rows=[]
    with evaluation_rng(trainer.args,525252),torch.no_grad():
        native=load_file('controlled_backend',root/'code/model/b_linear.py')
        losses=load_file('controlled_losses',root/'code/model/loss.py')
        head=native.B_Linear(25,1024,160,128,loss=losses.OCSoftmax(160,class_weight=[5/9,5]),
            loss_bpl=losses.BPL(30)).to(trainer.args['device']).eval()
        state={k.removeprefix('module.'):v for k,v in trainer.classifier.state_dict().items()}
        head.load_state_dict(state,strict=True)
        # Model construction must not shift the common ASP evaluation draws.
        with evaluation_rng(trainer.args,525252):
            for number,batch in enumerate(loader):
                x,mask=features(trainer.plm,batch['waves'],trainer.args['device'])
                scores=head(x,mask=mask)[1].flatten()
                assert torch.isfinite(scores).all()
                rows.extend((int(i),float(s),int(y)) for i,s,y in zip(batch['indices'],scores.cpu(),batch['labels']))
                if limit is not None and len(rows)>=limit:break
    assert versions==[p._version for p in trainer.classifier.parameters()]
    gathered=[None]*2;dist.all_gather_object(gathered,rows);rows=sorted(r for part in gathered for r in part)
    if limit is None:assert [r[0] for r in rows]==list(range(len(loader.dataset)))
    eer=None
    if len({r[2] for r in rows})==2:
        fpr,tpr,_=roc_curve([r[2] for r in rows],[r[1] for r in rows],pos_label=1,drop_intermediate=False)
        delta=fpr-(1-tpr);k=int(np.flatnonzero(delta>=0)[0])
        eer=float(100*fpr[0]) if k==0 else float(100*(fpr[k-1]-delta[k-1]/(delta[k]-delta[k-1])*(fpr[k]-fpr[k-1])))
    if trainer.logger is not None and limit is None:
        protocol=oc_evaluation_protocol(trainer.args,cfg['frontend_revision'])
        protocol.update(phase=phase,epoch=epoch,rows=len(rows),selection='user fixed20; 19LA monitoring every5; no dev')
        trainer.logger.log_text(f'{phase}/scores_{epoch}','index,score,label\n'+''.join(f'{i},{s:.17g},{y}\n' for i,s,y in rows))
        trainer.logger.log_text(f'{phase}/protocol_{epoch}',json.dumps(protocol,indent=2))
        trainer.logger.log_metric(f'{phase}/EER',eer,step=epoch)
    del head
    torch.cuda.empty_cache()
    return eer,rows


def worker(rank,root_string,smoke):
    import numpy as np
    import torch
    import torch.distributed as dist
    root=Path(root_string);cfg=json.loads((root/'experiment.json').read_text())
    # Import the archived main/train/model/loaders first; policy and evaluation
    # helpers remain in a separate, explicitly named native source tree.
    sys.path[:0]=[str(root/'baseline_code'),str(root/'code')]
    import main,train,arguments,data_loaders
    from mild_duration import MildTrainSet
    from baseline240_mild import MildOriginal240TrainSet,ConsumedDataLoader
    native_data=load_file('controlled_data_loaders',root/'code/data_loaders.py')
    base=train.ModelTrainer;original_loaders=data_loaders.get_loaders;original_class=data_loaders.TrainSet
    original_loader_class=data_loaders.DataLoader
    def loaders(args,items,val,evals):
        train_meta=load_file('controlled_train_metadata',root/'code/dataset/ASVspoof2019_la_train.py')
        metadata=train_meta.load_train_items_only(args['path_19LA'])
        assert len(items)==len(metadata)==25380
        for item,record in zip(items,metadata):
            assert (item.path,item.label,item.speaker)==(record.path,record.label,record.speaker)
            item.type=record.type
        mild=MildTrainSet(args,items)
        namespace=dict(original_loaders.__globals__,
            TrainSet=lambda a,i:MildOriginal240TrainSet(a,i,original_class,mild,root),DataLoader=ConsumedDataLoader)
        loader_function=FunctionType(original_loaders.__code__,namespace,original_loaders.__name__,
            original_loaders.__defaults__,original_loaders.__closure__)
        result=loader_function(args,items,val,evals)
        def evaluation(data):
            ds=native_data.EnrollmentSet(dict(args,native_length=True,cap_long=True),data)
            return original_loader_class(ds,batch_size=8,sampler=native_data.ExactDistributedEvalSampler(ds,args),
                num_workers=0 if smoke else 4,collate_fn=native_data.collate_eval,
                generator=torch.Generator().manual_seed(1+rank))
        return (*result[:4],evaluation(val),{name:evaluation(db.trials) for name,db in evals.items()})
    data_loaders.get_loaders=loaders
    # Refuse the archived logger's destructive same-name branch.
    import logger.local
    original_logger_init=logger.local.LocalLogger.__init__
    def safe_logger_init(self,path,name,project,tags,*a,**kw):
        target=Path('/')/path/project/('/'.join(tags))/name
        if target.exists():raise FileExistsError('Refuse existing experiment: '+str(target))
        return original_logger_init(self,path,name,project,tags,*a,**kw)
    logger.local.LocalLogger.__init__=safe_logger_init

    class Trainer(base):
        def train(self,epoch):
            for loader in [self.train_loader_bona,self.train_loader_spoof]:
                loader.dataset.epoch=epoch;loader.dataset.visits={}
            assert not (root/f'consumed_epoch{epoch}_rank{rank}.csv').exists()
            self.audit={'epoch':epoch,'updates':0,'step':0,'oc':[],'bpl':[],'pair_counts':[]}
            super().train(epoch)  # Exact archived method; no gradient replay/microbatch.
            expected=3 if smoke else 108
            assert self.audit['step']==expected and self.audit['updates']==2*expected
            if not smoke:
                with (root/f'consumed_epoch{epoch}_rank{rank}.csv').open() as stream:
                    consumed=list(csv.DictReader(stream))
                assert len(consumed)==12906
                for y,name,total in [(1,'bona',1290),(0,'spoof',11616)]:
                    rows=[r for r in consumed if int(r['label'])==y]
                    adapter=getattr(self,'train_loader_'+name).dataset
                    expected_ids={adapter.indices[i] for i in getattr(self,'sampler_'+name)}
                    assert len(rows)==total and {int(r['index']) for r in rows}==expected_ids
            save(root/f'epoch{epoch}_rank{rank}.json',{'status':'passed','epoch':epoch,'steps':expected,
                'backend_updates':expected,'center_updates':expected,'BPL_pair_counts':self.audit['pair_counts'],
                'PLM_grads_absent':all(p.grad is None for p in self.plm.parameters())})

        def run(self):
            assert self.plm.module.config._commit_hash==cfg['frontend_revision']
            self.args['frontend_resolved_revision']=cfg['frontend_revision']
            plm_versions=[p._version for p in self.plm.parameters()]
            center=self.classifier.module.loss.weight.detach().clone()
            def oc_hook(module,inputs,output):self.audit['oc'].append(float(output[0].detach()))
            def bpl_hook(module,inputs,output):
                self.audit['bpl'].append((float(output.detach()),int(inputs[0].shape[0]*inputs[1].shape[0])))
            self.classifier.module.loss.register_forward_hook(oc_hook)
            self.classifier.module.loss_bpl.register_forward_hook(bpl_hook)
            def step_hook(optimizer,args,kwargs):
                a=self.audit;a['updates']+=1
                assert all(p.grad is None for p in self.plm.parameters())
                if a['updates']%2:
                    assert len(a['oc'])==len(a['bpl'])==1 and not self.classifier.module.loss.weight.requires_grad
                    a['loss1']=a['oc'][0]+a['bpl'][0][0]
                else:
                    assert len(a['oc'])==2 and self.classifier.module.loss.weight.requires_grad
                    a['step']+=1;a['pair_counts'].append(a['bpl'][0][1])
                    values={'oc':a['oc'][0],'bpl':a['bpl'][0][0],'loss1':a['loss1'],'loss2':a['oc'][1],
                        'positive_pairs':a['bpl'][0][1],'optimizer_steps':2,'center_updates':1}
                    assert all(np.isfinite(v) for v in values.values())
                    with (root/f'steps_{"smoke_" if smoke else ""}rank{rank}.jsonl').open('a') as out:
                        out.write(json.dumps({'epoch':a['epoch'],'batch':a['step'],**values})+'\n')
                    if a['step']==1 or a['step']%10==0:
                        save(root/f'train_progress_rank{rank}.json',{'epoch':a['epoch'],'batch':a['step'],
                            'batches':3 if smoke else 108,'values':values,'updated_unix':time.time(),'smoke':smoke})
                    a['oc']=[];a['bpl']=[]
            self.optimizer.register_step_post_hook(step_hook)
            if smoke:
                # Full-size and both original ragged-batch shapes, each on2GPUs.
                for name,sizes in [('bona',[12,12,6]),('spoof',[108,60,108])]:
                    old=getattr(self,'train_loader_'+name)
                    ids=list(iter(getattr(self,'sampler_'+name)));batches=[];offset=0
                    for size in sizes:batches.append(ids[offset:offset+size]);offset+=size
                    replacement=ConsumedDataLoader(old.dataset,batch_sampler=batches,num_workers=0)
                    setattr(self,'train_loader_'+name,replacement)
                self.train(0)
                # Full graph/actual original train method must update the center.
                assert not torch.equal(center,self.classifier.module.loss.weight)
                # Turn off training-only loss hooks before evaluation forwards.
                self.classifier.module.loss._forward_hooks.clear();self.classifier.module.loss_bpl._forward_hooks.clear()
                e1,rows1=controlled_test(self,root,'19LA',0,limit=32)
                e2,rows2=controlled_test(self,root,'19LA',0,limit=32)
                assert rows1==rows2
                save(root/f'train_gpu_smoke_rank{rank}.json',{'status':'passed','logical_steps':3,
                    'backend_updates':3,'center_updates':3,'pair_counts':self.audit['pair_counts'],
                    'controlled_evaluation_repeat_equal':True,'center_changed':True,
                    'frontend_versions_unchanged':plm_versions==[p._version for p in self.plm.parameters()]})
                return
            for epoch in range(1,21):
                self.train(epoch);self._synchronize()
                if epoch%5==0:
                    # Temporarily remove only audit hooks around evaluation.
                    hooks=self.classifier.module.loss._forward_hooks.copy()
                    self.classifier.module.loss._forward_hooks.clear()
                    controlled_test(self,root,'19LA',epoch)
                    self.classifier.module.loss._forward_hooks.update(hooks)
                    if self.logger is not None:
                        self.logger.save_model(f'CLS_{epoch}',self.classifier.state_dict())
                        self.logger.save_model(f'TrainState_{epoch}',{'epoch':epoch,
                            'classifier':self.classifier.state_dict(),'optimizer':self.optimizer.state_dict(),
                            'scheduler':self.lr_scheduler.state_dict(),
                            'args':{k:v for k,v in self.args.items() if not any(s in k.lower() for s in ['token','password','secret'])}})
                    self._synchronize()
            assert plm_versions==[p._version for p in self.plm.parameters()]
            save(root/f'train_done_rank{rank}.json',{'status':'complete','epoch':20,'logical_steps':2160,
                'backend_updates':2160,'center_updates':2160,'frontend_versions_unchanged':True})

    train.ModelTrainer=Trainer
    args,_,experiment=arguments.get_args()
    args.update(epoch=20,batch_size=120,world_size=2,gpu_ids=['0','1'],rand_seed=1,
        duration_policy_path=str(root/'duration_policy.json'),mild_transform_cache=str(root/'audio_cache'),
        activity_policy_path=None,native_length=True,cap_long=True,evaluation_mode='controlled_oc',
        evaluation_seed=1729,num_test_frames=404,num_seg=1,eval_batch_size=8,
        num_workers=0 if smoke else 4,port=str(cfg['smoke_port'] if smoke else cfg['train_port']),
        name=cfg['experiment_name']+('_gpu_smoke' if smoke else ''),
        path_log=str(root/'smoke_results') if smoke else '/results')
    # native_length above configures only controlled EnrollmentSet; archived
    # TrainSet has no native flag and always repeats/truncates as before.
    experiment.update(training_basis='archived240 full graph, unchanged ModelTrainer.train',
        epoch=20,global_batch_size=240,per_rank_batch=120,data_policy_sha256=cfg['duration_policy_sha256'])
    torch.cuda.set_device(rank)
    main.run(rank,args,experiment)
    dist.destroy_process_group()


if __name__=='__main__':
    import torch.multiprocessing as mp
    assert Path('/.dockerenv').is_file() and os.environ['CUDA_VISIBLE_DEVICES']=='0,1'
    cfg=json.loads((ROOT/'experiment.json').read_text())
    for tree,key in [('code','source_sha256'),('baseline_code','baseline_source_sha256')]:
        for name,expected in cfg[key].items():
            assert hashlib.sha256((ROOT/tree/name).read_bytes()).hexdigest()==expected,(tree,name)
    smoke='--smoke' in sys.argv
    assert not (ROOT/('smoke_results' if smoke else 'train.claim')).exists()
    if not smoke:(ROOT/'train.claim').mkdir()
    torch.multiprocessing.set_sharing_strategy('file_system')
    mp.spawn(worker,args=(str(ROOT),smoke),nprocs=2,join=True)
