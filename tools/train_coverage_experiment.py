"""Run the verified 50-epoch coverage experiment with non-random audit hooks."""
import hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(sys.argv[1])
sys.path.insert(0,str(ROOT/'code'))


def save(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def worker(rank,root_string,mode):
    import torch
    import torch.distributed as dist
    import arguments,main,train
    from native_steps import unwrap
    root=Path(root_string);config=json.loads((root/'experiment.json').read_text())
    base=train.ModelTrainer;original_step=train.train_batch
    audit={'step':0,'backend_updates':0,'center_updates':0,'source_sets':[set(),set(),set()]}

    def flush(logger):
        if logger is None:return
        for component in getattr(logger,'components',[]):
            for stream in getattr(component,'_metrics',{}).values():
                stream.flush()

    def audited_step(trainer,batch):
        assert batch['labels'].tolist()==[1]*12+[0]*108
        assert len(batch['long'])==120 and len(batch['bpl_right'])==12
        for role,indices in enumerate([batch['indices'][:12],batch['indices'][12:],batch['bpl_right_indices']]):
            audit['source_sets'][role].update(indices.tolist())
        values=original_step(trainer,batch)
        assert all(np.isfinite(value) for value in values.values())
        assert values['positive_pairs']==144 and values['optimizer_steps']==2 and values['centroid_updates']==1
        audit['step']+=1;audit['backend_updates']+=1;audit['center_updates']+=1
        if audit['step']==1 or audit['step']%10==0:
            save(root/f'train_progress_rank{rank}.json',{
                'epoch':audit['epoch'],'batch':audit['step'],'batches':108,'rank':rank,
                'values':values,'updated_unix':time.time(),'mode':mode})
        return values
    train.train_batch=audited_step

    class AuditedTrainer(base):
        def evaluation(self,epoch,final=False,avg=False,eval_only=False):
            # User-requested monitoring only; fixed50 remains the final model.
            return self.test('19LA',epoch)

        def train(self,epoch):
            audit.update(epoch=epoch,step=0,backend_updates=0,center_updates=0,source_sets=[set(),set(),set()])
            super().train(epoch)
            assert audit['step']==108 and audit['backend_updates']==audit['center_updates']==108
            gathered=[None]*2
            dist.all_gather_object(gathered,[sorted(ids) for ids in audit['source_sets']])
            merged=[set(gathered[0][j])|set(gathered[1][j]) for j in range(3)]
            ds=self.train_loader.dataset
            assert merged==[set(ds.class_indices[1]),set(ds.class_indices[0]),set(ds.class_indices[1])]
            flush(self.logger)
            save(root/f'coverage_epoch{epoch}_rank{rank}.json',{
                'epoch':epoch,'steps':108,'OC_bona_unique':len(merged[0]),'OC_spoof_unique':len(merged[1]),
                'BPL_right_bona_unique':len(merged[2]),'coverage_all_originals':True,
                'backend_updates_per_rank':108,'center_updates_per_rank':108})

        def test(self,phase,epoch):
            result=super().test(phase,epoch)
            flush(self.logger)
            return result

        def run(self):
            ds=self.train_loader.dataset
            assert self.train_sampler.source_sampling=='coverage'
            assert ds.policy_sha256==config['duration_policy_sha256']
            assert ds.activity_policy.sha256==config['activity_policy_sha256']
            assert self.args['frontend_resolved_revision']==config['frontend_revision']
            assert not any(parameter.requires_grad for parameter in self.plm.parameters())
            initial_center=unwrap(self.classifier).loss.weight.detach().clone()
            if mode=='smoke':
                # Smoke performs no optimization and is separate from the full run.
                original=iter(self.train_loader);batch=next(original)
                assert batch['labels'].tolist()==[1]*12+[0]*108
                del original
                from native_steps import features
                with torch.no_grad():
                    f,mask=features(self.plm,batch['long'][:8],self.args['device'])
                    z=unwrap(self.classifier)(f,mask=mask,return_embedding=True)
                    assert torch.isfinite(z).all()
                save(root/f'train_gpu_smoke_rank{rank}.json',{'status':'passed','optimizer_updates':0,
                    'source_sampling':'coverage','OC_inputs':120,'extra_BPL_inputs':12,
                    'frontend_revision':self.args['frontend_resolved_revision']})
                return
            super().run()
            assert not torch.equal(initial_center,unwrap(self.classifier).loss.weight)
            assert self.args['epoch']==50
            flush(self.logger)
            save(root/f'train_done_rank{rank}.json',{'status':'complete','epoch':50,
                'logical_steps':5400,'backend_updates':5400,'center_updates':5400,
                'OC_center_changed':True,'frontend_frozen':not any(p.requires_grad for p in self.plm.parameters()),
                'frontend_revision':self.args['frontend_resolved_revision']})
    train.ModelTrainer=AuditedTrainer
    args,_,experiment=arguments.get_args()
    args.update(bpl_pairing='matched_independent',matched_source_sampling='coverage',
        duration_policy_path=str(root/'duration_policy.json'),activity_policy_path=str(root/'activity_policy.json'),
        epoch=50,world_size=2,gpu_ids=['0','1'],batch_size=120,
        num_workers=0 if mode=='smoke' else 4,intermediate_evaluation_phase='19LA',
        port=str(config['smoke_port'] if mode=='smoke' else config['train_port']),
        name=config['experiment_name']+('_gpu_smoke' if mode=='smoke' else ''),
        path_log=str(root/'smoke_results') if mode=='smoke' else '/results')
    experiment.update({k:args[k] for k in ['bpl_pairing','matched_source_sampling','duration_policy_path',
        'activity_policy_path','epoch','batch_size','evaluation_mode','evaluation_seed']})
    experiment.update(global_OC_batch_size=240,batch_size_per_rank=120)
    main.run(rank,args,experiment)
    dist.destroy_process_group()


if __name__=='__main__':
    import torch
    assert Path('/.dockerenv').is_file() and os.environ['CUDA_VISIBLE_DEVICES']=='0,1'
    config=json.loads((ROOT/'experiment.json').read_text())
    assert not (ROOT/('smoke_results' if '--smoke' in sys.argv else 'train.claim')).exists()
    if '--smoke' not in sys.argv:(ROOT/'train.claim').mkdir()
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.multiprocessing.spawn(worker,args=(str(ROOT),'smoke' if '--smoke' in sys.argv else 'train'),nprocs=2,join=True)
