"""Isolated two-GPU OC+BPL smoke; two logical batches per matched condition."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys


def worker(rank,root_s,pairing):
    root=Path(root_s);sys.path.insert(0,str(root/'code'))
    import torch
    import torch.distributed as dist
    import main
    import arguments
    from native_steps import unwrap
    from types import SimpleNamespace
    original_db=main.ASVspoof2019_LA_Train
    original_trainer=main.train.ModelTrainer
    original_batch=main.train.train_batch
    stats=[];oc_hashes=[]

    def small_database(path):
        db=original_db(path)
        db.dev_set=[x for x in db.dev_set if x.label==1][:16]+[x for x in db.dev_set if x.label==0][:16]
        return db
    main.ASVspoof2019_LA_Train=small_database
    main.ASVspoof2019_LA_Eval=lambda path:SimpleNamespace(trials=[])
    main.ASVspoof2021_LA_Eval=lambda path:SimpleNamespace(trials=[])
    main.ASVspoof2021_DF_Eval=lambda path:SimpleNamespace(trials=[])

    def checked_batch(trainer,batch):
        assert batch['labels'].tolist()==[1]*12+[0]*108
        assert len(batch['long'])==120 and len(batch['bpl_right'])==12
        assert all(len(w) in range(8000,64001,8000) for w in batch['long']+batch['bpl_right'])
        if pairing=='matched_same_source':
            assert torch.equal(batch['indices'][:12],batch['bpl_right_indices'])
        else:
            assert not torch.equal(batch['indices'][:12],batch['bpl_right_indices'])
        h=hashlib.sha256()
        h.update(batch['indices'].numpy().tobytes())
        for wave in batch['long']:h.update(wave.numpy().tobytes())
        oc_hashes.append(h.hexdigest())
        result=original_batch(trainer,batch)
        assert result['oc_long']>0 and result['bpl']>0 and result['loss2']>0
        assert result['positive_pairs']==144 and result['optimizer_steps']==2 and result['centroid_updates']==1
        stats.append(result)
        return result
    main.train.train_batch=checked_batch

    def backend_sha(model):
        h=hashlib.sha256()
        for k,v in model.state_dict().items():
            if k!='loss.weight':h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
        return h.hexdigest()

    class Limited:
        def __init__(self,base):self.base=base;self.dataset=base.dataset
        def __len__(self):return 2
        def __iter__(self):return itertools.islice(iter(self.base),2)

    class CheckedTrainer(original_trainer):
        def test(self,phase,epoch):
            model=unwrap(self.classifier)
            before={k:v.detach().clone() for k,v in model.state_dict().items()}
            rng=torch.get_rng_state().clone(),torch.cuda.get_rng_state(rank).clone()
            result=super().test(phase,epoch)
            assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
            assert torch.equal(rng[0],torch.get_rng_state()) and torch.equal(rng[1],torch.cuda.get_rng_state(rank))
            assert not hasattr(self,'score_reference')
            if pairing=='checkpoint_oc':
                assert not self.optimizer.state and all(p.grad is None for p in model.parameters())
                prior=json.loads((root/'matched_independent_gpu_check.json').read_text())
                assert backend_sha(model)==prior['final_backend_sha256']
                previous=torch.tensor([r[1] for r in prior['evaluation_rows']],dtype=torch.float64)
                current=torch.tensor([r[1] for r in self.last_eval_rows],dtype=torch.float64)
                delta=float((previous-current).abs().max())
                if rank==0:
                    (root/'checkpoint_score_diagnostic.json').write_text(json.dumps({
                        'max_absolute_difference':delta,'previous':previous.tolist(),
                        'current':current.tolist()},indent=2))
                torch.testing.assert_close(previous,current,atol=1e-6,rtol=0)
                assert self.calculate_EER(previous.tolist(),[r[2] for r in prior['evaluation_rows']])==result[0]
                assert prior['evaluation_protocol']==self.evaluation_protocol
                if rank==0:
                    (root/'checkpoint_oc_gpu_check.json').write_text(json.dumps({
                        'status':'passed','optimizer_updates':0,'trained_OC_weight_unchanged':True,
                        'scores_equivalent_atol_1e6':True,'maximum_score_difference':delta,
                        'EER_equal':True,'rows':len(self.last_eval_rows)},indent=2))
            return result

        def run(self):
            self.train_loader=Limited(self.train_loader)
            model=unwrap(self.classifier);initial=backend_sha(model);center=model.loss.weight.detach().clone()
            calls=[];original_step=self.optimizer.step
            def count_step(*args,**kwargs):calls.append(1);return original_step(*args,**kwargs)
            self.optimizer.step=count_step
            super().run()
            assert len(calls)==4 and len(stats)==2
            assert model.loss.weight in self.optimizer.state
            assert set(float(v['step']) for v in self.optimizer.state.values())=={2.}
            assert not torch.equal(center,model.loss.weight) and initial!=backend_sha(model)
            assert not any(p.requires_grad for p in self.plm.parameters())
            assert len(self.last_eval_rows)==32
            first=self.last_eval_rows.copy();self.test('V',999);assert first==self.last_eval_rows
            gathered=[None,None]
            dist.all_gather_object(gathered,(backend_sha(model),model.loss.weight.detach().cpu().tolist()))
            assert gathered[0]==gathered[1]
            if rank==0:
                record={'status':'passed','pairing':pairing,'world_size':2,'logical_batches':2,
                    'optimizer_step_calls':len(calls),'backend_updates':2,'OC_centroid_updates':2,
                    'OC_centroid_optimizer_state_present':True,'OC_spoof_inputs_per_batch':108,
                    'OC_bona_inputs_per_batch':12,'BPL_pairs_per_GPU':144,'extra_BPL_bona_inputs':12,
                    'OC_and_BPL_variable_lengths_valid':True,'OC_input_hashes':oc_hashes,
                    'initial_backend_sha256':initial,'final_backend_sha256':backend_sha(model),
                    'DDP_backend_and_trained_OC_center_equal':True,'evaluation_repeat_exact':True,
                    'evaluation_preserves_model_and_rng':True,'evaluation_rows':self.last_eval_rows,
                    'evaluation_protocol':self.evaluation_protocol,
                    'policy_sha256':self.args['duration_policy_sha256'],
                    'source_fingerprint':self.args['bpl_source_fingerprint']}
                (root/f'{pairing}_gpu_check.json').write_text(json.dumps(record,indent=2))

    main.train.ModelTrainer=CheckedTrainer
    args,system,experiment=arguments.get_args()
    args.update(bpl_pairing='legacy_long_short' if pairing=='checkpoint_oc' else pairing,
                duration_policy_path=str(root/'policy.json'),epoch=1,world_size=2,batch_size=120,num_workers=2,
                port={'matched_independent':'29671','matched_same_source':'29672','checkpoint_oc':'29673'}[pairing],
                path_log=str(root/'smoke_results'),name='CODE_CHECK_'+pairing)
    if pairing=='checkpoint_oc':
        args['evaluate_checkpoint']=str(root/'smoke_results/CM/LA/CODE_CHECK_matched_independent/model/CLS_1.pt')
    experiment.update(bpl_pairing=args['bpl_pairing'],epoch=1)
    main.run(rank,args,experiment)
    dist.destroy_process_group()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root');args=parser.parse_args()
    assert Path('/.dockerenv').is_file() and os.environ.get('CUDA_VISIBLE_DEVICES')=='0,1'
    import torch.multiprocessing as mp
    for pairing in ['matched_independent','matched_same_source','checkpoint_oc']:
        mp.spawn(worker,args=(args.root,pairing),nprocs=2,join=True)
    root=Path(args.root)
    a=json.loads((root/'matched_independent_gpu_check.json').read_text())
    b=json.loads((root/'matched_same_source_gpu_check.json').read_text())
    for key in ['initial_backend_sha256','policy_sha256','source_fingerprint','evaluation_protocol','OC_input_hashes']:
        assert a[key]==b[key],key
    assert json.loads((root/'checkpoint_oc_gpu_check.json').read_text())['status']=='passed'
    (root/'gpu_checks_complete.json').write_text(json.dumps({'status':'passed','conditions':[a,b]},indent=2))
