"""Fixed50 legacy/coverage paired evaluation under controlled_oc_v1.

One frozen frontend pass per batch is shared, with identical pooling RNG for
both heads. No optimizer is constructed, and hidden states are never saved.
"""
import csv,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(sys.argv[1]);sys.path.insert(0,str(ROOT/'code'))
from tools.evaluate_B_C_checkpoints import make_items,Waves,collate,metrics,EXPECTED,PHASES
from tools.train_coverage_experiment import save


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker(rank,root_string,smoke):
    import torch
    from transformers import Wav2Vec2Model
    import arguments,model,native_steps,data_loaders
    from positive_bpl import evaluation_rng,oc_evaluation_protocol
    root=Path(root_string);destination=root/('evaluation_smoke' if smoke else 'evaluation')
    config=json.loads((root/'experiment.json').read_text())
    torch.set_num_threads(2);torch.cuda.set_device(rank)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    args,_,_=arguments.get_args();args.update(cap_long=True,rank=rank,world_size=2,device=f'cuda:{rank}')
    paths={'legacy240':config['legacy_checkpoint'],
        'coverage':config['smoke_checkpoint'] if smoke else config['result_path']+'/model/CLS_50.pt'}
    assert sha(paths['legacy240'])==config['legacy_checkpoint_sha256']
    if smoke:assert sha(paths['coverage'])==config['smoke_checkpoint_sha256']
    if not smoke:
        proof=json.loads((root/'final_training_audit.json').read_text())
        assert proof['status']=='passed' and sha(paths['coverage'])==proof['CLS50_sha256']
    plm=Wav2Vec2Model.from_pretrained(args['PLM_name'],local_files_only=True).to(args['device']).eval().requires_grad_(False)
    revision=getattr(plm.config,'_commit_hash',None)
    assert revision==config['frontend_revision']
    heads={}
    for name,path in paths.items():
        obj=model.B_Linear(args['num_layers'],args['hidden_size'],args['output_size'],args['agg_size'],
            loss=model.OCSoftmax(args['output_size'],r_real=args['r_real'],r_fake=args['r_fake'],
                alpha=args['alpha'],class_weight=[5/9,5]),loss_bpl=model.BPL(args['alpha_cl'])).to(args['device']).eval()
        state=torch.load(path,map_location='cpu',weights_only=True)
        obj.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True)
        assert all(m.noise_std==1e-5 for m in obj.modules() if hasattr(m,'noise_std'))
        heads[name]=obj
    protocol=oc_evaluation_protocol(args,revision)
    protocol.update(checkpoints={name:{'path':path,'sha256':sha(path)} for name,path in paths.items()},
        input='native, front64600, no repeat, masks',rank=rank,
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,
        optimizer_updates=0,frozen_frontend_shared_between_heads=True,
        paired_pooling_noise=True,actual_torch_seed=1729+525252+rank)
    save(destination/f'protocol_rank{rank}.json',protocol)
    fields=['index','uid','speaker','label','original_samples','used_samples','attack','score_legacy240','score_coverage']
    for phase in PHASES:
        items=make_items(args,phase);dataset=Waves(items)
        indices=list(range(rank,len(items),2))
        if smoke:
            indices=sorted([i for y in [1,0] for i in [j for j in indices if items[j].label==y][:16]])
        loader=torch.utils.data.DataLoader(dataset,batch_size=8,sampler=indices,num_workers=4,
            collate_fn=collate,pin_memory=False,generator=torch.Generator().manual_seed(1+rank))
        output=destination/f'scores_{phase}_rank{rank}.csv';partial=output.with_suffix('.partial.csv')
        assert not output.exists() and not partial.exists()
        n=0;started=time.monotonic();trial_hash=hashlib.sha256()
        with partial.open('x',newline='') as stream,torch.no_grad(),evaluation_rng(args,525252):
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            for batch_number,(waves,metadata) in enumerate(loader):
                if batch_number==0:
                    original=data_loaders.EnrollmentSet(args,items)
                    for wave,row in zip(waves,metadata):
                        expected=original[row['index']][0]
                        assert np.array_equal(wave.numpy(),np.asarray(expected,dtype='float32'))
                features,mask=native_steps.features(plm,waves,args['device'])
                before=torch.cuda.get_rng_state(rank);after=None;outputs={}
                for name,head in heads.items():
                    torch.cuda.set_rng_state(before,rank)
                    scores=head(features,mask=mask)[1].flatten()
                    assert torch.isfinite(scores).all()
                    outputs[name]=scores.cpu().tolist()
                    current=torch.cuda.get_rng_state(rank)
                    if after is None:after=current
                    else:assert torch.equal(after,current),'Paired heads consumed different pooling RNG'
                if batch_number==0:
                    # Compare ordinary independent frontend+head evaluation to
                    # this shared-frontend path using identical stochastic state.
                    for name,head in heads.items():
                        torch.cuda.set_rng_state(before,rank)
                        own_features,own_mask=native_steps.features(plm,waves,args['device'])
                        independent=head(own_features,mask=own_mask)[1].flatten()
                        expected=torch.tensor(outputs[name],device=args['device'])
                        assert float((independent-expected).abs().max())<1e-6
                        del own_features,own_mask,independent
                    torch.cuda.set_rng_state(after,rank)
                for j,row in enumerate(metadata):
                    row.update(score_legacy240=outputs['legacy240'][j],score_coverage=outputs['coverage'][j])
                    trial_hash.update(f"{row['index']}|{row['uid']}|{row['label']}\n".encode())
                    writer.writerow(row);n+=1
                del features,mask,outputs,scores
                if batch_number%50==0:
                    stream.flush()
                    save(destination/f'progress_rank{rank}.json',{'phase':phase,'rows':n,'rank_rows':len(indices),
                        'elapsed_seconds':time.monotonic()-started,'updated_unix':time.time()})
            stream.flush();os.fsync(stream.fileno())
        assert n==len(indices);partial.replace(output)
        save(destination/f'done_{phase}_rank{rank}.json',{'rows':n,'sha256':sha(output),
            'trial_sha256':trial_hash.hexdigest(),'seconds':time.monotonic()-started})
        if smoke:break
    for name,path in paths.items():assert sha(path)==protocol['checkpoints'][name]['sha256']


def summarize(root):
    rows=[];thresholds={};by_length=[]
    for phase in PHASES:
        frames=[]
        for rank in range(2):
            path=root/'evaluation'/f'scores_{phase}_rank{rank}.csv'
            done=json.loads((path.parent/f'done_{phase}_rank{rank}.json').read_text())
            assert sha(path)==done['sha256'];frames.append(pd.read_csv(path))
        frame=pd.concat(frames,ignore_index=True).sort_values('index').reset_index(drop=True)
        assert frame['index'].tolist()==list(range(EXPECTED[phase])) and frame.uid.is_unique
        assert not frame.isna().any().any()
        assert np.array_equal(frame.used_samples,np.minimum(frame.original_samples,64600))
        frame.to_parquet(root/'evaluation'/f'paired_scores_{phase}.parquet',index=False)
        frame['length_bin']=pd.cut(frame.original_samples/16000,[0,1,2,3,4,6,np.inf],right=False)
        for name in ['legacy240','coverage']:
            column='score_'+name;m=metrics(frame,column)
            if phase=='dev':thresholds[name]=m['discrete_EER_threshold']
            threshold=thresholds[name];accept=frame[column].to_numpy()>=threshold;y=frame.label.to_numpy()
            rows.append(dict(phase=phase,model=name,**m,dev_threshold=threshold,
                bona_reject_percent=100*float((~accept[y==1]).mean()),
                spoof_accept_percent=100*float(accept[y==0].mean())))
            for label in [1,0]:
                for length,part in frame[frame.label==label].groupby('length_bin',observed=True):
                    errors=part[column]<threshold if label else part[column]>=threshold
                    by_length.append({'phase':phase,'model':name,'label':label,'length_bin':str(length),
                        'n':len(part),'errors':int(errors.sum()),'error_percent':100*float(errors.mean())})
    table=pd.DataFrame(rows);table.to_csv(root/'comparison.csv',index=False)
    pd.DataFrame(by_length).to_csv(root/'comparison_by_length.csv',index=False)
    changes=[]
    for phase in PHASES:
        values=table[table.phase==phase].set_index('model')
        changes.append({'phase':phase,'legacy_EER_percent':float(values.loc['legacy240','EER_percent']),
            'coverage_EER_percent':float(values.loc['coverage','EER_percent']),
            'EER_change_percentage_points':float(values.loc['coverage','EER_percent']-values.loc['legacy240','EER_percent'])})
    save(root/'comparison.json',{'status':'complete','fixed_epoch':50,'metrics':rows,'EER_changes':changes,
        'phase_trial_counts':EXPECTED,'evaluation_protocol':'controlled_oc_v1',
        'target_used_for_checkpoint_or_policy_selection':False,'optimizer_updates_during_evaluation':0,
        'limitations':['Multiple training input/pairing policies change together.',
            'Legacy240 historical codec execution was not established.',
            'Single training seed; no causal or statistical superiority claim from one run.']})
    save(root/'comparison_complete.json',{'status':'complete','finished_unix':time.time()})


if __name__=='__main__':
    import torch.multiprocessing as mp
    smoke='--smoke' in sys.argv
    assert os.environ['CUDA_VISIBLE_DEVICES']=='0,1' and Path('/.dockerenv').is_file()
    (ROOT/('evaluation_smoke' if smoke else 'evaluation')).mkdir(exist_ok=False)
    mp.spawn(worker,args=(str(ROOT),smoke),nprocs=2,join=True)
    if smoke:save(ROOT/'eval_smoke_complete.json',{'status':'passed','optimizer_updates':0,'trials':64})
    else:summarize(ROOT)
