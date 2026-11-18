"""Evaluate every saved B/C checkpoint on common B inputs with paired RNG.

Standalone analysis only: never imports a training logger or edits training code.
SSL hidden states are transient and never saved. Only paired derived scores persist.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

CODE=Path('/home/mentee/code/EBM')
RESULTS={
 'B':Path('/home/mentee/results/CM/LA/EBM_e50_bs240_lr1e5_B_zeropad_short303'),
 'C':Path('/home/mentee/results/CM/LA/EBM_e50_bs240_lr1e5_C_nativeLong_short303')}
OLD_RESULT=Path('/home/mentee/results/CM/LA/EBM_e50_bs240_lr1e5')
TRAIN_ROOT=Path('/home/mentee/analysis/ebm_train_B_20261106_213627')
EPOCHS=[f'{condition}_e{epoch}' for condition in ['B','C'] for epoch in range(5,51,5)]
PHASES=['dev','19LA','21LA','21DF']
EXPECTED={'dev':24844,'19LA':71237,'21LA':148176,'21DF':533928}
FIELDS=['index','uid','speaker','label','original_samples','used_samples','attack']+[f'score_{e}' for e in EPOCHS]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,value):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)

def make_items(args,phase):
    from dataset import ASVspoof2019_LA_Train,ASVspoof2019_LA_Eval,ASVspoof2021_LA_Eval,ASVspoof2021_DF_Eval
    if phase=='dev':items=ASVspoof2019_LA_Train(args['path_19LA']).dev_set
    elif phase=='19LA':items=ASVspoof2019_LA_Eval(args['path_19LA']).trials
    elif phase=='21LA':items=ASVspoof2021_LA_Eval(args['path_21LA']).trials
    else:items=ASVspoof2021_DF_Eval(args['path_21DF']).trials
    assert len(items)==EXPECTED[phase],(phase,len(items))
    if phase in ['21LA','21DF']:
        base=args['path_21LA'] if phase=='21LA' else args['path_21DF']
        key=Path(base)/('keys/LA/CM/trial_metadata.txt' if phase=='21LA' else 'keys/DF/CM/trial_metadata.txt')
        mapping={}
        for line in key.read_text().splitlines():
            tokens=line.split()
            mapping[tokens[1]]=tokens[0]
        for item in items:item.speaker=mapping[Path(item.path).stem]
    return items

class Waves:
    def __init__(self,items):self.items=items
    def __len__(self):return len(self.items)
    def __getitem__(self,i):
        import soundfile as sf
        item=self.items[i];wave,sr=sf.read(item.path)
        assert sr==16000 and wave.ndim==1 and len(wave)>=720,(i,sr,wave.shape)
        original=len(wave);wave=wave[:64600].astype('float32')
        row={'index':i,'uid':Path(item.path).stem,'speaker':getattr(item,'speaker','unknown'),
             'label':int(item.label),'original_samples':original,'used_samples':len(wave),
             'attack':getattr(item,'type','unknown')}
        return wave,row

def collate(rows):
    import torch
    return [torch.from_numpy(a) for a,_ in rows],[m for _,m in rows]

def metrics(frame,column):
    from sklearn.metrics import roc_curve
    y=frame.label.to_numpy();s=frame[column].to_numpy()
    assert np.isfinite(s).all() and set(y)=={0,1}
    fpr,tpr,thresholds=roc_curve(y,s,pos_label=1,drop_intermediate=False)
    difference=fpr-(1-tpr);k=int(np.flatnonzero(difference>=0)[0])
    if k==0:eer=float(fpr[0])
    else:
        fraction=-difference[k-1]/(difference[k]-difference[k-1])
        eer=float(fpr[k-1]+fraction*(fpr[k]-fpr[k-1]))
    # Actual attainable dev operating point; fixed rule, never selected on targets.
    j=min(range(len(fpr)),key=lambda i:(abs(difference[i]),fpr[i],-thresholds[i]))
    return {'EER_percent':100*eer,'discrete_EER_threshold':float(thresholds[j]),
            'threshold_rule':'min abs(FPR-FNR), then lower FPR, then higher threshold',
            'n':len(y),'bonafide':int(y.sum()),'spoof':int((y==0).sum())}

def worker(rank,root_s,smoke):
    import torch
    import soundfile as sf
    from transformers import Wav2Vec2Model
    root=Path(root_s);sys.path.insert(0,str(CODE))
    import arguments,model,data_loaders,native_steps
    torch.set_num_threads(2);torch.cuda.set_device(rank);device=f'cuda:{rank}'
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(20261107+rank)
    args,_,_=arguments.get_args()
    assert args['native_length'] and args['short_oc_weight']==args['padding_loss_weight']==0
    args=dict(args,cap_long=True)  # Evaluation remains B-capped under running C source.
    assert (TRAIN_ROOT/'run.exit').read_text().strip()=='0'
    plm=Wav2Vec2Model.from_pretrained(args['PLM_name'],local_files_only=True).to(device).eval().requires_grad_(False)
    def classifier(checkpoint):
        obj=model.B_Linear(args['num_layers'],args['hidden_size'],args['output_size'],args['agg_size'],
            loss=model.OCSoftmax(args['output_size'],r_real=args['r_real'],r_fake=args['r_fake'],alpha=args['alpha'],class_weight=[5/9,5]),
            loss_bpl=model.BPL(args['alpha_cl'])).to(device).eval()
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        state={k.removeprefix('module.'):v for k,v in state.items()}
        obj.load_state_dict(state,strict=True)
        return obj,state
    provenance=json.loads((root/'provenance.json').read_text())
    heads={};validation={'checkpoints':{},'frontend_revision':getattr(plm.config,'_commit_hash',None)}
    assert validation['frontend_revision']=='1a640f32ac3e39899438a2931f9924c02f080a54'
    for epoch in EPOCHS:
        condition,saved_epoch=epoch.split('_e');saved_epoch=int(saved_epoch)
        checkpoint=RESULTS[condition]/f'model/CLS_{saved_epoch}.pt'
        assert sha(checkpoint)==provenance['checkpoints'][str(epoch)]['sha256']
        heads[epoch],state_head=classifier(checkpoint)
        state_path=RESULTS[condition]/f'model/TrainState_{saved_epoch}.pt'
        assert sha(state_path)==provenance['checkpoints'][epoch]['TrainState_sha256']
        state=torch.load(state_path,map_location='cpu',weights_only=True)
        assert state['epoch']==saved_epoch and state['args']['batch_size']==120
        assert state['args']['native_length'] and state['args']['cap_long']==(condition=='B')
        assert state['args']['short_oc_weight']==state['args']['padding_loss_weight']==0
        assert all(torch.equal(v,state['classifier']['module.'+k]) for k,v in state_head.items())
        steps=[int(v['step']) for v in state['optimizer']['state'].values()]
        assert min(steps)==max(steps)==saved_epoch*108
        validation['checkpoints'][str(epoch)]={'strict_load':True,'matches_training_state':True,'Adam_steps':saved_epoch*108}
        del state,state_head

    def score(obj,x,mask):
        z=obj(x,mask=mask,return_embedding=True)
        return (torch.nn.functional.normalize(z,dim=1)@torch.nn.functional.normalize(obj.loss.weight,dim=1).T).flatten()

    # QC uses noiseless pooling only for numerical invariance; full evaluation
    # restores the implementation's default noise_std=1e-5 even in eval mode.
    probe=pd.read_parquet('/home/mentee/analysis/embedding_geometry/20261106_variable_length_01/manifest.parquet').set_index('selection')
    probe_items=[SimpleNamespace(path=probe.loc[k].path,label=int(probe.loc[k].y),type=probe.loc[k].attack_id)
                 for k in ['train_y0_minimum','train_y1_minimum','DF_remaining_maximum']]
    dataset=Waves(probe_items)
    original=data_loaders.EnrollmentSet(args,probe_items)
    for i in range(3):assert np.array_equal(dataset[i][0],np.asarray(original[i][0],dtype='float32'))
    waves,metadata=collate([dataset[i] for i in range(3)])
    for obj in heads.values():
        for m in obj.modules():
            if hasattr(m,'noise_std'):m.noise_std=0.
    with torch.no_grad():
        f,mask=native_steps.features(plm,waves,device)
        error=0.;native_error=0.
        for obj in heads.values():
            together=score(obj,f,mask)
            _,forward_score=obj(f,mask=mask)
            native_error=max(native_error,float((together-forward_score.flatten()).abs().max()))
            single=[]
            for w in waves:
                solo,sm=native_steps.features(plm,[w],device);single.append(score(obj,solo,sm));del solo,sm
            error=max(error,float((together-torch.cat(single)).abs().max()))
        assert error<1e-4 and native_error<1e-6,(error,native_error)
    for obj in heads.values():
        for m in obj.modules():
            if hasattr(m,'noise_std'):m.noise_std=1e-5
    validation.update(batch_single_score_error=error,builtin_forward_score_error=native_error,
                      input_matches_source_EnrollmentSet=True,full_evaluation_pooling_noise_std=1e-5)
    save(root/f'QC_rank{rank}_{"smoke" if smoke else "full"}.json',validation)
    del f,mask,waves
    for phase_number,phase in enumerate(PHASES):
        destination=root/f'scores_{phase}_rank{rank}{"_smoke" if smoke else ""}.csv'
        assert not destination.exists(),'Do not overwrite completed scores'
        items=make_items(args,phase)
        dataset=Waves(items)
        indices=list(range(rank,len(dataset),2))
        if smoke:indices=indices[:24]
        loader=torch.utils.data.DataLoader(dataset,batch_size=8,sampler=indices,num_workers=4,collate_fn=collate,pin_memory=False)
        torch.manual_seed(20261107+rank+1000*phase_number)
        partial=destination.with_suffix('.partial.csv');n=0;start=time.time()
        with partial.open('x',newline='') as stream,torch.no_grad():
            writer=csv.DictWriter(stream,fieldnames=FIELDS);writer.writeheader()
            for batch_number,(waves,metadata) in enumerate(loader):
                f,mask=native_steps.features(plm,waves,device)
                rng=torch.cuda.get_rng_state(rank)
                scores={};after=None
                for epoch,obj in heads.items():
                    torch.cuda.set_rng_state(rng,rank)
                    value=score(obj,f,mask)
                    assert torch.isfinite(value).all()
                    current=torch.cuda.get_rng_state(rank)
                    if after is None:after=current
                    else:assert torch.equal(after,current),'Checkpoint pooling RNG mismatch'
                    scores[epoch]=value.cpu().tolist()
                    del value
                for j,row in enumerate(metadata):
                    row.update({f'score_{epoch}':values[j] for epoch,values in scores.items()})
                    writer.writerow(row);n+=1
                del f,mask,scores
                if batch_number%50==0:
                    stream.flush()
                    save(root/f'progress_rank{rank}{"_smoke" if smoke else ""}.json',
                        {'phase':phase,'rows':n,'expected_rank_rows':len(indices),'elapsed_seconds':time.time()-start,'updated_unix':time.time(),'smoke':smoke})
            stream.flush();os.fsync(stream.fileno())
        assert n==len(indices)
        partial.replace(destination)
        save(root/f'done_{phase}_rank{rank}{"_smoke" if smoke else ""}.json',
            {'phase':phase,'rows':n,'sha256':sha(destination),'seconds':time.time()-start,'smoke':smoke})
        if smoke:break

def summarize(root):
    all_metrics={}
    dev_thresholds={}
    for phase in PHASES:
        parts=[]
        for rank in range(2):
            path=root/f'scores_{phase}_rank{rank}.csv'
            info=json.loads((root/f'done_{phase}_rank{rank}.json').read_text())
            assert info['sha256']==sha(path)
            parts.append(pd.read_csv(path,dtype={'uid':str,'speaker':str,'attack':str}))
        frame=pd.concat(parts,ignore_index=True).sort_values('index').reset_index(drop=True)
        assert frame['index'].tolist()==list(range(EXPECTED[phase]))
        assert frame.uid.is_unique and not frame.isna().any().any()
        assert (frame.used_samples==np.minimum(frame.original_samples,64600)).all()
        phase_metrics={}
        for epoch in EPOCHS:
            label=str(epoch);column=f'score_{epoch}'
            m=metrics(frame,column)
            if phase=='dev':dev_thresholds[label]=m['discrete_EER_threshold']
            threshold=dev_thresholds[label];accept=frame[column].to_numpy()>=threshold;y=frame.label.to_numpy()
            m.update(dev_threshold=threshold,normal_reject_percent=float(100*(~accept[y==1]).mean()),
                     spoof_accept_percent=float(100*accept[y==0].mean()))
            phase_metrics[label]=m
        frame.to_parquet(root/f'checkpoint_scores_{phase}.parquet',index=False)
        all_metrics[phase]=phase_metrics
    save(root/'metrics.json',all_metrics)
    save(root/'complete.json',{'status':'evaluation_complete_analysis_pending','phase_counts':EXPECTED,
         'protocol':'B/C epochs5,10,15,20,25,30,35,40,45,50; common B cap/masks/batch8; pooling noise1e-5; common RNG',
         'no_target_checkpoint_selection':True,'thresholds_selected_on_dev_only':True})

def main():
    assert Path('/.dockerenv').is_file(),'GPU evaluation must run in the existing container'
    assert Path('/home/mentee/analysis/ebm_train_C_20261108_061305/run.exit').read_text().strip()=='0','C must finish successfully first'
    import torch.multiprocessing as mp
    parser=argparse.ArgumentParser();parser.add_argument('root');parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();root=Path(args.root)
    assert os.environ['CUDA_VISIBLE_DEVICES']=='0,1'
    mp.spawn(worker,args=(str(root),args.smoke),nprocs=2,join=True)
    if args.smoke:
        reference=pd.read_parquet('/home/mentee/analysis/ebm_B_final_eval_20261107_215451/paired_scores_dev.parquet').set_index('uid')
        errors=[]
        for rank in range(2):
            probe=pd.read_csv(root/f'scores_dev_rank{rank}_smoke.csv')
            expected=reference.loc[probe.uid,'score_B'].to_numpy()
            error=float(np.max(np.abs(probe.score_B_e50.to_numpy()-expected)))
            assert error<1e-5,('Epoch50 historical score mismatch',rank,error)
            errors.append(error)
        save(root/'smoke_complete.json',{'status':'passed','rank_rows':24,'historical_epoch50_max_errors':errors})
    else:summarize(root)

if __name__=='__main__':main()
