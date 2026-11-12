"""Fixed final-checkpoint evaluation of both models with common B inputs.

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
B_RESULT=Path('/home/mentee/results/CM/LA/EBM_e50_bs240_lr1e5_B_zeropad_short303')
OLD_RESULT=Path('/home/mentee/results/CM/LA/EBM_e50_bs240_lr1e5')
TRAIN_ROOT=Path('/home/mentee/analysis/ebm_train_B_20261106_213627')
PHASES=['dev','19LA','21LA','21DF']
EXPECTED={'dev':24844,'19LA':71237,'21LA':148176,'21DF':533928}
FIELDS=['index','uid','speaker','label','original_samples','used_samples','attack','score_B','score_legacy240']

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
    assert args['native_length'] and args['cap_long'] and args['short_oc_weight']==args['padding_loss_weight']==0
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
    trained,state_B=classifier(B_RESULT/'model/CLS_50.pt')
    legacy,state_old=classifier(OLD_RESULT/'model/BestCLS_50_21DF.pt')
    assert state_B.keys()==state_old.keys()
    state=torch.load(B_RESULT/'model/TrainState_50.pt',map_location='cpu',weights_only=True)
    assert state['epoch']==50 and state['args']['batch_size']==120
    assert state['args']['native_length'] and state['args']['cap_long']
    assert state['args']['short_oc_weight']==state['args']['padding_loss_weight']==0
    assert all(torch.equal(v,state['classifier']['module.'+k]) for k,v in state_B.items())
    adam_steps=[int(v['step']) for v in state['optimizer']['state'].values()]
    assert min(adam_steps)==max(adam_steps)==5400,sorted(set(adam_steps))
    validation={'epoch':50,'classifier_matches_TrainState':True,'Adam_steps_min':min(adam_steps),
                'Adam_steps_max':max(adam_steps),'strict_both_checkpoints':True,
                'frontend_model':args['PLM_name'],'frontend_revision':getattr(plm.config,'_commit_hash',None),
                'frontend_feat_extract_norm':plm.config.feat_extract_norm}
    del state,state_B,state_old

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
    for obj in [trained,legacy]:
        for m in obj.modules():
            if hasattr(m,'noise_std'):m.noise_std=0.
    with torch.no_grad():
        f,mask=native_steps.features(plm,waves,device)
        error=0.;native_error=0.
        for obj in [trained,legacy]:
            together=score(obj,f,mask)
            _,forward_score=obj(f,mask=mask)
            native_error=max(native_error,float((together-forward_score.flatten()).abs().max()))
            single=[]
            for w in waves:
                solo,sm=native_steps.features(plm,[w],device);single.append(score(obj,solo,sm));del solo,sm
            error=max(error,float((together-torch.cat(single)).abs().max()))
        assert error<1e-4 and native_error<1e-6,(error,native_error)
    for obj in [trained,legacy]:
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
                sB=score(trained,f,mask)
                after=torch.cuda.get_rng_state(rank)
                torch.cuda.set_rng_state(rng,rank)
                sOld=score(legacy,f,mask)
                assert torch.equal(after,torch.cuda.get_rng_state(rank)),'Paired models must consume the same pooling RNG'
                assert torch.isfinite(sB).all() and torch.isfinite(sOld).all()
                for row,b,o in zip(metadata,sB.cpu().tolist(),sOld.cpu().tolist()):
                    row.update(score_B=b,score_legacy240=o);writer.writerow(row);n+=1
                del f,mask,sB,sOld
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
        for label,column in [('B','score_B'),('legacy240','score_legacy240')]:
            m=metrics(frame,column)
            if phase=='dev':dev_thresholds[label]=m['discrete_EER_threshold']
            threshold=dev_thresholds[label];accept=frame[column].to_numpy()>=threshold;y=frame.label.to_numpy()
            m.update(dev_threshold=threshold,normal_reject_percent=float(100*(~accept[y==1]).mean()),
                     spoof_accept_percent=float(100*accept[y==0].mean()))
            phase_metrics[label]=m
        frame.to_parquet(root/f'paired_scores_{phase}.parquet',index=False)
        all_metrics[phase]=phase_metrics
    save(root/'metrics.json',all_metrics)
    save(root/'complete.json',{'status':'evaluation_complete_analysis_pending','phase_counts':EXPECTED,
         'protocol':'both fixed50 checkpoints, same B cap/masks/batch8, default pooling noise1e-5 with common random numbers',
         'no_target_checkpoint_selection':True,'thresholds_selected_on_dev_only':True})

def main():
    import torch.multiprocessing as mp
    parser=argparse.ArgumentParser();parser.add_argument('root');parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();root=Path(args.root)
    assert os.environ['CUDA_VISIBLE_DEVICES']=='0,1'
    mp.spawn(worker,args=(str(root),args.smoke),nprocs=2,join=True)
    if args.smoke:save(root/'smoke_complete.json',{'status':'passed','rank_rows':24})
    else:summarize(root)

if __name__=='__main__':main()
