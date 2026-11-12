"""Locked metadata-only selection, then bounded fixed-model input intervention."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

SEED='EBM-B-mechanism-20261107-v1'
TARGETS=['19LA','21LA','21DF']
LABELS=['under2','2to3.0275','3.0275to4.0375','atleast4.0375']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n')

def lock(evalroot):
    assert (evalroot/'complete.json').exists() and (evalroot/'run.exit').read_text().strip()=='0'
    root=evalroot/'mechanism_2x2'
    root.mkdir(exist_ok=False)
    rows=[];inputs={}
    for phase in TARGETS:
        path=evalroot/f'paired_scores_{phase}.parquet'
        inputs[phase]=sha(path)
        # Intentionally never read scores or select using outcome/embedding.
        f=pd.read_parquet(path,columns=['index','uid','speaker','label','original_samples'])
        f['phase']=phase
        f['length_bin']=pd.cut(f.original_samples,[0,32000,48440,64600,np.inf],right=False,labels=LABELS).astype(str)
        f['priority']=[hashlib.sha256((SEED+'|'+phase+'|'+uid).encode()).hexdigest() for uid in f.uid]
        for _,g in f.groupby(['label','length_bin'],sort=True):
            g=g.sort_values('priority').copy()
            g['within_speaker_order']=g.groupby('speaker').cumcount()
            rows.append(g.sort_values(['within_speaker_order','priority']).head(32))
    f=pd.concat(rows,ignore_index=True);f['selection_index']=np.arange(len(f))
    assert len(f)<=768 and not f[['phase','uid']].duplicated().any()
    path=root/'selected.parquet';f.to_parquet(path,index=False)
    protocol={'status':'locked_before_intervention','selection_seed':SEED,'selection_fields_only':['phase','uid','speaker','label','original_samples'],
        'selection':'phase x class x prespecified original length bin, max32 each, UID SHA256 and speaker round-robin',
        'N':len(f),'bins_samples':[0,32000,48440,64600],'quota':32,'selected_sha256':sha(path),
        'source_parquet_hashes':inputs,'noise_std':0.,'noise_scope':'diagnostic only; full benchmark retained1e-5',
        'models':['B','legacy240'],'inputs':['native','wrap'],'input_content':'same original first64600 samples; wrap only when shorter',
        'no_op_control':'original_samples>=64600 gives identical input in both policies',
        'limits':['balanced diagnostic subset, not target population proportions','input policy changes length and context too','learned latent axes differ across models','no new training']}
    save(root/'protocol.json',protocol)
    print(json.dumps({'N':len(f),'selected_sha256':protocol['selected_sha256'],'root':str(root)}),flush=True)

def worker(rank,evalroot_s):
    import torch
    import soundfile as sf
    from transformers import Wav2Vec2Model
    evalroot=Path(evalroot_s);root=evalroot/'mechanism_2x2'
    sys.path.insert(0,str(evalroot));import evaluate_B_final as ref
    sys.path.insert(0,str(ref.CODE));import arguments,model,native_steps
    protocol=json.loads((root/'protocol.json').read_text())
    assert sha(root/'selected.parquet')==protocol['selected_sha256']
    selected=pd.read_parquet(root/'selected.parquet')
    torch.set_num_threads(2);torch.cuda.set_device(rank);device=f'cuda:{rank}'
    torch.manual_seed(20261108+rank)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    args,_,_=arguments.get_args()
    maps={}
    for phase in TARGETS:
        items=ref.make_items(args,phase)
        maps[phase]={Path(v.path).stem:v for v in items}
    plm=Wav2Vec2Model.from_pretrained(args['PLM_name'],local_files_only=True).to(device).eval().requires_grad_(False)
    assert plm.config._commit_hash=='1a640f32ac3e39899438a2931f9924c02f080a54'
    models=[];centroids=[]
    audit=json.loads((evalroot/'training_audit.json').read_text())
    for key in ['B_CLS50','legacy240_CLS50']:
        item=audit['checkpoints'][key];assert sha(item['path'])==item['sha256']
        obj=model.B_Linear(args['num_layers'],args['hidden_size'],args['output_size'],args['agg_size'],
            loss=model.OCSoftmax(args['output_size'],r_real=args['r_real'],r_fake=args['r_fake'],alpha=args['alpha'],class_weight=[5/9,5]),
            loss_bpl=model.BPL(args['alpha_cl'])).to(device).eval()
        state=torch.load(item['path'],map_location='cpu',weights_only=True)
        obj.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True)
        for module in obj.modules():
            if hasattr(module,'noise_std'):module.noise_std=0.
        models.append(obj);centroids.append(obj.loss.weight.detach().cpu().numpy().reshape(-1))
    subset=selected.iloc[rank::2]
    z=np.empty((len(subset),2,2,160),dtype='float32')
    ids=subset.selection_index.to_numpy();start=time.time()
    with torch.no_grad():
        for n in range(0,len(subset),8):
            batch=subset.iloc[n:n+8];native=[];wrapped=[]
            for row in batch.itertuples():
                item=maps[row.phase][row.uid];assert item.label==row.label
                a,sr=sf.read(item.path);assert sr==16000 and a.ndim==1 and len(a)==row.original_samples
                a=a[:64600].astype('float32');assert len(a)>=720
                b=np.pad(a,(0,64600-len(a)),mode='wrap') if len(a)<64600 else a.copy()
                native.append(torch.from_numpy(a));wrapped.append(torch.from_numpy(b))
            for view,waves in enumerate([native,wrapped]):
                f,mask=native_steps.features(plm,waves,device)
                for m,obj in enumerate(models):
                    out=obj(f,mask=mask,return_embedding=True)
                    assert torch.isfinite(out).all()
                    z[n:n+len(batch),m,view]=out.cpu().numpy()
                del f,mask,out
            if n%32==0:save(root/f'progress_rank{rank}.json',{'rows':n+len(batch),'expected_rows':len(subset),'elapsed_seconds':time.time()-start})
    np.savez_compressed(root/f'embeddings_rank{rank}.npz',indices=ids,z=z,centroids=np.stack(centroids))
    save(root/f'rank{rank}_complete.json',{'rows':len(subset),'selected_sha256':protocol['selected_sha256'],'seconds':time.time()-start,'sha256':sha(root/f'embeddings_rank{rank}.npz')})

def collect(evalroot):
    root=evalroot/'mechanism_2x2';selected=pd.read_parquet(root/'selected.parquet')
    z=np.empty((len(selected),2,2,160),dtype='float32');indices=[];centroids=None
    for rank in range(2):
        p=root/f'embeddings_rank{rank}.npz';info=json.loads((root/f'rank{rank}_complete.json').read_text());assert info['sha256']==sha(p)
        d=np.load(p);i=d['indices'];z[i]=d['z'];indices.extend(i.tolist())
        if centroids is None:centroids=d['centroids']
        else:assert np.array_equal(centroids,d['centroids'])
    assert sorted(indices)==list(range(len(selected))) and np.isfinite(z).all()
    unit=z/np.linalg.norm(z,axis=-1,keepdims=True)
    c=centroids/np.linalg.norm(centroids,axis=-1,keepdims=True)
    scores=np.einsum('nmvd,md->nmv',unit,c)
    angles=np.degrees(np.arccos(np.clip(scores,-1,1)))
    long=selected.original_samples.to_numpy()>=64600
    no_op_z=float(np.abs(z[long,:,0]-z[long,:,1]).max())
    no_op_score=float(np.abs(scores[long,:,0]-scores[long,:,1]).max())
    assert no_op_z<1e-4 and no_op_score<1e-5,(no_op_z,no_op_score)
    np.savez_compressed(root/'paired_embeddings.npz',z=z,centroids=centroids,scores=scores,angles=angles)
    records=[]
    for row in selected.itertuples():
        i=row.selection_index
        for m,name in enumerate(['B','legacy240']):
            records.append({'selection_index':i,'phase':row.phase,'uid':row.uid,'speaker':row.speaker,'label':row.label,'length_bin':row.length_bin,
                'model':name,'score_native':float(scores[i,m,0]),'score_wrap':float(scores[i,m,1]),
                'delta_score_native_minus_wrap':float(scores[i,m,0]-scores[i,m,1]),'angle_native':float(angles[i,m,0]),'angle_wrap':float(angles[i,m,1])})
    pd.DataFrame(records).to_parquet(root/'intervention_scores.parquet',index=False)
    save(root/'complete.json',{'status':'intervention_complete_analysis_pending','N':len(selected),'paired_rows':len(records),
        'noise_std':0.,'models':['B','legacy240'],'views':['native','wrap'],'no_op_max_z_error':no_op_z,'no_op_max_score_error':no_op_score,
        'selected_sha256':sha(root/'selected.parquet'),'embeddings_sha256':sha(root/'paired_embeddings.npz')})

def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['lock','run']);parser.add_argument('evalroot');a=parser.parse_args()
    root=Path(a.evalroot)
    if a.mode=='lock':lock(root)
    else:
        import torch.multiprocessing as mp
        assert (root/'run.exit').read_text().strip()=='0'
        mp.spawn(worker,args=(str(root),),nprocs=2,join=True);collect(root)

if __name__=='__main__':main()
