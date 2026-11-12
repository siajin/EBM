"""Locked frozen-model native/repeat and same-crop dose controls; container only."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd

CAP=64600
HEADS=['legacy240','B']
META=['phase','index','uid','speaker','label','original_samples','attack']
FIELDS=META+['condition','used_samples','wave_sha256','gpu_rank']+[f'score_{m}' for m in HEADS]

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
    p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)

def smooth(x,positions,radius=80):
    out=x.copy()
    for pos in positions:
        lo=max(0,int(pos)-radius);hi=min(len(x)-1,int(pos)+radius)
        out[lo:hi+1]=np.linspace(x[lo],x[hi],hi-lo+1,dtype=np.float32)
    return out

def repeat(x,n):
    assert n>=len(x)>720
    return np.resize(x,n).copy()

def dose_views(wave):
    assert len(wave)>CAP
    natural=wave[:CAP].copy()
    result=[('natural4',natural)]
    for n in [16000,32000]:
        for end in ['head','tail']:
            x=wave[:n].copy() if end=='head' else wave[-n:].copy()
            name=f'{end}_{n//16000}s_'
            wrap=repeat(x,CAP);seams=list(range(n,CAP,n));sham=list(range(n//2,CAP,n))
            result.extend([(name+'native',x),(name+'repeat2',repeat(x,32000)),
                           (name+'repeat3',repeat(x,48000)),(name+'repeat4',wrap),
                           (name+'smooth_seams4',smooth(wrap,seams)),
                           (name+'smooth_sham4',smooth(wrap,sham[:len(seams)])),
                           (name+'reflect4',np.pad(x,(0,CAP-n),mode='reflect'))])
        result.append((f'natural4_smooth_{n//16000}s_positions',smooth(natural,range(n,CAP,n))))
    assert len(result)==31 and len({name for name,_ in result})==31
    return result

def read_selection(root,probe,smoke):
    name='probe_selected.csv' if probe else 'selected.csv'
    p=json.loads((root/'protocol.json').read_text());assert sha(root/name)==p['selection_hashes'][name]
    f=pd.read_csv(root/name,dtype={'speaker':str,'uid':str,'attack':str})
    assert not f.duplicated(['phase','uid']).any() and not f[META].isna().any().any()
    if smoke:
        f=f.assign(short=f.original_samples<CAP).groupby(['phase','label','short'],sort=True).head(2)
    return f.reset_index(drop=True),p

def worker(rank,root_string,probe,smoke):
    import soundfile as sf
    import torch
    from transformers import Wav2Vec2Model
    root=Path(root_string);code=root/'code';sys.path.insert(0,str(code))
    import arguments,model,native_steps
    # Helper is the frozen locally authored evaluator; its dataset reader uses the frozen code above.
    sys.path.insert(0,str(root));from probe_common import make_items
    torch.set_num_threads(2);torch.cuda.set_device(rank);device=f'cuda:{rank}'
    torch.manual_seed(20261111+rank)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    frame,protocol=read_selection(root,probe,smoke);frame=frame.iloc[rank::2].reset_index(drop=True)
    tag=('probe' if probe else 'full')+('_smoke' if smoke else '')
    dest=root/f'{tag}_rank{rank}.csv';partial=dest.with_suffix('.partial.csv')
    assert not dest.exists() and not partial.exists(),'Refuse overwrite'
    args,_,_=arguments.get_args()
    plm=Wav2Vec2Model.from_pretrained(args['PLM_name'],local_files_only=True).to(device).eval().requires_grad_(False)
    assert getattr(plm.config,'_commit_hash',None)=='1a640f32ac3e39899438a2931f9924c02f080a54'
    heads={}
    for name in HEADS:
        source=protocol['checkpoints'][name];assert sha(source['path'])==source['sha256']
        obj=model.B_Linear(args['num_layers'],args['hidden_size'],args['output_size'],args['agg_size'],
            loss=model.OCSoftmax(args['output_size'],r_real=args['r_real'],r_fake=args['r_fake'],alpha=args['alpha'],class_weight=[5/9,5]),loss_bpl=model.BPL(args['alpha_cl'])).to(device).eval().requires_grad_(False)
        state=torch.load(source['path'],map_location='cpu',weights_only=True)
        obj.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True)
        for module in obj.modules():
            if hasattr(module,'noise_std'):module.noise_std=0.
        heads[name]=obj
    def infer(waves):
        x,mask=native_steps.features(plm,[torch.from_numpy(w) for w in waves],device)
        scores={}
        for name,head in heads.items():
            z=head(x,mask=mask,return_embedding=True)
            s=torch.nn.functional.normalize(z,dim=1)@torch.nn.functional.normalize(head.loss.weight,dim=1).T
            scores[name]=s[:,0].cpu().numpy().copy();assert np.isfinite(scores[name]).all()
        return [dict((f'score_{m}',float(scores[m][i])) for m in HEADS) for i in range(len(waves))]
    items={phase:make_items(args,phase) for phase in frame.phase.unique()}
    start=time.time();rows=0;qc={};noops=0
    with partial.open('x',newline='') as stream,torch.inference_mode():
        writer=csv.DictWriter(stream,fieldnames=FIELDS);writer.writeheader()
        for offset in range(0,len(frame),8):
            records=frame.iloc[offset:offset+8].to_dict('records');tasks=[];cache={};unique={}
            for r in records:
                item=items[r['phase']][int(r['index'])]
                assert Path(item.path).stem==r['uid'] and int(item.label)==r['label']
                assert str(item.speaker)==r['speaker'] and str(item.type)==r['attack']
                wave,sr=sf.read(item.path,dtype='float32')
                assert sr==16000 and wave.ndim==1 and len(wave)==r['original_samples'] and np.isfinite(wave).all()
                native=wave[:CAP].copy()
                vs=dose_views(wave) if probe else [('native',native),('wrap4',repeat(native,CAP))]
                if not probe and len(native)==CAP:
                    assert np.array_equal(vs[0][1],vs[1][1]);noops+=1
                for condition,w in vs:
                    h=hashlib.sha256(w.tobytes()).hexdigest();unique.setdefault(h,w)
                    tasks.append((r,condition,len(w),h))
            pairs=list(unique.items())
            for begin in range(0,len(pairs),8):
                batch=pairs[begin:begin+8];measured=infer([w for _,w in batch])
                if smoke and not qc:
                    one=infer([batch[0][1]])[0]
                    err=max(abs(one[k]-measured[0][k]) for k in one);assert err<3e-5,err
                    again=infer([batch[0][1]])[0];assert one==again
                    qc={'batch_single_score_max_error':err,'repeat_single_exact':True,'noise_std':0.,'optimizer_updates':0}
                cache.update((h,s) for (h,_),s in zip(batch,measured))
            for r,condition,n,h in tasks:
                row={k:r[k] for k in META};row.update(condition=condition,used_samples=n,wave_sha256=h,gpu_rank=rank);row.update(cache[h]);writer.writerow(row);rows+=1
            stream.flush()
            save(root/f'{tag}_progress_rank{rank}.json',dict(utterances=min(offset+8,len(frame)),expected=len(frame),rows=rows,elapsed_seconds=time.time()-start,updated_unix=time.time()))
        os.fsync(stream.fileno())
    partial.replace(dest)
    assert rows==len(frame)*(31 if probe else 2)
    for source in protocol['checkpoints'].values():assert sha(source['path'])==source['sha256']
    save(root/f'{tag}_done_rank{rank}.json',dict(rows=rows,sha256=sha(dest),utterances=len(frame),no_op_utterances=noops,qc=qc,elapsed_seconds=time.time()-start))

def validate(root,probe,smoke):
    frame,p=read_selection(root,probe,smoke);tag=('probe' if probe else 'full')+('_smoke' if smoke else '')
    parts=[];hashes={}
    for rank in range(2):
        file=root/f'{tag}_rank{rank}.csv';d=json.loads((root/f'{tag}_done_rank{rank}.json').read_text())
        assert sha(file)==d['sha256'];hashes[file.name]=d['sha256'];f=pd.read_csv(file);assert len(f)==d['rows'];parts.append(f)
    f=pd.concat(parts,ignore_index=True);assert not f.isna().any().any()
    assert not f.duplicated(['phase','uid','condition']).any() and len(f)==len(frame)*(31 if probe else 2)
    for _,g in f.groupby('condition'):
        a=g.sort_values(['phase','index']);b=frame.sort_values(['phase','index'])
        for col in META:assert a[col].tolist()==b[col].tolist(),col
    assert np.isfinite(f[[f'score_{m}' for m in HEADS]]).all().all()
    if not probe:
        a=f.query("condition=='native'").set_index(['phase','index']);b=f.query("condition=='wrap4'").set_index(['phase','index'])
        same=a.original_samples>=CAP
        assert a.loc[same,'wave_sha256'].equals(b.loc[same,'wave_sha256'])
        assert np.array_equal(a.loc[same,[f'score_{m}' for m in HEADS]],b.loc[same,[f'score_{m}' for m in HEADS]])
    save(root/f'{tag}_complete.json',dict(status='validated_inference_complete_analysis_pending',rows=len(f),utterances=len(frame),sha256=hashes,protocol_sha256=sha(root/'protocol.json'),noise_std=0,optimizer_updates=0))

def main():
    a=argparse.ArgumentParser();a.add_argument('root',type=Path);a.add_argument('--probe',action='store_true');a.add_argument('--smoke',action='store_true');cli=a.parse_args()
    assert Path('/.dockerenv').exists() and os.environ.get('CUDA_VISIBLE_DEVICES')=='0,1'
    import torch.multiprocessing as mp
    mp.spawn(worker,args=(str(cli.root),cli.probe,cli.smoke),nprocs=2,join=True);validate(cli.root,cli.probe,cli.smoke)

if __name__=='__main__':main()
