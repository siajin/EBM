"""Visualize an EBM checkpoint using a reproducible balanced 21DF sample.

Run inside the mentee container. No training, network access, or source edits.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

sys.dont_write_bytecode = True
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['MPLCONFIGDIR'] = '/home/mentee/.cache/matplotlib'

ROOT = Path('/home/mentee/results/CM/LA/EBM')
CHECKPOINT = ROOT / 'model/BestCLS_80_21DF.pt'
SOURCE = ROOT / 'backup/scripts'
DATA = Path('/data/ASVspoof2021_DF')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def sample_trials(n, seed):
    groups = {0: [], 1: []}
    protocol = DATA / 'keys/DF/CM/trial_metadata.txt'
    for line in protocol.read_text().splitlines():
        fields = line.split()
        if len(fields) < 8 or fields[7] != 'eval':
            continue
        label = int(fields[5] == 'bonafide')
        groups[label].append({'id': fields[1], 'label': label, 'attack': fields[4]})
    rng = random.Random(seed)
    selected = []
    for label in (1, 0):
        if len(groups[label]) < n:
            raise ValueError('Requested sample exceeds population')
        selected.extend(rng.sample(groups[label], n))
    rng.shuffle(selected)
    for row in selected:
        path = DATA / 'ASVspoof2021_DF_eval/flac' / (row['id'] + '.wav')
        if not path.is_file():
            raise FileNotFoundError(path)
    return selected, {str(k): len(v) for k, v in groups.items()}, digest(protocol)


def worker(rank, cfg, trials):
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import Wav2Vec2Model
    sys.path.insert(0, str(SOURCE))
    from model import B_Linear, BPL, OCSoftmax

    torch.set_num_threads(2)
    torch.manual_seed(cfg['seed'] + rank)
    np.random.seed(cfg['seed'] + rank)
    device = torch.device('cpu' if cfg['cpu'] else f'cuda:{rank}')
    if not cfg['cpu']:
        torch.cuda.set_device(rank)
    state = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    state = {k.removeprefix('module.'): v for k, v in state.items()}
    net = B_Linear(num_layer=25, hidden_size=1024, output_size=160, agg_size=128,
                   loss=OCSoftmax(160, r_real=.9, r_fake=.2, alpha=20),
                   loss_bpl=BPL(alpha=30))
    net.load_state_dict(state, strict=True)
    net.to(device).eval()
    plm = Wav2Vec2Model.from_pretrained(cfg['plm_path'], local_files_only=True).to(device).eval()
    center = torch.nn.functional.normalize(net.loss.weight.detach(), dim=1)
    assert center.shape == (1, 160)


    names=['native','identity']
    for dose in ['025','050','100']:names.extend([f'zero_front_{dose}',f'zero_back_{dose}',f'own_front_{dose}',f'own_back_{dose}'])
    names += ['ramp_front_100','ramp_back_100']
    def infer(views):
        arr=np.stack(views);assert arr.ndim==2
        # Equal-length versions of ONE original file. No batch padding.
        h=torch.stack(plm(torch.from_numpy(arr).to(device),output_hidden_states=True).hidden_states,dim=1)
        z=net.layer_norm(net.linear(net.AHP_ASP(net.bridge_module(h))))
        z=torch.nn.functional.normalize(z,dim=1)
        assert torch.isfinite(z).all()
        return z.cpu().numpy()
    def onset(a):
        chunks=[a[i:min(i+320,len(a))].astype(np.float64) for i in range(0,len(a),320)]
        rms=np.array([np.sqrt(np.mean(x*x)+1e-20) for x in chunks])
        assert rms.max()>1e-8
        return int(np.flatnonzero(rms>=rms.max()*.01)[0])*320
    def assemble(a,q):
        front=np.concatenate([q,a]);back=np.concatenate([a,q])
        assert len(front)==len(back)==len(a)+len(q)
        assert np.array_equal(front[len(q):],a) and np.array_equal(back[:len(a)],a)
        assert np.array_equal(front[:len(q)],back[len(a):])
        # Parts equality implies identical sample multiset, without sorting away time information.
        energy_error=abs(np.sum(front.astype(np.float64)**2)-np.sum(back.astype(np.float64)**2))
        assert energy_error<1e-8
        rms=np.sqrt(np.mean(a.astype(np.float64)**2)+1e-20)
        jump=[abs(float(q[-1])-float(a[0]))/rms,abs(float(a[-1])-float(q[0]))/rms]
        return front,back,energy_error,jump
    # Complete the prespecified new-file sample first; old files are exploratory.
    inds=list(range(rank,len(trials),cfg['workers']))
    inds.sort(key=lambda i:(trials[i]['cohort']!='new_files',i))
    used=[];emb=[];eligible=[];prefixes=[];checks=[];seams=[];lengths=[]
    started=time.time()
    def save(name):
        z=np.stack(emb).astype(np.float32)
        score=np.einsum('nkd,d->nk',z,center.detach().cpu().numpy()[0])
        np.savez_compressed(Path(cfg['out'])/name,indices=used,embeddings=z,scores=score,eligible=eligible,
          prefixes=prefixes,checks=checks,seams=seams,lengths=lengths,center=center.detach().cpu().numpy()[0],names=np.array(names))
    with torch.inference_mode():
        for count,idx in enumerate(inds):
            t=trials[idx]
            a,sr=sf.read(DATA/'ASVspoof2021_DF_eval/flac'/(t['id']+'.wav'),dtype='float32')
            assert sr==16000 and a.ndim==1 and len(a)==t['samples']
            assert hashlib.sha256(a.tobytes()).hexdigest()==t['pcm_sha256']
            left=onset(a);ok=left>=320;qsource=a[:left]
            allz=np.full((16,160),np.nan,dtype=np.float32);allz[:2]=infer([a,a.copy()])
            maxerr=0.;jumps=np.full((3,2,2),np.nan)
            cap=1 if len(a)>192000 else (2 if len(a)>128000 else 4)
            for k,dose in enumerate([4000,8000,16000]):
                q0=np.zeros(dose,dtype=np.float32);f,bb,err,jump=assemble(a,q0);maxerr=max(maxerr,err)
                views=[f,bb];slots=[2+4*k,3+4*k];jumps[k,0]=jump
                if ok:
                    q=np.resize(qsource,dose);f,bb,err,jump=assemble(a,q);maxerr=max(maxerr,err)
                    views.extend([f,bb]);slots.extend([4+4*k,5+4*k]);jumps[k,1]=jump
                for lo in range(0,len(views),cap):allz[slots[lo:lo+cap]]=infer(views[lo:lo+cap])
            # Native samples remain unchanged; only added zeros near the splice get 10ms ramps.
            qf=np.zeros(16000,dtype=np.float32);qb=qf.copy()
            qf[-160:]=np.linspace(0,a[0],160,dtype=np.float32)
            qb[:160]=np.linspace(a[-1],0,160,dtype=np.float32)
            ff=np.concatenate([qf,a]);bb=np.concatenate([a,qb])
            assert np.array_equal(ff[16000:],a) and np.array_equal(bb[:len(a)],a)
            assert ff[15999]==ff[16000] and bb[len(a)-1]==bb[len(a)]
            for lo in range(0,2,cap):allz[14+lo:14+min(lo+cap,2)]=infer([ff,bb][lo:lo+cap])
            valid=np.isfinite(allz).all(1)
            assert valid.sum()==(16 if ok else 10)
            assert np.max(np.abs(np.linalg.norm(allz[valid],axis=1)-1))<2e-6
            emb.append(allz);used.append(idx);eligible.append(ok);prefixes.append(left)
            checks.append([maxerr,1,1]);seams.append(jumps);lengths.append(len(a))
            if count==0 or (count+1)%100==0:
                print(f'worker {rank}: {count+1}/{len(inds)}, cohort={t["cohort"]}, elapsed={time.time()-started:.0f}s',flush=True)
            if (count+1)%500==0:save(f'partial_{rank}.npz')
    save(f'shard_{rank}.npz')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--out',default='/results/analysis/EBM80_native_quiet_strict/inference')
    args=parser.parse_args()
    assert Path('/.dockerenv').exists(), 'Container required'
    import torch
    from huggingface_hub import snapshot_download
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0,1' and torch.cuda.device_count()==2
    planroot=Path('/results/analysis/EBM80_native_quiet_strict')
    trials=json.loads((planroot/'sample_manifest.json').read_text())
    plan=json.loads((planroot/'plan.json').read_text());assert not plan['exact_pcm_duplicate_pairs']
    if args.smoke:
        trials=[t for cohort in ['previous','new_files'] for label in [1,0]
           for t in [v for v in trials if v['cohort']==cohort and v['label']==label][:2]]
    out=Path(args.out);out.mkdir(exist_ok=False)
    cfg={'out':str(out),'workers':2,'seed':7301,'cpu':False,
         'plm_path':snapshot_download('facebook/wav2vec2-xls-r-300m',local_files_only=True)}
    provenance=dict(cfg,checkpoint=str(CHECKPOINT),checkpoint_sha256=digest(CHECKPOINT),
       script_sha256=digest(Path(__file__)),plan_sha256=digest(planroot/'plan.json'),
       manifest_sha256=digest(planroot/'sample_manifest.json'),torch_version=torch.__version__,
       source_sha256={f:digest(SOURCE/f) for f in ['model/b_linear.py','model/loss.py']})
    (out/'sample_manifest.json').write_text(json.dumps(trials,indent=2))
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2))
    torch.multiprocessing.spawn(worker,args=(cfg,trials),nprocs=2,join=True)
    (out/'COMPLETED.json').write_text(json.dumps({'completed_trials':len(trials),'workers':2,'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}))
    print('INFERENCE_COMPLETE',len(trials),flush=True)

if __name__=='__main__':main()
