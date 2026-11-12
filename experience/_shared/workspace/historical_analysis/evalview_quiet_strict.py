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



    names=['evaluation_original','identity','zero_front_100','zero_back_100','own_front_100','own_back_100']
    def infer(views):
        h=torch.stack(plm(torch.from_numpy(np.stack(views)).to(device),output_hidden_states=True).hidden_states,dim=1)
        z=net.layer_norm(net.linear(net.AHP_ASP(net.bridge_module(h))))
        return torch.nn.functional.normalize(z,dim=1).cpu().numpy()
    inds=list(range(rank,len(trials),2));used=[];emb=[];eligible=[];prefixes=[]
    started=time.time()
    with torch.inference_mode():
        for k,idx in enumerate(inds):
            t=trials[idx];full,sr=sf.read(DATA/'ASVspoof2021_DF_eval/flac'/(t['id']+'.wav'),dtype='float32')
            assert sr==16000 and full.ndim==1 and len(full)==t['samples']
            assert hashlib.sha256(full.tobytes()).hexdigest()==t['pcm_sha256']
            a=full[:64600];a=np.pad(a,(0,64600-len(a)),mode='wrap')
            frames=[a[q:q+320].astype(float) for q in range(0,len(a),320)]
            rms=np.array([np.sqrt(np.mean(q*q)+1e-20) for q in frames]);left=int(np.flatnonzero(rms>=rms.max()*.01)[0])*320
            zz=np.full((6,160),np.nan,dtype=np.float32);zz[:2]=infer([a,a.copy()])
            views=[];ok=left>=320
            for q in ([np.zeros(16000,dtype=np.float32),np.resize(a[:left],16000)] if ok else [np.zeros(16000,dtype=np.float32)]):
                f=np.r_[q,a];b=np.r_[a,q]
                assert f.shape==b.shape==(80600,) and np.array_equal(f[16000:],a) and np.array_equal(b[:64600],a)
                assert np.array_equal(f[:16000],b[64600:])
                assert abs(np.sum(f.astype(float)**2)-np.sum(b.astype(float)**2))<1e-8
                views.extend([f,b])
            zz[2:2+len(views)]=infer(views);emb.append(zz);used.append(idx);eligible.append(ok);prefixes.append(left)
            if k==0 or (k+1)%200==0:print(f'evaluation worker {rank}: {k+1}/{len(inds)}, {time.time()-started:.0f}s',flush=True)
    z=np.stack(emb);cc=center.detach().cpu().numpy()[0]
    np.savez_compressed(Path(cfg['out'])/f'shard_{rank}.npz',indices=used,embeddings=z,scores=np.sum(z*cc,axis=-1),eligible=eligible,prefixes=prefixes,center=cc,names=np.array(names))

def main():
    assert Path('/.dockerenv').exists()
    import torch
    from huggingface_hub import snapshot_download
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0,1' and torch.cuda.device_count()==2
    root=Path('/results/analysis/EBM80_native_quiet_strict');out=root/'evaluation_view';out.mkdir(exist_ok=False)
    trials=[t for t in json.loads((root/'sample_manifest.json').read_text()) if t['cohort']=='new_files'];assert len(trials)==2000
    cfg={'out':str(out),'workers':2,'seed':4193,'cpu':False,'plm_path':snapshot_download('facebook/wav2vec2-xls-r-300m',local_files_only=True)}
    (out/'sample_manifest.json').write_text(json.dumps(trials,indent=2))
    (out/'provenance.json').write_text(json.dumps(dict(cfg,script_sha256=digest(Path(__file__)),checkpoint_sha256=digest(CHECKPOINT),
        purpose='Secondary scope check: preserve the complete conventional head/wrap64600 evaluation input, then append/prepend identical1s. Not part of the eight primary native hypotheses.'),indent=2))
    torch.multiprocessing.spawn(worker,args=(cfg,trials),nprocs=2,join=True)
    (out/'COMPLETED.json').write_text(json.dumps({'n':len(trials)}));print('EVALUATION_VIEW_COMPLETE',flush=True)

if __name__=='__main__':main()
