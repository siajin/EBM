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




    used=[];scores=[];started=time.time()
    with torch.inference_mode():
        for k,idx in enumerate(range(rank,len(trials),2)):
            t=trials[idx];full,sr=sf.read(DATA/'ASVspoof2021_DF_eval/flac'/(t['id']+'.wav'),dtype='float32')
            assert sr==16000 and hashlib.sha256(full.tobytes()).hexdigest()==t['pcm_sha256']
            a=full[:64600];a=np.pad(a,(0,64600-len(a)),mode='wrap');views=np.stack([a*g for g in [1.,.5,.25,.1]])
            h=torch.stack(plm(torch.from_numpy(views).to(device),output_hidden_states=True).hidden_states,dim=1)
            z=net.layer_norm(net.linear(net.AHP_ASP(net.bridge_module(h))))
            scores.append((torch.nn.functional.normalize(z,dim=1)@center.T).squeeze(1).cpu().numpy());used.append(idx)
            if k==0 or (k+1)%250==0:print(f'gain worker {rank}: {k+1}/1000, {time.time()-started:.0f}s',flush=True)
    np.savez_compressed(Path(cfg['out'])/f'shard_{rank}.npz',indices=used,scores=scores)

def main():
    assert Path('/.dockerenv').exists()
    import torch
    from huggingface_hub import snapshot_download
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0,1' and torch.cuda.device_count()==2
    root=Path('/results/analysis/EBM80_cue_factor_audit');out=root/'gain_followup';out.mkdir(exist_ok=False)
    trials=json.loads((root/'sample_manifest.json').read_text());cfg=dict(out=str(out),workers=2,cpu=False,seed=62419,plm_path=snapshot_download('facebook/wav2vec2-xls-r-300m',local_files_only=True))
    (out/'plan.json').write_text(json.dumps(dict(cfg,gains=[1.,.5,.25,.1],scope='Exploratory dose response after seeing gain0.1 sensitivity; not prespecified confirmation',script_sha256=digest(Path(__file__)),checkpoint_sha256=digest(CHECKPOINT)),indent=2))
    torch.multiprocessing.spawn(worker,args=(cfg,trials),nprocs=2,join=True);(out/'COMPLETED.json').write_text(json.dumps({'n':2000}));print('GAIN_COMPLETE',flush=True)

if __name__=='__main__':main()
