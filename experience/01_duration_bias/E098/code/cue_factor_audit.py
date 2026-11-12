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




    names=['baseline','identity','zero_lead','zero_tail','zero_both','shuffle_lead','shuffle_tail','shuffle_both','phase_both','quiet_gain_01','quiet_gain_10','global_gain_01']
    inds=list(range(rank,len(trials),2));used=[];scores=[];features=[];checks=[];started=time.time()
    def infer(a):
        h=torch.stack(plm(torch.from_numpy(np.stack(a)).to(device),output_hidden_states=True).hidden_states,dim=1)
        z=net.layer_norm(net.linear(net.AHP_ASP(net.bridge_module(h))))
        return (torch.nn.functional.normalize(z,dim=1)@center.T).squeeze(1).cpu().numpy()
    with torch.inference_mode():
        for k,idx in enumerate(inds):
            t=trials[idx];full,sr=sf.read(DATA/'ASVspoof2021_DF_eval/flac'/(t['id']+'.wav'),dtype='float32')
            assert sr==16000 and len(full)==t['samples'] and hashlib.sha256(full.tobytes()).hexdigest()==t['pcm_sha256']
            a=full[:64600];a=np.pad(a,(0,64600-len(a)),mode='wrap')
            frames=[a[q:q+320].astype(float) for q in range(0,len(a),320)]
            rms=np.array([np.sqrt(np.mean(v*v)+1e-20) for v in frames]);active=np.flatnonzero(rms>=rms.max()*.01)
            lead=np.zeros(64600,dtype=bool);trail=lead.copy();lead[:active[0]*320]=True;trail[min((active[-1]+1)*320,64600):]=True;mask=lead|trail
            assert not (lead&trail).any()
            shuffle=a.copy();phase=a.copy();rng=np.random.default_rng(51820+idx);eerr=0.;serr=0.;merr=0.
            for q,v in enumerate(frames):
                lo=q*320;hi=lo+len(v)
                if not mask[lo]:continue
                shuffle[lo:hi]=rng.permutation(v)
                ft=np.fft.rfft(v);pt=ft.copy();last=len(pt)-1 if len(v)%2==0 else len(pt)
                pt[1:last]=abs(ft[1:last])*np.exp(1j*rng.uniform(-np.pi,np.pi,last-1))
                phase[lo:hi]=np.fft.irfft(pt,n=len(v)).astype(np.float32)
                vv=phase[lo:hi].astype(float);energy=np.mean(v*v)
                eerr=max(eerr,abs(np.mean(vv*vv)-energy)/max(energy,1e-20))
                serr=max(serr,float(np.max(abs(abs(np.fft.rfft(vv))**2-abs(ft)**2))/max(float(np.max(abs(ft)**2)),1e-20)))
                merr=max(merr,abs(vv.mean()-v.mean()))
                assert np.array_equal(np.sort(shuffle[lo:hi]),np.sort(a[lo:hi]))
            views=[a,a.copy()]
            for m in [lead,trail,mask]:
                b=a.copy();b[m]=0;views.append(b)
            for m in [lead,trail,mask]:
                b=a.copy();b[m]=shuffle[m];views.append(b)
            views.append(phase)
            for gain in [.1,10]:
                b=a.copy();b[mask]*=gain;views.append(b)
            views.append(a*.1)
            assert len(views)==12 and eerr<1e-5 and serr<1e-5 and merr<1e-8
            assert all(np.array_equal(b[~mask],a[~mask]) for b in views[:11])
            ss=np.concatenate([infer(views[j:j+4]) for j in range(0,12,4)])
            scores.append(ss);used.append(idx)
            features.append([lead.sum()/16000,trail.sum()/16000,np.mean(a.astype(float)**2),float(np.mean(a[mask].astype(float)**2)) if mask.any() else 0.,mask.sum()/64600])
            checks.append([eerr,serr,merr,1.])
            if k==0 or (k+1)%200==0:print(f'factor worker {rank}: {k+1}/{len(inds)}, {time.time()-started:.0f}s',flush=True)
    np.savez_compressed(Path(cfg['out'])/f'shard_{rank}.npz',indices=used,scores=scores,features=features,checks=checks,names=np.array(names))

def main():
    assert Path('/.dockerenv').exists()
    import torch
    from huggingface_hub import snapshot_download
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0,1' and torch.cuda.device_count()==2
    out=Path('/results/analysis/EBM80_cue_factor_audit');out.mkdir(exist_ok=False)
    trials=[t for t in json.loads(Path('/results/analysis/EBM80_native_quiet_strict/sample_manifest.json').read_text()) if t['cohort']=='new_files'];assert len(trials)==2000
    cfg=dict(out=str(out),workers=2,cpu=False,seed=7921,plm_path=snapshot_download('facebook/wav2vec2-xls-r-300m',local_files_only=True))
    (out/'sample_manifest.json').write_text(json.dumps(trials,indent=2))
    (out/'plan.json').write_text(json.dumps(dict(cfg,checkpoint=str(CHECKPOINT),checkpoint_sha256=digest(CHECKPOINT),script_sha256=digest(Path(__file__)),
      purpose='Follow-up factor diagnosis. Same new-file cohort, not untouched confirmation. Fixed conventional64600 input and fixed edge positions. No retraining.',
      contrasts='Zero versus native; shuffle preserves frame histogram/energy/mean; phase randomization preserves frame power spectrum/energy/mean; gains change quiet amplitude alone; lead/tail separately and jointly; whole-waveform gain reference.',
      decision_threshold='Use existing 19LA dev threshold from leading_quiet_paired summary, freeze across all conditions. EER is descriptive sample ranking.',
      inference='Original checkpoint eval including original pooling noise. GPU0,1. No source edits. All non-edge samples bit-identical except explicit global gain control.'),indent=2))
    torch.multiprocessing.spawn(worker,args=(cfg,trials),nprocs=2,join=True)
    (out/'COMPLETED.json').write_text(json.dumps({'n':2000}));print('FACTOR_AUDIT_COMPLETE',flush=True)

if __name__=='__main__':main()
