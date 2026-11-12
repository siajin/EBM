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




    bankroot=Path('/results/analysis/EBM80_quiet_donor_audit')
    bank=np.load(bankroot/'donor_frames_private.npz')['frames'].astype(float)
    dm=json.loads((bankroot/'donor_manifest_private.json').read_text());assign=np.load(bankroot/'assignment_private.npz')['pairs']
    used=[];scores=[];checks=[];gains=[];jumps=[];counts=[];started=time.time()
    def infer(views):
        h=torch.stack(plm(torch.from_numpy(np.stack(views)).to(device),output_hidden_states=True).hidden_states,dim=1)
        z=net.layer_norm(net.linear(net.AHP_ASP(net.bridge_module(h))))
        return (torch.nn.functional.normalize(z,dim=1)@center.T).squeeze(1).cpu().numpy()
    with torch.inference_mode():
        for k,idx in enumerate(range(rank,len(trials),2)):
            t=trials[idx];raw,sr=sf.read(DATA/'ASVspoof2021_DF_eval/flac'/(t['id']+'.wav'),dtype='float32')
            assert sr==16000 and hashlib.sha256(raw.tobytes()).hexdigest()==t['pcm_sha256']
            a=raw[:64600];a=np.pad(a,(0,64600-len(a)),mode='wrap');ff=np.array(t['eligible_frame_indices'],dtype=int)
            ss=np.full(17,np.nan);ss[:2]=infer([a,a.copy()]);cc=np.zeros(4);gg=np.full(17,np.nan);jj=np.full(17,np.nan)
            if len(ff):
                rng=np.random.default_rng(915037+idx);own=np.stack([a[q*320+16:q*320+304] for q in ff]).astype(float)
                means=own.mean(1);stds=own.std(1);assert np.min(stds)>=1e-6
                donorids=assign[idx].reshape(-1);common_k=min(8,len(ff),*(len(dm[int(j)]['frame_ids']) for j in donorids))
                pattern=np.tile(np.arange(common_k),int(np.ceil(len(ff)/common_k)))[:len(ff)];rng.shuffle(pattern)
                selected=[]
                selected.append(own[rng.choice(len(ff),common_k,replace=False)][pattern])
                selected.append(np.stack([rng.permutation(v) for v in own]))
                selected.append(rng.normal(size=(common_k,288))[pattern])
                for donor in donorids:
                    frameids=np.array(dm[int(donor)]['frame_ids']);chosen=rng.choice(frameids,common_k,replace=False);selected.append(bank[chosen][pattern])
                assert len(selected)==15
                views=[];energyerr=0.;meanerr=0.;originalrms=np.sqrt(np.mean(a.astype(float)**2))
                outside=np.ones(64600,dtype=bool)
                for q in ff:outside[q*320+16:q*320+304]=False
                for vi,q in enumerate(selected,2):
                    qm=q.mean(1);qs=q.std(1);assert np.min(qs)>0
                    repl=((q-qm[:,None])*(stds/qs)[:,None]+means[:,None]).astype(np.float32)
                    b=a.copy()
                    for row,frameid in enumerate(ff):b[frameid*320+16:frameid*320+304]=repl[row]
                    assert np.array_equal(b[outside],a[outside]) and len(b)==len(a)
                    energyerr=max(energyerr,float(np.max(abs(np.mean(repl.astype(float)**2,axis=1)-np.mean(own**2,axis=1))/np.maximum(np.mean(own**2,axis=1),1e-20))))
                    meanerr=max(meanerr,float(np.max(abs(repl.astype(float).mean(1)-means))))
                    # Both added inner joins, relative to recipient whole-waveform RMS. Actual20ms frame edges unchanged.
                    jump=[]
                    for frameid in ff:jump.extend([abs(float(b[frameid*320+16])-float(b[frameid*320+15])),abs(float(b[frameid*320+304])-float(b[frameid*320+303]))])
                    jj[vi]=max(jump)/originalrms;gg[vi]=float(np.max(stds/qs));views.append(b)
                assert energyerr<1e-5 and meanerr<1e-8
                for j in range(0,len(views),4):ss[2+j:2+j+len(views[j:j+4])]=infer(views[j:j+4])
                cc=np.array([energyerr,meanerr,1.,common_k]);counts.append([len(ff),common_k])
            else:counts.append([0,0])
            scores.append(ss);checks.append(cc);gains.append(gg);jumps.append(jj);used.append(idx)
            if k==0 or (k+1)%200==0:print(f'donor worker {rank}: {k+1}/1000, {time.time()-started:.0f}s',flush=True)
    np.savez_compressed(Path(cfg['out'])/f'shard_{rank}.npz',indices=used,scores=scores,checks=checks,gains=gains,jumps=jumps,counts=counts)

def main():
    assert Path('/.dockerenv').exists()
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0,1' and torch.cuda.device_count()==2
    root=Path('/results/analysis/EBM80_quiet_donor_audit');out=root/'inference';out.mkdir(exist_ok=False)
    dm=json.loads((root/'donor_manifest_private.json').read_text());trials=json.loads((root/'recipient_manifest_private.json').read_text());assert len(trials)==2000
    groups={}
    for j,t in enumerate(dm):groups.setdefault((t['source'],t['speaker'],t['codec']),{0:[],1:[]})[t['label']].append(j)
    groups={k:g for k,g in groups.items() if g[0] and g[1]};rng=np.random.default_rng(177213);sources=['asvspoof','vcc2018','vcc2020'];splits={};speakers={}
    for so in sources:
        allsp=np.array(sorted({k[1] for k in groups if k[0]==so}));rng.shuffle(allsp);splits[so]=[set(allsp[::2]),set(allsp[1::2])];speakers[so]=[sorted(s) for s in splits[so]];assert not (splits[so][0]&splits[so][1])
    cells={(so,rep):{codec:[k for k in groups if k[0]==so and k[1] in splits[so][rep] and k[2]==codec] for codec in sorted({k[2] for k in groups})} for so in sources for rep in [0,1]}
    common=sorted(set.intersection(*(set(codec for codec,keys in c.items() if keys) for c in cells.values())));assert common
    pairs=np.empty((2000,3,2,2),dtype=np.int64);codec_matches=0
    for i,t in enumerate(trials):
        codec=t['codec'] if t['codec'] in common else str(rng.choice(common));codec_matches+=codec==t['codec']
        for si,so in enumerate(sources):
            for rep in [0,1]:
                keys=cells[(so,rep)][codec];key=keys[int(rng.integers(len(keys)))];pool=groups[key]
                pairs[i,si,rep]=[int(rng.choice(pool[1])),int(rng.choice(pool[0]))]
                a,b=[dm[j] for j in pairs[i,si,rep]];assert (a['source'],a['speaker'],a['codec'])==(b['source'],b['speaker'],b['codec']) and a['label']==1 and b['label']==0
    np.savez_compressed(root/'assignment_private.npz',pairs=pairs)
    cfg=dict(out=str(out),workers=2,cpu=False,seed=9241,plm_path=snapshot_download('facebook/wav2vec2-xls-r-300m',local_files_only=True))
    names=['baseline','identity','self_reuse','self_shuffle','gaussian']+[f'{so}_{rep}_{lab}' for so in sources for rep in ['A','B'] for lab in ['donorB','donorS']]
    plan=dict(cfg,checkpoint=str(CHECKPOINT),checkpoint_sha256=digest(CHECKPOINT),script_sha256=digest(Path(__file__)),sources=sources,names=names,donor_speakers_by_disjoint_bank=speakers,common_codecs=common,recipient_codec_matched_n=codec_matches,
      intervention='Only inner288samples of eligible leadingquiet20ms frames. Original first/last16samples retained. Recipient inner mean/variance restored. All other samples identical.',
      frame_sampling='Each recipient uses samek<=8uniqueframes and same repetition pattern in all12donorconditions. Files matchedsource/speaker/codec within B/S pair. Codec held across all sources/banks. Banks donor-speaker disjoint.',
      primary='Per donor-source and recipient-label: donorB-minus-donorS, mean of two banks; restrict to original eligible recipients and maxgain<=10 for bothB/S in bothbanks. Report eachbank separately. Allgain responses secondary.',
      uncertainty='Donorbank replication, leave-one-donor-speaker ranges and recipient speaker cluster intervals conditional on realized donor bank; no claim of independent rows or exact universal population proof.',
      limitations=['New hybrid quiet can be unnatural. Internal joins may change though original frame edges retained.','Low-energy proxy may contain genuine synthesis/channel traces; donorlabel effect does not itself prove spurious learning.','Donor speaker counts VCC2018 only4; finite bank scope and selected lowgain subsets constrain inference.'])
    (root/'plan.json').write_text(json.dumps(plan,indent=2));torch.multiprocessing.spawn(worker,args=(cfg,trials),nprocs=2,join=True);(out/'COMPLETED.json').write_text(json.dumps({'n':2000}));print('DONOR_AUDIT_COMPLETE',flush=True)

if __name__=='__main__':main()
