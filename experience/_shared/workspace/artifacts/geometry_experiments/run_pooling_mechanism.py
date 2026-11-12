"""Bounded, frozen follow-up to the completed padding intervention."""
import argparse
import json
from pathlib import Path
import shlex
import time
from run_padding_causal import connect,run,save,digest,OUT as BASE
OUT=BASE/'pooling_mechanism'
REMOTE='/home/mentee/analysis/embedding_geometry/20261106_pooling_mechanism_01'

def setup():
    OUT.mkdir(exist_ok=False)
    protocol={'stage':'locked_before_followup','parent':'padding_causal_20261106_v1','same_selected_utterances':471,
        'aim':'Disentangle unpadding context-length changes from acoustic repetition and identify pooled-stat numerical mechanism',
        'noisy_native':'reproduce parent wrap/original scores; final pooled mean/std swaps, exact two-order Shapley decomposition',
        'noise0_control':'identical original-only SSL frame sequence copied2x/4x; no added phonetic information or waveform change',
        'factorial':'dup2: none, bridge-only, final-only, both std scale correction; dup4: none/both',
        'correction':'sqrt(k*(k*T-1)/(T-1)); analytical unbiased-standard-deviation correction, not a fitted coefficient',
        'limits':['Adaptive follow-up after a large observed unpadding effect; exploratory mechanistic test','Hybrid pooled statistics need not be realizable by an audio waveform','All pools use std(weighted frames) rather than the usual weighted variance','Native fixed waveform input always has201frames; duplication length sensitivity alone does not prove the initial fixed-length domain failure cause','Not a learned detector or deployment fix; tiny pooling noise suppressed only for the algebraic duplication control'],
        'created_epoch':time.time(),'script_sha256':digest(Path(__file__).with_name('pooling_mechanism_sidecar.py'))}
    save(OUT/'protocol.json',protocol)
    c=connect()
    try:
        assert run(c,'ps -p 352729 -o pgid=,tpgid=').split()==['352729','352729']
        assert run(c,'test -f /proc/352729/root/.dockerenv && echo yes').strip()=='yes'
        gpu=run(c,'nvidia-smi -i 6,7 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits')
        assert all(int(a.split(',')[0])<100 and int(a.split(',')[1])==0 for a in gpu.splitlines())
        run(c,'mkdir '+shlex.quote(REMOTE))
        with c.open_sftp() as s:
            s.put(str(Path(__file__).with_name('pooling_mechanism_sidecar.py')),REMOTE+'/pooling_mechanism_sidecar_v1.py')
            s.put(str(OUT/'protocol.json'),REMOTE+'/protocol.json')
            launch='''#!/bin/bash
set -u
test -f /.dockerenv || exit 90
export CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
ROOT="'''+REMOTE+'''"
cd /home/mentee/code/EBM || exit 91
python -u "$ROOT/pooling_mechanism_sidecar_v1.py" --root "$ROOT" > "$ROOT/run.log" 2>&1
rc=$?; echo "$rc" > "$ROOT/run.exit"
exit "$rc"
'''
            with s.open(REMOTE+'/run.sh','w') as f:f.write(launch)
        assert run(c,'ps -p 352729 -o pgid=,tpgid=').split()==['352729','352729']
        run(c,'screen -S 352153.exp -p 0 -X stuff '+shlex.quote('bash '+REMOTE+'/run.sh\n'))
        save(OUT/'launched.json',{'epoch':time.time(),'gpu_before':gpu})
    finally:c.close()
    print('Mechanistic follow-up launched on the same471 original IDs.',flush=True)

def status(sync=False):
    c=connect();out={}
    try:
        with c.open_sftp() as s:
            for name in ['run.exit','rank0_progress.json','rank1_progress.json','complete.json']:
                try:
                    with s.open(REMOTE+'/'+name) as f:v=f.read().decode()
                    out[name]=json.loads(v) if name.endswith('.json') else v.strip()
                except FileNotFoundError:pass
            if out.get('run.exit') not in [None,'0']:
                with s.open(REMOTE+'/run.log') as f:out['error_tail']=f.read().decode()[-4500:]
            if sync:
                assert out.get('run.exit')=='0' and out['complete.json']['status']=='pass'
                for n,sha in out['complete.json']['files'].items():
                    if not (OUT/n).exists():s.get(REMOTE+'/'+n,str(OUT/n))
                    assert digest(OUT/n)==sha
                if not (OUT/'complete.json').exists():s.get(REMOTE+'/complete.json',str(OUT/'complete.json'))
        save(OUT/'last_status.json',out);print(json.dumps(out,ensure_ascii=False,indent=2),flush=True)
    finally:c.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['setup','status','sync']);a=p.parse_args();setup() if a.action=='setup' else status(a.action=='sync')
