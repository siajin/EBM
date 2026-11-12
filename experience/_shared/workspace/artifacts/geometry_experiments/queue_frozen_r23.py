"""Prepare derived ID locks and submit R2/R3 only after existing R1 has ended."""
import argparse
from pathlib import Path
import shlex
import time
import paramiko
import pandas as pd
from frozen_common import *

REMOTE='/home/mentee/analysis/embedding_geometry/20261105_frozen_all_01'
DEST=REMOTE+'/R23'
LOCAL=DEFAULT_ROOT/'local_analysis/frozen_all_v1/R23_submission'

def connect():
    c=paramiko.SSHClient();c.load_system_host_keys();c.connect('172.16.162.222',username='mentee',password=Path('C:/Users/sihwa/.ssh/mentee-password.txt').read_text().splitlines()[0],look_for_keys=False,allow_agent=False,timeout=15);return c

def setup():
    LOCAL.mkdir(exist_ok=False)
    for name in ['DF_pilot','21LA_pilot','DF_small','21LA_small']:
        d=panel(name);d.rows[['utterance_id','y']].to_parquet(LOCAL/(name+'.parquet'),index=False)
    v=panel('V');chosen=[]
    for y,n in [(1,32),(0,96)]:
        rows=v.rows[v.rows.y==y].reset_index(drop=True);chosen.append(rows.iloc[stable_indices(rows,'V-QC-20261105-v1|',n)])
    frame=pd.concat(chosen).sort_values('utterance_id');frame[['utterance_id','y']].to_parquet(LOCAL/'V_QC.parquet',index=False)
    lock=dict(status='prepared_not_yet_executed',R1_remote=REMOTE,R23_remote=DEST,V_QC_SHA256=digest(LOCAL/'V_QC.parquet'),input_files={p.name:digest(p) for p in LOCAL.glob('*.parquet')},conditions=7,full_rows_per_dataset=20000,small_rows_per_dataset=2000)
    c=connect()
    try:
        with c.open_sftp() as s:
            s.mkdir(DEST);s.mkdir(DEST+'/inputs')
            for p in LOCAL.glob('*.parquet'):s.put(str(p),DEST+'/inputs/'+p.name)
            s.put(str(Path(__file__).with_name('frozen_r23.py')),DEST+'/frozen_r23_v1.py')
            launch='''#!/bin/bash
set -u
test -f /.dockerenv || exit 90
export CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
ROOT="'''+DEST+'''"
cd /home/mentee/code/EBM || exit 91
python -u "$ROOT/frozen_r23_v1.py" prepare --root "$ROOT" > "$ROOT/prepare.log" 2>&1
rc=$?; echo "$rc" > "$ROOT/prepare.exit"
if [ "$rc" -ne 0 ]; then echo "$rc" > "$ROOT/run.exit"; exit "$rc"; fi
for dataset in DF_pilot 21LA_pilot; do
 python -u "$ROOT/frozen_r23_v1.py" extract --root "$ROOT" --dataset "$dataset" > "$ROOT/$dataset.log" 2>&1
 rc=$?; echo "$rc" > "$ROOT/$dataset.exit"
 if [ "$rc" -ne 0 ]; then echo "$rc" > "$ROOT/run.exit"; exit "$rc"; fi
done
echo 0 > "$ROOT/run.exit"
'''
            with s.open(DEST+'/run_r23.sh','w') as f:f.write(launch)
    finally:c.close()
    save(LOCAL/'submission.json',lock);print('R2/R3 prepared, waiting for R1; no input sent to active experiment.',flush=True)

def watch():
    assert (LOCAL/'submission.json').is_file() and not (LOCAL/'launched.json').exists()
    while True:
        c=connect()
        try:
            with c.open_sftp() as s:
                try:
                    with s.open(REMOTE+'/run.exit') as f:rc=f.read().decode().strip()
                except FileNotFoundError:rc=None
            if rc is not None:
                if rc!='0':save(LOCAL/'blocked.json',dict(status='upstream_failed',R1_exit=rc));raise RuntimeError('R1 failed; queue preserved, no restart')
                def run(cmd):
                    _,out,err=c.exec_command(cmd);text=out.read().decode();error=err.read().decode();code=out.channel.recv_exit_status()
                    if code:raise RuntimeError((text,error,code))
                    return text
                idle=run('ps -p 352729 -o pgid=,tpgid=').split()==['352729','352729']
                gpu=run('nvidia-smi -i 6,7 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits')
                available=all(int(x.split(',')[0])<100 and int(x.split(',')[1])==0 for x in gpu.splitlines())
                if idle and available:
                    assert run('test -f /proc/352729/root/.dockerenv && echo yes').strip()=='yes'
                    command='bash '+DEST+'/run_r23.sh\n';run('screen -S 352153.exp -p 0 -X stuff '+shlex.quote(command))
                    save(LOCAL/'launched.json',dict(status='launched_after_R1',UTC=time.time(),R1_exit=rc));print('R2/R3 launched into verified idle existing container.',flush=True);return
            save(LOCAL/'waiting.json',dict(status='waiting_for_existing_R1_to_finish',checked_epoch=time.time()))
        finally:c.close()
        time.sleep(30)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');o=p.parse_args()
    if o.watch:watch()
    else:setup()
