"""Server-side one-shot queue: wait for R23, submit once to its idle container.

No SSH credentials are uploaded. The queue survives closing the local PC.
"""
import argparse
import json
from pathlib import Path
import shlex
import time

ROOT='/home/mentee/analysis/embedding_geometry/20261105_19la_train_xlsr_last_01'
UPSTREAM='/home/mentee/analysis/embedding_geometry/20261105_frozen_all_01/R23'
SHELL=352729
SCREEN='352153.exp'

WATCHER=r'''import json,pathlib,subprocess,time,os
ROOT=pathlib.Path(__ROOT__)
UPSTREAM=pathlib.Path(__UPSTREAM__)
EXPECTED_NS=__NAMESPACE__
SHELL=352729
def save(value):
 p=ROOT/'queue_status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(dict(checked_epoch=time.time(),**value),indent=2));t.replace(p)
def run(args):
 p=subprocess.run(args,capture_output=True,universal_newlines=True)
 if p.returncode:raise RuntimeError('Readiness command failed: '+repr(args)+' '+p.stderr[:300])
 return p.stdout.strip()
try:
 save(dict(status='waiting_for_existing_R23',upstream=str(UPSTREAM)))
 while True:
  rcpath=UPSTREAM/'run.exit'
  if rcpath.exists():
   rc=rcpath.read_text().strip()
   if rc!='0':save(dict(status='upstream_failed_no_launch',exit_code=rc));break
   assert (UPSTREAM/'21LA_pilot.exit').read_text().strip()=='0'
   if os.readlink('/proc/%d/ns/mnt'%SHELL)!=EXPECTED_NS:raise RuntimeError('Container namespace changed; no submission')
   assert pathlib.Path('/proc/%d/root/.dockerenv'%SHELL).is_file()
   idle=run(['ps','-p',str(SHELL),'-o','pgid=,tpgid=']).split()==[str(SHELL),str(SHELL)]
   lines=run(['nvidia-smi','-i','6,7','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits']).splitlines()
   available=len(lines)==2 and all(int(x.split(',')[0])<100 and int(x.split(',')[1])==0 for x in lines)
   if idle and available:
    if any((ROOT/p).exists() for p in ['launched.json','run.exit','prepare.log']):raise RuntimeError('Already submitted; refusing a duplicate')
    assert run(['ps','-p',str(SHELL),'-o','pgid=,tpgid=']).split()==[str(SHELL),str(SHELL)]
    # Write the reservation before the one and only input; never retry it.
    (ROOT/'launched.json').write_text(json.dumps(dict(submitted_epoch=time.time(),screen='352153.exp:0',upstream_exit=rc)))
    run(['screen','-S','352153.exp','-p','0','-X','stuff','bash '+str(ROOT/'run_last_layer.sh')+'\n'])
    save(dict(status='submitted_to_existing_idle_container',upstream_exit=rc));break
  save(dict(status='waiting_for_existing_R23',upstream=str(UPSTREAM)))
  time.sleep(15)
except Exception as e:
 save(dict(status='queue_failed_no_retry',error=str(e)));raise
'''


def connect():
    import paramiko
    client=paramiko.SSHClient();client.load_system_host_keys()
    client.connect('172.16.162.222',username='mentee',
        password=Path('C:/Users/sihwa/.ssh/mentee-password.txt').read_text().splitlines()[0],
        look_for_keys=False,allow_agent=False,timeout=15)
    return client


def setup():
    raise RuntimeError('Server-disk job superseded by laptop-only streaming; use receive_xlsr_last_layer.py')
    c=connect()
    try:
        def run(command):
            _,out,err=c.exec_command(command);a=out.read().decode();b=err.read().decode();rc=out.channel.recv_exit_status()
            if rc:raise RuntimeError((a,b,rc))
            return a.strip()
        namespace=run('readlink /proc/352729/ns/mnt')
        assert run('test -f /proc/352729/root/.dockerenv && echo yes')=='yes'
        # Reserving a new sidecar directory does not touch the active shell.
        run('mkdir '+shlex.quote(ROOT))
        watcher=WATCHER.replace('__ROOT__',repr(ROOT)).replace('__UPSTREAM__',repr(UPSTREAM)).replace('__NAMESPACE__',repr(namespace))
        launcher='''#!/bin/bash
set -u
ROOT="'''+ROOT+'''"
test -f /.dockerenv || { echo 90 > "$ROOT/run.exit"; exit 90; }
export CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
cd /home/mentee/code/EBM || { echo 91 > "$ROOT/run.exit"; exit 91; }
for stage in prepare qc extract; do
 python -u "$ROOT/xlsr_last_layer_extract_v1.py" "$stage" --root "$ROOT" > "$ROOT/$stage.log" 2>&1
 rc=$?
 echo "$rc" > "$ROOT/$stage.exit"
 if [ "$rc" -ne 0 ]; then echo "$rc" > "$ROOT/run.exit"; exit "$rc"; fi
done
echo 0 > "$ROOT/run.exit"
'''
        with c.open_sftp() as s:
            s.put(str(Path(__file__).with_name('xlsr_last_layer_extract.py')),ROOT+'/xlsr_last_layer_extract_v1.py')
            for name,body in [('wait_and_launch.py',watcher),('run_last_layer.sh',launcher)]:
                with s.open(ROOT+'/'+name,'w') as f:f.write(body)
            with s.open(ROOT+'/request.json','w') as f:
                f.write(json.dumps(dict(dataset='19LA_train',count=25380,shape=[25380,201,1024],dtype='float32',
                    storage='server_only',user_requested=True,upstream=UPSTREAM,requested_epoch=time.time()),indent=2))
        # A stdlib-only supervisor on the host waits; all GPU work runs later
        # inside the already-existing container via its idle screen shell.
        payload="import subprocess,pathlib;root=pathlib.Path("+repr(ROOT)+");log=(root/'queue.log').open('ab');p=subprocess.Popen(['python3','-u',str(root/'wait_and_launch.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);print(p.pid)"
        pid=run('python3 -c '+shlex.quote(payload))
        print(json.dumps(dict(status='queued',remote_root=ROOT,supervisor_pid=int(pid)),indent=2))
    finally:c.close()


def status():
    c=connect()
    try:
        with c.open_sftp() as s:
            for name in ['queue_status.json','prepare.exit','qc.exit','qc/summary.json','extract.exit','run.exit',
                         'extract/rank0_progress.json','extract/rank1_progress.json','extract/summary.json']:
                try:
                    with s.open(ROOT+'/'+name) as f:body=f.read().decode()
                    print(name,body)
                except FileNotFoundError:pass
    finally:c.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['setup','status']);o=p.parse_args()
    setup() if o.action=='setup' else status()
