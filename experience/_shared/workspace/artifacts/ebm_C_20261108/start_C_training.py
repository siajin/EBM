"""Launch C exactly once after the bounded two-GPU check; never restarts a run."""
import json
from pathlib import Path
import shlex
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'geometry_experiments'))
from run_padding_causal import connect
HERE=Path(__file__).resolve().parent
launch=json.loads((HERE/'launch.json').read_text());root=launch['root']
def run(c,cmd):
    _,o,e=c.exec_command(cmd,timeout=40);out=o.read().decode(errors='replace');err=e.read().decode(errors='replace')
    assert o.channel.recv_exit_status()==0,err[-2000:]
    return out.strip()
c=connect()
try:
    assert run(c,'cat '+root+'/verification.exit')=='0'
    assert run(c,'test ! -e '+root+'/training_launch.claim && echo absent')=='absent'
    assert run(c,'test ! -e '+launch['host_result_path']+' && echo absent')=='absent'
    assert run(c,'ps -p 352729 -o pgid=,tpgid=').split()==['352729','352729']
    gpu=run(c,'nvidia-smi -i 6,7 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits')
    assert all(int(s.split(',')[0])<100 and int(s.split(',')[1])==0 for s in gpu.splitlines())
    assert run(c,'git -C /home/mentee/code/EBM rev-parse HEAD')==launch['commit']
    assert not run(c,'git -C /home/mentee/code/EBM status --porcelain')
    script='''#!/bin/bash
set -u
ROOT="'''+root+'''"
mkdir "$ROOT/training_launch.claim" || exit 95
test -f /.dockerenv || { echo 90 > "$ROOT/run.exit"; exit 90; }
export CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd /home/mentee/code/EBM || { echo 91 > "$ROOT/run.exit"; exit 91; }
python -u "$ROOT/preflight_C_training.py" "$ROOT" > "$ROOT/preflight.json" 2> "$ROOT/preflight.error"
rc=$?
if [ "$rc" -ne 0 ]; then echo "$rc" > "$ROOT/run.exit"; exit "$rc"; fi
date -Iseconds > "$ROOT/started.txt"
python -u main.py > "$ROOT/train.log" 2>&1 &
pid=$!
echo "$pid" > "$ROOT/parent_pid.txt"
wait "$pid"
rc=$?
echo "$rc" > "$ROOT/run.exit"
date -Iseconds > "$ROOT/finished.txt"
exit "$rc"
'''
    with c.open_sftp() as s:
        s.put(str(HERE/'preflight_C_training.py'),root+'/preflight_C_training.py')
        with s.open(root+'/run.sh','w') as f:f.write(script)
    assert run(c,'ps -p 352729 -o pgid=,tpgid=').split()==['352729','352729']
    run(c,'screen -S 352153.exp -p 0 -X stuff '+shlex.quote('bash '+root+'/run.sh\n'))
    launch.update(status='training_launch_requested',training_launch_requested=True)
    (HERE/'launch.json').write_text(json.dumps(launch,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(launch,indent=2))
finally:c.close()
