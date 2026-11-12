"""Launch the explicitly requested B training once in the existing container."""
import json
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'geometry_experiments'))
from run_padding_causal import connect

branch='train/ebm-B-zeropad-e50-bs240-20261106'
name='EBM_e50_bs240_lr1e5_B_zeropad_short303'
local=Path.cwd()/'EBM'
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=local,text=True).strip()
assert subprocess.check_output(['git','branch','--show-current'],cwd=local,text=True).strip()==branch
assert not subprocess.check_output(['git','status','--porcelain'],cwd=local,text=True).strip()
assert subprocess.check_output(['git','rev-parse','@{upstream}'],cwd=local,text=True).strip()==commit

def run(c,cmd):
    _,o,e=c.exec_command(cmd,timeout=40)
    text=o.read().decode();err=e.read().decode()
    assert o.channel.recv_exit_status()==0,err[-1500:]
    return text.strip()

c=connect()
try:
    assert run(c,'ps -p 352729 -o pgid=,tpgid=').split()==['352729','352729']
    assert run(c,'test -f /proc/352729/root/.dockerenv && echo yes')=='yes'
    gpu=run(c,'nvidia-smi -i 6,7 --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits')
    assert all(int(row.split(',')[1])<100 and int(row.split(',')[2])==0 for row in gpu.splitlines())
    assert not run(c,'git -C /home/mentee/code/EBM status --porcelain')
    assert run(c,'git -C /home/mentee/code/EBM branch --show-current')=='baseline/e50-bs240-lr1e5'
    result='/home/mentee/results/CM/LA/'+name
    assert run(c,'test ! -e '+shlex.quote(result)+' && echo absent')=='absent'
    run(c,'git -C /home/mentee/code/EBM fetch origin')
    run(c,'git -C /home/mentee/code/EBM switch --track origin/'+branch)
    assert run(c,'git -C /home/mentee/code/EBM rev-parse HEAD')==commit
    root='/home/mentee/analysis/ebm_train_B_'+run(c,'date +%Y%m%d_%H%M%S')
    run(c,'mkdir '+shlex.quote(root))
    launch={'status':'launch_requested','branch':branch,'commit':commit,'condition':'B',
            'experiment_name':name,'root':root,'host_result_path':result,
            'host_gpu_readback_before_launch':gpu,'existing_screen':'352153.exp:0',
            'CUDA_VISIBLE_DEVICES':'0,1','baseline_training_started':False}
    script='''#!/bin/bash
set -u
ROOT="'''+root+'''"
mkdir "$ROOT/launch.claim" || exit 95
test -f /.dockerenv || { echo 90 > "$ROOT/run.exit"; exit 90; }
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd /home/mentee/code/EBM || { echo 91 > "$ROOT/run.exit"; exit 91; }
python -u "$ROOT/preflight_B_training.py" "$ROOT" > "$ROOT/preflight.json" 2> "$ROOT/preflight.error"
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
        s.put(str(Path(__file__).with_name('preflight_B_training.py')),root+'/preflight_B_training.py')
        for file,text in [('launch.json',json.dumps(launch,indent=2)+'\n'),('run.sh',script)]:
            with s.open(root+'/'+file,'w') as f:f.write(text)
    # Recheck immediately before delivering one command to the idle container shell.
    assert run(c,'ps -p 352729 -o pgid=,tpgid=').split()==['352729','352729']
    readback=run(c,'nvidia-smi -i 6,7 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits')
    assert all(int(row.split(',')[0])<100 and int(row.split(',')[1])==0 for row in readback.splitlines())
    run(c,'screen -S 352153.exp -p 0 -X stuff '+shlex.quote('bash '+root+'/run.sh\n'))
    Path(__file__).with_name('B_training_launch.json').write_text(json.dumps(launch,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(launch),flush=True)
finally:c.close()
