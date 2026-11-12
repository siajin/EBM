"""Deploy the isolated C branch and run bounded validation in the idle container."""
import json
from pathlib import Path
import shlex
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'geometry_experiments'))
from run_padding_causal import connect

LOCAL=Path(__file__).resolve().parent
CODE=Path.cwd()/'EBM'
BRANCH='train/ebm-C-native-long-e50-bs240-20261108'
BASE='2c2c6ff65bbadf33b811983f7c648c97fffb81a1'
NAME='EBM_e50_bs240_lr1e5_C_nativeLong_short303'
def run(c,cmd):
    _,o,e=c.exec_command(cmd,timeout=40);s=o.read().decode(errors='replace');err=e.read().decode(errors='replace')
    assert o.channel.recv_exit_status()==0,err[-2000:]
    return s.strip()

assert not subprocess.check_output(['git','status','--porcelain'],cwd=CODE,text=True).strip()
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip()
assert subprocess.check_output(['git','rev-parse','@{upstream}'],cwd=CODE,text=True).strip()==commit
assert not (LOCAL/'launch.json').exists(),'Do not duplicate deployment/validation'
c=connect()
try:
    assert run(c,'ps -p 352729 -o pgid=,tpgid=').split()==['352729','352729']
    assert run(c,'test -f /proc/352729/root/.dockerenv && echo yes')=='yes'
    gpu=run(c,'nvidia-smi -i 6,7 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits')
    assert all(int(s.split(',')[0])<100 and int(s.split(',')[1])==0 for s in gpu.splitlines())
    assert not run(c,'git -C /home/mentee/code/EBM status --porcelain')
    assert run(c,'git -C /home/mentee/code/EBM rev-parse HEAD')==BASE
    assert run(c,'test ! -e /home/mentee/results/CM/LA/'+NAME+' && echo absent')=='absent'
    run(c,'git -C /home/mentee/code/EBM fetch origin '+BRANCH)
    run(c,'git -C /home/mentee/code/EBM switch --track origin/'+BRANCH)
    assert run(c,'git -C /home/mentee/code/EBM rev-parse HEAD')==commit
    root='/home/mentee/analysis/ebm_train_C_'+run(c,'date +%Y%m%d_%H%M%S')
    run(c,'mkdir '+shlex.quote(root))
    launch={'status':'bounded_verification_requested','branch':BRANCH,'commit':commit,'B_parent':BASE,'experiment_name':NAME,
            'root':root,'host_result_path':'/home/mentee/results/CM/LA/'+NAME,'CUDA_VISIBLE_DEVICES':'0,1',
            'screen':'352153.exp:0','training_started':False,'B_preserved':True}
    script='''#!/bin/bash
set -u
ROOT="'''+root+'''"
mkdir "$ROOT/verification.claim" || exit 95
test -f /.dockerenv || exit 90
export CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd /home/mentee/code/EBM || exit 91
python -u "$ROOT/verify_C.py" "$ROOT" > "$ROOT/verification.log" 2>&1
rc=$?
echo "$rc" > "$ROOT/verification.exit"
exit "$rc"
'''
    with c.open_sftp() as s:
        s.put(str(LOCAL/'verify_C.py'),root+'/verify_C.py')
        for name,text in [('launch.json',json.dumps(launch,indent=2)+'\n'),('run_verification.sh',script)]:
            with s.open(root+'/'+name,'w') as f:f.write(text)
    assert run(c,'ps -p 352729 -o pgid=,tpgid=').split()==['352729','352729']
    (LOCAL/'launch.json').write_text(json.dumps(launch,indent=2)+'\n',encoding='utf-8')
    run(c,'screen -S 352153.exp -p 0 -X stuff '+shlex.quote('bash '+root+'/run_verification.sh\n'))
    print(json.dumps(launch,indent=2))
finally:c.close()
