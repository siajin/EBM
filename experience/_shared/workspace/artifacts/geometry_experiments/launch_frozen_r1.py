"""One-time submission into the existing idle container shell; never restart."""
from pathlib import Path
import shlex
import paramiko

ROOT='/home/mentee/analysis/embedding_geometry/20261105_frozen_all_01'
def main():
    c=paramiko.SSHClient();c.load_system_host_keys()
    try:
        c.connect('172.16.162.222',username='mentee',password=Path('C:/Users/sihwa/.ssh/mentee-password.txt').read_text().splitlines()[0],look_for_keys=False,allow_agent=False,timeout=15)
        def run(cmd):
            _,out,err=c.exec_command(cmd);a=out.read().decode();b=err.read().decode();code=out.channel.recv_exit_status()
            if code: raise RuntimeError((cmd,a,b,code))
            return a
        # Current shell must be in the known container and in the foreground.
        status=run('ps -p 352729 -o pgid=,tpgid=; readlink /proc/352729/ns/mnt; screen -S 352153.exp -Q windows')
        assert status.splitlines()[0].split()==['352729','352729'],status
        assert '0' in status and 'bash' in status
        assert run('test -f /proc/352729/root/.dockerenv && echo yes').strip()=='yes'
        gpu=run('nvidia-smi -i 6,7 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits')
        assert all(int(line.split(',')[0])<100 and int(line.split(',')[1])==0 for line in gpu.splitlines()),gpu
        run('mkdir '+shlex.quote(ROOT))  # Fail if the run already exists.
        with c.open_sftp() as s:
            s.put(str(Path(__file__).with_name('frozen_r1_extract.py')),ROOT+'/frozen_r1_extract_v1.py')
            launch='''#!/bin/bash
set -u
test -f /.dockerenv || exit 90
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
ROOT="'''+ROOT+'''"
cd /home/mentee/code/EBM || exit 91
for dataset in Q 19LA_eval DF_remaining; do
  folder="$ROOT/$dataset"
  mkdir "$folder" || exit 92
  for stage in prepare qc extract; do
    python -u "$ROOT/frozen_r1_extract_v1.py" "$stage" --dataset "$dataset" --root "$folder" > "$folder/$stage.log" 2>&1
    rc=$?
    echo "$rc" > "$folder/$stage.exit"
    if [ "$rc" -ne 0 ]; then echo "$rc" > "$ROOT/run.exit"; exit "$rc"; fi
  done
done
echo 0 > "$ROOT/run.exit"
'''
            with s.open(ROOT+'/run_r1.sh','w') as f:f.write(launch)
        # Check foreground again immediately before the only screen input.
        assert run('ps -p 352729 -o pgid=,tpgid=').split()==['352729','352729']
        command='bash '+ROOT+'/run_r1.sh\n'
        run('screen -S 352153.exp -p 0 -X stuff '+shlex.quote(command))
        print('Submitted new R1 run:',ROOT)
    finally:c.close()
if __name__=='__main__':main()
