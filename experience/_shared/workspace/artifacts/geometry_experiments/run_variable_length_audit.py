"""Launch one isolated feasibility probe or retrieve only derived diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import time
from run_padding_causal import connect,run
from audit_variable_length_metadata import OUT

REMOTE='/home/mentee/analysis/embedding_geometry/20261106_variable_length_01'


def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def setup():
    assert not (OUT/'launched.json').exists()
    c=connect()
    try:
        status=run(c,'date -Is; screen -S exp -Q windows; ps -p 352729 -o pgid=,tpgid=')
        assert run(c,'ps -p 352729 -o pgid=,tpgid=').split()==['352729','352729']
        assert run(c,'test -f /proc/352729/root/.dockerenv && echo yes').strip()=='yes'
        gpu=run(c,'nvidia-smi -i 6,7 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits')
        assert all(int(q.split(',')[0])<100 and int(q.split(',')[1])==0 for q in gpu.splitlines())
        run(c,'mkdir '+shlex.quote(REMOTE))
        with c.open_sftp() as s:
            for name in ['probe_selection.parquet','probe_selection_lock.json']:s.put(str(OUT/name),REMOTE+'/'+name)
            s.put(str(Path(__file__).with_name('variable_length_sidecar.py')),REMOTE+'/variable_length_sidecar_v1.py')
            script='''#!/bin/bash
set -u
test -f /.dockerenv || exit 90
export CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
ROOT="'''+REMOTE+'''"
cd /home/mentee/code/EBM || exit 91
for stage in prepare probe; do
 python -u "$ROOT/variable_length_sidecar_v1.py" "$stage" --root "$ROOT" > "$ROOT/$stage.log" 2>&1
 rc=$?; echo "$rc" > "$ROOT/$stage.exit"
 if [ "$rc" -ne 0 ]; then echo "$rc" > "$ROOT/run.exit"; exit "$rc"; fi
done
echo 0 > "$ROOT/run.exit"
'''
            with s.open(REMOTE+'/run.sh','w') as f:f.write(script)
        assert run(c,'ps -p 352729 -o pgid=,tpgid=').split()==['352729','352729']
        run(c,'screen -S 352153.exp -p 0 -X stuff '+shlex.quote('bash '+REMOTE+'/run.sh\n'))
        save(OUT/'launched.json',dict(epoch=time.time(),remote=REMOTE,status_before=status,gpu_before=gpu))
        print('Launched isolated header/mask/gradient probe in existing container on CUDA_VISIBLE_DEVICES=0,1')
    finally:c.close()


def status(sync=False):
    result={};c=connect()
    try:
        with c.open_sftp() as s:
            for name in ['prepare.exit','probe.exit','run.exit','prepare_complete.json','rank0_progress.json','rank1_progress.json','complete.json']:
                try:
                    with s.open(REMOTE+'/'+name) as f:q=f.read().decode()
                    result[name]=json.loads(q) if name.endswith('.json') else q.strip()
                except FileNotFoundError:pass
            if result.get('run.exit') not in [None,'0']:
                for name in ['prepare.log','probe.log']:
                    try:
                        with s.open(REMOTE+'/'+name) as f:result[name]=f.read().decode()[-5500:]
                    except FileNotFoundError:pass
            if sync:
                assert result.get('run.exit')=='0'
                for name,sha in result['complete.json']['files'].items():
                    target=OUT/name
                    if not target.exists():s.get(REMOTE+'/'+name,str(target))
                    assert hashlib.sha256(target.read_bytes()).hexdigest()==sha
                if not (OUT/'complete.json').exists():s.get(REMOTE+'/complete.json',str(OUT/'complete.json'))
                result['sync']='derived JSON diagnostics only, SHA256 verified'
        save(OUT/'last_status.json',result)
        # Avoid repeating28 archive hashes in routine progress.
        compact={k:({j:v for j,v in q.items() if not j.endswith('hashes')} if isinstance(q,dict) else q) for k,q in result.items()}
        print(json.dumps(compact,ensure_ascii=False,indent=2))
    finally:c.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['setup','status','sync']);a=p.parse_args()
    setup() if a.action=='setup' else status(a.action=='sync')
