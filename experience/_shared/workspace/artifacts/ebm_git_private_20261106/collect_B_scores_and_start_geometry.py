"""Retrieve only derived evaluation products and launch the bounded diagnostic."""
import hashlib
import json
from pathlib import Path
import shlex
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'geometry_experiments'))
from run_padding_causal import connect

folder=Path(__file__).resolve().parent
statefile=folder/'B_final_evaluation_state.json'
state=json.loads(statefile.read_text());root=state['root']
out=Path.cwd()/'artifacts/geometry_cache/local_analysis/B_final_analysis_20261108_v1'
out.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(c,cmd):
    _,o,e=c.exec_command(cmd,timeout=45);s=o.read().decode(errors='replace');err=e.read().decode(errors='replace')
    assert o.channel.recv_exit_status()==0,err[-1500:]
    return s.strip()

c=connect()
try:
    assert run(c,'cat '+root+'/run.exit')=='0','Full evaluation must finish first'
    script=r'''
import hashlib,json
from pathlib import Path
r=Path(ROOT)
files=['metrics.json','complete.json','training_audit.json','training_history.json','training_loss_identity_audit.json','QC_rank0_full.json','QC_rank1_full.json']+[f'paired_scores_{p}.parquet' for p in ['dev','19LA','21LA','21DF']]
d={'root':str(r),'files':{n:{'bytes':(r/n).stat().st_size,'sha256':hashlib.sha256((r/n).read_bytes()).hexdigest()} for n in files}}
print(json.dumps(d))
'''.replace('ROOT',repr(root))
    manifest=json.loads(run(c,"python - <<'PY'\n"+script+"\nPY"))
    with c.open_sftp() as s:
        for name,meta in manifest['files'].items():
            target=out/name
            if target.exists() and sha(target)==meta['sha256']:continue
            temporary=target.with_suffix(target.suffix+'.download')
            s.get(root+'/'+name,str(temporary))
            assert sha(temporary)==meta['sha256'],name
            temporary.replace(target)
    (out/'download_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    state.update(evaluation_complete=True,local_score_folder=str(out),stage='score_analysis_and_geometry_preparing')
    statefile.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    if state.get('geometry_launch_requested'):
        print('Derived files collected; geometry launch already recorded. Do not duplicate.');sys.exit(0)
    assert run(c,'ps -p 352729 -o pgid=,tpgid=').split()==['352729','352729']
    assert run(c,'test -f /proc/352729/root/.dockerenv && echo yes')=='yes'
    gpu=run(c,'nvidia-smi -i 6,7 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits')
    assert all(int(x.split(',')[0])<100 and int(x.split(',')[1])==0 for x in gpu.splitlines())
    assert run(c,'test ! -e '+root+'/mechanism_2x2 && echo absent')=='absent'
    script='''#!/bin/bash
set -u
ROOT="'''+root+'''"
mkdir "$ROOT/geometry_launch.claim" || exit 95
test -f /.dockerenv || exit 90
export CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd /home/mentee/code/EBM || exit 91
python -u "$ROOT/geometry_B_intervention.py" lock "$ROOT" > "$ROOT/geometry_lock.log" 2>&1
rc=$?; if [ "$rc" -ne 0 ]; then echo "$rc" > "$ROOT/geometry.exit"; exit "$rc"; fi
python -u "$ROOT/geometry_B_intervention.py" run "$ROOT" > "$ROOT/geometry.log" 2>&1
rc=$?; echo "$rc" > "$ROOT/geometry.exit"
exit "$rc"
'''
    with c.open_sftp() as s:
        s.put(str(folder/'geometry_B_intervention.py'),root+'/geometry_B_intervention.py')
        with s.open(root+'/run_geometry.sh','w') as f:f.write(script)
    assert run(c,'ps -p 352729 -o pgid=,tpgid=').split()==['352729','352729']
    run(c,'screen -S 352153.exp -p 0 -X stuff '+shlex.quote('bash '+root+'/run_geometry.sh\n'))
    state.update(stage='paired_score_analysis_and_geometry_running',geometry_launch_requested=True,geometry_root=root+'/mechanism_2x2')
    statefile.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'scores_downloaded':True,'folder':str(out),'files':len(manifest['files']),'geometry_launch_requested':True,'root':root}),flush=True)
finally:c.close()
