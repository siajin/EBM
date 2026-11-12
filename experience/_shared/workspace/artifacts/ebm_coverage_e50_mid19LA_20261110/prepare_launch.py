"""Freeze and launch the authorized train -> fixed50 paired comparison once."""
import hashlib,json,pathlib,shlex,sys
import paramiko

HERE=pathlib.Path(__file__).resolve().parent
WORKSPACE=HERE.parents[1]
ROOT='/home/mentee/analysis/ebm_coverage_e50_20261110_v2_mid19LA'
NAME='EBM_e50_bs240_lr1e5_OC_BPL_coverage_activity_20261110_mid19LA'
assert not (HERE/'state.json').exists(),'Inspect existing state; do not launch twice'
client=paramiko.SSHClient();client.load_system_host_keys()
client.connect('172.16.162.222',username='mentee',
    password=pathlib.Path('C:/Users/sihwa/.ssh/mentee-password.txt').read_text().splitlines()[0],
    look_for_keys=False,allow_agent=False,timeout=12)


def run(command):
    _,out,err=client.exec_command(command)
    body=out.read().decode();error=err.read().decode()
    assert out.channel.recv_exit_status()==0,error[-1500:]
    return body.strip()


try:
    if '--prepare-only' not in sys.argv:assert run('ps -p 352729 -o pgid=,tpgid=').split()==['352729','352729']
    run('test -f /proc/352729/root/.dockerenv')
    assert not run('git -C /home/mentee/code/EBM status --porcelain')
    commit=run('git -C /home/mentee/code/EBM rev-parse HEAD')
    assert commit=='cf199af03044c40837c08ce31d633e2c048bde08'
    gpu=run('nvidia-smi -i 6,7 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits')
    if '--prepare-only' not in sys.argv:assert all(int(row.split(',')[1])<100 and int(row.split(',')[2])==0 for row in gpu.splitlines())
    run('test ! -e /proc/352729/root/results/CM/LA/'+NAME)
    run('mkdir '+ROOT)
    previous='/home/mentee/analysis/ebm_source_coverage_20261110_v1'
    run('cp '+previous+'/duration_policy.json '+previous+'/activity_policy.json '+previous+'/activity_sampling.npz '+ROOT+'/')
    manifest=json.loads((WORKSPACE/'artifacts/source_coverage_20261110/source_manifest.json').read_text())
    for name in ['tools/train_coverage_experiment.py','tools/evaluate_coverage_comparison.py',
                 'tools/check_coverage_training.py','tools/evaluate_B_C_checkpoints.py','tests/test_intermediate_19la.py']:
        manifest[name]=''
    with client.open_sftp() as transfer:
        for name in manifest:
            path=WORKSPACE/'EBM'/name;destination=ROOT+'/code/'+name
            parts=pathlib.PurePosixPath(destination).parent.parts;current=''
            for part in parts:
                current=current+'/'+part if current else part
                try:transfer.mkdir(current)
                except IOError:pass
            transfer.put(str(path),destination)
            manifest[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    checkpoint=json.loads((WORKSPACE/'artifacts/speed_duration_feasibility_20261109/probe/protocol.json').read_text())['checkpoints']
    config={'experiment_name':NAME,'result_path':'/results/CM/LA/'+NAME,
        'host_result_path':'/home/mentee/results/CM/LA/'+NAME,'root':ROOT,
        'live_commit':commit,'source_sha256':manifest,'source_repo_branch':'experiment/positive-bpl-native-pairs-20261110',
        'legacy_checkpoint':checkpoint['legacy240']['path'],'legacy_checkpoint_sha256':checkpoint['legacy240']['sha256'],
        'smoke_checkpoint':checkpoint['C']['path'],'smoke_checkpoint_sha256':checkpoint['C']['sha256'],
        'frontend_revision':'1a640f32ac3e39899438a2931f9924c02f080a54',
        'duration_policy_sha256':'06a03d3d568cbecba375a225e976117b26862c673cce8acab4da888dccbccfd4',
        'activity_policy_sha256':'0c0064b434fd6011b5c6ff5ba9a419c1fe769b9015fc7e59a6ccbeb126074a53',
        'train_port':29695,'smoke_port':29694,'intermediate_evaluation_phase':'19LA','intermediate_evaluation_every_epochs':5,'user_authorized_restart':True,'supersedes_root':'/home/mentee/analysis/ebm_coverage_e50_20261110_v1','training_epochs':50,'training_seed':1,
        'OC_global_batch_size':240,'OC_batch_size_per_rank':120,'extra_BPL_per_rank':12,
        'source_sampling':'coverage','steps_per_epoch':108,'evaluation_seed':1729,
        'evaluation_phases':{'dev':24844,'19LA':71237,'21LA':148176,'21DF':533928},
        'model_selection':'fixed50 for both; no target selection',
        'CUDA_VISIBLE_DEVICES':'0,1','host_gpu_readback_at_preparation':gpu}
    script='''#!/bin/bash
set -u
ROOT=ROOT_VALUE
mkdir "$ROOT/pipeline.claim" || exit 95
test -f /.dockerenv || exit 90
export CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=1 EBM_BASELINE_ROOT=/home/mentee/code/EBM
export PYTHONPATH="$ROOT/code"
cd "$ROOT/code" || exit 91
step() {
 stage="$1"; shift
 echo "$stage" > "$ROOT/stage.txt"
 "$@" > "$ROOT/$stage.log" 2>&1 &
 pid=$!; echo "$pid" > "$ROOT/$stage.pid"
 wait "$pid"; rc=$?
 echo "$rc" > "$ROOT/$stage.exit"
 if [ "$rc" -ne 0 ]; then echo "$rc" > "$ROOT/pipeline.exit"; date -Iseconds > "$ROOT/failed.txt"; exit "$rc"; fi
}
date -Iseconds > "$ROOT/preparation_started.txt"
step unit python -m unittest discover -s tests -p 'test_*.py' -v
step preflight python -u tools/check_coverage_training.py "$ROOT"
step train_gpu_smoke python -u tools/train_coverage_experiment.py "$ROOT" --smoke
step eval_smoke python -u tools/evaluate_coverage_comparison.py "$ROOT" --smoke
date -Iseconds > "$ROOT/training_started.txt"
step train python -u tools/train_coverage_experiment.py "$ROOT"
date -Iseconds > "$ROOT/training_finished.txt"
step training_audit python -u tools/check_coverage_training.py "$ROOT" --final
step evaluation python -u tools/evaluate_coverage_comparison.py "$ROOT"
echo complete > "$ROOT/stage.txt"
echo 0 > "$ROOT/pipeline.exit"
date -Iseconds > "$ROOT/finished.txt"
'''.replace('ROOT_VALUE',shlex.quote(ROOT))
    for name,value in [('experiment.json',json.dumps(config,indent=2)),('pipeline.sh',script)]:
        (HERE/name).write_bytes(value.encode('utf-8'))
        with client.open_sftp() as transfer:transfer.put(str(HERE/name),ROOT+'/'+name)
    state={'status':'prepared','root':ROOT,'experiment_name':NAME,'training_complete':False,
        'evaluation_complete':False,'comparison_complete':False,'live_server_repo_changed':False,
        'regular_schedule_reenabled':False,'pipeline_queued':True,'configuration':config}
    (HERE/'state.json').write_text(json.dumps(state,indent=2))
    if '--prepare-only' in sys.argv:
        print(json.dumps({'status':'prepared_not_launched','root':ROOT,'experiment_name':NAME},indent=2))
        sys.exit(0)
    assert run('ps -p 352729 -o pgid=,tpgid=').split()==['352729','352729']
    gpu=run('nvidia-smi -i 6,7 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits')
    assert all(int(row.split(',')[0])<100 and int(row.split(',')[1])==0 for row in gpu.splitlines())
    run('screen -S 352153.exp -p 0 -X stuff '+shlex.quote('bash '+ROOT+'/pipeline.sh'+chr(13)))
    state['status']='preflight_and_smokes_running'
    (HERE/'state.json').write_text(json.dumps(state,indent=2))
    print(json.dumps({k:state[k] for k in ['status','root','experiment_name','pipeline_queued']},indent=2))
finally:client.close()
