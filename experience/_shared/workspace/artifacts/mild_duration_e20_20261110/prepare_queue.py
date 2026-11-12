"""Freeze the authorized20epoch specification and queue after existing DF work."""
import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import sys
import paramiko

HERE=Path(__file__).resolve().parent
WORKSPACE=HERE.parents[1]
ROOT='/home/mentee/analysis/ebm_mild_duration_e20_20261110_v1'
NAME='EBM_e20_bs240_lr1e5_OC_BPL_speed08_12_crop20_20261110'
PRIOR='/home/mentee/analysis/ebm_coverage_e20_eval_20261110_v1'
assert not (HERE/'state.json').exists(),'Inspect existing registration; never queue twice'
client=paramiko.SSHClient();client.load_system_host_keys()
client.connect('172.16.162.222',username='mentee',
    password=Path('C:/Users/sihwa/.ssh/mentee-password.txt').read_text().splitlines()[0],
    look_for_keys=False,allow_agent=False,timeout=12)


def run(command):
    _,out,err=client.exec_command(command)
    body=out.read().decode();error=err.read().decode()
    assert out.channel.recv_exit_status()==0,error[-1500:]
    return body.strip()


def transfer_file(path,destination):
    with client.open_sftp() as transfer:
        current=''
        for part in PurePosixPath(destination).parent.parts:
            current=current+'/'+part if current else part
            try: transfer.mkdir(current)
            except IOError: pass
        transfer.put(str(path),destination)


try:
    run('test -f /proc/352729/root/.dockerenv')
    assert '352153.exp' in run('screen -ls')
    assert not run('git -C /home/mentee/code/EBM status --porcelain')
    commit=run('git -C /home/mentee/code/EBM rev-parse HEAD')
    assert commit=='cf199af03044c40837c08ce31d633e2c048bde08'
    start=run("awk '{print $22}' /proc/352729/stat")
    run('test ! -e '+ROOT)
    run('test ! -e /proc/352729/root/results/CM/LA/'+NAME)
    old=json.loads((WORKSPACE/'artifacts/ebm_coverage_e20_eval_20261110/evaluation_config.json').read_text())
    manifest=dict(old['source_sha256'])
    for name in ['mild_duration.py','tests/test_mild_duration.py','tools/build_mild_duration_policy.py',
        'tools/prepare_mild_audio.py','tools/train_mild_duration.py','tools/check_mild_training.py',
        'tools/finish_mild_experiment.py','tools/queue_mild_experiment.py',
        'tools/evaluate_coverage_checkpoint.py','tools/compare_coverage_epoch20.py']:
        manifest[name]=''
    run('mkdir '+ROOT)
    for name in manifest:
        path=WORKSPACE/'EBM'/name
        transfer_file(path,ROOT+'/code/'+name)
        manifest[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    policy=HERE/'duration_policy.json'
    transfer_file(policy,ROOT+'/duration_policy.json')
    checkpoints=json.loads((WORKSPACE/'artifacts/speed_duration_feasibility_20261109/probe/protocol.json').read_text())['checkpoints']
    checkpoints['stopped_coverage20']={'path':old['checkpoint'],'sha256':old['checkpoint_sha256']}
    config={'experiment_name':NAME,'root':ROOT,'result_path':'/results/CM/LA/'+NAME,
        'host_result_path':'/home/mentee/results/CM/LA/'+NAME,'source_sha256':manifest,
        'live_commit':commit,'source_repo_branch':'experiment/positive-bpl-native-pairs-20261110',
        'protected_checkpoints':checkpoints,'duration_policy_sha256':hashlib.sha256(policy.read_bytes()).hexdigest(),
        'frontend_revision':old['frontend_revision'],'training_epochs':20,'training_seed':1,
        'source_sampling':'uniform_coverage','source_originals':{'bonafide':2580,'spoof':22800},
        'OC_global_batch':240,'OC_per_rank':120,'BPL_extra_per_rank':12,'BPL_pairs_per_rank':144,
        'BPL_pairing':'matched_independent','steps_per_epoch':108,'logical_training_steps':2160,
        'objective':'original OC+BPL; backend then recomputed OC-center Adam updates',
        'lr':1e-5,'weight_decay':.0001,'micro_batch_size':8,'RawBoost':'original types0..8; type8 LnL+ISD',
        'codec':False,'VAD_matching':False,'silence_removal':False,'old_short_duration_policy':False,
        'speed_bounds':[.8,1.2],'crop_fraction_bounds':[0,.2],'pitch_preserving_transform':'SoX tempo -s',
        'source_uid_weights':'uniform within class, unchanged expected speaker/attack proportions',
        'target_choice':'independent random conditional target on each visit; fixed seeded crop per variant',
        'evaluation_input':'front64600 cap; no repeat; masks; batch8/rank; trained OC cosine; original ASPnoise',
        'evaluation_seed':1729,'intermediate_evaluation_phase':'19LA','intermediate_evaluation_every_epochs':5,
        'final_evaluation_phases':{'19LA':71237,'21LA':148176,'21DF':533928},
        'model_selection':'fixed20 chosen before this run; no dev, no best-target selection',
        'train_port':29719,'smoke_port':29718,'CUDA_VISIBLE_DEVICES':'0,1',
        'wait_for_evaluation_root':PRIOR,'container_shell_start_ticks':start,
        'new_paid_resource_cost_USD':0,'new_spending_ceiling_USD':0,'automatic_retries':0,
        'queue_wait_ceiling_hours':8,'pipeline_wall_ceiling_hours':48,'generated_cache_ceiling_bytes':20*1024**3,
        'cleanup':'Watcher exits after launch; own stage process groups bounded; temp files only in unique new root. '
                  'No resource rental. Preserve cache, checkpoints, logs, frozen sources and existing results.',
        'registered_KST':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()}
    pipeline='''#!/bin/bash
set -u
ROOT=ROOT_VALUE
mkdir "$ROOT/pipeline.claim" || exit 95
test -f /.dockerenv || exit 90
export CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=1 EBM_BASELINE_ROOT=/home/mentee/code/EBM
export PYTHONPATH="$ROOT/code"
cd "$ROOT/code" || exit 91
pid=''
cleanup() {
 rc="$1"
 trap - TERM INT
 if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
  pgid=$(ps -p "$pid" -o pgid= | tr -d ' ')
  if [ "$pgid" = "$pid" ]; then
   kill -TERM -- "-$pid" 2>/dev/null || true
   for tick in $(seq 1 15); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
   if kill -0 "$pid" 2>/dev/null; then kill -KILL -- "-$pid" 2>/dev/null || true; fi
  fi
 fi
 echo "$rc" > "$ROOT/pipeline.exit"
 date -Iseconds > "$ROOT/failed.txt"
 exit "$rc"
}
trap 'cleanup 143' TERM
trap 'cleanup 130' INT
step() {
 stage="$1"; bound="$2"; shift 2
 echo "$stage" > "$ROOT/stage.txt"
 timeout --signal=TERM --kill-after=60s "$bound" "$@" > "$ROOT/$stage.log" 2>&1 &
 pid=$!; echo "$pid" > "$ROOT/$stage.pid"
 wait "$pid"; rc=$?; pid=''
 echo "$rc" > "$ROOT/$stage.exit"
 if [ "$rc" -ne 0 ]; then cleanup "$rc"; fi
}
date -Iseconds > "$ROOT/preparation_started.txt"
step unit 15m python -m unittest discover -s tests -p 'test_*.py' -v
step cache 4h python -u tools/prepare_mild_audio.py "$ROOT"
step preflight 45m python -u tools/check_mild_training.py "$ROOT"
step train_gpu_smoke 30m python -u tools/train_mild_duration.py "$ROOT" --smoke
date -Iseconds > "$ROOT/training_started.txt"
step train 32h python -u tools/train_mild_duration.py "$ROOT"
date -Iseconds > "$ROOT/training_finished.txt"
step training_audit 15m python -u tools/check_mild_training.py "$ROOT" --final
step evaluation_config 5m python -u tools/finish_mild_experiment.py "$ROOT" --configure
step evaluation_smoke 15m python -u tools/evaluate_coverage_checkpoint.py "$ROOT" --smoke
step evaluation 8h python -u tools/evaluate_coverage_checkpoint.py "$ROOT"
step comparison 30m python -u tools/finish_mild_experiment.py "$ROOT"
echo complete > "$ROOT/stage.txt"
echo 0 > "$ROOT/pipeline.exit"
date -Iseconds > "$ROOT/finished.txt"
'''.replace('ROOT_VALUE',shlex.quote(ROOT))
    for name,value in [('experiment.json',json.dumps(config,indent=2)+'\n'),('pipeline.sh',pipeline)]:
        (HERE/name).write_bytes(value.encode())
        transfer_file(HERE/name,ROOT+'/'+name)
    # Verify every frozen file remotely without touching any active source/job.
    verify="import hashlib,json,pathlib; r=pathlib.Path("+repr(ROOT)+"); c=json.loads((r/'experiment.json').read_text()); assert all(hashlib.sha256((r/'code'/n).read_bytes()).hexdigest()==s for n,s in c['source_sha256'].items()); print(len(c['source_sha256']))"
    assert int(run('/usr/bin/python3 -c '+shlex.quote(verify)))==len(manifest)
    run('bash -n '+ROOT+'/pipeline.sh')
    state={'status':'prepared_not_queued','root':ROOT,'experiment_name':NAME,'training_started':False,
        'training_complete':False,'evaluation_complete':False,'comparison_complete':False,
        'configuration':config,'regular_schedule_reenabled':False}
    (HERE/'state.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
    if '--prepare-only' in sys.argv:
        print(json.dumps({'status':state['status'],'source_files':len(manifest),'root':ROOT}));sys.exit(0)
    launch="""import json,pathlib,subprocess,time
r=pathlib.Path(ROOT_VALUE)
assert not (r/'queue.claim').exists()
stream=(r/'queue.log').open('xb')
p=subprocess.Popen(['/usr/bin/python3','-u',str(r/'code/tools/queue_mild_experiment.py'),str(r)],stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
(r/'queue_host.pid').write_text(str(p.pid)+'\\n')
print(json.dumps({'queue_host_pid':p.pid,'registered_unix':time.time()}))
""".replace('ROOT_VALUE',repr(ROOT))
    receipt=json.loads(run('/usr/bin/python3 -c '+shlex.quote(launch)))
    (HERE/'queue_receipt.json').write_text(json.dumps(receipt,indent=2))
    state.update(status='queued_after_existing_DF_and_analysis',queue_host_pid=receipt['queue_host_pid'])
    (HERE/'state.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
    print(json.dumps({'status':state['status'],'root':ROOT,**receipt},indent=2))
finally: client.close()
