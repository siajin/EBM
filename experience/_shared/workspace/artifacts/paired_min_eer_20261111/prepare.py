"""Isolate and queue only the newly requested post-hoc evaluation; never edit active code."""
import hashlib
import json
from pathlib import Path
import shlex

import paramiko

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
PLAN = json.loads((HERE/'plan.json').read_text())
ROOT = PLAN['root']


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value): Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def run(client, code):
    _, stdout, stderr = client.exec_command('python -c '+shlex.quote(code), timeout=90)
    text = stdout.read().decode(); error = stderr.read().decode()
    assert stdout.channel.recv_exit_status() == 0, error[-2500:]
    return json.loads(text)


def main():
    assert not (HERE/'prepared.json').exists(), 'Already prepared; use check_run.py'
    additions = ['paired_evaluation.py', 'paired_queue_guard.py', 'tools/run_paired_evaluation.py',
        'tools/queue_paired_evaluation.py', 'tools/evaluate_B_C_checkpoints.py',
        'tools/probe_repetition_shortcut.py', 'tools/review_repetition_results.py', 'tests/test_paired_evaluation.py']
    c = paramiko.SSHClient(); c.load_system_host_keys()
    c.connect('172.16.162.222', username='mentee', password=(Path.home()/'.ssh/mentee-password.txt').read_text().splitlines()[0],
              look_for_keys=False, allow_agent=False, timeout=15)
    try:
        # CPU/file operations only. Snapshot comes from the lab Git archive internally.
        observation = run(c, """
import hashlib,json,subprocess,shutil
from pathlib import Path
root=Path(ROOT); live=Path('/home/mentee/code/EBM'); active=Path(DEPENDENCY)
assert not root.exists(),'Unique new root already exists; do not overwrite'
assert Path('/proc/352729/root/.dockerenv').is_file()
assert shutil.disk_usage('/home/mentee').free > 10*1024**3
assert not subprocess.check_output(['git','-C',str(live),'status','--porcelain'],text=True)
commit=subprocess.check_output(['git','-C',str(live),'rev-parse','HEAD'],text=True).strip()
assert commit=='cf199af03044c40837c08ce31d633e2c048bde08'
active_cfg=json.loads((active/'experiment.json').read_text())
for name,h in active_cfg['source_sha256'].items(): assert hashlib.sha256((active/'code'/name).read_bytes()).hexdigest()==h,name
root.mkdir(); (root/'code').mkdir()
p=subprocess.Popen(['git','-C',str(live),'archive','HEAD'],stdout=subprocess.PIPE)
subprocess.run(['tar','-x','-C',str(root/'code')],stdin=p.stdout,check=True);p.stdout.close();assert p.wait()==0
refs={}; result=Path('/home/mentee/results/CM/LA')
for family,experiment in [('legacy240','EBM_e50_bs240_lr1e5'),('B','EBM_e50_bs240_lr1e5_B_zeropad_short303'),('C','EBM_e50_bs240_lr1e5_C_nativeLong_short303')]:
 for epoch in (range(15,51,5) if family=='legacy240' else range(5,51,5)):
  model=result/experiment/'model'; path=model/(f'BestCLS_{epoch}_21DF.pt' if family=='legacy240' else f'CLS_{epoch}.pt')
  entry=dict(family=family,epoch=epoch,path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
  if family!='legacy240':
   state=model/f'TrainState_{epoch}.pt';entry.update(TrainState_path=str(state),TrainState_sha256=hashlib.sha256(state.read_bytes()).hexdigest())
  refs[f'{family}_e{epoch}']=entry
assert len(refs)==28
print(json.dumps(dict(reference_checkpoints=refs,live_commit=commit,container_shell_start_ticks=Path('/proc/352729/stat').read_text().split()[21],active_source_verified=True)))
""".replace('ROOT', repr(ROOT)).replace('DEPENDENCY', repr(PLAN['wait_for_root'])))
        cfg = dict(PLAN, **observation)
        cfg['frontend_revision'] = PLAN['primary_input']['frontend_revision']
        selection = PROJECT/'artifacts/repetition_shortcut_20261111/probe_selected.csv'
        cfg['probe_selection_sha256'] = sha(selection)
        run_text = '''#!/bin/bash
set -u
ROOT="ROOT_VALUE"
test -f /.dockerenv || exit 90
mkdir "$ROOT/pipeline.claim" || exit 91
export CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$ROOT/code:$ROOT/code/tests"
date -Is > "$ROOT/started.txt"
echo $$ > "$ROOT/container_parent.pid"
timeout --signal=TERM --kill-after=90s 72h bash "$ROOT/pipeline.sh" > "$ROOT/pipeline.log" 2>&1
rc=$?
echo "$rc" > "$ROOT/run.exit"
date -Is > "$ROOT/finished.txt"
exit "$rc"
'''.replace('ROOT_VALUE', ROOT)
        pipeline = '''#!/bin/bash
set -u
ROOT="ROOT_VALUE"
cd "$ROOT/code" || exit 92
stage() {
 local label="$1"
 shift
 echo "$label" > "$ROOT/stage.txt"
 "$@" > "$ROOT/$label.log" 2>&1
 local rc=$?
 echo "$rc" > "$ROOT/$label.exit"
 return "$rc"
}
stage unit python -u -m unittest discover -s tests -p test_paired_evaluation.py -v || exit $?
stage freeze python -u tools/run_paired_evaluation.py "$ROOT" --mode freeze || exit $?
stage smoke python -u tools/run_paired_evaluation.py "$ROOT" --mode smoke || exit $?
stage full python -u tools/run_paired_evaluation.py "$ROOT" --mode full || exit $?
stage probe python -u tools/run_paired_evaluation.py "$ROOT" --mode probe || exit $?
stage report python -u tools/run_paired_evaluation.py "$ROOT" --mode report || exit $?
echo complete > "$ROOT/stage.txt"
'''.replace('ROOT_VALUE', ROOT)
        (HERE/'run.sh').write_text(run_text, encoding='utf-8', newline='\n')
        (HERE/'pipeline.sh').write_text(pipeline, encoding='utf-8', newline='\n')
        with c.open_sftp() as s:
            for name in additions:
                parent = str(Path(name).parent).replace('\\', '/')
                try: s.mkdir(ROOT+'/code/'+parent)
                except OSError: pass
                s.put(str(PROJECT/'EBM'/name), ROOT+'/code/'+name)
            for filename in ['run.sh', 'pipeline.sh', 'plan.json']:
                s.put(str(HERE/filename), ROOT+'/'+filename)
            s.put(str(selection), ROOT+'/probe_selected.csv')
        cfg['payload_sha256'] = {name: sha(HERE/name) for name in ['run.sh', 'pipeline.sh', 'plan.json']}
        cfg['payload_sha256']['probe_selected.csv'] = sha(selection)
        added_hashes = {name: sha(PROJECT/'EBM'/name) for name in additions}
        validation = run(c, """
import hashlib,json,subprocess
from pathlib import Path
root=Path(ROOT)
expected=json.loads(ADDED)
for name,h in expected.items(): assert hashlib.sha256((root/'code'/name).read_bytes()).hexdigest()==h,name
hashes={str(p.relative_to(root/'code')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'code').rglob('*.py')}
for p in (root/'code').rglob('*.py'):compile(p.read_bytes(),str(p),'exec')
subprocess.run(['bash','-n',str(root/'run.sh')],check=True);subprocess.run(['bash','-n',str(root/'pipeline.sh')],check=True)
print(json.dumps(dict(source_sha256=hashes,python_syntax='passed',shell_syntax='passed',source_files=len(hashes))))
""".replace('ROOT', repr(ROOT)).replace('ADDED', repr(json.dumps(added_hashes))))
        cfg['source_sha256'] = validation['source_sha256']
        save(HERE/'experiment.json', cfg); save(HERE/'preparation_validation.json', validation)
        (HERE/'experiment.sha256').write_text(sha(HERE/'experiment.json')+'\n')
        with c.open_sftp() as s:
            s.put(str(HERE/'experiment.json'), ROOT+'/experiment.json')
            s.put(str(HERE/'experiment.sha256'), ROOT+'/experiment.sha256')
        # Launch exactly one CPU watcher. It cannot touch the busy training screen.
        launch = run(c, """
import json,os,subprocess
from pathlib import Path
root=Path(ROOT); cfg=json.loads((root/'experiment.json').read_text())
assert not (root/'queue.pid').exists() and not (root/'queue.claim').exists()
environment=dict(os.environ,PYTHONPATH=str(root/'code'))
with (root/'queue.log').open('xb') as log:
 p=subprocess.Popen(['python','-u',str(root/'code/tools/queue_paired_evaluation.py'),str(root)],stdout=log,stderr=subprocess.STDOUT,env=environment,start_new_session=True)
(root/'queue.pid').write_text(str(p.pid)+'\\n')
print(json.dumps(dict(queue_pid=p.pid,status='watcher_started_no_GPU_launch',root=str(root))))
""".replace('ROOT', repr(ROOT)))
        save(HERE/'prepared.json', launch)
        save(HERE/'state.json', dict(status='waiting_for_existing_training_and_evaluation', root=ROOT,
            queue_registered=True, evaluation_started=False, **{'queue_pid': launch['queue_pid']},
            spending_ceiling_usd=0, automatic_retries=0))
        print(json.dumps(dict(**launch, source_files=validation['source_files'], reference_checkpoints=28)))
    finally: c.close()


if __name__ == '__main__': main()
