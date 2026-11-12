"""Prepare one isolated, read-only checkpoint evaluation on the existing server."""
import hashlib
import json
from pathlib import Path
import shlex
import paramiko

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
OLD = '/home/mentee/analysis/ebm_short32_original240_e50_20261111_v2'
RESULT = '/home/mentee/results/CM/LA/EBM_e50_bs240_lr1e5_original240_OC_BPL43_short32_LE2_20261111_v2'
ROOT = '/home/mentee/analysis/ebm_short32_fixed25_eval_20261111_v1'
EVALUATOR = PROJECT/'EBM/tools/evaluate_short32_fixed25.py'

client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('172.16.162.222', username='mentee',
               password=(Path.home()/'.ssh/mentee-password.txt').read_text().splitlines()[0],
               look_for_keys=False, allow_agent=False, timeout=12)
sftp = client.open_sftp()

def run(command, timeout=30):
    _, stdout, stderr = client.exec_command(command, timeout=timeout)
    output = stdout.read().decode()
    error = stderr.read().decode()
    code = stdout.channel.recv_exit_status()
    if code:
        raise RuntimeError(f'command failed ({code}): {output[-1000:]} {error[-1000:]}')
    return output

def read_json(path):
    with sftp.open(path) as stream:
        return json.load(stream)

assert run(f'cat {OLD}/run.exit').strip() == '143'
assert run("nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader -i 6,7").count('6 MiB, 0 %') == 2
assert not run(f"ps -u mentee -o args= | grep -F {shlex.quote(OLD)} | grep -v grep || true").strip()
manifest = read_json(f'{RESULT}/short32_stages/checkpoint_25.complete.json')
record = read_json(f'{RESULT}/short32_stages/eval_25_19LA.json')
experiment = read_json(f'{OLD}/experiment.json')
assert manifest['status'] == record['status'].replace('complete', 'committed')
assert record['rows'] == 71237 and record['checkpoint_sha256'] == manifest['files']['CLS_25.pt']['sha256']
for name, info in manifest['files'].items():
    path = f'{RESULT}/model/{name}'
    assert run(f'sha256sum {shlex.quote(path)}').split()[0] == info['sha256']
    assert sftp.stat(path).st_size == info['bytes']
for relative, digest in experiment['source_sha256'].items():
    assert run(f'sha256sum {shlex.quote(OLD+"/code/"+relative)}').split()[0] == digest

# A unique root makes a prior attempt impossible to overwrite or restart.
run(f'test ! -e {shlex.quote(ROOT)} && mkdir {shlex.quote(ROOT)}')
config = {
    'epoch': 25,
    'code_root': OLD+'/code',
    'source_sha256': experiment['source_sha256'],
    'checkpoint': RESULT+'/model/CLS_25.pt',
    'checkpoint_sha256': manifest['files']['CLS_25.pt']['sha256'],
    'train_state': RESULT+'/model/TrainState_25.pt',
    'train_state_sha256': manifest['files']['TrainState_25.pt']['sha256'],
    'frontend_revision': experiment['frontend_revision'],
    'reference_19LA_scores': RESULT+'/short32_stages/eval_25_19LA_scores.csv',
    'phases': {'21LA': 148176, '21DF': 533928},
    'class_counts': {'21LA': [14816, 133360], '21DF': [14869, 519059]},
    'evaluator_sha256': hashlib.sha256(EVALUATOR.read_bytes()).hexdigest(),
    'optimizer_updates': 0,
    'spending_ceiling_usd': 0,
    'automatic_retries': 0,
}
with sftp.open(ROOT+'/evaluation_config.json', 'w') as stream:
    stream.write(json.dumps(config, indent=2).encode())
sftp.put(str(EVALUATOR), ROOT+'/evaluate_short32_fixed25.py')
shell = f'''#!/usr/bin/env bash
set -uo pipefail
ROOT={shlex.quote(ROOT)}
export CUDA_VISIBLE_DEVICES=0,1
test -f /.dockerenv && test "${{CUDA_VISIBLE_DEVICES:-}}" = 0,1 || exit 90
test ! -e "$ROOT/launch.claim" || exit 91
mkdir "$ROOT/launch.claim" || exit 92
printf 'smoke\\n' > "$ROOT/stage.txt"
python "$ROOT/evaluate_short32_fixed25.py" "$ROOT" --smoke > "$ROOT/smoke.log" 2>&1
rc=$?
printf '%s\\n' "$rc" > "$ROOT/smoke.exit"
if [ "$rc" -ne 0 ]; then printf 'smoke_failed\\n' > "$ROOT/stage.txt"; printf '%s\\n' "$rc" > "$ROOT/run.exit"; exit "$rc"; fi
printf 'evaluation\\n' > "$ROOT/stage.txt"
python "$ROOT/evaluate_short32_fixed25.py" "$ROOT" > "$ROOT/evaluation.log" 2>&1
rc=$?
printf '%s\\n' "$rc" > "$ROOT/evaluation.exit"
if [ "$rc" -eq 0 ]; then printf 'complete\\n' > "$ROOT/stage.txt"; else printf 'evaluation_failed\\n' > "$ROOT/stage.txt"; fi
printf '%s\\n' "$rc" > "$ROOT/run.exit"
exit "$rc"
'''
with sftp.open(ROOT+'/run.sh', 'w') as stream:
    stream.write(shell.encode())
run(f'chmod 755 {shlex.quote(ROOT+"/run.sh")} && bash -n {shlex.quote(ROOT+"/run.sh")}')
assert run(f'sha256sum {shlex.quote(ROOT+"/evaluate_short32_fixed25.py")}').split()[0] == config['evaluator_sha256']
print(json.dumps({'root': ROOT, 'checkpoint_sha256': config['checkpoint_sha256'],
                  'source_files_verified': len(config['source_sha256']),
                  'evaluator_sha256': config['evaluator_sha256']}))
sftp.close()
client.close()
