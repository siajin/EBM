"""Prepare and launch one all-checkpoint B/C evaluation in the existing container."""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
sys.path.insert(0, str(WORKSPACE / 'artifacts/geometry_experiments'))
from run_padding_causal import connect

ROOT = '/home/mentee/analysis/ebm_B_C_checkpoints_20261109'
EVALUATOR = WORKSPACE / 'EBM/tools/evaluate_B_C_checkpoints.py'
assert json.loads((HERE / 'summary_validation.json').read_text())['status'] == 'passed'
assert not (HERE / 'state.json').exists(), 'Inspect existing state; do not repeat launch'

def run(client, source):
    _, out, err = client.exec_command("python - <<'PY'\n" + source + "\nPY", timeout=60)
    raw = out.read().decode(errors='replace')
    error = err.read().decode(errors='replace')
    assert out.channel.recv_exit_status() == 0, error[-2000:]
    return json.loads(raw)

prepare = r'''
import datetime,hashlib,json,subprocess
from pathlib import Path
root=Path(ROOT);code=Path('/home/mentee/code/EBM')
assert not root.exists(),'New run directory already exists: inspect instead of overwriting'
assert Path('/proc/352729/root/.dockerenv').is_file()
assert subprocess.check_output(['ps','-p','352729','-o','pgid=,tpgid='],text=True).split()==['352729','352729']
gpu=subprocess.check_output(['nvidia-smi','-i','6,7','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
assert all(int(x.split(',')[0])<100 and int(x.split(',')[1])==0 for x in gpu.splitlines()),gpu
assert not subprocess.check_output(['git','status','--porcelain'],cwd=code,text=True).strip()
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip()
assert commit=='cf199af03044c40837c08ce31d633e2c048bde08'
for name in ['ebm_train_B_20261106_213627','ebm_train_C_20261108_061305']:
 assert (Path('/home/mentee/analysis')/name/'run.exit').read_text().strip()=='0'
oldroot=Path('/home/mentee/analysis/ebm_B_checkpoints_20261108')
assert not any((oldroot/name).exists() for name in ['launch.claim','run.exit','smoke.exit','complete.json'])
previous=json.loads((oldroot/'provenance.json').read_text())
source_hashes={name:hashlib.sha256((code/name).read_bytes()).hexdigest() for name in previous['source_hashes']}
assert source_hashes==previous['source_hashes']
checkpoints={}
for condition,suffix in [('B','B_zeropad'),('C','C_nativeLong')]:
 result=Path('/home/mentee/results/CM/LA')/('EBM_e50_bs240_lr1e5_'+suffix+'_short303')
 for epoch in range(5,51,5):
  path=result/f'model/CLS_{epoch}.pt';train=result/f'model/TrainState_{epoch}.pt'
  sha=hashlib.sha256(path.read_bytes()).hexdigest();train_sha=hashlib.sha256(train.read_bytes()).hexdigest()
  if condition=='B' and str(epoch) in previous['checkpoints']:
   assert sha==previous['checkpoints'][str(epoch)]['sha256']
   assert train_sha==previous['checkpoints'][str(epoch)]['TrainState_sha256']
  checkpoints[f'{condition}_e{epoch}']={'path':str(path),'bytes':path.stat().st_size,'sha256':sha,'TrainState_path':str(train),'TrainState_sha256':train_sha}
audit=json.loads(Path('/home/mentee/analysis/ebm_B_final_eval_20261107_215451/training_audit.json').read_text())
for entry in audit['checkpoints'].values():assert hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest()==entry['sha256']
provenance={'prepared_KST':datetime.datetime.now().isoformat(timespec='seconds'),'root':str(root),'commit':commit,'source_hashes':source_hashes,'checkpoints':checkpoints,'epochs':list(range(5,51,5)),'models':list(checkpoints),'phases':['dev','19LA','21LA','21DF'],'phase_counts':{'dev':24844,'19LA':71237,'21LA':148176,'21DF':533928},'protocol':{'max_samples':64600,'native_nonrepeating_input':True,'masks':True,'batch_per_rank':8,'pooling_noise_std':1e-5,'paired_common_RNG':True,'CUDA_VISIBLE_DEVICES':'0,1','hidden_states_saved':False,'no_target_checkpoint_selection':True},'gpu_before_launch':gpu.strip(),'protected_checkpoints':audit['checkpoints']}
root.mkdir()
(root/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
print(json.dumps(provenance))
'''.replace('ROOT', repr(ROOT))

script = '''#!/bin/bash
set -u
ROOT="''' + ROOT + '''"
test -f /.dockerenv || exit 90
test "$(cat /home/mentee/analysis/ebm_train_C_20261108_061305/run.exit 2>/dev/null)" = 0 || exit 91
mkdir "$ROOT/launch.claim" || exit 95
export CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd /home/mentee/code/EBM || exit 92
date -Iseconds > "$ROOT/started.txt"
python -u "$ROOT/evaluate_B_C_checkpoints.py" "$ROOT" --smoke > "$ROOT/smoke.log" 2>&1 &
pid=$!; echo "$pid" > "$ROOT/smoke_parent_pid.txt"
wait "$pid"
rc=$?; echo "$rc" > "$ROOT/smoke.exit"
if [ "$rc" -ne 0 ]; then echo "$rc" > "$ROOT/run.exit"; date -Iseconds > "$ROOT/finished.txt"; exit "$rc"; fi
python -u "$ROOT/evaluate_B_C_checkpoints.py" "$ROOT" > "$ROOT/eval.log" 2>&1 &
pid=$!; echo "$pid" > "$ROOT/parent_pid.txt"
wait "$pid"
rc=$?; echo "$rc" > "$ROOT/run.exit"
date -Iseconds > "$ROOT/finished.txt"
exit "$rc"
'''

client = connect()
try:
    provenance = run(client, prepare)
    provenance['evaluator_sha256'] = hashlib.sha256(EVALUATOR.read_bytes()).hexdigest()
    with client.open_sftp() as s:
        s.put(str(EVALUATOR), ROOT + '/evaluate_B_C_checkpoints.py')
        with s.open(ROOT + '/run.sh', 'w') as f:
            f.write(script)
        with s.open(ROOT + '/provenance.json', 'w') as f:
            f.write(json.dumps(provenance, indent=2) + '\n')
    (HERE / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    state = {'status':'prepared','root':ROOT,'models':provenance['models'],'epochs':provenance['epochs'],'phases':provenance['phases'],'evaluation_complete':False,'analysis_complete':False,'automation_status':'PAUSED','regular_schedule_reenabled':False,'supersedes_B8_root':'/home/mentee/analysis/ebm_B_checkpoints_20261108','evaluator':'EBM/tools/evaluate_B_C_checkpoints.py'}
    (HERE / 'state.json').write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
    launch = r'''
import ast,hashlib,json,subprocess
from pathlib import Path
root=Path(ROOT);code=Path('/home/mentee/code/EBM');p=json.loads((root/'provenance.json').read_text())
assert not (root/'launch.claim').exists() and not (root/'run.exit').exists()
assert not Path('/home/mentee/analysis/ebm_B_checkpoints_20261108/launch.claim').exists()
ast.parse((root/'evaluate_B_C_checkpoints.py').read_text())
assert hashlib.sha256((root/'evaluate_B_C_checkpoints.py').read_bytes()).hexdigest()==p['evaluator_sha256']
subprocess.check_call(['bash','-n',str(root/'run.sh')])
assert not subprocess.check_output(['git','status','--porcelain'],cwd=code,text=True).strip()
assert subprocess.check_output(['ps','-p','352729','-o','pgid=,tpgid='],text=True).split()==['352729','352729']
assert Path('/proc/352729/root/.dockerenv').is_file()
gpu=subprocess.check_output(['nvidia-smi','-i','6,7','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
assert all(int(x.split(',')[0])<100 and int(x.split(',')[1])==0 for x in gpu.splitlines())
subprocess.check_call(['screen','-S','352153.exp','-p','0','-X','stuff','bash '+str(root/'run.sh')+'\n'])
print(json.dumps({'status':'launch_sent','root':str(root),'models':len(p['models'])}))
'''.replace('ROOT', repr(ROOT))
    outcome = run(client, launch)
    state.update(outcome)
    (HERE / 'state.json').write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(outcome))
finally:
    client.close()
