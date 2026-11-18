"""Launch the prepared evaluation only after C exits and its container shell is idle."""
import json
from pathlib import Path
import shlex
import sys

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE / 'artifacts/geometry_experiments'))
from run_padding_causal import connect

STATE = WORKSPACE / 'artifacts/ebm_B_checkpoints_20261108/state.json'


def main():
    state = json.loads(STATE.read_text(encoding='utf-8'))
    root = state['root']
    remote = r'''
import json,subprocess,hashlib
from pathlib import Path
root=Path(ROOT)
croot=Path('/home/mentee/analysis/ebm_train_C_20261108_061305')
def report(status,**extra):
 print(json.dumps(dict(status=status,root=str(root),**extra)))
 raise SystemExit(0)
if (root/'run.exit').exists():report('finished',exit=int((root/'run.exit').read_text()))
if (root/'launch.claim').exists():report('already_claimed_inspect_progress')
if not (croot/'run.exit').exists():report('waiting_for_C')
if (croot/'run.exit').read_text().strip()!='0':report('C_failed_do_not_launch')
assert Path('/home/mentee/results/CM/LA/EBM_e50_bs240_lr1e5_C_nativeLong_short303/model/CLS_50.pt').is_file()
gpu=subprocess.check_output(['nvidia-smi','-i','6,7','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
if not all(int(x.split(',')[0])<100 and int(x.split(',')[1])==0 for x in gpu.splitlines()):report('waiting_for_idle_GPUs',gpu=gpu)
shell=352729
assert Path(f'/proc/{shell}/root/.dockerenv').is_file(),'Existing container changed: inspect manually'
foreground=subprocess.check_output(['ps','-p',str(shell),'-o','pgid=,tpgid='],text=True).split()
if foreground!=[str(shell),str(shell)]:report('waiting_for_idle_container_shell')
assert not subprocess.check_output(['git','status','--porcelain'],cwd='/home/mentee/code/EBM',text=True).strip()
p=json.loads((root/'provenance.json').read_text())
for name,digest in p['source_hashes'].items():
 assert hashlib.sha256((Path('/home/mentee/code/EBM')/name).read_bytes()).hexdigest()==digest
for entry in p['checkpoints'].values():
 assert hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest()==entry['sha256']
assert hashlib.sha256((root/'evaluate_B_checkpoints.py').read_bytes()).hexdigest()==p['evaluator_sha256']
# This is the sole scheduler/launcher. A claim is made by run.sh before GPU use.
subprocess.check_call(['screen','-S','352153.exp','-p','0','-X','stuff','bash '+str(root/'run.sh')+'\n'])
report('launch_sent')
'''.replace('ROOT', repr(root))
    client = connect()
    try:
        _, out, err = client.exec_command("python - <<'PY'\n" + remote + "\nPY", timeout=45)
        raw = out.read().decode(); error = err.read().decode()
        if out.channel.recv_exit_status():
            raise RuntimeError(error[-1800:])
        result = json.loads(raw)
        state['last_launch_check'] = result
        STATE.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(result))
    finally:
        client.close()


if __name__ == '__main__':
    main()
