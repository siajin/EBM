"""Read-only final-evaluation progress; never starts or retries a job."""
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'geometry_experiments'))
from run_padding_causal import connect

folder=Path(__file__).resolve().parent
state=json.loads((folder/'B_final_evaluation_state.json').read_text())
source=r'''
import datetime,hashlib,json,os,subprocess,time
from pathlib import Path
root=Path(ROOT)
data={'checked_KST':datetime.datetime.now().isoformat(timespec='seconds'),'root':str(root),'exits':{},'progress':{},'done':{},'QC':{},'evaluation_complete':False}
for name in ['smoke.exit','run.exit']:
 p=root/name
 if p.exists():data['exits'][name]=int(p.read_text())
for prefix,field in [('progress_rank','progress'),('done_','done'),('QC_rank','QC')]:
 for p in root.glob(prefix+'*.json'):
  if '_smoke' not in p.name:data[field][p.name]=json.loads(p.read_text())
for name in ['metrics.json','complete.json']:
 p=root/name
 if p.exists():data[name]=json.loads(p.read_text())
data['evaluation_complete']=data['exits'].get('run.exit')==0 and 'complete.json' in data
data['evaluator_sha256']=hashlib.sha256((root/'evaluate_B_final.py').read_bytes()).hexdigest()
data['GPU_readback']=subprocess.check_output(['nvidia-smi','-i','6,7','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
code=Path('/home/mentee/code/EBM')
pre=json.loads(Path('/home/mentee/analysis/ebm_train_B_20261106_213627/preflight.json').read_text())
data['training_source_unchanged']=all(hashlib.sha256((code/n).read_bytes()).hexdigest()==h for n,h in pre['python_hashes'].items())
log=root/'eval.log'
data['log_tail']=log.read_text(errors='replace')[-1800:] if log.exists() else ''
text=log.read_text(errors='replace') if log.exists() else ''
data['error_markers']=[marker for marker in ['Traceback (most recent call last)','CUDA out of memory','RuntimeError:','AssertionError'] if marker in text]
data['processes']=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  if not (p/'comm').read_text().startswith('python'):continue
  if os.readlink(str(p/'cwd'))!=str(code):continue
  cmd=(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode(errors='replace')
  fields={line.split(':',1)[0]:line.split(':',1)[1].strip() for line in (p/'status').read_text().splitlines() if ':' in line}
  role='eval_parent' if 'evaluate_B_final.py' in cmd else ('rank' if 'spawn_main' in cmd else 'worker/helper')
  data['processes'].append({'host_pid':int(p.name),'role':role,'state':fields['State'],'namespace_pids':fields.get('NSpid','')})
 except (OSError,PermissionError):pass
print(json.dumps(data))
'''.replace('ROOT',repr(state['root']))
c=connect()
try:
    _,o,e=c.exec_command("python - <<'PY'\n"+source+"\nPY",timeout=40)
    text=o.read().decode(errors='replace');err=e.read().decode(errors='replace')
    assert o.channel.recv_exit_status()==0,err[-1000:]
    result=json.loads(text)
    assert result['evaluator_sha256']==state['evaluator_sha256'],'Evaluation script changed'
    (folder/'B_final_evaluation_last_check.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    if result['evaluation_complete']:
        state.update(stage='evaluation_complete_analysis_pending',evaluation_complete=True)
        (folder/'B_final_evaluation_state.json').write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result),flush=True)
finally:c.close()
