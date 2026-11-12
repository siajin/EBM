"""Observe the evaluation without modifying remote files or processes."""
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'artifacts/geometry_experiments'))
from run_padding_causal import connect

source=r'''
import datetime,json,subprocess
from pathlib import Path
root=Path('/home/mentee/analysis/ebm_B_C_checkpoints_20261109');out={'checked_KST':datetime.datetime.now().isoformat(timespec='seconds'),'root':str(root),'claim':(root/'launch.claim').exists()}
for name in ['started.txt','smoke.exit','run.exit','finished.txt','smoke_parent_pid.txt','parent_pid.txt']:
 out[name]=(root/name).read_text().strip() if (root/name).exists() else None
for name in ['smoke_complete.json','complete.json','progress_rank0.json','progress_rank1.json','progress_rank0_smoke.json','progress_rank1_smoke.json','QC_rank0_smoke.json','QC_rank1_smoke.json','QC_rank0_full.json','QC_rank1_full.json']:
 path=root/name
 if path.exists():
  try:out[name]=json.loads(path.read_text())
  except json.JSONDecodeError:out[name]={'status':'currently_writing'}
for name in ['smoke.log','eval.log']:
 path=root/name
 if path.exists():
  log=path.read_text(errors='replace');out[name+'_tail']=log.splitlines()[-14:]
  out[name+'_errors']=[m for m in ['Traceback (most recent call last)','CUDA out of memory','RuntimeError:','AssertionError'] if m in log]
out['done_files']=[p.name for p in root.glob('done_*.json')]
out['GPU6_7']=subprocess.check_output(['nvidia-smi','-i','6,7','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
processes=[]
for path in Path('/proc').iterdir():
 if not path.name.isdigit():continue
 try:
  if not (path/'comm').read_text().startswith('python') or (path/'cwd').resolve()!=Path('/home/mentee/code/EBM'):continue
  cmd=(path/'cmdline').read_bytes().replace(bytes([0]),b' ').decode(errors='replace')
  if 'evaluate_B_C_checkpoints.py' not in cmd and 'spawn_main' not in cmd:continue
  fields=dict((line.split(':',1)[0],line.split(':',1)[1].strip()) for line in (path/'status').read_text().splitlines() if ':' in line)
  processes.append({'pid':int(path.name),'ppid':int(fields['PPid']),'state':fields['State'],'role':'parent' if 'evaluate_B_C_checkpoints.py' in cmd else 'worker'})
 except (OSError,PermissionError):pass
parents={p['pid'] for p in processes if p['role']=='parent'}
out['processes']=[dict(p,role='rank' if p['ppid'] in parents else p['role']) for p in processes if p['role']=='parent' or p['ppid'] in parents]
print(json.dumps(out))
'''
client=connect()
try:
 _,out,err=client.exec_command("python - <<'PY'\n"+source+"\nPY",timeout=35)
 raw=out.read().decode();error=err.read().decode();assert out.channel.recv_exit_status()==0,error[-1500:]
 observation=json.loads(raw)
 (HERE/'last_check.json').write_text(json.dumps(observation,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(observation,indent=2))
finally:client.close()
