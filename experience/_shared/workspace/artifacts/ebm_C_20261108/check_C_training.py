"""Read-only C observation. Never edits source, stops or restarts training."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'geometry_experiments'))
from run_padding_causal import connect
HERE=Path(__file__).resolve().parent
launch=json.loads((HERE/'launch.json').read_text());root=launch['root']
source=r'''
import datetime,hashlib,json,os,re,subprocess,time
from pathlib import Path
root=Path(ROOT);code=Path('/home/mentee/code/EBM')
launch=json.loads((root/'launch.json').read_text())
pre=json.loads((root/'preflight.json').read_text())
logpath=root/'train.log';log=logpath.read_text(errors='replace') if logpath.exists() else ''
matches=list(re.finditer(r'epoch([0-9]+): loss1=([-0-9.]+) loss2=([-0-9.]+).*?([0-9]+)/108',log))
last=matches[-1].groups() if matches else None
result=Path(launch['host_result_path']);backup=result/'backup/scripts'
checks={name:hashlib.sha256((code/name).read_bytes()).hexdigest()==sha for name,sha in pre['python_hashes'].items()}
backup_ok=all((backup/name).is_file() and hashlib.sha256((backup/name).read_bytes()).hexdigest()==sha for name,sha in pre['python_hashes'].items())
processes=[]
for path in Path('/proc').iterdir():
 if not path.name.isdigit():continue
 try:
  if not (path/'comm').read_text().startswith('python') or os.readlink(str(path/'cwd'))!=str(code):continue
  cmd=(path/'cmdline').read_bytes().replace(bytes([0]),b' ').decode(errors='replace')
  if 'main.py' not in cmd and 'spawn_main' not in cmd:continue
  fields=dict((s.split(':',1)[0],s.split(':',1)[1].strip()) for s in (path/'status').read_text().splitlines() if ':' in s)
  processes.append({'host_pid':int(path.name),'host_parent_pid':int(fields['PPid']),'role':'parent' if 'main.py' in cmd else 'worker','state':fields['State'],'namespace_pids':fields.get('NSpid')})
 except (OSError,PermissionError):pass
parent_hosts={p['host_pid'] for p in processes if p['role']=='parent'}
for p in processes:
 if p['role']!='parent' and p['host_parent_pid'] in parent_hosts:p['role']='rank'
metrics={};finite=True
for name in ['loss1','loss2','oc_long','bpl','oc_short','padding','active_pairs']:
 p=result/'metric'/f'{name}.txt'
 if p.exists():
  lines=p.read_text().splitlines();values=[float(s.split(':')[-1]) for s in lines]
  finite=finite and all(__import__('math').isfinite(v) for v in values)
  metrics[name]={'rows_flushed':len(lines),'last_values':values[-3:]}
audit=json.loads(Path('/home/mentee/analysis/ebm_B_final_eval_20261107_215451/training_audit.json').read_text())
protected={k:hashlib.sha256(Path(v['path']).read_bytes()).hexdigest()==v['sha256'] for k,v in audit['checkpoints'].items()}
status={'checked_KST':datetime.datetime.now().isoformat(timespec='seconds'),'experiment_name':launch['experiment_name'],'root':str(root),
 'run_exit':int((root/'run.exit').read_text()) if (root/'run.exit').exists() else None,
 'epoch':int(last[0]) if last else None,'last_rendered_batch':int(last[3]) if last else 0,
 'loss1':float(last[1]) if last else None,'loss2':float(last[2]) if last else None,
 'log_age_seconds':round(time.time()-logpath.stat().st_mtime,2) if logpath.exists() else None,
 'source_hashes_unchanged':all(checks.values()),'logger_backup_matches_source':backup_ok,
 'working_tree_clean':not subprocess.check_output(['git','status','--porcelain'],cwd=code,text=True).strip(),
 'B_checkpoints_unchanged':protected,'processes':processes,'metrics':metrics,'all_flushed_loss_values_finite':finite,
 'dev_log':(result/'metric/V/EER.txt').read_text().splitlines() if (result/'metric/V/EER.txt').exists() else [],
 'checkpoints':[p.name for p in (result/'model').glob('*.pt')],
 'error_markers':[s for s in ['Traceback (most recent call last)','CUDA out of memory','RuntimeError:','NCCL error','AssertionError'] if s in log],
 'GPU6_7':subprocess.check_output(['nvidia-smi','-i','6,7','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()}
print(json.dumps(status))
'''.replace('ROOT',repr(root))
c=connect()
try:
    _,o,e=c.exec_command("python - <<'PY'\n"+source+"\nPY",timeout=40)
    raw=o.read().decode(errors='replace');err=e.read().decode(errors='replace')
    assert o.channel.recv_exit_status()==0,err[-1500:]
    status=json.loads(raw)
    (HERE/'last_check.json').write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8')
    with c.open_sftp() as s:
        with s.open(root+'/last_check.json','w') as f:f.write(json.dumps(status,indent=2)+'\n')
    print(json.dumps(status,indent=2))
finally:c.close()
