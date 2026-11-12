"""Read-only training observation; never stops, retries, or edits a run."""
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'geometry_experiments'))
from run_padding_causal import connect

folder=Path(__file__).resolve().parent
launch=json.loads((folder/'B_training_launch.json').read_text())
root=launch['root']
source=r'''
import datetime,hashlib,json,re,subprocess,time
from pathlib import Path
root=Path(ROOT)
launch=json.loads((root/'launch.json').read_text())
preflight=json.loads((root/'preflight.json').read_text())
code=Path('/home/mentee/code/EBM')
def git(*a):return subprocess.check_output(['git',*a],cwd=code,text=True).strip()
log=(root/'train.log').read_text(errors='replace')
epoch=re.findall(r'epoch([0-9]+): loss1=([-0-9.]+) loss2=([-0-9.]+)',log)
progress=re.findall(r'([0-9]+)/108 +'+re.escape('['),log)
errors=re.findall(r'Traceback [(]most recent call last[)]|CUDA out of memory|DistBackendError|NCCL error|RuntimeError:|loss[12]=(?:nan|inf|-inf)',log,re.I)
host_result=Path(launch['host_result_path'])
backup=host_result/'backup/scripts'
source_ok=all(hashlib.sha256((code/p).read_bytes()).hexdigest()==h for p,h in preflight['python_hashes'].items())
backup_ok=all((backup/p).is_file() and hashlib.sha256((backup/p).read_bytes()).hexdigest()==h for p,h in preflight['python_hashes'].items())
metrics={}
for name in ['loss1','loss2','padding','active_pairs','oc_long','oc_short','bpl']:
 p=host_result/'metric'/f'{name}.txt'
 if p.exists():metrics[name]=p.read_text().splitlines()[-3:]
old=json.loads(Path('/home/mentee/analysis/ebm_server_baseline_restore_20261106_2133/restore.json').read_text())
oldroot=Path('/home/mentee/results/CM/LA/EBM_e50_bs240_lr1e5')
old_unchanged=all((oldroot/p).is_file() and hashlib.sha256((oldroot/p).read_bytes()).hexdigest()==h for p,h in old['original_result_hashes'].items())
exitfile=root/'run.exit'
status={'checked_KST':datetime.datetime.now().isoformat(timespec='seconds'),
 'run_root':str(root),'condition':'B','experiment_name':launch['experiment_name'],
 'branch':git('branch','--show-current'),'commit':git('rev-parse','HEAD'),
 'working_tree_clean':not git('status','--porcelain'),
 'exit_code':int(exitfile.read_text()) if exitfile.exists() else None,
 'last_epoch':int(epoch[-1][0]) if epoch else None,
 'last_completed_batch':int(progress[-1]) if progress else 0,'batches_per_epoch':108,
 'last_loss1':float(epoch[-1][1]) if epoch else None,
 'last_loss2':float(epoch[-1][2]) if epoch else None,
 'error_markers':errors,'log_age_seconds':round(time.time()-(root/'train.log').stat().st_mtime,2),
 'GPU_readback':subprocess.check_output(['nvidia-smi','-i','6,7','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip(),
 'source_hashes_unchanged':source_ok,'logger_backup_matches_launched_source':backup_ok,
 'old_baseline_results_unchanged':old_unchanged,'metrics_tail':metrics,
 'parent_container_pid':int((root/'parent_pid.txt').read_text()),
 'preflight_passed':preflight['status']=='passed'}
print(json.dumps(status))
'''.replace('ROOT',repr(root))
c=connect()
try:
    _,o,e=c.exec_command("python - <<'PY'\n"+source+"\nPY",timeout=45)
    text=o.read().decode();err=e.read().decode()
    assert o.channel.recv_exit_status()==0,err[-1500:]
    status=json.loads(text)
    (folder/'B_training_last_check.json').write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8')
    with c.open_sftp() as s:
        with s.open(root+'/last_check.json','w') as f:f.write(json.dumps(status,indent=2)+'\n')
    print(json.dumps(status),flush=True)
finally:c.close()
