import hashlib,json,os,subprocess,time
from pathlib import Path
root=Path('/home/mentee/analysis/ebm_short32_original240_e50_20261111_v1');dependency=Path('/home/mentee/analysis/ebm_repetition_shortcut_20261111_v1')
deadline=time.time()+8*3600

def state(status,**kw):
 p=root/'queue_state.json';temp=p.with_suffix('.tmp')
 temp.write_text(json.dumps(dict(status=status,updated_unix=time.time(),deadline=deadline,dependency=str(dependency),**kw),indent=2)+'\n');temp.replace(p)

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def verify():
 cfg=json.loads((root/'experiment.json').read_text())
 for n,h in cfg['source_sha256'].items():assert digest(root/'code'/n)==h,n
 for n,h in json.loads((root/'payload_hashes.json').read_text()).items():assert digest(root/n)==h,n
 for ref in cfg['reference_checkpoints'].values():
  assert digest(ref['path'])==ref['sha256']
  if 'TrainState_path' in ref:assert digest(ref['TrainState_path'])==ref['TrainState_sha256']
 assert Path('/proc/352729/root/.dockerenv').is_file(),'Container unavailable'
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd='/home/mentee/code/EBM',text=True).strip()=='cf199af03044c40837c08ce31d633e2c048bde08'
 assert not subprocess.check_output(['git','status','--porcelain'],cwd='/home/mentee/code/EBM',text=True).strip(),'Operational Git changed'
 assert not (root/'launch.claim').exists(),'Already launched'
 assert not Path(cfg['host_result_path']).exists(),'Existing result directory'
 assert os.statvfs(root).f_bavail*os.statvfs(root).f_frsize>20*1024**3,'Insufficient free disk'

def idle():
 if subprocess.check_output(['ps','-p','352729','-o','pgid=,tpgid='],text=True).split()!=['352729','352729']:return False
 rows=subprocess.check_output(['nvidia-smi','-i','6,7','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()
 return len(rows)==2 and all(int(x.split(',')[0])<100 and int(x.split(',')[1])==0 for x in rows)

try:
 (root/'queue.claim').mkdir(exist_ok=False)
 verify();state('waiting_dependency_success_and_two_gpu_idle');stable=0
 while time.time()<deadline:
  f=dependency/'run.exit'
  if f.is_file():
   rc=f.read_text().strip()
   if rc!='0':state('aborted_dependency_failed',dependency_exit=rc);break
   complete=dependency/'analysis_complete.json'
   success=complete.is_file() and json.loads(complete.read_text()).get('status')=='analysis_complete'
   stable=stable+1 if success and idle() else 0
   if stable>=3:
    verify();assert idle()
    subprocess.check_call(['screen','-S','352153.exp','-p','0','-X','stuff','bash '+str(root/'run.sh')+'\n'])
    state('command_sent_once');break
  time.sleep(20)
 else:state('expired_no_launch')
except Exception as e:
 state('failed_no_retry',error=repr(e));raise
