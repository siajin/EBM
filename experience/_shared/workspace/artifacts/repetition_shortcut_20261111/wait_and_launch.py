import hashlib,json,subprocess,time
from pathlib import Path
root=Path('/home/mentee/analysis/ebm_repetition_shortcut_20261111_v1');dependency=Path('/home/mentee/analysis/ebm_baseline240_mild_e20_20261111_v2');deadline=time.time()+48*3600
def state(status,**kw):
 d={'status':status,'updated_unix':time.time(),'dependency':str(dependency),'deadline':deadline,**kw};p=root/'queue_state.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2));tmp.replace(p)
def verify():
 p=json.loads((root/'protocol.json').read_text())
 for file,h in json.loads((root/'payload_hashes.json').read_text()).items():assert hashlib.sha256((root/file).read_bytes()).hexdigest()==h,file
 for file,h in json.loads((root/'source_hashes.json').read_text()).items():assert hashlib.sha256((root/'code'/file).read_bytes()).hexdigest()==h,file
 for c in p['checkpoints'].values():assert hashlib.sha256(Path(c['path']).read_bytes()).hexdigest()==c['sha256']
 assert Path('/proc/352729/root/.dockerenv').is_file()
 assert not (root/'launch.claim').exists()
 assert not subprocess.check_output(['git','status','--porcelain'],cwd='/home/mentee/code/EBM',text=True).strip()
def idle():
 if subprocess.check_output(['ps','-p','352729','-o','pgid=,tpgid='],text=True).split()!=['352729','352729']:return False
 gpu=subprocess.check_output(['nvidia-smi','-i','6,7','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 return len(gpu.splitlines())==2 and all(int(x.split(',')[0])<100 and int(x.split(',')[1])==0 for x in gpu.splitlines())
try:
 stable=0;verify();state('waiting_dependency_success_and_idle')
 while time.time()<deadline:
  exitfile=dependency/'pipeline.exit'
  if exitfile.exists():
   rc=exitfile.read_text().strip()
   if rc!='0':state('aborted_dependency_failed',dependency_exit=rc);break
   stable=stable+1 if idle() else 0
   if stable>=3:
    verify();assert idle();subprocess.check_call(['screen','-S','352153.exp','-p','0','-X','stuff','bash '+str(root/'run.sh')+'\n'])
    state('command_sent_once');break
  time.sleep(20)
 else:state('expired_no_launch')
except Exception as e:
 state('failed_no_retry',error=repr(e));raise
