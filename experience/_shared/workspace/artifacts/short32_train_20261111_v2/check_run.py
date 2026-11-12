"""Read-only status: never launches, stops, retries or downloads models/audio."""
import datetime
import json
from pathlib import Path
import paramiko
HERE=Path(__file__).resolve().parent
cfg=json.loads((HERE/'experiment.json').read_text(encoding='utf-8'))
root=cfg['root'];results=cfg['host_result_path']+'/short32_stages'
c=paramiko.SSHClient();c.load_system_host_keys()
c.connect('172.16.162.222',username='mentee',password=(Path.home()/'.ssh/mentee-password.txt').read_text().splitlines()[0],look_for_keys=False,allow_agent=False,timeout=15)
s=c.open_sftp()
def read(path,limit=None):
 try:
  with s.open(path) as f:
   if limit is not None:f.seek(max(0,f.stat().st_size-limit))
   raw=f.read().decode(errors='replace')
  return json.loads(raw) if path.endswith('.json') else raw.strip()
 except FileNotFoundError:return None
try:
 out={'checked_KST':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),'root':root,'files':{}}
 names=['manual_launch.json','started.txt','queue_state.json','queue_launch.json','stage.txt','run.exit','unit.exit','preflight.exit','smoke.exit','train.exit','audit.exit','preflight.json','final_audit.json','smoke_progress_rank0.json','smoke_progress_rank1.json','smoke_done_rank0.json','smoke_done_rank1.json','train_progress_rank0.json','train_progress_rank1.json','train_done_rank0.json','train_done_rank1.json','training_started_rank0.txt','training_started_rank1.txt']
 for name in names:
  value=read(root+'/'+name)
  if value is not None:out['files'][name]=value
 out['checkpoint50']=read(results+'/checkpoint_50.complete.json')
 out['evaluation_summary']=read(results+'/complete.json')
 out['running_evaluations']=[]
 try:
  for a in s.listdir_attr(results):
   if a.filename.startswith('eval_') and a.filename.endswith('.json'):
    value=read(results+'/'+a.filename)
    if value and value.get('status')=='running':out['running_evaluations'].append(value)
 except FileNotFoundError:pass
 stage=out['files'].get('stage.txt')
 if stage and stage!='complete':out['log_tail']=read(root+'/'+stage+'.log',3500)
 code="import json,subprocess;from pathlib import Path;r=Path("+repr(root)+");p=int((r/'queue.pid').read_text()) if (r/'queue.pid').exists() else 0;print(json.dumps({'queue_process_alive':Path('/proc',str(p)).exists(),'GPU':subprocess.check_output(['nvidia-smi','-i','6,7','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)}))"
 _,o,e=c.exec_command('python -c '+__import__('shlex').quote(code),timeout=20)
 health=o.read().decode();error=e.read().decode();assert o.channel.recv_exit_status()==0,error
 out.update(json.loads(health))
 f=out['files'];rc=f.get('run.exit');summary=out.get('evaluation_summary')
 trained=out['checkpoint50'] is not None
 started=any('training_started_rank'+str(i)+'.txt' in f for i in [0,1])
 status=('complete' if rc=='0' else 'training_complete_evaluation_incomplete' if rc is not None and trained else 'failed_'+str(stage)+'_no_retry' if rc is not None else 'training_complete_evaluating' if trained else 'training_running' if started else str(stage)+'_running' if stage else f.get('queue_state.json',{}).get('status','unknown'))
 state=json.loads((HERE/'state.json').read_text(encoding='utf-8'))
 state.update(status=status,queue_registered=False,training_started=started,training_complete=trained,GPU_smoke_started='smoke_progress_rank0.json' in f or 'smoke_done_rank0.json' in f,evaluation_complete=bool(summary and summary.get('status')=='complete'),last_check=out['checked_KST'])
 (HERE/'state.json').write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
 (HERE/'last_check.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({'status':status,'training_started':started,'training_complete':trained,**out},indent=2))
finally:s.close();c.close()
