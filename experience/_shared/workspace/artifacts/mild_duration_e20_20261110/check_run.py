"""Read-only status receipt; never starts, stops or retries remote work."""
import datetime
import json
from pathlib import Path
import paramiko

HERE=Path(__file__).resolve().parent
state=json.loads((HERE/'state.json').read_text())
root=state['root']
c=paramiko.SSHClient();c.load_system_host_keys()
c.connect('172.16.162.222',username='mentee',
    password=Path('C:/Users/sihwa/.ssh/mentee-password.txt').read_text().splitlines()[0],
    look_for_keys=False,allow_agent=False,timeout=12)
sftp=c.open_sftp()


def read(name,tail=None):
    try:
        with sftp.open(root+'/'+name) as stream:
            if tail is not None: stream.seek(max(0,stream.stat().st_size-tail))
            return stream.read().decode()
    except IOError: return None


try:
    snapshot={'checked_KST':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),
              'stage':(read('stage.txt') or '').strip(),'exits':{},'progress':[]}
    for stage in ['unit','cache','preflight','train_gpu_smoke','train','training_audit',
                  'evaluation_smoke','evaluation','comparison','pipeline']:
        value=read(stage+'.exit')
        if value is not None: snapshot['exits'][stage]=int(value)
    for name in ['queue_state.json','queue_failed.json','superseded.json','cache_progress.json','cache_complete.json',
                 'preflight.json','train_gpu_smoke_rank0.json','train_gpu_smoke_rank1.json',
                 'train_done_rank0.json','train_done_rank1.json','final_training_audit.json',
                 'phase.json','complete.json','metrics_21LA.json','metrics_21DF.json',
                 'comparison.json','analysis_complete.json']:
        value=read(name)
        if value:
            snapshot[name.removesuffix('.json')]=json.loads(value)
            (HERE/name).write_text(value,encoding='utf-8')
    for rank in [0,1]:
        value=read(f'train_progress_rank{rank}.json')
        if value: snapshot['progress'].append(json.loads(value))
    log=read((snapshot['stage'] or 'queue')+'.log',6000)
    if log: (HERE/'latest_log_tail.txt').write_text(log,encoding='utf-8')
    for label,command in [('queue_process',f"ps -p {state['queue_host_pid']} -o pid=,etime=,args="),
                          ('gpu','nvidia-smi -i 6,7 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits')]:
        _,o,e=c.exec_command(command);snapshot[label]=o.read().decode().strip();e.read()
    state['training_started']=read('training_started.txt') is not None
    state['training_complete']=snapshot['exits'].get('train')==0
    state['evaluation_complete']='complete' in snapshot
    state['comparison_complete']='analysis_complete' in snapshot
    state['status']=('superseded_by_user_request' if 'superseded' in snapshot else
        'complete' if snapshot['exits'].get('pipeline')==0 else
        'failed' if 'queue_failed' in snapshot or any(snapshot['exits'].values()) else
        snapshot['stage']+'_running' if snapshot['stage'] else
        snapshot.get('queue_state',{}).get('status','queued'))
    state['last_check']=snapshot
    (HERE/'state.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
    (HERE/'last_check.json').write_text(json.dumps(snapshot,indent=2),encoding='utf-8')
    print(json.dumps({'status':state['status'],**snapshot},indent=2))
finally: sftp.close();c.close()
