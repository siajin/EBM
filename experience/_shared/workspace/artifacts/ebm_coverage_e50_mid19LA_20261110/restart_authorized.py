"""Apply the user's explicit restart approval to this experiment group only."""
import datetime,hashlib,json,pathlib,shlex
import paramiko
HERE=pathlib.Path(__file__).resolve().parent
PREVIOUS=HERE.parent/'ebm_coverage_e50_20261110'
ROOT='/home/mentee/analysis/ebm_coverage_e50_20261110_v2_mid19LA'
old=json.loads((PREVIOUS/'experiment.json').read_text());new=json.loads((HERE/'experiment.json').read_text())
changed={name for name in old['source_sha256'] if old['source_sha256'][name]!=new['source_sha256'][name]}
assert changed-{'POSITIVE_BPL.md'}=={'tools/train_coverage_experiment.py'},changed
assert 'tests/test_intermediate_19la.py' in new['source_sha256']
assert not (HERE/'stop_receipt.json').exists(),'Inspect state instead of stopping twice'
client=paramiko.SSHClient();client.load_system_host_keys()
client.connect('172.16.162.222',username='mentee',
    password=pathlib.Path('C:/Users/sihwa/.ssh/mentee-password.txt').read_text().splitlines()[0],
    look_for_keys=False,allow_agent=False,timeout=12)
program=r'''
import os,signal,subprocess,pathlib,time,json,datetime
root=pathlib.Path('/home/mentee/analysis/ebm_coverage_e50_20261110_v1')
assert (root/'stage.txt').read_text().strip()=='train'
assert not (root/'train.exit').exists()
rows={}
for line in subprocess.check_output(['ps','-eo','pid=,ppid=,pgid=,args='],text=True).splitlines():
 parts=line.split(None,3)
 if len(parts)==4:rows[int(parts[0])]={'pid':int(parts[0]),'ppid':int(parts[1]),'pgid':int(parts[2]),'args':parts[3]}
leader=991393
assert rows[leader]['ppid']==352729 and rows[leader]['pgid']==leader
assert rows[leader]['args']=='bash /home/mentee/analysis/ebm_coverage_e50_20261110_v1/pipeline.sh'
members=[v for v in rows.values() if v['pgid']==leader]
for member in members:
 pid=member['pid'];visited=set()
 while pid!=leader:
  assert pid not in visited and pid in rows,member
  visited.add(pid);pid=rows[pid]['ppid']
assert subprocess.check_output(['ps','-p','352729','-o','pgid=,tpgid='],text=True).split()==['352729',str(leader)]
receipt={'user_explicit_restart_authorized':True,'reason':'Intermediate evaluation changed from dev to full19LA.',
 'old_root':str(root),'signal':'SIGTERM','verified_job_pgid':leader,'verified_group_members':members,
 'old_results_preserved':True,'timestamp_KST':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()}
os.killpg(leader,signal.SIGTERM)
def remaining():
 alive=[]
 for member in members:
  try:
   body=pathlib.Path('/proc/'+str(member['pid'])+'/stat').read_text();status=body[body.rfind(')')+2:].split()[0]
   if status!='Z' and os.getpgid(member['pid'])==leader:alive.append(member['pid'])
  except (FileNotFoundError,ProcessLookupError):pass
 return alive
for _ in range(40):
 if not remaining():break
 time.sleep(.5)
left=remaining()
if left:
 for pid in left:os.kill(pid,signal.SIGKILL)
 time.sleep(2)
assert not remaining(),remaining()
receipt['all_verified_job_processes_stopped']=True
for _ in range(30):
 gpu=subprocess.check_output(['nvidia-smi','-i','6,7','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 if all(int(row.split(',')[0])<100 and int(row.split(',')[1])==0 for row in gpu.splitlines()):break
 time.sleep(1)
assert all(int(row.split(',')[0])<100 and int(row.split(',')[1])==0 for row in gpu.splitlines()),gpu
assert subprocess.check_output(['ps','-p','352729','-o','pgid=,tpgid='],text=True).split()==['352729','352729']
receipt['gpu_after_stop']=gpu
receipt['exit_markers_written_by_stop_controller']=True
(root/'user_stop.json').write_text(json.dumps(receipt,indent=2))
(root/'train.exit').write_text('143\n');(root/'pipeline.exit').write_text('143\n')
(root/'stage.txt').write_text('user_stopped_for_19LA_restart\n')
print(json.dumps(receipt))
'''
try:
    stdin,stdout,stderr=client.exec_command('python3 -')
    stdin.write(program);stdin.channel.shutdown_write()
    body=stdout.read().decode();error=stderr.read().decode()
    assert stdout.channel.recv_exit_status()==0,error[-2500:]
    receipt=json.loads(body);(HERE/'stop_receipt.json').write_text(json.dumps(receipt,indent=2))
    old_state=json.loads((PREVIOUS/'state.json').read_text())
    old_state.update(status='superseded_by_user_authorized_19LA_restart',
        user_stopped_to_change_intermediate_evaluation=True,replacement_root=ROOT,
        replacement_state_path=str(HERE/'state.json'),training_complete=False)
    (PREVIOUS/'state.json').write_text(json.dumps(old_state,indent=2))
    new.update(host_gpu_readback_before_launch=receipt['gpu_after_stop'],
        restart_authorization='User explicitly selected restart to apply19LA intermediate evaluation.',
        restart_from_epoch=1,old_results_preserved=True)
    (HERE/'experiment.json').write_text(json.dumps(new,indent=2))
    with client.open_sftp() as transfer:transfer.put(str(HERE/'experiment.json'),ROOT+'/experiment.json')
    _,out,err=client.exec_command('screen -S 352153.exp -p 0 -X stuff '+shlex.quote('bash '+ROOT+'/pipeline.sh'+chr(13)))
    assert out.channel.recv_exit_status()==0,err.read().decode()
    state=json.loads((HERE/'state.json').read_text())
    state.update(status='preflight_and_smokes_running',user_restart_authorized=True,
        intermediate_evaluation_phase='19LA',intermediate_evaluation_every_epochs=5,
        comparison_primary_phases=['19LA','21LA','21DF'],dev_intermediate_evaluation=False,
        previous_root_preserved=old['root'],pipeline_queued=True)
    (HERE/'state.json').write_text(json.dumps(state,indent=2))
    print(json.dumps({'status':state['status'],'root':ROOT,'old_job_stopped':True,
        'old_results_preserved':True,'intermediate_evaluation':'full19LA every5 epochs'},indent=2))
finally:client.close()
