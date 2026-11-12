"""One-shot completion pipeline. Waits, syncs, verifies and records results.

Never kills, restarts or resubmits an experiment. R2/R3 submission is handled by
its separate single-use queue, which checks the existing container is idle.
"""
import json
from pathlib import Path
import subprocess
import sys
import time
import paramiko
from frozen_common import DEFAULT_ROOT,save

OUT=DEFAULT_ROOT/'local_analysis/frozen_all_v1';HERE=Path(__file__).parent
REMOTE='/home/mentee/analysis/embedding_geometry/20261105_frozen_all_01'
STATUS=OUT/'completion_pipeline'
INVENTORY=r'''
from pathlib import Path
import json,time
p=Path('/home/mentee/analysis/embedding_geometry/20261105_frozen_all_01')
r={'checked_epoch':time.time(),'exits':{},'progress':{}}
for name in ['Q/extract.exit','19LA_eval/extract.exit','DF_remaining/extract.exit','run.exit','R23/prepare.exit','R23/DF_pilot.exit','R23/21LA_pilot.exit','R23/run.exit']:
 f=p/name
 if f.exists():r['exits'][name]=f.read_text().strip()
for dataset in ['Q','19LA_eval','DF_remaining','R23/DF_pilot','R23/21LA_pilot']:
 for rank in [0,1]:
  f=p/dataset/('rank%d_progress.json'%rank) if dataset.startswith('R23') else p/dataset/'extract'/('rank%d_progress.json'%rank)
  if f.exists():r['progress'][dataset+'::'+str(rank)]=json.loads(f.read_text())
print(json.dumps(r))
'''

def remote_status():
    c=paramiko.SSHClient();c.load_system_host_keys()
    try:
        c.connect('172.16.162.222',username='mentee',password=Path('C:/Users/sihwa/.ssh/mentee-password.txt').read_text().splitlines()[0],look_for_keys=False,allow_agent=False,timeout=15)
        inp,out,err=c.exec_command('python3 -u -',timeout=30);inp.write(INVENTORY);inp.channel.shutdown_write();text=out.read().decode();error=err.read().decode();assert out.channel.recv_exit_status()==0,error
        return json.loads(text)
    finally:c.close()

def call(script,args,label):
    with (STATUS/(label+'.log')).open('w',encoding='utf8') as log:
        rc=subprocess.call([sys.executable,'-u','-X','utf8',str(HERE/script),*args],stdout=log,stderr=subprocess.STDOUT)
    save(STATUS/(label+'.json'),dict(exit=rc,finished_epoch=time.time()))
    if rc:raise RuntimeError('Completion step failed, original run preserved: '+label)

def main():
    STATUS.mkdir(exist_ok=False);last_exits={};audited_n0=False;audited_probes=False;seen=set();last_progress=None
    while True:
        try:remote=remote_status();save(STATUS/'remote_status.json',remote)
        except Exception as error:
            save(STATUS/'connection_wait.json',dict(status='waiting_for_connection',error_type=type(error).__name__,epoch=time.time()));time.sleep(30);continue
        exits=remote['exits'];failed={k:v for k,v in exits.items() if v!='0'}
        if failed:save(STATUS/'server_failure.json',dict(status='server_stage_failed_no_restart',exits=failed))
        for source,target in [('Q','Q'),('19LA_eval','19LA_eval'),('DF_remaining','DF_full')]:
            if exits.get(source+'/extract.exit')=='0' and not (OUT/'R1'/target/'complete.json').exists():
                if (OUT/'R1'/target).exists():raise RuntimeError('Incomplete expansion evaluation preserved; inspect '+target)
                call('sync_geometry_cache.py',[],'sync_'+source)
                call('frozen_expansion_evaluate.py',['--dataset',target],'evaluate_'+target)
        for dataset in ['DF_pilot','21LA_pilot']:
            if exits.get('R23/'+dataset+'.exit')=='0' and not (OUT/'R2_R3'/dataset/'complete.json').exists():
                if (OUT/'R2_R3'/dataset).exists():raise RuntimeError('Incomplete R23 analysis preserved; inspect '+dataset)
                call('frozen_r23_cache.py',[],'sync_R23_'+dataset)
                call('analyze_frozen_r23.py',['--dataset',dataset],'evaluate_R23_'+dataset)
        if (OUT/'N0_complete.json').exists() and not audited_n0:
            call('audit_frozen_all.py',[],'audit_N0_heads');audited_n0=True
        probe_ready=all((OUT/'P3_P4'/path/'complete.json').exists() for path in ['A','B/DF_pilot','B/21LA_pilot'])
        if probe_ready and not audited_probes:
            call('audit_frozen_probes.py',[],'audit_probes');audited_probes=True
        signature=(tuple(sorted(exits.items())),audited_n0,audited_probes)
        if signature!=last_progress:
            call('refresh_frozen_progress.py',[],'refresh_progress');last_progress=signature
        finished=audited_n0 and audited_probes and all((OUT/'R1'/n/'complete.json').exists() for n in ['Q','19LA_eval','DF_full']) and all((OUT/'R2_R3'/n/'complete.json').exists() for n in ['DF_pilot','21LA_pilot'])
        save(STATUS/'status.json',dict(status='complete' if finished else ('partial_server_failure' if failed else 'running'),checked_epoch=time.time(),N0_audited=audited_n0,probes_audited=audited_probes,server_exits=exits))
        if finished:
            call('finalize_frozen_report.py',[],'finalize_report');save(OUT/'all_experiments_complete.json',dict(status='complete',finished_epoch=time.time(),all_actual_experiment_markers_and_audits_present=True,EBM_joint_retraining=False));print('All frozen experiments, derived caches, evaluations and audits completed.',flush=True);return
        if failed and audited_n0 and audited_probes:
            print('Local experiments complete; server failure preserved for inspection, no automatic restart.',flush=True);return
        time.sleep(30)

if __name__=='__main__':main()
