"""Read-only status of the already launched 21LA cache extraction."""
import json
from pathlib import Path
import sys

import paramiko

PAYLOAD = r'''
from pathlib import Path
import json,datetime,time,subprocess
root=Path('/home/mentee/analysis/embedding_geometry/20261105_21la_full_03')
r={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'run':root.name,'exits':{},'ranks':[]}
for stage in ['prepare','qc','extract','run']:
 p=root/(stage+'.exit')
 if p.exists():r['exits'][stage]=p.read_text().strip()
for rank in [0,1]:
 p=root/'extract'/('rank%d_progress.json'%rank)
 if p.exists():r['ranks'].append({'rank':rank,**json.loads(p.read_text()),'age_seconds':round(time.time()-p.stat().st_mtime,2)})
if len(r['ranks'])==2:
 r['completed']=sum(x['completed'] for x in r['ranks'])
 r['total']=sum(x['total'] for x in r['ranks'])
 r['percent']=round(100*r['completed']/r['total'],2)
 r['aggregate_per_second']=round(sum(x['utterances_per_second'] for x in r['ranks']),2)
 r['ETA_minutes']=round(max((x['total']-x['completed'])/x['utterances_per_second'] for x in r['ranks'])/60,2)
r['ready_for_sync']=r['exits'].get('extract')=='0' and (root/'extract/complete').is_file()
if r['ready_for_sync']:r['summary']=json.loads((root/'extract/summary.json').read_text())
for stage,code in r['exits'].items():
 if code!='0' and stage!='run':
  p=root/(stage+'.log')
  if p.exists():r[stage+'_error_tail']=p.read_text(errors='replace')[-3500:]
q=subprocess.run(['nvidia-smi','-i','6,7','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True)
r['assigned_GPU']=q.stdout.strip().splitlines()
print(json.dumps(r))
'''


def main():
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    try:
        client.connect('172.16.162.222', username='mentee',
                       password=(Path.home() / '.ssh/mentee-password.txt').read_text().splitlines()[0],
                       look_for_keys=False, allow_agent=False, timeout=15)
        stdin, stdout, stderr = client.exec_command('python3 -u -', timeout=30)
        stdin.write(PAYLOAD)
        stdin.channel.shutdown_write()
        result = stdout.read().decode()
        error = stderr.read().decode(errors='replace')
        if stdout.channel.recv_exit_status():
            raise RuntimeError('Read-only status failed: ' + error[-1500:])
        status = json.loads(result)
        path = Path(__file__).resolve().parents[1] / 'geometry_cache/local_analysis/la21_extraction_status.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(status, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(status))
    finally:
        client.close()


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    main()
