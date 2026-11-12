"""Read-only progress check for fixed epoch-25 evaluation."""
from pathlib import Path
import json
import paramiko

root = '/home/mentee/analysis/ebm_short32_fixed25_eval_20261111_v1'
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('172.16.162.222', username='mentee',
               password=(Path.home()/'.ssh/mentee-password.txt').read_text().splitlines()[0],
               look_for_keys=False, allow_agent=False, timeout=12)
sftp = client.open_sftp()

def read(path):
    try:
        with sftp.open(root+'/'+path) as stream:
            return stream.read().decode()
    except FileNotFoundError:
        return None

fields = ['stage.txt','smoke.exit','evaluation.exit','run.exit','phase.json',
          'smoke_complete.json','metrics_21LA.json','metrics_21DF.json','complete.json',
          'smoke/done_19LA_rank0.json','smoke/done_19LA_rank1.json',
          'smoke/progress_rank0.json','smoke/progress_rank1.json',
          'evaluation/progress_rank0.json','evaluation/progress_rank1.json']
result = {}
for name in fields:
    raw = read(name)
    if raw is not None:
        try:
            result[name] = json.loads(raw)
        except ValueError:
            result[name] = raw.strip()
for name in ['smoke.log','evaluation.log']:
    raw=read(name)
    if raw is not None:
        result[name] = raw[-1500:]
_, stdout, _ = client.exec_command('nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader -i 6,7', timeout=20)
result['gpu'] = stdout.read().decode().strip().splitlines()
_, stdout, _ = client.exec_command("screen -S exp -p 0 -X hardcopy /home/mentee/analysis/ebm_short32_fixed25_eval_20261111_v1/screen_capture.txt; tail -12 /home/mentee/analysis/ebm_short32_fixed25_eval_20261111_v1/screen_capture.txt", timeout=20)
result['screen_tail'] = stdout.read().decode().strip().splitlines()
print(json.dumps(result, ensure_ascii=False, indent=2))
sftp.close()
client.close()
