import json
from pathlib import Path
import paramiko

root = '/home/mentee/analysis/ebm_short32_original240_e50_20261111_v2'
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('172.16.162.222', username='mentee',
               password=(Path.home()/'.ssh/mentee-password.txt').read_text().splitlines()[0],
               look_for_keys=False, allow_agent=False, timeout=12)
commands = {
    'stop': f"for p in run.exit finished.txt train.exit stage.txt; do printf '%s ' \"$p\"; cat {root}/\"$p\" 2>/dev/null || true; done",
    'manifest': "find /home/mentee/results/CM/LA/EBM_e50_bs240_lr1e5_original240_OC_BPL43_short32_LE2_20261111_v2 -path '*checkpoint_25.complete.json' -o -path '*eval_25_19LA.json' | sort",
    'container': "sed -n '215,300p' /home/mentee/code/IRLab/MENTEE_SERVER.md 2>/dev/null || screen -S exp -Q windows",
    'gpu': "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader -i 6,7",
    'root': f"ls -lah {root} | head -35; ls -lah {root}/code | head -12",
    'shell': "ps -p 352729 -o pid,ppid,stat,args; pgrep -a -P 352729 || true; screen -S exp -Q windows",
}
for name, command in commands.items():
    _, stdout, stderr = client.exec_command(command, timeout=25)
    print(name, json.dumps({'exit': stdout.channel.recv_exit_status(), 'stdout': stdout.read().decode(), 'stderr': stderr.read().decode()}))
remote_result='/home/mentee/results/CM/LA/EBM_e50_bs240_lr1e5_original240_OC_BPL43_short32_LE2_20261111_v2'
sftp=client.open_sftp()
for label,path in {
    'checkpoint25':f'{remote_result}/short32_stages/checkpoint_25.complete.json',
    'eval25':f'{remote_result}/short32_stages/eval_25_19LA.json',
    'experiment':f'{root}/experiment.json',
}.items():
    with sftp.open(path) as stream:
        data=json.load(stream)
    if label=='experiment':
        data={key:value for key,value in data.items() if key in ('source_sha256','frontend_revision','container_root','result_path','model_path','experiment_name')}
    print(label,json.dumps(data))
sftp.close()
client.close()
