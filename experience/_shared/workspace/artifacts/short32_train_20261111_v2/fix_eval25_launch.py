"""Set the assigned GPU mapping before the first actual evaluation launch."""
from pathlib import Path
import paramiko

root='/home/mentee/analysis/ebm_short32_fixed25_eval_20261111_v1'
client=paramiko.SSHClient(); client.load_system_host_keys()
client.connect('172.16.162.222', username='mentee',
               password=(Path.home()/'.ssh/mentee-password.txt').read_text().splitlines()[0],
               look_for_keys=False, allow_agent=False, timeout=12)
sftp=client.open_sftp()
assert not any(name in sftp.listdir(root) for name in ('launch.claim','stage.txt','run.exit','smoke'))
path=root+'/run.sh'
with sftp.open(path) as stream: source=stream.read().decode()
assert 'export CUDA_VISIBLE_DEVICES=0,1' not in source
source=source.replace('set -uo pipefail\n','set -uo pipefail\nexport CUDA_VISIBLE_DEVICES=0,1\n',1)
with sftp.open(path,'w') as stream: stream.write(source.encode())
_, stdout, _=client.exec_command('bash -n '+path, timeout=20)
assert stdout.channel.recv_exit_status()==0
print('GPU mapping added before first evaluator process')
sftp.close();client.close()
