"""Launch exactly one prepared evaluator inside the existing screen container shell."""
from pathlib import Path
import paramiko

root = '/home/mentee/analysis/ebm_short32_fixed25_eval_20261111_v1'
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('172.16.162.222', username='mentee',
               password=(Path.home()/'.ssh/mentee-password.txt').read_text().splitlines()[0],
               look_for_keys=False, allow_agent=False, timeout=12)

def run(command):
    _, stdout, stderr = client.exec_command(command, timeout=20)
    result = stdout.read().decode()
    code = stdout.channel.recv_exit_status()
    assert code == 0, (code, result, stderr.read().decode())
    return result

assert run(f'test -f {root}/evaluation_config.json && test ! -e {root}/launch.claim && test ! -e {root}/run.exit && echo ready').strip() == 'ready'
assert run("nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader -i 6,7").count('6 MiB, 0 %') == 2
assert '352729' in run('ps -p 352729 -o pid=,stat=,args=')
assert 'bash' in run('screen -S exp -Q windows')
# Screen sends the literal command to the existing container shell. The launcher
# itself has an atomic claim, so another invocation cannot start a second eval.
run("screen -S exp -p 0 -X stuff 'bash /home/mentee/analysis/ebm_short32_fixed25_eval_20261111_v1/run.sh\n'")
print('sent one evaluator command to the existing container screen')
client.close()
