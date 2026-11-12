"""Read-only final source/checkpoint/foreground/GPU audit; never prints auth."""
import hashlib, json, pathlib, shlex
import paramiko

directory = pathlib.Path(__file__).resolve().parent
root = '/home/mentee/analysis/ebm_activity_joint_20261110_v5'
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('172.16.162.222', username='mentee',
               password=pathlib.Path('C:/Users/sihwa/.ssh/mentee-password.txt').read_text().splitlines()[0],
               look_for_keys=False, allow_agent=False, timeout=12)
protocol = json.loads((directory.parent / 'speed_duration_feasibility_20261109/probe/protocol.json').read_text())
program = '''
import hashlib,json,pathlib,subprocess
protocol = json.loads(PROTOCOL)
result = {}
for name, item in protocol['checkpoints'].items():
    for path_key, sha_key in [('path','sha256'),('TrainState_path','TrainState_sha256')]:
        if path_key not in item: continue
        path = pathlib.Path('/proc/352729/root' + item[path_key])
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == item[sha_key], name + ' checkpoint changed'
        result[name+'_'+path_key] = actual
result['git_branch'] = subprocess.check_output(['git','-C','/home/mentee/code/EBM','branch','--show-current'],text=True).strip()
result['git_commit'] = subprocess.check_output(['git','-C','/home/mentee/code/EBM','rev-parse','HEAD'],text=True).strip()
result['git_status'] = subprocess.check_output(['git','-C','/home/mentee/code/EBM','status','--porcelain'],text=True)
result['foreground'] = subprocess.check_output(['ps','-p','352729','-o','pgid=,tpgid='],text=True).strip()
result['gpu'] = subprocess.check_output(['nvidia-smi','-i','6,7','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
assert result['git_commit']=='cf199af03044c40837c08ce31d633e2c048bde08'
assert result['git_status']==''
print(json.dumps(result))
'''.replace('PROTOCOL', repr(json.dumps(protocol)))
stdin, stdout, stderr = client.exec_command('python3 -')
stdin.write(program)
stdin.channel.shutdown_write()
body = stdout.read().decode()
error = stderr.read().decode()
assert stdout.channel.recv_exit_status() == 0, error
result = json.loads(body)
manifest = json.loads((directory / 'v5_source_manifest.json').read_text())
sftp = client.open_sftp()
checked = []
for name, expected in manifest.items():
    if name == 'POSITIVE_BPL.md':
        continue  # Narrative validation results may be appended after the run.
    current = hashlib.sha256((directory.parents[1] / 'EBM' / name).read_bytes()).hexdigest()
    remote = hashlib.sha256(sftp.open(root + '/code/' + name).read()).hexdigest()
    assert current == remote == expected, name + ' source mismatch'
    checked.append(name)
result.update(source_files_equal=len(checked), source_manifest='v5_source_manifest.json',
              live_server_repo_changed=False)
sftp.close()
client.close()
(directory / 'final_audit.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
