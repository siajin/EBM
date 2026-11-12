"""Execute analysis on the mentee server without downloading source data."""
import argparse
import pathlib
import sys
import paramiko

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

parser = argparse.ArgumentParser()
parser.add_argument('--python', action='store_true')
parser.add_argument('--upload', metavar='REMOTE_PATH')
args = parser.parse_args()
payload = sys.stdin.read()
client = paramiko.SSHClient()
client.load_system_host_keys()
try:
    client.connect('172.16.162.222', username='mentee',
                   password=pathlib.Path('C:/Users/sihwa/.ssh/mentee-password.txt').read_text().splitlines()[0],
                   look_for_keys=False, allow_agent=False, timeout=15)
    if args.upload:
        with client.open_sftp() as sftp:
            try:
                if sftp.stat(args.upload).st_size:
                    raise FileExistsError(args.upload)
            except FileNotFoundError:
                pass
            with sftp.open(args.upload, 'w') as remote_file:
                remote_file.write(payload)
        print('Saved analysis script:', args.upload)
        sys.exit(0)
    stdin, stdout, stderr = client.exec_command('python3 -u -' if args.python else 'bash -s', timeout=30)
    stdin.write(payload)
    stdin.channel.shutdown_write()
    for line in stdout:
        print(line, end='', flush=True)
    errors = stderr.read().decode('utf-8', errors='replace')
    if errors:
        print(errors, file=sys.stderr, end='')
    sys.exit(stdout.channel.recv_exit_status())
finally:
    client.close()
