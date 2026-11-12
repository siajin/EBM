"""Read-only server status and derived metric receipt for fixed epoch20 eval."""
import datetime
import json
from pathlib import Path
import re

import paramiko

HERE = Path(__file__).resolve().parent
state = json.loads((HERE / 'state.json').read_text())
root = state['root']
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('172.16.162.222', username='mentee',
    password=Path('C:/Users/sihwa/.ssh/mentee-password.txt').read_text().splitlines()[0],
    look_for_keys=False, allow_agent=False, timeout=12)
sftp = client.open_sftp()


def read(name):
    try:
        return sftp.open(root + '/' + name).read().decode()
    except IOError:
        return None


try:
    snapshot = {'checked_KST': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),
                'stage': (read('stage.txt') or '').strip(), 'exits': {}, 'metrics': {}, 'progress': []}
    for stage in ['smoke', 'evaluation', 'run']:
        value = read(stage + '.exit')
        if value is not None:
            snapshot['exits'][stage] = int(value)
    phase = read('phase.json')
    if phase:
        snapshot['phase'] = json.loads(phase)
    for name in ['smoke_checkpoint_audit.json', 'checkpoint_audit.json', 'smoke_complete.json',
                 'complete.json', 'metrics_21LA.json', 'metrics_21DF.json',
                 'comparison.json', 'analysis_complete.json', 'analysis_failed.json']:
        value = read(name)
        if value:
            (HERE / name).write_text(value, encoding='utf-8')
            if name.startswith('metrics_'):
                snapshot['metrics'][name.removeprefix('metrics_').removesuffix('.json')] = json.loads(value)
    directory = 'smoke' if snapshot['stage'] == 'smoke' else 'evaluation'
    for rank in [0, 1]:
        progress = read(f'{directory}/progress_rank{rank}.json')
        if progress:
            snapshot['progress'].append(dict(rank=rank, **json.loads(progress)))
    snapshot['completed_rank_phases'] = []
    for phase in ['21LA', '21DF']:
        for rank in [0, 1]:
            value = read(f'evaluation/done_{phase}_rank{rank}.json')
            if value:
                snapshot['completed_rank_phases'].append(dict(phase=phase, rank=rank, **json.loads(value)))
    smoke_done = []
    for phase in ['19LA', '21LA', '21DF']:
        for rank in [0, 1]:
            value = read(f'smoke/done_{phase}_rank{rank}.json')
            if value:
                data = json.loads(value)
                smoke_done.append(dict(phase=phase, rank=rank, **data))
    snapshot['smoke_checks'] = smoke_done
    log = read(snapshot['stage'] + '.log') if snapshot['stage'] in ['smoke', 'evaluation'] else None
    if log:
        snapshot['error_markers'] = re.findall(
            r'Traceback|CUDA out of memory|RuntimeError|AssertionError|NCCL error', log)
        (HERE / 'latest_log_tail.txt').write_text(log[-6000:], encoding='utf-8')
    _, stdout, stderr = client.exec_command(
        'nvidia-smi -i 6,7 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits')
    snapshot['gpu'] = stdout.read().decode().strip()
    assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
    state['evaluation_complete'] = snapshot['exits'].get('run') == 0 and (HERE / 'complete.json').exists()
    state['comparison_complete'] = (HERE / 'analysis_complete.json').exists()
    state['status'] = ('complete' if state['evaluation_complete'] and
                       (not state.get('comparison_requested') or state['comparison_complete']) else
                       'evaluation_complete_analysis_pending' if state['evaluation_complete'] else
                       'failed' if any(snapshot['exits'].values()) else snapshot['stage'] + '_running')
    state['last_check'] = snapshot
    (HERE / 'last_check.json').write_text(json.dumps(snapshot, indent=2), encoding='utf-8')
    (HERE / 'state.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
    print(json.dumps(snapshot, indent=2))
finally:
    sftp.close()
    client.close()
