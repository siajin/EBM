"""Bounded host CPU watcher; never sends input while the training shell is busy."""
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

from paired_queue_guard import dependency_ready
import hashlib

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""): h.update(b)
    return h.hexdigest()

def save(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2)+"\n")
    temp.replace(path)


def main(root):
    assert sha(root/'experiment.json') == (root/'experiment.sha256').read_text().strip(), 'Configuration changed'
    cfg = json.loads((root/'experiment.json').read_text())
    (root/'queue.claim').mkdir()
    deadline = time.time()+cfg['queue_max_hours']*3600
    idle = 0
    while time.time() < deadline:
        assert Path('/proc/352729/root/.dockerenv').is_file(), 'Existing container missing'
        assert Path('/proc/352729/stat').read_text().split()[21] == cfg['container_shell_start_ticks'], 'PID reused'
        ready = dependency_ready(cfg)
        shell = subprocess.check_output(['ps', '-p', '352729', '-o', 'pgid=,tpgid='], text=True).split()
        gpu = subprocess.check_output(['nvidia-smi', '-i', '6,7', '--query-gpu=index,memory.used,utilization.gpu',
                                     '--format=csv,noheader,nounits'], text=True).strip()
        free = len(gpu.splitlines()) == 2 and all(int(l.split(',')[1]) < 100 and int(l.split(',')[2]) == 0 for l in gpu.splitlines())
        idle = idle+1 if ready and shell == ['352729', '352729'] and free else 0
        save(root/'queue_state.json', dict(status='waiting_for_existing_training_and_evaluation',
            dependency_terminal=ready, idle_checks=idle, updated_unix=time.time(), GPU=gpu))
        if idle >= 3: break
        time.sleep(20)
    else: raise TimeoutError('Bounded wait expired; no evaluation launched')
    for name, expected in cfg['source_sha256'].items(): assert sha(root/'code'/name) == expected, name
    for name, expected in cfg['payload_sha256'].items(): assert sha(root/name) == expected, name
    for ck in cfg['reference_checkpoints'].values(): assert sha(ck['path']) == ck['sha256']
    live = '/home/mentee/code/EBM'
    assert subprocess.check_output(['git', '-C', live, 'rev-parse', 'HEAD'], text=True).strip() == cfg['live_commit']
    assert not subprocess.check_output(['git', '-C', live, 'status', '--porcelain'], text=True)
    (root/'launch.claim').mkdir()
    subprocess.run(['screen', '-S', '352153.exp', '-p', '0', '-X', 'stuff',
        'bash '+shlex.quote(str(root/'run.sh'))+'\r'], check=True)
    for _ in range(60):
        if (root/'pipeline.claim').exists():
            save(root/'queue_state.json', dict(status='evaluation_launched', launched_unix=time.time(), GPU=gpu))
            return
        time.sleep(1)
    raise TimeoutError('Sent command once; no pipeline claim; no retry')


if __name__ == '__main__':
    root = Path(sys.argv[1])
    try: main(root)
    except Exception as error:
        save(root/'queue_failed.json', dict(status='failed_no_retry', error=str(error), unix=time.time()))
        raise
