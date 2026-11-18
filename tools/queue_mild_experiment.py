"""One bounded host watcher, launches only after prior eval AND analysis finish."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time


def save(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)


def check_source(root,config):
    for name,expected in config['source_sha256'].items():
        assert hashlib.sha256((root/'code'/name).read_bytes()).hexdigest()==expected,name
    for entry in config['protected_checkpoints'].values():
        assert hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest()==entry['sha256']
    assert subprocess.check_output(['git','-C','/home/mentee/code/EBM','rev-parse','HEAD'],text=True).strip()==config['live_commit']
    assert not subprocess.check_output(['git','-C','/home/mentee/code/EBM','status','--porcelain'],text=True)


def main():
    root=Path(sys.argv[1]);config=json.loads((root/'experiment.json').read_text())
    (root/'queue.claim').mkdir()
    previous=Path(config['wait_for_evaluation_root']);deadline=time.time()+8*3600
    while True:
        assert Path('/proc/352729/root/.dockerenv').is_file(),'Existing container changed'
        current_start=Path('/proc/352729/stat').read_text().split()[21]
        assert current_start==config['container_shell_start_ticks'],'Container shell PID reused'
        if (previous/'run.exit').exists():
            assert int((previous/'run.exit').read_text())==0,'Prior evaluation did not finish successfully'
            assert not (previous/'analysis_failed.json').exists(),'Prior analysis failed'
            if (previous/'analysis_complete.json').exists():
                receipt=json.loads((previous/'analysis_complete.json').read_text())
                assert receipt['status']=='complete'
                assert hashlib.sha256((previous/'comparison.json').read_bytes()).hexdigest()==receipt['comparison_sha256']
                break
        if time.time()>deadline: raise TimeoutError('8h queue bound reached; no new job launched')
        save(root/'queue_state.json',{'status':'waiting_for_existing_DF_and_comparison','updated_unix':time.time()})
        time.sleep(30)
    check_source(root,config)
    # Gate on actual idle foreground and GPUs, never type into the preceding job.
    idle_deadline=time.time()+600
    while True:
        shell=subprocess.check_output(['ps','-p','352729','-o','pgid=,tpgid='],text=True).split()
        gpu=subprocess.check_output(['nvidia-smi','-i','6,7','--query-gpu=index,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'],text=True).strip()
        free=all(int(line.split(',')[1])<100 and int(line.split(',')[2])==0 for line in gpu.splitlines())
        if shell==['352729','352729'] and free: break
        if time.time()>idle_deadline: raise TimeoutError('Previous work done but GPUs/shell are not idle')
        time.sleep(10)
    assert not Path('/proc/352729/root'+config['result_path']).exists()
    (root/'queue_launch.claim').mkdir()
    subprocess.run(['screen','-S','352153.exp','-p','0','-X','stuff',
        'timeout --signal=TERM --kill-after=60s 48h bash '+shlex.quote(str(root/'pipeline.sh'))+'\r'],check=True)
    for _ in range(60):
        if (root/'pipeline.claim').exists():
            save(root/'queue_state.json',{'status':'pipeline_launched','launched_unix':time.time(),
                'waited_for_prior_evaluation_and_comparison':True,'idle_GPU_readback':gpu})
            return
        time.sleep(1)
    raise TimeoutError('No pipeline claim observed after sending command')


if __name__=='__main__':
    try: main()
    except Exception as error:
        save(Path(sys.argv[1])/'queue_failed.json',{'status':'failed','error':str(error),'unix':time.time()})
        raise
