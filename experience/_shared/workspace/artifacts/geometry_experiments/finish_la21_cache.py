"""One-shot completion: watch existing extraction, sync, validate offline.

Never starts, stops or restarts a server experiment. The user's cache request
authorizes transfer of these completed derived outputs only.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts/geometry_cache/local_analysis/21la_cache_finish_v1'


def invoke(script):
    result = subprocess.run([sys.executable, '-X', 'utf8', str(SCRIPTS / script)],
                            cwd=ROOT, capture_output=True, text=True, encoding='utf-8', timeout=600)
    (OUT / (script + '.stdout.txt')).write_text(result.stdout, encoding='utf-8')
    (OUT / (script + '.stderr.txt')).write_text(result.stderr, encoding='utf-8')
    if result.returncode:
        raise RuntimeError(f'{script} failed: ' + result.stderr[-1800:])
    return result.stdout


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    try:
        failures = 0
        while True:
            result = subprocess.run([sys.executable, '-X', 'utf8', str(SCRIPTS / 'la21_cache_status.py')],
                                    cwd=ROOT, capture_output=True, text=True, encoding='utf-8', timeout=45)
            if result.returncode:
                failures += 1
                print(json.dumps({'status': 'status_read_retry', 'attempt': failures}), flush=True)
                if failures >= 3:
                    raise RuntimeError('Three status reads failed: ' + result.stderr[-1500:])
                time.sleep(30)
                continue
            failures = 0
            status = json.loads(result.stdout)
            assert status['run'] == '20261105_21la_full_03'
            with (OUT / 'polls.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(status) + '\n')
            print(json.dumps({k: status[k] for k in ['UTC', 'exits', 'completed', 'total', 'percent', 'ETA_minutes', 'ready_for_sync'] if k in status}), flush=True)
            if any(code != '0' for code in status['exits'].values()):
                raise RuntimeError('Extraction ended unsuccessfully; no restart performed')
            if status['ready_for_sync']:
                break
            time.sleep(40)
        for script in ['sync_geometry_cache.py', 'verify_la21_cache.py', 'analyze_geometry_cache.py']:
            print(json.dumps({'status': 'running_completion_step', 'script': script}), flush=True)
            invoke(script)
        verification = json.loads((ROOT / 'artifacts/geometry_cache/local_analysis/21la_cache_validation_v1/verification.json').read_text(encoding='utf-8'))
        assert verification['status'] == 'pass'
        report = {'status': 'complete', 'UTC': datetime.now(timezone.utc).isoformat(),
                  'dataset': '21LA_eval', 'rows': verification['rows'],
                  'cache_fingerprint': verification['cache_fingerprint'],
                  'native_metrics': verification['native_metrics_independent'],
                  'new_training': False, 'server_experiments_started_by_this_helper': False}
        (OUT / 'complete.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(report), flush=True)
    except Exception as error:
        (OUT / 'failure.json').write_text(json.dumps({'status': 'failed', 'error': str(error)}, indent=2) + '\n', encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
