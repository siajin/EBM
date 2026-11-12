"""Read-only monitor; --sync retrieves only compact derived results, never audio/models."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

import paramiko

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--sync', action='store_true'); cli = parser.parse_args()
    cfg = json.loads((HERE/'experiment.json').read_text()); root = cfg['root']
    c = paramiko.SSHClient(); c.load_system_host_keys()
    c.connect('172.16.162.222', username='mentee', password=(Path.home()/'.ssh/mentee-password.txt').read_text().splitlines()[0],
              look_for_keys=False, allow_agent=False, timeout=15)
    s = c.open_sftp()
    def read(path):
        try:
            with s.open(path) as f: raw = f.read().decode()
            return json.loads(raw) if path.endswith('.json') else raw.strip()
        except FileNotFoundError: return None
    try:
        names = ['queue_state.json', 'queue_failed.json', 'queue_cancelled.json', 'queue.pid', 'stage.txt', 'run.exit',
                 'unit.exit', 'freeze.exit', 'smoke.exit', 'full.exit', 'probe.exit', 'report.exit', 'complete.json']
        out = {name: read(root+'/'+name) for name in names}
        out['dependency_run_exit'] = read(cfg['wait_for_root']+'/run.exit')
        out['training_progress'] = [read(cfg['wait_for_root']+f'/train_progress_rank{r}.json') for r in [0, 1]]
        out['checked_KST'] = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
        for phase in cfg['phases']:
            out[phase+'_progress'] = [read(root+f'/full_{phase}_progress_rank{r}.json') for r in [0, 1]]
        if out['queue_cancelled.json'] and not out['stage.txt']:
            status = 'cancelled_before_evaluation_launch'
        elif out['run.exit'] == '0' and out['complete.json']:
            status = 'complete'
        elif out['run.exit'] is not None: status = 'failed_no_retry'
        elif out['queue_failed.json']: status = 'queue_failed_no_launch'
        elif out['stage.txt']: status = 'evaluation_'+out['stage.txt']+'_running'
        else: status = (out['queue_state.json'] or {}).get('status', 'unknown')
        if cli.sync:
            assert status == 'complete', 'Results not complete; sync refused'
            destination = HERE/'results'; destination.mkdir(exist_ok=True)
            allowed = ['complete.json', 'report.md', 'checkpoint_validation.json', 'probe_analysis.json', 'probe_effects.csv']
            for phase in cfg['phases']:
                allowed += [f'analysis_{phase}.json']+[f'{name}_{phase}.csv' for name in
                    ['full_metrics', 'minima', 'length_eer', 'transitions', 'attack_codec_errors', 'duration_weighting']]
            for filename in allowed: s.get(root+'/'+filename, str(destination/filename))
            assert hashlib.sha256((destination/'report.md').read_bytes()).hexdigest() == out['complete.json']['report_sha256']
            out['synced_derived_results'] = allowed
        state = dict(status=status, root=root, queue_registered=not bool(out['stage.txt']) and not bool(out['queue_cancelled.json']),
            evaluation_started=bool(out['stage.txt']), evaluation_complete=status == 'complete',
            last_check=out['checked_KST'], spending_ceiling_usd=0, automatic_retries=0)
        for path, value in [(HERE/'state.json', state), (HERE/'last_check.json', out)]:
            path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
        print(json.dumps(dict(**state, files=out), indent=2))
    finally: s.close(); c.close()


if __name__ == '__main__': main()
