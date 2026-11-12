"""Read-only queue status; optional retrieval of derived scores and summaries only."""
import argparse
import json
from pathlib import Path
import sys
import hashlib
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'artifacts/geometry_experiments'))
from run_padding_causal import connect
REMOTE='/home/mentee/analysis/ebm_repetition_shortcut_20261111_v1'

def main(sync=False):
    result={};c=connect()
    try:
        with c.open_sftp() as s:
            for file in ['queue_state.json','queue.pid','manual_queue_superseded.json','manual_launch.json','manual_stop_readback.json','stage.txt','started.txt','run.exit','finished.txt','full_smoke.exit','probe_smoke.exit','full.exit','probe.exit','analysis.exit','full_smoke_progress_rank0.json','full_smoke_progress_rank1.json','probe_smoke_progress_rank0.json','probe_smoke_progress_rank1.json','full_progress_rank0.json','full_progress_rank1.json','probe_progress_rank0.json','probe_progress_rank1.json','full_complete.json','probe_complete.json','analysis_complete.json']:
                try:
                    with s.open(REMOTE+'/'+file) as stream:value=stream.read().decode()
                    result[file]=json.loads(value) if file.endswith('.json') else value.strip()
                except FileNotFoundError:pass
            # Bounded tails, never inspect screen contents or unrelated logs.
            for file in ['queue.log','full_smoke.log','probe_smoke.log','full.log','probe.log','analysis.log','pipeline.log']:
                try:
                    with s.open(REMOTE+'/'+file) as stream:
                        size=stream.stat().st_size;stream.seek(max(0,size-3000));result[file+'_tail']=stream.read().decode(errors='replace').splitlines()[-6:]
                except FileNotFoundError:pass
            if sync:
                assert result.get('run.exit')=='0' and result.get('analysis_complete.json',{}).get('status')=='analysis_complete','Not complete'
                allow=['full_complete.json','probe_complete.json','analysis_complete.json','new_full_metrics.csv','new_length_attack_codec_errors.csv','new_paired_effects.csv','new_dose_controls.csv']
                for kind in ['full','probe']:
                    for rank in [0,1]:allow.extend([f'{kind}_rank{rank}.csv',f'{kind}_done_rank{rank}.json'])
                dest=HERE/'new_results';dest.mkdir(exist_ok=True)
                for file in allow:s.get(REMOTE+'/'+file,str(dest/file))
                for kind in ['full','probe']:
                    for rank in [0,1]:
                        done=json.loads((dest/f'{kind}_done_rank{rank}.json').read_text())
                        assert hashlib.sha256((dest/f'{kind}_rank{rank}.csv').read_bytes()).hexdigest()==done['sha256']
                result['derived_retrieval']='all_allowed_files_hash_verified'
        source="""
import hashlib,json,subprocess
from pathlib import Path
root=Path(ROOT);dep=Path('/home/mentee/analysis/ebm_baseline240_mild_e20_20261111_v2')
pid=int((root/'queue.pid').read_text());cmd=Path('/proc')/str(pid)/'cmdline'
qcmd=cmd.read_bytes().replace(b'\\0',b' ').decode() if cmd.exists() else ''
manifest=json.loads((root/'payload_hashes.json').read_text())
snapshot=json.loads((root/'source_hashes.json').read_text())
protocol=json.loads((root/'protocol.json').read_text())
ok=all(hashlib.sha256((root/f).read_bytes()).hexdigest()==h for f,h in manifest.items())
snapshot_ok=all(hashlib.sha256((root/'code'/f).read_bytes()).hexdigest()==h for f,h in snapshot.items())
checkpoint_ok=all(hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256'] for r in protocol['checkpoints'].values())
progress={}
for name in ['train_progress_rank0.json','train_progress_rank1.json']:
 if (dep/name).exists():progress[name]=json.loads((dep/name).read_text())
print(json.dumps({'queue_process_alive':str(root/'wait_and_launch.py') in qcmd,'new_payload_hashes_unchanged':ok,'new_snapshot_hashes_unchanged':snapshot_ok,'reference_checkpoint_hashes_unchanged':checkpoint_ok,'operational_git_clean':not subprocess.check_output(['git','status','--porcelain'],cwd='/home/mentee/code/EBM',text=True).strip(),'existing_pipeline_stage':(dep/'stage.txt').read_text().strip(),'existing_pipeline_exit':(dep/'pipeline.exit').read_text().strip() if (dep/'pipeline.exit').exists() else None,'new_launch_claim_exists':(root/'launch.claim').exists(),'existing_progress':progress}))
""".replace('ROOT',repr(REMOTE))
        _,stdout,stderr=c.exec_command("python - <<'PY'\n"+source+"\nPY",timeout=40)
        observation=stdout.read().decode();error=stderr.read().decode();assert stdout.channel.recv_exit_status()==0,error[-2000:]
        result['preservation_and_dependency']=json.loads(observation)
        if 'run.exit' in result:
            result['effective_status']='complete' if result['run.exit']=='0' else 'failed_no_retry'
        elif 'stage.txt' in result:
            result['effective_status']='running_'+result['stage.txt']
        else:
            result['effective_status']=result.get('queue_state.json',{}).get('status','unknown')
    finally:c.close()
    (HERE/'last_check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    brief={k:v for k,v in result.items() if k!='analysis_complete.json'}
    print(json.dumps(brief,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sync',action='store_true');main(p.parse_args().sync)
