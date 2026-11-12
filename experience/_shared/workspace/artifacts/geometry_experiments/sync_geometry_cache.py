"""Sync only completed, explicitly requested derived experiment outputs.

Never transfers source audio, lab source code, original manifests, credentials
or checkpoints. No experiment, inference or remote process is started.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import paramiko


DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "geometry_cache"

# Fixed read-only inventory, executed on the SSH host without GPU libraries.
REMOTE_INVENTORY = r'''
from pathlib import Path
import hashlib,json

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()

def passed(p):
 return p.is_file() and p.read_text().strip()=='0'

base=Path('/home/mentee/analysis/embedding_geometry')
specs=[('19la_tv','20261105_fast_p0_01',{'T':'T','V':'V'},40363),
       ('21df_pilot','20261105_df_fast_01',{'DF_pilot':'R21DF_pilot'},20000),
       ('21la_eval','20261105_21la_full_03',{'21LA_eval':'R21LA'},148176),
       ('frozen_Q','20261105_frozen_all_01/Q',{'Q':'Q'},9861),
       ('frozen_19la_eval','20261105_frozen_all_01/19LA_eval',{'19LA_eval':'R19LA'},71237),
       ('frozen_DF_remaining','20261105_frozen_all_01/DF_remaining',{'DF_remaining':'R21DF_remaining'},513928)]
result=[]
for key,run_name,datasets,expected_count in specs:
 root=base/run_name
 item={'key':key,'run_name':run_name,'remote_root':str(root),'datasets':datasets}
 if not (passed(root/'extract.exit') and passed(root/'extract/complete')):
  item['status']='not_ready'
  item['progress']={}
  for rank in [0,1]:
   p=root/'extract'/('rank%d_progress.json'%rank)
   if p.is_file():item['progress'][str(rank)]=json.loads(p.read_text())
  result.append(item)
  continue
 audit=json.loads((root/'input_audit.json').read_text())
 summary=json.loads((root/'extract/summary.json').read_text())
 run=json.loads((root/'extract/run.json').read_text())
 assert audit['status']=='pass' and summary['status']=='pass'
 assert summary['sample_count']==expected_count
 assert summary['Q_evaluated']==('Q' in datasets)
 required=['input_audit.json','extract/summary.json','extract/run.json','extract/scores.parquet']
 for rank in [0,1]:
  required.extend('extract/rank%d_%s'%(rank,suffix) for suffix in
                  ['z.npy','h.npy','centroid.npy','rows.parquet','model.json'])
 optional=[]
 if key=='19la_tv':
  item['analysis_complete']=passed(root/'fast_analysis.exit') and passed(root/'fast_analysis/complete')
  if item['analysis_complete']:
   optional=['fast_analysis/summary.json','fast_analysis/profile_lock.json',
             'fast_analysis/C_V_predictions.parquet','fast_analysis/A_fisher_by_attack.npy']
   if (root/'output_verification_v1.json').is_file():optional.append('output_verification_v1.json')
 elif key=='21df_pilot':
  optional=['registry.json']
  item['analysis_complete']=passed(root/'evaluate.exit') and passed(root/'DF_evaluation/complete')
  if item['analysis_complete']:
   optional.extend('DF_evaluation/'+n for n in ['summary.json','metrics.json','segments.json','predictions.parquet'])
 else:
  item['analysis_complete']=False
  optional=['qc/summary.json']
 files=[]
 for name in required+optional:
  p=root/name
  assert p.is_file() and not p.is_symlink()
  p.resolve().relative_to(root.resolve())
  files.append({'path':name,'bytes':p.stat().st_size,'sha256':digest(p)})
 model=json.loads((root/'extract/rank0_model.json').read_text())
 identity={'run_name':run_name,'checkpoint_sha256':audit['checkpoint_sha256'],
           'manifest_sha256':audit['manifest_sha256'],'extraction_script_sha256':run['script_sha256'],
           'frontend_commit':model.get('frontend_commit'),'extraction_start_epoch':run['start_epoch']}
 if run.get('core_extractor_sha256'):
  identity.update(core_extractor_sha256=run['core_extractor_sha256'],
                  SSL_frontend_batch_size=run['SSL_frontend_batch_size'],
                  classifier_microbatch_size=run['classifier_microbatch_size'],
                  frontend_execution_mode=run['frontend_execution_mode'])
 fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
 item.update(status='ready',identity=identity,cache_fingerprint=fingerprint,
             sample_count=summary['sample_count'],files=files,
             frontend_weight_hash_verified=False)
 result.append(item)
print(json.dumps({'schema':1,'runs':result}))
'''


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sync(cache_root, password_path):
    cache_root = cache_root.resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    downloaded = skipped = transferred_bytes = 0
    try:
        client.connect("172.16.162.222", username="mentee",
                       password=password_path.read_text().splitlines()[0],
                       look_for_keys=False, allow_agent=False, timeout=15)
        stdin, stdout, stderr = client.exec_command("python3 -u -", timeout=30)
        stdin.write(REMOTE_INVENTORY)
        stdin.channel.shutdown_write()
        payload = stdout.read().decode("utf-8")
        error = stderr.read().decode("utf-8", errors="replace")
        if stdout.channel.recv_exit_status() != 0:
            raise RuntimeError("Read-only cache inventory failed: " + error[-2000:])
        inventory = json.loads(payload)
        catalog_path = cache_root / "catalog.json"
        old = json.loads(catalog_path.read_text(encoding="utf-8")) if catalog_path.exists() else {"runs": {}}
        catalog = {"schema": 1, "updated_utc": datetime.now(timezone.utc).isoformat(),
                   "local_result_export_requested": "2026-11-05; derived experiment outputs only",
                   "runs": old["runs"], "not_extracted": ["Q", "19LA_eval", "21LA_eval", "DF_remaining", "21DF_full"]}
        with client.open_sftp() as sftp:
            for run in inventory["runs"]:
                if run["status"] != "ready":
                    print(json.dumps({"run": run["key"], "status": run["status"], "progress": run.get("progress")}), flush=True)
                    if catalog['runs'].get(run['key'], {}).get('status') != 'ready':
                        catalog["runs"][run["key"]] = run
                    continue
                folder = cache_root / "runs" / run["run_name"] / run["cache_fingerprint"][:16]
                folder.mkdir(parents=True, exist_ok=True)
                previous_path = folder / "cache_manifest.json"
                if previous_path.exists():
                    previous = json.loads(previous_path.read_text(encoding="utf-8"))
                    assert previous["cache_fingerprint"] == run["cache_fingerprint"]
                for file in run["files"]:
                    relative = Path(file["path"])
                    assert not relative.is_absolute() and ".." not in relative.parts
                    target = folder / relative
                    target.resolve().relative_to(cache_root)
                    if target.exists():
                        if target.stat().st_size != file["bytes"] or digest(target) != file["sha256"]:
                            raise RuntimeError("Local cache checksum mismatch: " + str(target))
                        skipped += 1
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_suffix(target.suffix + ".part")
                    sftp.get(run["remote_root"] + "/" + file["path"], str(temporary))
                    assert temporary.stat().st_size == file["bytes"]
                    assert digest(temporary) == file["sha256"], "Transfer checksum mismatch"
                    temporary.replace(target)
                    downloaded += 1
                    transferred_bytes += file["bytes"]
                entry = {**run, "local_folder": folder.relative_to(cache_root).as_posix(),
                         "verified_utc": datetime.now(timezone.utc).isoformat()}
                save_json(previous_path, entry)
                catalog["runs"][run["key"]] = entry
                print(json.dumps({"run": run["key"], "status": "cached", "rows": run["sample_count"],
                                  "analysis_complete": run["analysis_complete"]}), flush=True)
        ready_datasets = {name for entry in catalog['runs'].values()
                          if entry.get('status') == 'ready' and entry.get('local_folder')
                          for name in entry.get('datasets', {})}
        catalog['not_extracted'] = [name for name in catalog['not_extracted'] if name not in ready_datasets]
        save_json(catalog_path, catalog)
    finally:
        client.close()
    result = {"downloaded_files": downloaded, "skipped_verified_files": skipped,
              "downloaded_bytes": transferred_bytes, "catalog": str(cache_root / "catalog.json")}
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--password-file", type=Path, default=Path.home() / ".ssh/mentee-password.txt")
    args = parser.parse_args()
    sync(args.cache_root, args.password_file)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
