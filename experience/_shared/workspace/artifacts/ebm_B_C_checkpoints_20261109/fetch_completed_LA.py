"""Retrieve only finalized derived LA scores, verifying the completion hashes."""
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'artifacts/geometry_experiments'))
from run_padding_causal import connect

ROOT='/home/mentee/analysis/ebm_B_C_checkpoints_20261109'
DEST=HERE/'LA_analysis'
DEST.mkdir(exist_ok=True)
client=connect()
try:
 records={}
 with client.open_sftp() as s:
  for phase in ['19LA','21LA']:
   for rank in range(2):
    name=f'scores_{phase}_rank{rank}.csv';done=f'done_{phase}_rank{rank}.json'
    with s.open(ROOT+'/'+done) as f:record=json.loads(f.read().decode())
    destination=DEST/name
    if not destination.exists():
     partial=destination.with_suffix('.download');s.get(ROOT+'/'+name,str(partial))
     assert hashlib.sha256(partial.read_bytes()).hexdigest()==record['sha256']
     partial.replace(destination)
    assert hashlib.sha256(destination.read_bytes()).hexdigest()==record['sha256']
    (DEST/done).write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    records[name]={'remote':ROOT+'/'+name,'sha256':record['sha256'],'rows':record['rows'],'bytes':destination.stat().st_size}
 (DEST/'sources.json').write_text(json.dumps(records,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(records,indent=2))
finally:client.close()
