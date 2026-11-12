"""Copy only completed derived dev scores and verify the producer's SHA256."""
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'artifacts/geometry_experiments'))
from run_padding_causal import connect

ROOT='/home/mentee/analysis/ebm_B_C_checkpoints_20261109'
DEST=HERE/'dev_analysis'
DEST.mkdir(exist_ok=True)
client=connect()
try:
 records={}
 with client.open_sftp() as s:
  for rank in range(2):
   name=f'scores_dev_rank{rank}.csv';done=f'done_dev_rank{rank}.json'
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
