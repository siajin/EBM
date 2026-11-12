"""Read server metrics and retain aggregate B/C comparisons; no GPU work."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'geometry_experiments'))
from run_padding_causal import connect

REMOTE = r'''
import datetime,hashlib,json,math,re,statistics
from pathlib import Path
names={'B':'EBM_e50_bs240_lr1e5_B_zeropad_short303','C':'EBM_e50_bs240_lr1e5_C_nativeLong_short303'}
out={'checked_KST':datetime.datetime.now().isoformat(timespec='seconds'),'batches_per_epoch':108,'models':{}}
for key,name in names.items():
 root=Path('/home/mentee/results/CM/LA')/name
 data={};sources={}
 for metric in ['loss1','loss2','oc_long','bpl','oc_short']:
  path=root/'metric'/f'{metric}.txt';raw=path.read_bytes()
  values=[float(line.rsplit(':',1)[-1]) for line in raw.decode().splitlines()]
  assert all(math.isfinite(v) for v in values)
  data[metric]=values;sources[metric]={'path':str(path),'rows':len(values),'sha256_snapshot':hashlib.sha256(raw).hexdigest()}
 devraw=(root/'metric/V/EER.txt').read_text()
 dev={int(e):float(v) for e,v in re.findall(r'\[(\d+)\] V/EER: ([^\s]+)',devraw)}
 out['models'][key]={'sources':sources,'dev_EER_percent':dev,'epochs':[]}
 for epoch in range(1,21):
  row={'epoch':epoch,'batches':108}
  for metric,values in data.items():
   block=values[(epoch-1)*108:epoch*108];assert len(block)==108
   row[metric]=statistics.fmean(block)
  out['models'][key]['epochs'].append(row)
out['comparison']=[]
for b,c in zip(out['models']['B']['epochs'],out['models']['C']['epochs']):
 out['comparison'].append({'epoch':b['epoch'],**{m+'_C_minus_B':c[m]-b[m] for m in ['loss1','loss2','oc_long','bpl','oc_short']}})
print(json.dumps(out))
'''

client = connect()
try:
    _, stdout, stderr = client.exec_command("python - <<'PY'\n" + REMOTE + "\nPY", timeout=40)
    raw = stdout.read().decode()
    error = stderr.read().decode()
    assert stdout.channel.recv_exit_status() == 0, error[-1500:]
    result = json.loads(raw)
finally:
    client.close()

destination = HERE / 'B_C_comparison_20261108_2000.json'
destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'checked_KST': result['checked_KST'], 'models': {
    key: {'dev': model['dev_EER_percent'], 'epochs': [row for row in model['epochs'] if row['epoch'] in [1,5,10,15,20]],
          'mean_epochs16_20': {m: statistics.fmean(row[m] for row in model['epochs'][15:20]) for m in ['loss1','loss2','oc_long','bpl','oc_short']}}
    for key, model in result['models'].items()}}, indent=2))
