import hashlib,json,sys
from pathlib import Path
import torch
root=Path(sys.argv[1]);cfg=json.loads((root/'experiment.json').read_text())
for n,h in cfg['source_sha256'].items():assert hashlib.sha256((root/'code'/n).read_bytes()).hexdigest()==h,n
for rank in [0,1]:
 done=json.loads((root/f'train_done_rank{rank}.json').read_text());assert done['logical_steps']==done['backend_updates']==done['center_updates']==5400
 metrics=[json.loads(s) for s in (root/f'train_metrics_rank{rank}.jsonl').read_text().splitlines()]
 assert len(metrics)==5400 and {x['epoch'] for x in metrics}==set(range(1,51))
 assert all(x['bpl43_pairs']==x['bona']**2 and x['bpl32_pairs']==x['eligible']**2 for x in metrics)
model=Path(cfg['host_result_path'])/'model';stages=model.parent/'short32_stages'
for epoch in range(5,51,5):
 marker=json.loads((stages/f'checkpoint_{epoch}.complete.json').read_text());assert marker['epoch']==epoch and marker['status']=='committed'
 for name,info in marker['files'].items():assert hashlib.sha256((model/name).read_bytes()).hexdigest()==info['sha256'],name
state=torch.load(model/'TrainState_50.pt',map_location='cpu',weights_only=True)
assert state['epoch']==50 and {int(x['step']) for x in state['optimizer']['state'].values()}=={5400}
assert len(state['rng_by_rank'])==2
report=json.loads((stages/'complete.json').read_text());assert report['saved_epochs']==list(range(5,51,5))
expected={'19LA':71237,'21LA':148176,'21DF':533928}
for result in report['results']:
 if result['status']=='complete':
  record=json.loads((stages/f"eval_{result['epoch']}_{result['phase']}.json").read_text());assert record['rows']==expected[result['phase']]
for ref in cfg['reference_checkpoints'].values():
 assert hashlib.sha256(Path(ref['path']).read_bytes()).hexdigest()==ref['sha256']
 if 'TrainState_path' in ref:assert hashlib.sha256(Path(ref['TrainState_path']).read_bytes()).hexdigest()==ref['TrainState_sha256']
(root/'final_audit.json').write_text(json.dumps(dict(status=report['status'],training_epochs=50,steps_per_rank=5400,checkpoints=10,evaluation_results=report['results'],minimum_eer=report['minimum_eer'],source_and_references_unchanged=True),indent=2)+'\n')
if report['status']!='complete':raise SystemExit(2)
