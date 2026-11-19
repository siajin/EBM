"""Verify actual three-update two-GPU smoke, evaluation and all protected artifacts."""
import csv,hashlib,json,math,subprocess,sys
from pathlib import Path
import torch
from role_lifecycle import load_scoring_head
from role_objectives import RoleConfig
from role_runtime import validate_smoke_receipt
r=Path(sys.argv[1]);s=r/'smoke';receipt=json.loads((s/'smoke_complete.json').read_text())
args=json.loads((s/'experiment.json').read_text())['args'];validate_smoke_receipt(args,s/'smoke_complete.json')
assert receipt['status']=='passed' and receipt['loader_batches_seen']==108 and receipt['eval_phases']==['19LA']
assert receipt['spoof_bpl_disabled_graph_checked'] and receipt['dual_bpl_graph_checked']
for rank in receipt['ranks']:
 assert [v['batch'] for v in rank['values']]==[1,106,108]
 assert [v['bona_pair_count'] for v in rank['values']]==[144,144,36]
 assert all(v['pair_count']==0 and v['pair_spoof']==0 and v['spoof_short_gradient_norm']==0 and v['bona_short_gradient_norm']>0 for v in rank['values'])
 assert all(v['optimizer_steps']==2 and v['center_updates']==1 and v['center_update_norm']>0 and v['backend_gradient_norm']>0 and v['center_gradient_norm']>0 for v in rank['values'])
 assert all(math.isfinite(v) for row in rank['values'] for v in row.values() if isinstance(v,float))
old_smoke=Path('/home/mentee/analysis/ebm_spoof_uid_bonaallcross_pair1_pull09_e50_20261119_v1/smoke/smoke_complete.json')
assert receipt['data_fingerprint']==json.loads(old_smoke.read_text())['data_fingerprint']
head,manifest=load_scoring_head(s,0,allow_smoke=True);assert head.role==RoleConfig(bona_bpl_weight=1,spoof_bpl_weight=0) and manifest['purpose']=='smoke'
try:load_scoring_head(s,0)
except ValueError:pass
else:raise AssertionError('Smoke must not masquerade as trained model')
state=torch.load(s/'model/TrainState_0.pt',map_location='cpu',weights_only=True)
assert len(state['optimizer']['state'])==182 and all(float(v['step'])==3 for v in state['optimizer']['state'].values())
assert len(state['rng_by_rank'])==2
rows=list(csv.DictReader((s/'stages/eval_0_19LA_scores.csv').open()));meta=json.loads((s/'stages/eval_0_19LA.json').read_text())
assert len(rows)==64 and [int(v['index']) for v in rows]==list(range(64)) and len({v['uid'] for v in rows})==64
assert sum(int(v['label']) for v in rows)==32 and meta['protocol']['input']=='wrap64600/no-mask' and meta['scope']=='smoke_subset'
assert all(math.isfinite(float(v['score_bona'])) and float(v['score_bona'])==-float(v['raw_center_cosine']) for v in rows)
ordered=sorted([(float(v['score_bona']),int(v['label'])) for v in rows],reverse=True);fp=tp=0;prior=(0.,1.);eer=None;i=0
while i<len(ordered):
 score=ordered[i][0]
 while i<len(ordered) and ordered[i][0]==score:tp+=ordered[i][1];fp+=1-ordered[i][1];i+=1
 point=(fp/32,1-tp/32)
 if point[0]>=point[1]:
  d0=prior[0]-prior[1];d1=point[0]-point[1];mix=-d0/(d1-d0);eer=100*(prior[0]+mix*(point[0]-prior[0]));break
 prior=point
assert eer is not None and abs(eer-meta['eer_percent'])<1e-12
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
for n,h in json.loads((r/'source_manifest.json').read_text()).items():assert sha(r/'code'/n)==h,n
for n,h in json.loads((r/'control_manifest.json').read_text()).items():assert sha(r/n)==h,n
protected=json.loads((r/'preservation.json').read_text())
for p,h in protected['files'].items():assert sha(Path(p))==h,p
assert subprocess.check_output(['git','-C','/home/mentee/code/EBM','status','--porcelain'],text=True)==''
out=dict(status='passed',actual_two_GPU=True,loader_batches_seen=108,trained_batches=[1,106,108],bona_pairs=[144,144,36],spoof_pairs=[0,0,0],spoof_short_gradient_exact_zero=True,bona_short_gradient_positive=True,center_synchronized=True,v6_save_restore_passed=True,Adam_states=182,Adam_steps=3,eval_phases=['19LA'],smoke_eval_rows=64,independent_ROC_error_pp=abs(eer-meta['eer_percent']),initialization=json.loads((r/'runtime_acceptance.json').read_text()),source_model_prior_controls_unchanged=True,data_fingerprint_matches_prior=True,full_training_started=False,new_spending_usd=0,automatic_retries=0)
(r/'gpu_acceptance.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
