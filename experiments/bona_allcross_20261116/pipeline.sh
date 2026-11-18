#!/usr/bin/env bash
set -uo pipefail
umask 077
ROOT=/home/mentee/analysis/ebm_spoof_dualbpl_pair1_pull09_e50_20261116_v1
RESULT=/results/CM/LA/EBM_e50_bs240_lr1e5_spoof_dualBPL_pair1_pull09_wrapeval_20261116_v1
REVISION=1a640f32ac3e39899438a2931f9924c02f080a54
test -f /.dockerenv || exit 90
test -f "$ROOT/prepared.json" || exit 91
mkdir "$ROOT/launch.claim" || exit 92
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH="$ROOT/code:$ROOT/code/tests"
export EBM_REFERENCE_GIT_ROOT=/home/mentee/code/EBM
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
date -Iseconds > "$ROOT/started.txt"
printf '%s\n' "$$" > "$ROOT/container_pipeline.pid"
START=$(date +%s)
DEADLINE=$((START + 168*3600))
trap 'rc=$?; printf "%s\n" "$rc" > "$ROOT/run.exit"' EXIT
cd "$ROOT/code" || exit 93

python - "$ROOT" <<'PY'
import hashlib,json,sys
from pathlib import Path
r=Path(sys.argv[1]);manifest=json.loads((r/'source_manifest.json').read_text())
for name,expected in manifest.items():
    assert hashlib.sha256((r/'code'/name).read_bytes()).hexdigest()==expected,name
print('Frozen source verified:',len(manifest))
PY
rc=$?
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

printf 'unit\n' > "$ROOT/stage.txt"
timeout --signal=TERM --kill-after=30s 2700s python -m unittest test_dual_bpl test_role_relaxed test_role_iterator_lifecycle test_role_code_audit test_role_experiment test_short32_bpl test_short32_lifecycle test_positive_bpl -v > "$ROOT/unit.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$ROOT/unit.exit"
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

printf 'validate\n' > "$ROOT/stage.txt"
python tools/train_role_experiment.py --check-runtime --pair-target-cosine 1.0 --r-pull 0.9 --bona-bpl-weight 1.0 --eval-input front64600_wrap --frontend-revision "$REVISION" > "$ROOT/validate.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$ROOT/validate.exit"
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

printf 'runtime_acceptance\n' > "$ROOT/stage.txt"
python - "$ROOT" <<'PY'
import json,sys,torch
from pathlib import Path
from role_model import RoleBLinear
from role_objectives import RoleConfig
from role_lifecycle import load_scoring_head
r=Path(sys.argv[1]);torch.set_num_threads(1)
torch.manual_seed(1);old=RoleBLinear(RoleConfig());old_rng=torch.get_rng_state().clone()
torch.manual_seed(1);new=RoleBLinear(RoleConfig(bona_bpl_weight=1))
assert torch.equal(old_rng,torch.get_rng_state())
assert all(torch.equal(v,new.state_dict()[k]) for k,v in old.state_dict().items())
assert sum(p.numel() for p in new.parameters())==3744384
records=[]
for name,target,pull in [('EBM_e50_bs240_lr1e5_spoof_center_OC_SPL_20261112_v1',1.,.9),('EBM_e50_bs240_lr1e5_spoof_pair09_pull08_wrapeval_20261113_v1',.9,.8)]:
 head,manifest=load_scoring_head('/results/CM/LA/'+name,50)
 assert head.role==RoleConfig(pair_target_cosine=target,r_pull=pull)
 assert head.role.bona_bpl_weight==0 and head.role.score_sign==-1
 records.append(dict(name=name,role=head.role.metadata(),format=manifest['artifact_format']))
out=dict(status='passed',initial_state_exact=True,initial_rng_exact=True,parameters=3744384,old_checkpoints_read_only=records)
(r/'runtime_acceptance.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out))
PY
rc=$?
printf '%s\n' "$rc" > "$ROOT/runtime_acceptance.exit"
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

printf 'smoke\n' > "$ROOT/stage.txt"
python tools/train_role_experiment.py --mode smoke --compact-class spoof --pair-target-cosine 1.0 --r-pull 0.9 --bona-bpl-weight 1.0 --eval-input front64600_wrap --epochs 50 --seed 1 --num-workers 4 --frontend-revision "$REVISION" --run-root "$ROOT/smoke" --timeout-hours 2 > "$ROOT/smoke.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$ROOT/smoke.exit"
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

printf 'smoke_acceptance\n' > "$ROOT/stage.txt"
python - "$ROOT" <<'PY'
import csv,json,math,sys
from pathlib import Path
r=Path(sys.argv[1]);s=r/'smoke';receipt=json.loads((s/'smoke_complete.json').read_text())
assert receipt['dual_bpl_graph_checked'] is True
for rank in receipt['ranks']:
 assert [v['bona_pair_count'] for v in rank['values']]==[144,144,36]
 assert [v['pair_count'] for v in rank['values']]==[11664,3600,11664]
 assert all(math.isfinite(v['bona_short_gradient_norm']) and v['bona_short_gradient_norm']>0 for v in rank['values'])
checks=[]
for phase in ['19LA','21LA','21DF']:
 rows=list(csv.DictReader((s/'stages'/f'eval_0_{phase}_scores.csv').open()))
 meta=json.loads((s/'stages'/f'eval_0_{phase}.json').read_text())
 assert len(rows)==64 and [int(v['index']) for v in rows]==list(range(64))
 assert len({v['uid'] for v in rows})==64
 assert sum(int(v['label']) for v in rows)==32
 assert all(math.isfinite(float(v['score_bona'])) and float(v['score_bona'])==-float(v['raw_center_cosine']) for v in rows)
 assert meta['protocol']['input']=='wrap64600/no-mask' and meta['scope']=='smoke_subset'
 ordered=sorted([(float(v['score_bona']),int(v['label'])) for v in rows],reverse=True)
 fp=tp=0;prior=(0.,1.);eer=None;i=0
 while i<len(ordered):
  score=ordered[i][0]
  while i<len(ordered) and ordered[i][0]==score:
   tp+=ordered[i][1];fp+=1-ordered[i][1];i+=1
  point=(fp/32,1-tp/32)
  if point[0]>=point[1]:
   d0=prior[0]-prior[1];d1=point[0]-point[1]
   mix=-d0/(d1-d0);eer=100*(prior[0]+mix*(point[0]-prior[0]));break
  prior=point
 assert eer is not None and abs(eer-meta['eer_percent'])<1e-12
 checks.append(dict(phase=phase,rows=64,independent_eer_percent=eer,error=abs(eer-meta['eer_percent'])))
out=dict(status='passed',gradient_and_pair_counts=True,wrapped_eval_checks=checks)
(r/'smoke_acceptance.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
PY
rc=$?
printf '%s\n' "$rc" > "$ROOT/smoke_acceptance.exit"
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

NOW=$(date +%s)
LEFT=$((DEADLINE-NOW))
if [ "$LEFT" -le 0 ]; then exit 124; fi
HOURS=$(python -c "print($LEFT / 3600)")
printf 'train\n' > "$ROOT/stage.txt"
python tools/train_role_experiment.py --mode train --compact-class spoof --pair-target-cosine 1.0 --r-pull 0.9 --bona-bpl-weight 1.0 --eval-input front64600_wrap --epochs 50 --seed 1 --num-workers 4 --frontend-revision "$REVISION" --run-root "$RESULT" --smoke-receipt "$ROOT/smoke/smoke_complete.json" --timeout-hours "$HOURS" > "$ROOT/train.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$ROOT/train.exit"
if [ "$rc" -ne 0 ]; then exit "$rc"; fi
printf 'complete\n' > "$ROOT/stage.txt"
