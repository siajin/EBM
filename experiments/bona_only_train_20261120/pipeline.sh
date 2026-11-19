#!/usr/bin/env bash
set -uo pipefail
umask 077
ROOT=/home/mentee/analysis/ebm_bona_only_allcross_pair1_pull09_e50_20261120_v1
RESULT=/results/CM/LA/EBM_e50_bs240_lr1e5_bonaOnlyAllcrossBPL_spoofOff_pair1_pull09_wrapeval_20261120_v1
SMOKE=/home/mentee/analysis/ebm_bona_only_bpl_validation_20261120_v1/smoke
REVISION=1a640f32ac3e39899438a2931f9924c02f080a54
test -f /.dockerenv || exit 90
test -f "$ROOT/prepared.json" || exit 91
mkdir "$ROOT/launch.claim" || exit 92
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH="$ROOT/code:$ROOT/code/tests:$ROOT"
export EBM_REFERENCE_GIT_ROOT=/home/mentee/code/EBM
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
date -Iseconds > "$ROOT/started.txt"
printf '%s\n' "$$" > "$ROOT/container_pipeline.pid"
trap 'rc=$?; printf "%s\n" "$rc" > "$ROOT/run.exit"' EXIT
DEADLINE=$(( $(date +%s) + 168*3600 ))
cd "$ROOT/code" || exit 93
printf 'preflight\n' > "$ROOT/stage.txt"
python "$ROOT/preflight.py" "$ROOT" > "$ROOT/preflight.log" 2>&1
rc=$?; printf '%s\n' "$rc" > "$ROOT/preflight.exit"
if [ "$rc" -ne 0 ]; then exit "$rc"; fi
LEFT=$((DEADLINE-$(date +%s)))
if [ "$LEFT" -le 0 ]; then exit 124; fi
HOURS=$(python -c "print($LEFT/3600)")
printf 'train\n' > "$ROOT/stage.txt"
python tools/train_role_experiment.py --mode train --compact-class spoof --spoof-bpl-weight 0 --spoof-bpl-pairing all_cross --bona-bpl-weight 1 --bona-bpl-pairing all_cross --pair-target-cosine 1 --r-pull .9 --eval-input front64600_wrap --final-eval-policy deferred_shared --epochs 50 --seed 1 --num-workers 4 --frontend-revision "$REVISION" --run-root "$RESULT" --smoke-receipt "$SMOKE/smoke_complete.json" --timeout-hours "$HOURS" > "$ROOT/train.log" 2>&1
rc=$?; printf '%s\n' "$rc" > "$ROOT/train.exit"
if [ "$rc" -ne 0 ]; then exit "$rc"; fi
printf 'shared_prepare\n' > "$ROOT/stage.txt"
python "$ROOT/finalize_bona_only.py" prepare_shared "$ROOT" "$RESULT" > "$ROOT/shared_prepare.log" 2>&1
rc=$?; printf '%s\n' "$rc" > "$ROOT/shared_prepare.exit"
if [ "$rc" -ne 0 ]; then exit "$rc"; fi
for MODE in smoke full; do
 LEFT=$((DEADLINE-$(date +%s)))
 if [ "$LEFT" -le 0 ]; then exit 124; fi
 HOURS=$(python -c "print($LEFT/3600)")
 printf 'shared_%s\n' "$MODE" > "$ROOT/stage.txt"
 python "$ROOT/shared_eval/shared_eval.py" "$ROOT/shared_eval" "$MODE" --timeout-hours "$HOURS" > "$ROOT/shared_${MODE}.log" 2>&1
 rc=$?; printf '%s\n' "$rc" > "$ROOT/shared_eval/${MODE}.exit"
 if [ "$rc" -ne 0 ]; then printf '%s\n' "$rc" > "$ROOT/shared_eval/run.exit"; exit "$rc"; fi
done
printf '0\n' > "$ROOT/shared_eval/run.exit"
LEFT=$((DEADLINE-$(date +%s)))
if [ "$LEFT" -le 0 ]; then exit 124; fi
if [ "$LEFT" -gt 1800 ]; then LEFT=1800; fi
printf 'compare\n' > "$ROOT/stage.txt"
timeout --signal=TERM --kill-after=30s "$LEFT"s python "$ROOT/finalize_bona_only.py" compare "$ROOT" "$RESULT" > "$ROOT/compare.log" 2>&1
rc=$?; printf '%s\n' "$rc" > "$ROOT/compare.exit"
if [ "$rc" -ne 0 ]; then exit "$rc"; fi
printf 'complete\n' > "$ROOT/stage.txt"
