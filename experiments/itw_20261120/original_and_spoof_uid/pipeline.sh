#!/usr/bin/env bash
set -uo pipefail
umask 077
ROOT=/home/mentee/analysis/ebm_itw_all_models_20261120_v1
SOURCE=/home/mentee/analysis/ebm_spoof_uid_bonaallcross_pair1_pull09_e50_20261119_v1/code
test -f /.dockerenv || exit 90
test -f "$ROOT/manifest.json" || exit 91
mkdir "$ROOT/launch.claim" || exit 92
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH="$SOURCE:$ROOT:$SOURCE/tests"
export EBM_ITW_SOURCE_ROOT="$SOURCE"
export EBM_REFERENCE_GIT_ROOT=/home/mentee/code/EBM
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
date -Iseconds > "$ROOT/started.txt"
trap 'rc=$?; printf "%s\n" "$rc" > "$ROOT/run.exit"' EXIT
DEADLINE=$(( $(date +%s) + 24*3600 ))
cd "$ROOT" || exit 93
printf 'unit\n' > "$ROOT/stage.txt"
timeout --signal=TERM --kill-after=30s 2700s python -m unittest test_itw -v > "$ROOT/unit.log" 2>&1
rc=$?; printf '%s\n' "$rc" > "$ROOT/unit.exit"
if [ "$rc" -ne 0 ]; then exit "$rc"; fi
for MODE in smoke full; do
 LEFT=$((DEADLINE-$(date +%s)))
 if [ "$LEFT" -le 0 ]; then exit 124; fi
 HOURS=$(python -c "print($LEFT/3600)")
 printf '%s\n' "$MODE" > "$ROOT/stage.txt"
 python "$ROOT/evaluate_itw.py" "$ROOT" "$MODE" --timeout-hours "$HOURS" > "$ROOT/${MODE}.log" 2>&1
 rc=$?; printf '%s\n' "$rc" > "$ROOT/${MODE}.exit"
 if [ "$rc" -ne 0 ]; then exit "$rc"; fi
done
printf 'report\n' > "$ROOT/stage.txt"
python "$ROOT/report.py" "$ROOT" > "$ROOT/report.log" 2>&1
rc=$?; printf '%s\n' "$rc" > "$ROOT/report.exit"
if [ "$rc" -ne 0 ]; then exit "$rc"; fi
printf 'complete\n' > "$ROOT/stage.txt"
