#!/usr/bin/env bash
set -uo pipefail
umask 077
ROOT=/home/mentee/analysis/ebm_bona_only_bpl_validation_20261120_v1
test -f /.dockerenv || exit 90
test -f "$ROOT/prepared.json" || exit 91
mkdir "$ROOT/launch.claim" || exit 92
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH="$ROOT/code:$ROOT/code/tests:$ROOT"
export EBM_REFERENCE_GIT_ROOT=/home/mentee/code/EBM
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
trap 'rc=$?; printf "%s\n" "$rc" > "$ROOT/run.exit"' EXIT
cd "$ROOT/code" || exit 93
printf 'unit\n' > "$ROOT/stage.txt"
timeout --signal=TERM --kill-after=30s 2700s python -m unittest test_bona_only_bpl test_finalize test_shared_eval test_spoof_uid_bpl test_bona_uid_bpl test_dual_bpl test_role_relaxed test_role_iterator_lifecycle test_role_code_audit test_role_experiment test_short32_bpl test_short32_lifecycle test_positive_bpl -v > "$ROOT/unit.log" 2>&1
rc=$?; printf '%s\n' "$rc" > "$ROOT/unit.exit"
if [ "$rc" -ne 0 ]; then exit "$rc"; fi
printf 'runtime\n' > "$ROOT/stage.txt"
python "$ROOT/check_runtime.py" "$ROOT" > "$ROOT/runtime.log" 2>&1
rc=$?; printf '%s\n' "$rc" > "$ROOT/runtime.exit"
if [ "$rc" -ne 0 ]; then exit "$rc"; fi
printf 'smoke\n' > "$ROOT/stage.txt"
python tools/train_role_experiment.py --mode smoke --compact-class spoof --spoof-bpl-weight 0 --spoof-bpl-pairing all_cross --bona-bpl-weight 1 --bona-bpl-pairing all_cross --pair-target-cosine 1 --r-pull .9 --eval-input front64600_wrap --final-eval-policy deferred_shared --epochs 50 --seed 1 --num-workers 4 --frontend-revision 1a640f32ac3e39899438a2931f9924c02f080a54 --run-root "$ROOT/smoke" --timeout-hours 2 > "$ROOT/smoke.log" 2>&1
rc=$?; printf '%s\n' "$rc" > "$ROOT/smoke.exit"
if [ "$rc" -ne 0 ]; then exit "$rc"; fi
printf 'acceptance\n' > "$ROOT/stage.txt"
python "$ROOT/check_gpu.py" "$ROOT" > "$ROOT/acceptance.log" 2>&1
rc=$?; printf '%s\n' "$rc" > "$ROOT/acceptance.exit"
if [ "$rc" -ne 0 ]; then exit "$rc"; fi
printf 'validation_complete_no_training\n' > "$ROOT/stage.txt"
