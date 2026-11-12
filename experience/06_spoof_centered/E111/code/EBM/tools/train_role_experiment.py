"""Validate, smoke, or train an isolated original240 class-role experiment.

Default is read-only validation. No work starts on import or without an explicit
smoke/train mode. There is no resume/retry/queue and no change to old entry points.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import math
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def wait_workers(context, timeout_seconds):
    """Bound the complete job and stop only verified children/groups we own."""
    deadline = time.monotonic() + timeout_seconds
    completed = False
    owned_groups = set()
    def remember_groups():
        if os.name == 'posix':
            for process in context.processes:
                try:
                    if process.is_alive() and os.getpgid(process.pid) == process.pid:
                        owned_groups.add(process.pid)
                except ProcessLookupError:
                    pass
    def stop_groups(sig):
        for group in owned_groups:
            try:
                os.killpg(group, sig)
            except ProcessLookupError:
                pass
    try:
        while time.monotonic() < deadline:
            remember_groups()
            if context.join(timeout=min(1., max(.001, deadline-time.monotonic()))):
                completed = True
                return
        raise TimeoutError('Role experiment exceeded its total wall-clock limit')
    finally:
        if not completed:
            import signal
            remember_groups()
            stop_groups(signal.SIGTERM)
            for process in context.processes:
                if process.is_alive() and process.pid not in owned_groups:
                    process.terminate()
            for process in context.processes:
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5)
            if os.name == 'posix':
                stop_groups(signal.SIGKILL)


def main(argv=None):
    from role_runtime import (options, signature, source_fingerprint, validate_runtime,
                              validate_smoke_receipt, verify_reference, worker)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['validate', 'smoke', 'train'], default='validate')
    parser.add_argument('--compact-class', choices=['spoof', 'bonafide'], default='spoof')
    parser.add_argument('--run-root')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--frontend-revision')
    parser.add_argument('--path-19LA', default='/data/ASVspoof2019')
    parser.add_argument('--path-21LA', default='/data/ASVspoof2021_LA_eval')
    parser.add_argument('--path-21DF', default='/data/ASVspoof2021_DF')
    parser.add_argument('--check-runtime', action='store_true')
    parser.add_argument('--smoke-receipt')
    parser.add_argument('--timeout-hours', type=float, default=168.)
    parsed = parser.parse_args(argv)
    if not math.isfinite(parsed.timeout_hours) or parsed.timeout_hours <= 0:
        parser.error('--timeout-hours must be finite and positive')
    args = options(parsed.compact_class, epoch=parsed.epochs, seed=parsed.seed, workers=parsed.num_workers,
                   frontend_revision=parsed.frontend_revision, path_19LA=parsed.path_19LA,
                   path_21LA=parsed.path_21LA, path_21DF=parsed.path_21DF, run_root=parsed.run_root)
    args['reference'] = verify_reference()
    args['source_sha256'] = source_fingerprint()
    args['timeout_hours'] = parsed.timeout_hours
    args['mode'] = parsed.mode
    if parsed.check_runtime or parsed.mode != 'validate':
        validate_runtime()
    if parsed.mode == 'validate':
        print(json.dumps(dict(status='validated_configuration_only', args=args,
                              signature=signature(args), launched=False,
                              remaining_gates='CPU regressions, existing GPU idle check and actual two-GPU smoke'),
                         indent=2, ensure_ascii=False))
        return 0
    if not parsed.run_root or not parsed.frontend_revision:
        parser.error('smoke/train require --run-root and a pinned --frontend-revision')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0,1':
        raise ValueError('Require the existing assigned container with CUDA_VISIBLE_DEVICES=0,1')
    import torch
    if torch.cuda.device_count() != 2:
        raise ValueError('The role experiment requires exactly the two assigned visible CUDA devices')
    if parsed.mode == 'train':
        if not parsed.smoke_receipt:
            parser.error('train requires --smoke-receipt from a matching successful smoke')
        validate_smoke_receipt(args, parsed.smoke_receipt)
        args['smoke_receipt'] = str(Path(parsed.smoke_receipt).resolve())
    root = Path(parsed.run_root).resolve()
    # Atomic exclusive claim: no destructive logger and no reuse of failed roots.
    root.mkdir(parents=True, exist_ok=False)
    args['run_root'] = str(root)
    from short32_lifecycle import write_json
    write_json(root/'experiment.json', dict(args=args, mode=parsed.mode, signature=signature(args)))
    rendezvous = (root/'distributed_rendezvous').as_uri()
    try:
        context = torch.multiprocessing.spawn(worker, args=(args, parsed.mode, rendezvous), nprocs=2, join=False)
        wait_workers(context, parsed.timeout_hours * 3600)
    except Exception as error:
        write_json(root/'run.exit.json', dict(exit_code=1, error=str(error), automatic_retries=0))
        raise
    write_json(root/'run.exit.json', dict(exit_code=0, mode=parsed.mode, automatic_retries=0))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
