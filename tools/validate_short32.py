"""Bounded local synthetic validation only; no training launcher or GPU model load."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import unittest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vendor', type=Path, help='Optional isolated local validation dependencies')
    parser.add_argument('--report-dir', type=Path, required=True)
    parser.add_argument('--patterns', nargs='+', default=['test_short32_bpl.py'])
    args = parser.parse_args()
    code = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(code))
    if args.vendor:
        sys.path.insert(0, str(args.vendor.resolve()))
    report_dir = args.report_dir.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    suite = unittest.TestSuite()
    for pattern in args.patterns:
        suite.addTests(unittest.defaultTestLoader.discover(str(code/'tests'), pattern=pattern))
    log = io.StringIO(); start = time.time()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    rendered = log.getvalue()
    (report_dir/'test_output.txt').write_text(rendered, encoding='utf-8')
    sources = ['short32_bpl.py', 'short32_original240.py', 'short32_lifecycle.py', 'native_steps.py', 'data_loaders.py', 'arguments.py',
               'main.py', 'train.py', 'data_processing/__init__.py', 'model/loss.py',
               'tests/test_short32_bpl.py', 'tests/test_short32_lifecycle.py', 'tools/validate_short32.py',
               'tests/test_short32_runner.py', 'tools/run_short32_experiment.py']
    report = dict(status='passed' if result.wasSuccessful() else 'failed',
                  tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                  skipped=[dict(test=str(t), reason=reason) for t, reason in result.skipped],
                  elapsed_seconds=time.time()-start, patterns=args.patterns,
                  new_experiment_training_started=False, new_experiment_optimizer_updates=0,
                  GPU_job_started=False, queue_registered=False, server_accessed=False,
                  validation_scope='Local synthetic audio/features, frozen original240 backend/train/autograd, CPU worker/DDP checks; scratch-model updates only. No lab audio/checkpoints/XLS-R loaded.',
                  source_sha256={name: hashlib.sha256((code/name).read_bytes()).hexdigest() for name in sources})
    (report_dir/'validation.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(rendered, end='')
    print(json.dumps(report, indent=2))
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
