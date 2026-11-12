"""Exercise real Train waveform loading, workers and collation without a model."""
import json, os, sys, time
from pathlib import Path
import numpy as np
import torch

root = Path(sys.argv[1])
sys.path.insert(0, str(root / 'code'))
import arguments
from dataset.ASVspoof2019_la_train import load_train_items_only
from data_loaders import get_loaders

if __name__ == '__main__':
    os.nice(10)
    torch.set_num_threads(1)
    started = time.monotonic()
    args, _, _ = arguments.get_args()
    args.update(matched_source_sampling='coverage',bpl_pairing='matched_independent',
                duration_policy_path=str(root / 'duration_policy.json'),
                activity_policy_path=str(root / 'activity_policy.json'),
                rank=0, world_size=2, batch_size=120, num_workers=2)
    items = load_train_items_only('/data/ASVspoof2019')
    loader, sampler, empty_validation, empty_evaluations = get_loaders(args, items, [], {})
    # This is a CPU input-path check; avoid CUDA pinned-memory initialization.
    loader.pin_memory = False
    sampler.set_epoch(1)
    assert len(empty_validation.dataset) == 0 and empty_evaluations == {}
    result = []
    iterator = iter(loader)
    for step in range(2):
        batch = next(iterator)
        assert batch['labels'].tolist() == [1] * 12 + [0] * 108
        assert len(batch['long']) == 120 and len(batch['bpl_right']) == 12
        lengths = [len(wave) for wave in batch['long'] + batch['bpl_right']]
        assert all(length in loader.dataset.targets for length in lengths)
        assert all(bool(torch.isfinite(wave).all()) for wave in batch['long'] + batch['bpl_right'])
        result.append({'step': step, 'OC_inputs': 120, 'extra_BPL_inputs': 12,
                       'seconds': sorted(set(length / 16000 for length in lengths))})
    del iterator
    assert not torch.cuda.is_initialized()
    (root / 'loader_check.json').write_text(json.dumps({
        'status': 'passed', 'workers': 2, 'pin_memory_in_CPU_check': False, 'batches': result,
        'dev_used': False, 'eval_used': False, 'model_training': False,
        'CUDA_initialized': False, 'elapsed_seconds': time.monotonic() - started,
    }, indent=2))
