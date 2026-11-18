"""Score preserved original240 fixed20, without fabricating an optimizer state."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(sys.argv[1]);sys.path.insert(0,str(ROOT/'code'))
from tools.train_baseline240_mild import save


def main():
    import torch.multiprocessing as mp
    from tools.evaluate_coverage_checkpoint import worker,summarize_phase
    cfg=json.loads((ROOT/'experiment.json').read_text())
    assert Path('/.dockerenv').is_file() and os.environ['CUDA_VISIBLE_DEVICES']=='0,1'
    original=cfg['protected_checkpoints']['original240_epoch20']
    assert hashlib.sha256(Path(original['path']).read_bytes()).hexdigest()==original['sha256']
    for name,expected in cfg['source_sha256'].items():
        assert hashlib.sha256((ROOT/'code'/name).read_bytes()).hexdigest()==expected
    root=ROOT/'original240_eval';root.mkdir(exist_ok=False)
    config={'epoch':20,'root':str(root),'code_root':str(ROOT/'code'),'source_sha256':cfg['source_sha256'],
        'checkpoint':original['path'],'checkpoint_sha256':original['sha256'],'frontend_revision':cfg['frontend_revision'],
        'phases':{'19LA':71237,'21LA':148176,'21DF':533928},
        'class_counts':{'19LA':[7355,63882],'21LA':[14816,133360],'21DF':[14869,519059]},
        'optimizer_updates':0,'checkpoint_kind':'preserved original240 epoch20; original archive has no TrainState',
        'evaluation_seed':1729,'frontend_basis':'same pinned pretrained frontend; original20 uses no_grad before T_0=150'}
    save(root/'evaluation_config.json',config);(root/'evaluation').mkdir()
    metrics=[]
    for phase in config['phases']:
        save(root/'phase.json',{'phase':phase,'updated_unix':time.time()})
        mp.spawn(worker,args=(str(root),phase,False),nprocs=2,join=True)
        metrics.append(summarize_phase(root,phase))
    assert hashlib.sha256(Path(original['path']).read_bytes()).hexdigest()==original['sha256']
    save(root/'complete.json',{'status':'complete','epoch':20,'metrics':metrics,
        'optimizer_updates':0,'checkpoint_sha256':original['sha256']})


if __name__=='__main__':main()
