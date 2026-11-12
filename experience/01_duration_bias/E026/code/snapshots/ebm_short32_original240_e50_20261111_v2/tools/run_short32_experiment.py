"""Bounded smoke/launch for frozen original240 + short32. No action on import."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))


def save(path, value):
    from short32_lifecycle import write_json
    write_json(path, value)


def verify(root):
    cfg = json.loads((root/'experiment.json').read_text())
    for name, expected in cfg['source_sha256'].items():
        if hashlib.sha256((root/'code'/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen source changed: '+name)
    return cfg


def smoke_items(items):
    """Original sampler produces N=12/12/6, K=0/12/2 vs0/3/0."""
    import soundfile as sf
    from torch.utils.data import DistributedSampler
    eligible, other, spoof = [], [], []
    for item in items:
        if item.label == 0:
            spoof.append(item)
        elif sf.info(item.path).frames <= 32000:
            eligible.append(item)
        else:
            other.append(item)
    slots = set()
    for rank, counts in [(0,[0,12,2]),(1,[0,3,0])]:
        sampler = DistributedSampler(range(60),num_replicas=2,rank=rank,shuffle=True)
        sampler.set_epoch(1); ids = list(sampler)
        for start, count in zip([0,12,24],counts):
            slots.update(ids[start:start+count])
    assert len(eligible) >= len(slots) and len(other) >= 60-len(slots) and len(spoof)>=336
    a, b = iter(eligible), iter(other)
    return [next(a) if i in slots else next(b) for i in range(60)] + spoof[:336]


def balanced_trials(items, count=32):
    return [x for y in [1,0] for x in [i for i in items if i.label==y][:count]]


def preflight(root, cfg):
    import soundfile as sf
    import arguments
    from short32_bpl import configure
    from dataset.ASVspoof2019_la_train import load_train_items_only
    a = arguments.get_args()[0]; configure(a)
    items = load_train_items_only(a['path_19LA'])
    counts = {0:0,1:0}; eligible = 0
    for item in items:
        info = sf.info(item.path)
        assert info.samplerate==16000 and info.channels==1 and info.frames>=720, item.path
        counts[item.label] += 1
        eligible += int(item.label==1 and info.frames<=32000)
    assert counts == {0:22800,1:2580} and eligible==109,(counts,eligible)
    selected = smoke_items(items)
    save(root/'preflight.json',dict(status='passed',sources=len(items),class_counts=counts,
         eligible_bona=eligible,smoke_bona=60,smoke_spoof=336,
         smoke_source_sha256=hashlib.sha256('\n'.join(x.path for x in selected).encode()).hexdigest(),
         frontend_revision=cfg['frontend_revision'],GPU_training_started=False))


def install_audit(t, root, smoke, cfg):
    import torch
    from short32_original240 import unwrap
    model = unwrap(t.classifier)
    revision = getattr(getattr(unwrap(t.plm),'config',None),'_commit_hash',None)
    assert revision == cfg['frontend_revision'],(revision,cfg['frontend_revision'])
    assert sum(p.numel() for p in model.parameters())-model.loss.weight.numel()==3744224
    versions = [p._version for p in t.plm.parameters()]
    t.audit = dict(backend=0,center=0,logical_steps=0,loss1=None,loss2=None,last_epoch=None)
    def forward(module, inputs, kwargs, output):
        if module.training:
            value = float(output[0].detach())
            assert torch.isfinite(output[0]).all(),'nonfinite loss'
            t.audit['loss1' if kwargs.get('x_short') is not None else 'loss2'] = value
            if kwargs.get('x_short') is not None:
                t.audit['oc_batch'] = int(inputs[0].shape[0])
    t.classifier.register_forward_hook(forward,with_kwargs=True)
    def before_step(optimizer,args,kwargs):
        assert all(p.grad is None for p in t.plm.parameters()),'frontend gradient'
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()),'nonfinite gradient'
    t.optimizer.register_step_pre_hook(before_step)
    def after_step(optimizer,args,kwargs):
        audit = t.audit
        center = model.loss.weight.requires_grad
        audit['center' if center else 'backend'] += 1
        if not center:
            return
        audit['logical_steps'] += 1
        assert audit['center']==audit['backend']==audit['logical_steps']
        assert versions == [p._version for p in t.plm.parameters()],'frontend updated'
        context = t.short32
        stats = dict(context.last_stats,loss1=audit['loss1'],loss2=audit['loss2'],
                     backend_updates=audit['backend'],center_updates=audit['center'],
                     logical_steps=audit['logical_steps'],updated_unix=time.time(),smoke=smoke)
        stats['oc_batch'] = audit['oc_batch']
        if smoke:
            expected_n=[12,12,6][context.step-1]
            expected_k=([0,12,2] if t.args['rank']==0 else [0,3,0])[context.step-1]
            assert stats['bona']==expected_n and stats['eligible']==expected_k,stats
            assert stats['oc_batch']==[120,72,114][context.step-1],stats
        file = root/f'{"smoke" if smoke else "train"}_metrics_rank{t.args["rank"]}.jsonl'
        with file.open('a') as out:
            out.write(json.dumps(stats)+'\n')
        if context.step==1 or context.step%10==0 or context.step==len(t.train_loader_bona):
            save(root/f'{"smoke" if smoke else "train"}_progress_rank{t.args["rank"]}.json',stats)
        if audit['last_epoch'] != context.epoch:
            audit['last_epoch'] = context.epoch
    t.optimizer.register_step_post_hook(after_step)


def worker(rank, root_string, mode, port):
    import torch
    import torch.distributed as dist
    import arguments, main
    import short32_original240 as adapter
    import short32_lifecycle as lifecycle
    from short32_bpl import configure
    root = Path(root_string); cfg = verify(root); smoke = mode=='smoke'
    assert torch.cuda.device_count()==2
    original_adapter = adapter.run
    def adapted(process_id,a,experiment,environment):
        if smoke:
            environment = dict(environment)
            original_train = environment['ASVspoof2019_LA_Train']
            def training(path):
                db = original_train(path); db.train_set = smoke_items(db.train_set); return db
            environment['ASVspoof2019_LA_Train'] = training
            for name in ['ASVspoof2019_LA_Eval','ASVspoof2021_LA_Eval','ASVspoof2021_DF_Eval']:
                factory = environment[name]
                def subset(path,factory=factory):
                    db = factory(path);db.trials=balanced_trials(db.trials);return db
                environment[name]=subset
        return original_adapter(process_id,a,experiment,environment)
    adapter.run = adapted
    original_lifecycle = lifecycle.run_training
    def audited(t):
        install_audit(t,root,smoke,cfg)
        if not smoke:
            (root/f'training_started_rank{rank}.txt').write_text(str(time.time()))
        result = original_lifecycle(t)
        expected = 3 if smoke else 108*50
        assert t.audit['logical_steps']==t.audit['center']==t.audit['backend']==expected,t.audit
        complete = json.loads((t.checkpoints.root/'complete.json').read_text())
        if smoke:
            assert complete['status']=='complete',complete
            assert all(x['status']=='complete' and x['epoch']==1 for x in complete['results'])
        save(root/f'{"smoke" if smoke else "train"}_done_rank{rank}.json',dict(status='passed',
             logical_steps=expected,backend_updates=t.audit['backend'],center_updates=t.audit['center'],
             frozen_frontend=True,lifecycle_status=complete['status'],result_directory=str(t.checkpoints.root.parent)))
        return result
    lifecycle.run_training = audited
    args,_,experiment = arguments.get_args()
    configure(args)
    args.update(epoch=1 if smoke else 50,batch_size=120,world_size=2,rand_seed=1,
                gpu_ids=['0','1'],usable_gpu='0,1',num_workers=4,port=str(port),
                name=cfg['experiment_name']+('_gpu_smoke' if smoke else ''),
                path_log=str(root/'smoke_results') if smoke else '/results',
                evaluation_seed=1729,eval_batch_size=8,distributed_timeout_seconds=600)
    experiment.update(short32_bpl=True,epoch=args['epoch'],global_batch_size=240,rand_seed=1,
                      objective='original240 OC+BPL43; conditional half43/half32 for native<=2s bona',
                      frontend_revision=cfg['frontend_revision'],source_sha256=cfg['source_sha256'])
    torch.cuda.set_device(rank)
    try:main.run(rank,args,experiment)
    finally:
        if dist.is_initialized():dist.destroy_process_group()


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    parser.add_argument('--mode',choices=['preflight','smoke','train'],required=True)
    a = parser.parse_args();root = a.root.resolve(); cfg = verify(root)
    assert Path('/.dockerenv').is_file() and os.environ['CUDA_VISIBLE_DEVICES']=='0,1'
    if a.mode=='preflight':
        preflight(root,cfg); return
    claim = root/(a.mode+'.claim')
    claim.mkdir(exist_ok=False)
    import torch.multiprocessing as mp
    mp.set_sharing_strategy('file_system')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    save(root/(a.mode+'_launch.json'),dict(mode=a.mode,world_size=2,port=port,started_unix=time.time()))
    mp.spawn(worker,args=(str(root),a.mode,port),nprocs=2,join=True)


if __name__=='__main__':cli()
