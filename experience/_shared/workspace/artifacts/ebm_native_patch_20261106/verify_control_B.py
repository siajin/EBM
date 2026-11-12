"""Bounded two-GPU verification of control B, without experiment logging."""
import datetime
import json
from pathlib import Path
import socket
import sys
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd


def worker(rank, root_s, port):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from transformers import Wav2Vec2Model
    import soundfile as sf
    root = Path(root_s)
    sys.path.insert(0, '/home/mentee/code/EBM')
    import arguments, data_loaders, model, native_steps
    from dataset import ASVspoof2019_LA_Train
    torch.set_num_threads(2)
    torch.cuda.set_device(rank)
    device = f'cuda:{rank}'
    torch.manual_seed(20261106)
    torch.cuda.manual_seed(20261106)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group('nccl', init_method=f'tcp://127.0.0.1:{port}',
                            rank=rank, world_size=2, timeout=datetime.timedelta(minutes=5))
    args, _, _ = arguments.get_args()
    assert args['batch_size'] == 240
    args.update(rank=rank, world_size=2, batch_size=120, device=device, num_workers=0)
    checks = {}
    def ck(name, value):
        assert value, name
        checks[name] = True
    ck('B_default_auxiliary_weights_zero', args['native_length'] and args['cap_long'] and
       args['short_oc_weight'] == args['padding_loss_weight'] == 0)
    ck('existing_training_hyperparameters', args['epoch'] == 50 and args['lr'] == 1e-5
       and args['num_train_frames'] == 404 and args['num_train_frames_short'] == 303)
    probe = pd.read_parquet('/home/mentee/analysis/embedding_geometry/20261106_variable_length_01/manifest.parquet').set_index('selection')
    keys = ['train_y1_minimum','train_y0_minimum','train_y1_median',
            'train_y0_median','train_y1_maximum','train_y0_maximum','DF_remaining_maximum']
    cases = []
    for key in keys:
        r = probe.loc[key]
        item = SimpleNamespace(path=r.path, label=int(r.y), type=r.attack_id)
        d = data_loaders.TrainSet(args, [item])
        original, _ = sf.read(item.path)
        def forbidden(*a, **kw):
            raise AssertionError('Codec must stay off')
        d.codec_aug.convert = forbidden
        with patch.object(data_loaders.random, 'random', return_value=0.), \
             patch.object(data_loaders.random, 'randint', return_value=0):
            b = d[0]
            a = data_loaders.TrainSet(dict(args, native_length=False), [item])[0]
            c = data_loaders.TrainSet(dict(args, cap_long=False), [item])[0]
        ck('B_exact_retained_samples_'+key, np.array_equal(b[0],original[:64600]) and np.array_equal(b[1],original[:48440]))
        ck('A_B_same_content_and_caps_'+key, len(a[0])==64600 and len(a[1])==48440 and
           np.array_equal(a[0][:len(b[0])],b[0]) and np.array_equal(a[1][:len(b[1])],b[1]))
        ck('C_optional_full_long_'+key, np.array_equal(c[0],original))
        ev = data_loaders.EnrollmentSet(args,[item])[0]
        ck('B_eval_same_long_cap_'+key, np.array_equal(ev[0],original[:64600]))
        # RawBoost remains enabled and length preserving; do not invoke codec.
        for aug in range(9):
            chooser = lambda low, high, j=aug: j if (low, high)==(0,8) else 0
            with patch.object(data_loaders.random, 'random', return_value=0.), \
                 patch.object(data_loaders.random, 'randint', side_effect=chooser):
                long, short, _, _ = d[0]
            ck(f'RawBoost_{aug}_cap_finite_{key}',len(long)==min(len(original),64600) and
               len(short)==min(len(original),48440) and np.isfinite(long).all() and np.isfinite(short).all())
        cases.append({'selection':key,'original':len(original),'long':len(b[0]),'short':len(b[1])})

    database = ASVspoof2019_LA_Train(args['path_19LA'])
    ck('same_train25380_bona2580',len(database.train_set)==25380 and sum(i.label for i in database.train_set)==2580)
    train_loader, sampler, _, _ = data_loaders.get_loaders(args,database.train_set,[],{})
    base_sampler = data_loaders.BalancedLengthBatchSampler(
        data_loaders.TrainSet(dict(args,native_length=False),database.train_set),args)
    ck('A_B_identical_batch_uid_order',list(sampler)==list(base_sampler))
    ids = list(sampler)[0]
    ck('logical_batch120_bona12_spoof108',len(ids)==120 and [database.train_set[i].label for i in ids]==[1]*12+[0]*108)
    # Exercise the actual get_loaders collate with repetition forbidden entirely.
    original_pad = data_loaders.np.pad
    def no_wrap(a, width, mode='constant', **kw):
        assert mode != 'wrap', 'B must not create repeated audio, even on CPU'
        return original_pad(a,width,mode=mode,**kw)
    with patch.object(data_loaders.np,'pad',side_effect=no_wrap):
        batch = train_loader.collate_fn([train_loader.dataset[i] for i in ids])
    ck('no_wrapped_input_or_pair_mask',set(batch)=={'long','short','labels','indices'})
    ck('actual_loader_respects_caps',max(map(len,batch['long']))<=64600 and max(map(len,batch['short']))<=48440)

    plm = Wav2Vec2Model.from_pretrained(args['PLM_name'],local_files_only=True).to(device).eval().requires_grad_(False)
    cls = model.B_Linear(args['num_layers'],args['hidden_size'],args['output_size'],args['agg_size'],
        loss=model.OCSoftmax(args['output_size'],r_real=args['r_real'],r_fake=args['r_fake'],alpha=args['alpha'],class_weight=[5/9,5]),
        loss_bpl=model.BPL(args['alpha_cl'])).to(device).eval()
    for m in cls.modules():
        if hasattr(m,'noise_std'): m.noise_std=0.
    waves = [torch.from_numpy(sf.read(probe.loc[k].path,dtype='float32')[0][:64600])
             for k in ['train_y0_minimum','train_y1_minimum','train_y0_maximum']]
    with torch.no_grad():
        solo=[]
        for w in waves:
            f,m=native_steps.features(plm,[w],device)
            solo.append(cls(f,mask=m,return_embedding=True))
        f,m=native_steps.features(plm,waves,device)
        z=cls(f,mask=m,return_embedding=True)
        error=float((z-torch.cat(solo)).abs().max())
    ck('mixed_zero_padding_vs_individual_embedding',error<1e-4)
    ck('feature_mask_exact_valid_lengths',m.sum(1).tolist()==[(len(w)-400)//320+1 for w in waves])
    # The objective contains only original long OC + local all-pairs BPL.
    embeddings={name:torch.randn(120,args['output_size'],device=device,requires_grad=True) for name in ['long','short']}
    objective,bona,stat=native_steps._objective(cls,embeddings,batch,args)
    expected=cls.loss(embeddings['long'],batch['labels'].to(device),logical_batch_size=120)[0]+cls.loss_bpl(embeddings['long'][:12],embeddings['short'][:12])
    ck('objective_exact_original_OC_plus_local_BPL',bona==12 and torch.equal(objective,expected) and stat['padding']==0 and stat['active_pairs']==0)
    del embeddings,objective,expected,f,m,z,solo,waves
    for mod in cls.modules():
        if hasattr(mod,'noise_std'): mod.noise_std=1e-5
    cls.train()
    ddp=DDP(cls,device_ids=[rank],find_unused_parameters=True)
    opt=torch.optim.Adam(ddp.parameters(),lr=args['lr'],weight_decay=args['weight_decay'])
    before_linear=cls.linear.weight.detach().clone()
    before_centroid=cls.loss.weight.detach().clone()
    versions=[p._version for p in plm.parameters()]
    engine=SimpleNamespace(classifier=ddp,plm=plm,args=args,optimizer=opt)
    torch.cuda.reset_peak_memory_stats(rank)
    stats=native_steps.train_batch(engine,batch)
    ck('real_two_GPU_global240_update_finite',all(np.isfinite(v) for v in stats.values()))
    ck('only_two_views_in_update_stats',set(k for k in stats if k.startswith('oc_'))=={'oc_long','oc_short'})
    ck('encoder_and_centroid_updated',not torch.equal(before_linear,cls.linear.weight) and not torch.equal(before_centroid,cls.loss.weight))
    ck('frontend_frozen',versions==[p._version for p in plm.parameters()] and all(p.grad is None for p in plm.parameters()))
    report={'rank':rank,'checks':checks,'cases':cases,'batch_single_max_embedding_error':error,
            'stats':stats,'peak_allocated_MiB':torch.cuda.max_memory_allocated(rank)/2**20,
            'global_batch':240,'local_BPL_pairs':144,'not_a_training_run':True}
    (root/f'rank{rank}.json').write_text(json.dumps(report,indent=2)+'\n')
    dist.destroy_process_group()


if __name__=='__main__':
    import torch.multiprocessing as mp
    root=sys.argv[1]
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    mp.spawn(worker,args=(root,port),nprocs=2,join=True)
    Path(root,'complete.json').write_text(json.dumps({'status':'passed','GPUs':2,'full_training_started':False})+'\n')
