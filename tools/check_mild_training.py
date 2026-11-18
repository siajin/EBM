"""Fail closed on policy/cache/source, coverage and final optimizer audits."""
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import numpy as np

ROOT=Path(sys.argv[1]);sys.path.insert(0,str(ROOT/'code'))
from tools.train_coverage_experiment import save


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import torch, arguments
    from mild_duration import MildTrainSet
    from positive_bpl import MatchedBatchSampler, validate_options, validate_evaluation_options
    from dataset.ASVspoof2019_la_train import load_train_items_only
    config=json.loads((ROOT/'experiment.json').read_text())
    assert Path('/.dockerenv').is_file() and os.environ['CUDA_VISIBLE_DEVICES']=='0,1'
    assert subprocess.check_output(['git','-C','/home/mentee/code/EBM','rev-parse','HEAD'],text=True).strip()==config['live_commit']
    assert not subprocess.check_output(['git','-C','/home/mentee/code/EBM','status','--porcelain'],text=True)
    for name,expected in config['source_sha256'].items(): assert sha(ROOT/'code'/name)==expected,name
    for checkpoint in config['protected_checkpoints'].values(): assert sha(checkpoint['path'])==checkpoint['sha256']
    assert sha(ROOT/'duration_policy.json')==config['duration_policy_sha256']
    result=Path(config['result_path'])
    if '--final' not in sys.argv:
        args,_,_=arguments.get_args()
        args.update(duration_policy_path=str(ROOT/'duration_policy.json'),activity_policy_path=None,
            mild_transform_cache=str(ROOT/'audio_cache'),bpl_pairing='matched_independent',
            matched_source_sampling='coverage',batch_size=120,world_size=2,rank=0,epoch=20)
        assert args['rand_seed']==1 and args['lr']==args['lr_min']==1e-5 and args['weight_decay']==.0001
        validate_options(args);validate_evaluation_options(args)
        ds=MildTrainSet(args,load_train_items_only(args['path_19LA']))
        assert ds.activity_policy is None and not result.exists() and torch.cuda.device_count()==2
        for port in [config['smoke_port'],config['train_port']]:
            with socket.socket() as connection: connection.bind(('127.0.0.1',port))
        # Decode+digest every cached variant once before training, not just file names.
        count=0
        for row in ds.records:
            for variant in row['variants']:
                wave=np.load(ROOT/'audio_cache'/variant['file'],allow_pickle=False)
                assert wave.shape==(variant['output_samples'],) and wave.dtype==np.float32 and np.isfinite(wave).all()
                assert hashlib.sha256(wave.tobytes()).hexdigest()==ds.wave_hashes[variant['file']]
                count+=1
        samplers=[MatchedBatchSampler(ds,dict(args,rank=rank)) for rank in [0,1]]
        hist=[np.zeros(620,dtype=np.int64) for _ in range(3)]
        for epoch in range(1,21):
            seen=[set(),set(),set()]
            for sampler in samplers:
                sampler.set_epoch(epoch)
                for batch in sampler:
                    for role,specs in enumerate([batch[:12],batch[12:120],batch[120:]]):
                        seen[role].update(s.index for s in specs)
                        lengths=[ds.records[s.index]['variants'][s.variant]['output_samples'] for s in specs]
                        hist[role]+=np.histogram(np.array(lengths)/16000,bins=np.arange(621)*.05)[0]
            assert seen==[set(ds.class_indices[1]),set(ds.class_indices[0]),set(ds.class_indices[1])]
        overlaps={}
        for width,multiple in [(.05,1),(.1,2),(.25,5)]:
            h=[v.reshape(-1,multiple).sum(1)/v.sum() for v in hist]
            overlaps[str(width)]={'OC_percent':float(100*np.minimum(h[0],h[1]).sum()),
                                 'BPL_right_vs_spoof_percent':float(100*np.minimum(h[2],h[1]).sum())}
        save(ROOT/'preflight.json',{'status':'passed','epoch':20,'coverage_verified_epochs':20,
            'cached_waveforms_verified':count,'source_fingerprint':ds.source_fingerprint,
            'finite20_epoch_histogram_overlap':overlaps,'expected_rounded_overlap':ds.policy['nominal_rounded_histogram_overlap_percent'],
            'uniform_sources':True,'VAD_policy':False,'dev_used':False,
            'coverage_plans':{y:p.summary for y,p in samplers[0].coverage_plans.items()},
            'devices':[torch.cuda.get_device_name(i) for i in [0,1]],'optimizer_updates':0})
    else:
        assert (ROOT/'train.exit').read_text().strip()=='0'
        state=torch.load(result/'model/TrainState_20.pt',map_location='cpu',weights_only=True)
        head=torch.load(result/'model/CLS_20.pt',map_location='cpu',weights_only=True)
        assert state['epoch']==state['args']['epoch']==20 and state['args']['batch_size']==120
        assert state['args']['mild_transform_cache']==str(ROOT/'audio_cache') and state['args']['activity_policy_path'] is None
        assert state['args']['duration_policy_sha256']==config['duration_policy_sha256']
        assert all(torch.equal(value,state['classifier'][name]) for name,value in head.items())
        assert all(torch.isfinite(value).all() for value in head.values())
        steps=[int(value['step']) for value in state['optimizer']['state'].values()]
        assert min(steps)==max(steps)==2160
        for rank in [0,1]:
            done=json.loads((ROOT/f'train_done_rank{rank}.json').read_text())
            assert done['backend_updates']==done['center_updates']==2160 and done['OC_center_changed']
            for epoch in range(1,21):
                coverage=json.loads((ROOT/f'coverage_epoch{epoch}_rank{rank}.json').read_text())
                assert coverage['coverage_all_originals'] and coverage['OC_bona_unique']==2580 and coverage['OC_spoof_unique']==22800
        for metric in ['loss1','loss2','oc_long','bpl','positive_pairs','centroid_updates','optimizer_steps']:
            lines=(result/'metric'/f'{metric}.txt').read_text().splitlines()
            assert len(lines)==2160 and all(math.isfinite(float(line.split(': ')[-1])) for line in lines)
        save(ROOT/'final_training_audit.json',{'status':'passed','epoch':20,'logical_steps':2160,
            'backend_updates_per_rank':2160,'center_updates_per_rank':2160,'every_original_every_epoch':True,
            'classifier_matches_TrainState':True,'CLS20_sha256':sha(result/'model/CLS_20.pt'),
            'TrainState20_sha256':sha(result/'model/TrainState_20.pt')})


if __name__=='__main__': main()
