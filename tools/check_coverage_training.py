"""Read-only preflight/final checkpoint audit for the frozen coverage run."""
import hashlib,json,os,socket,subprocess,sys
from pathlib import Path
ROOT=Path(sys.argv[1]);sys.path.insert(0,str(ROOT/'code'))
from tools.train_coverage_experiment import save


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__=='__main__':
    import torch,arguments
    from positive_bpl import MatchedTrainSet,coverage_source_plans,validate_options,validate_evaluation_options
    from dataset.ASVspoof2019_la_train import load_train_items_only
    config=json.loads((ROOT/'experiment.json').read_text())
    assert Path('/.dockerenv').is_file() and os.environ['CUDA_VISIBLE_DEVICES']=='0,1'
    assert subprocess.check_output(['git','-C','/home/mentee/code/EBM','rev-parse','HEAD'],text=True).strip()==config['live_commit']
    assert not subprocess.check_output(['git','-C','/home/mentee/code/EBM','status','--porcelain'],text=True)
    for name,expected in config['source_sha256'].items():assert sha(ROOT/'code'/name)==expected,name
    assert sha(config['legacy_checkpoint'])==config['legacy_checkpoint_sha256']
    assert sha(ROOT/'duration_policy.json')==config['duration_policy_sha256']
    assert sha(ROOT/'activity_policy.json')==config['activity_policy_sha256']
    if '--final' not in sys.argv:
        args,_,_=arguments.get_args()
        args.update(duration_policy_path=str(ROOT/'duration_policy.json'),activity_policy_path=str(ROOT/'activity_policy.json'),
            bpl_pairing='matched_independent',matched_source_sampling='coverage')
        assert args['epoch']==50 and args['batch_size']==240 and args['rand_seed']==1
        assert args['lr']==args['lr_min']==1e-5 and args['weight_decay']==.0001
        validate_options(args);validate_evaluation_options(args)
        ds=MatchedTrainSet(args,load_train_items_only(args['path_19LA']))
        plans=coverage_source_plans(ds,{1:2592,0:23328})
        assert not Path(config['result_path']).exists()
        assert torch.cuda.device_count()==2
        for port in [config['smoke_port'],config['train_port']]:
            with socket.socket() as connection:connection.bind(('127.0.0.1',port))
        save(ROOT/'preflight.json',{'status':'passed','epoch':50,'OC_global_batch':240,
            'OC_per_rank':120,'extra_BPL_per_rank':12,'steps_per_epoch':108,
            'coverage_plans':{y:p.summary for y,p in plans.items()},
            'activity_statistics':ds.activity_policy.statistics,'verified_source_files':len(config['source_sha256']),
            'devices':[torch.cuda.get_device_name(i) for i in range(2)]})
    else:
        assert (ROOT/'train.exit').read_text().strip()=='0'
        result=Path(config['result_path']);state=torch.load(result/'model/TrainState_50.pt',map_location='cpu',weights_only=True)
        head=torch.load(result/'model/CLS_50.pt',map_location='cpu',weights_only=True)
        assert state['epoch']==50 and state['args']['epoch']==50 and state['args']['batch_size']==120
        for name,value in {'bpl_pairing':'matched_independent','matched_source_sampling':'coverage',
                           'duration_policy_sha256':config['duration_policy_sha256'],
                           'activity_policy_sha256':config['activity_policy_sha256'],
                           'frontend_resolved_revision':config['frontend_revision']}.items():
            assert state['args'][name]==value,(name,state['args'].get(name))
        assert all(torch.equal(value,state['classifier'][name]) for name,value in head.items())
        assert all(torch.isfinite(value).all() for value in head.values())
        steps=[int(value['step']) for value in state['optimizer']['state'].values()]
        assert min(steps)==max(steps)==5400
        for rank in range(2):
            done=json.loads((ROOT/f'train_done_rank{rank}.json').read_text())
            assert done['backend_updates']==done['center_updates']==5400 and done['OC_center_changed']
            for epoch in range(1,51):
                coverage=json.loads((ROOT/f'coverage_epoch{epoch}_rank{rank}.json').read_text())
                assert coverage['coverage_all_originals'] and coverage['OC_bona_unique']==2580 and coverage['OC_spoof_unique']==22800
        for metric in ['loss1','loss2','oc_long','bpl','positive_pairs','centroid_updates']:
            lines=(result/'metric'/f'{metric}.txt').read_text().splitlines()
            assert len(lines)==5400,(metric,len(lines))
            assert all(__import__('math').isfinite(float(line.split(': ')[-1])) for line in lines)
        save(ROOT/'final_training_audit.json',{'status':'passed','epoch':50,'logical_steps_per_rank':5400,
            'Adam_steps_per_parameter':5400,'backend_updates_per_rank':5400,'center_updates_per_rank':5400,
            'coverage_all_originals_every_epoch':True,'metrics_rows':5400,'classifier_matches_TrainState':True,
            'CLS50_sha256':sha(result/'model/CLS_50.pt'),'TrainState50_sha256':sha(result/'model/TrainState_50.pt'),
            'frontend_revision':config['frontend_revision']})
