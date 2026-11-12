import os
import random
import datetime
import numpy as np

import torch
import torch.nn as nn

import warnings
import torchvision

torchvision.disable_beta_transforms_warning()
warnings.filterwarnings("ignore", message="Failed to load image Python extension")

import arguments
import data_loaders as data_loaders
import train
import model
import argparse
import hashlib
import json
from pathlib import Path
from positive_bpl import evaluation_mode, validate_options, validate_evaluation_options, oc_evaluation_protocol
from collections import OrderedDict

from dataset import ASVspoof2019_LA_Train, ASVspoof2021_DF_Eval, ASVspoof2019_LA_Eval, ASVspoof2021_LA_Eval
from dataset.ASVspoof2019_la_train import load_train_items_only
from logger import Logger
from utils import get_threshold
from data_processing import WaveformAugmetation
from transformers import AutoConfig, Wav2Vec2Model

def set_experiment_environment(args):
    # reproducible
    random.seed(args['rand_seed'])
    np.random.seed(args['rand_seed'])
    torch.manual_seed(args['rand_seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # DDP env
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args['port']
    args['rank'] = args['process_id']
    args['device'] = f'cuda:{args["process_id"]}'
    torch.cuda.set_device(args['process_id'])
    torch.distributed.init_process_group(   
            backend='nccl', world_size=args['world_size'], rank=args['rank'],
            timeout=datetime.timedelta(hours=12))
    torch.cuda.empty_cache()

def run(process_id, args, experiment_args):
    # check parent process
    args['process_id'] = process_id
    args['flag_parent'] = process_id == 0
    
    # args
    set_experiment_environment(args)
    if args['attack_type'] == 'LA':
        trainer = train.ModelTrainer()
    trainer.args = args

    # logger
    if args['flag_parent']:
        trainer.logger = Logger.Builder(args['name'], args['project']
            ).tags(args['tags']
            ).description(args['description']
            ).save_source_files(args['path_scripts']
            ).use_local(args['path_log']
            #).use_neptune(args['neptune_user'], args['neptune_token']
            ).build()
        trainer.logger.log_parameter(experiment_args)

    if args['attack_type'] == 'LA':
        # dataset
        train_db_19la = ASVspoof2019_LA_Train(args['path_19LA'])
        test_db_19la = ASVspoof2019_LA_Eval(args['path_19LA'])
        test_db_21la = ASVspoof2021_LA_Eval(args['path_21LA'])
        test_db_21df = ASVspoof2021_DF_Eval(args['path_21DF'])
        train_set = []
        train_set.extend(train_db_19la.train_set)
        test_set = {'19LA':test_db_19la, '21LA':test_db_21la, '21DF':test_db_21df}

        # set class weight
        class_weight = [1, 1]
        if args['use_class_weight']:
            class_num = [args['ratio_spoof'], args['ratio_bona']]
            class_weight = [sum(class_num) / (class_num[0] * 2),       # spoofed_weight
                            sum(class_num) / (class_num[1] * 2)]       # bonafide_weight
            print('class_weight: ',class_weight)
        
        # Data loader
        trainer.train_loader, trainer.train_sampler, trainer.val_loader, trainer.eval_loader = \
            data_loaders.get_loaders(args, train_set, train_db_19la.dev_set, test_set)
        trainer.num_val = len(train_db_19la.dev_set)
        if args.get('bpl_pairing', 'legacy_long_short') != 'legacy_long_short':
            ds = trainer.train_loader.dataset
            trainer.args['duration_policy_sha256'] = ds.policy_sha256
            trainer.args['bpl_source_fingerprint'] = ds.source_fingerprint
            if ds.activity_policy is not None:
                trainer.args['activity_policy_sha256']=ds.activity_policy.sha256
            if trainer.logger is not None:
                trainer.logger.log_text('bpl_sampling_policy', ds.policy_text)
                trainer.logger.log_parameter({'duration_policy_sha256': ds.policy_sha256,
                    'bpl_source_fingerprint': ds.source_fingerprint,
                    'OC_waveforms_per_rank': args['batch_size'],
                    'extra_BPL_waveforms_per_rank': args['bpl_pair_size'],
                    'objective': 'original OC+BPL, matched variable inputs for both classes; original centroid update'})
                trainer.logger.log_parameter({'matched_source_sampling':trainer.train_sampler.source_sampling})
                if trainer.train_sampler.coverage_plans is not None:
                    trainer.logger.log_text('source_coverage_plan',json.dumps(
                        {y:plan.summary for y,plan in trainer.train_sampler.coverage_plans.items()},indent=2))
                if ds.activity_policy is not None:
                    trainer.logger.log_parameter({'activity_policy_sha256':ds.activity_policy.sha256,
                        'activity_matching':'raw VAD2 class bins and mean within length; original source priors retained'})
                    trainer.logger.log_text('activity_sampling_policy',json.dumps(ds.activity_policy.metadata,indent=2))

    # Data augmentation
    trainer.da = None
    # trainer.da = WaveformAugmetation(['ACN'],
    # {'sr': 16000, 'ACN':{'min_snr_in_db': 10, 'max_snr_in_db': 40, 'min_f_decay': -2.0, 'max_f_decay': 2.0, 'p': 1}}).to(args['device'])
    
    # Speech pre-trained model
    config = AutoConfig.from_pretrained(
        args['PLM_name'], 
        finetuning_task="audio-classification",
        revision="main",
    )

    trainer.plm = Wav2Vec2Model.from_pretrained(
        args['PLM_name'],
        from_tf=bool(".ckpt" in args['PLM_name']),
        config=config,
        revision="main",
        ignore_mismatched_sizes=False,
    ).to(args['device'])
    trainer.plm.eval()
    trainer.plm.requires_grad_(False)
    trainer.evaluation_protocol = oc_evaluation_protocol(args, getattr(config, '_commit_hash', None))
    trainer.args['frontend_resolved_revision'] = getattr(config, '_commit_hash', None)

    # classifier
    trainer.classifier = model.B_Linear(
        num_layer = args['num_layers'],
        hidden_size = args['hidden_size'],
        output_size = args['output_size'],
        agg_size=args['agg_size'],
        loss = model.OCSoftmax(
            args['output_size'], r_real=args['r_real'],
            r_fake=args['r_fake'], alpha=args['alpha'], class_weight=class_weight),
        loss_bpl = model.BPL(alpha=args['alpha_cl'])
        ).to(args['device'])
    trainer.classifier = nn.SyncBatchNorm.convert_sync_batchnorm(trainer.classifier)
    trainer.classifier = nn.parallel.DistributedDataParallel(
        trainer.classifier, device_ids=[args['device']], find_unused_parameters=True
    )
    
    # optimizer
    trainer.optimizer = torch.optim.Adam(
        list(trainer.plm.parameters()) + list(trainer.classifier.parameters()),
        lr=args['lr'], 
        weight_decay=args['weight_decay']
    )


    # lr scheduler
    trainer.lr_step = 'epoch'
    trainer.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        trainer.optimizer,
        T_0=args['T_0'],
        T_mult=args['T_mult'],
        eta_min=args['lr_min']
    )


    # set thresghold for accuracy
    trainer.threshold = get_threshold(args['r_real'], args['r_fake'])
        
    if args.get('evaluate_checkpoint'):
        state = torch.load(args['evaluate_checkpoint'], map_location='cpu', weights_only=True)
        if state.get('args', {}).get('training_mode') == 'bpl_only':
            raise ValueError('The superseded BPL-only pilot has no trained OC center and is not a valid control')
        trainer.classifier.load_state_dict(state.get('classifier', state), strict=True)
        for phase in args.get('evaluate_phases', ['V']):
            trainer.test(phase, 0)
        trainer._synchronize()
    else:
        trainer.run()

if __name__ == '__main__':
    # get arguments
    args, system_args, experiment_args = arguments.get_args()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--bpl-pairing', choices=['legacy_long_short', 'matched_same_source', 'matched_independent'])
    parser.add_argument('--duration-policy', dest='duration_policy_path')
    parser.add_argument('--activity-policy', dest='activity_policy_path')
    parser.add_argument('--source-sampling', dest='matched_source_sampling', choices=['replacement','coverage'])
    parser.add_argument('--evaluation-mode', choices=['controlled_oc', 'original_oc'])
    parser.add_argument('--evaluation-seed', type=int)
    parser.add_argument('--evaluate-checkpoint')
    parser.add_argument('--evaluate-phases', nargs='+', choices=['V','19LA','21LA','21DF'])
    parser.add_argument('--name')
    parser.add_argument('--validate-only', action='store_true')
    parser_args = parser.parse_args()
    for key, value in vars(parser_args).items():
        if key != 'validate_only' and value is not None:
            args[key] = value
            if key != 'name':
                experiment_args[key] = value
    validate_options(args)
    validate_evaluation_options(args)
    if args.get('evaluate_checkpoint'):
        if evaluation_mode(args) != 'controlled_oc':
            raise ValueError('Compare checkpoints with controlled_oc for both models')
        checkpoint = Path(args['evaluate_checkpoint'])
        if not checkpoint.is_file():
            raise ValueError('Evaluation checkpoint does not exist')
        args['evaluation_checkpoint_sha256'] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if parser_args.name is None:
            args['name'] = 'EBM_controlled_OC_eval_'+args['evaluation_checkpoint_sha256'][:8]
    elif args.get('evaluate_phases'):
        raise ValueError('--evaluate-phases requires --evaluate-checkpoint')
    if args['bpl_pairing'] != 'legacy_long_short':
        from positive_bpl import MatchedTrainSet
        preflight = MatchedTrainSet(args, load_train_items_only(args['path_19LA']))
        if args.get('matched_source_sampling','replacement')=='coverage':
            from positive_bpl import coverage_source_plans
            plans=coverage_source_plans(preflight,{y:args['bpl_steps_per_epoch']*int(args['batch_size']*args[ratio])
                for y,ratio in [(1,'ratio_bona'),(0,'ratio_spoof')]})
            args['source_coverage_plan']={y:plan.summary for y,plan in plans.items()}
        args['duration_policy_sha256'] = preflight.policy_sha256
        if preflight.activity_policy is not None:
            args['activity_policy_sha256'] = preflight.activity_policy.sha256
        if parser_args.name is None and not args.get('evaluate_checkpoint'):
            args['name'] = f"EBM_e{args['epoch']}_OC_BPL_{args['bpl_pairing']}_{preflight.policy_sha256[:8]}"
            if preflight.activity_policy is not None:
                args['name'] += '_act_'+preflight.activity_policy.sha256[:8]
            if args.get('matched_source_sampling','replacement')=='coverage':
                args['name'] += '_src_coverage'
        del preflight
    if parser_args.validate_only:
        print({'objective': 'original OC+BPL, two optimizer steps including centroid',
               'bpl_pairing': args['bpl_pairing'], 'evaluation': evaluation_mode(args),
               'policy_sha256': args.get('duration_policy_sha256'),
               'activity_policy_sha256': args.get('activity_policy_sha256'),
               'source_sampling':args.get('matched_source_sampling','replacement'),
               'source_coverage_plan':args.get('source_coverage_plan'),
               'evaluation_only': bool(args.get('evaluate_checkpoint'))})
        raise SystemExit(0)

    # set reproducible
    random.seed(args['rand_seed'])
    np.random.seed(args['rand_seed'])
    torch.manual_seed(args['rand_seed'])

    # set gpu device
    if args['usable_gpu'] is None: 
        args['gpu_ids'] = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args['usable_gpu']
        args['gpu_ids'] = args['usable_gpu'].split(',')
    
    args['port'] = f'10{datetime.datetime.now().microsecond % 100}'
    assert 0 < len(args['gpu_ids']), 'Only GPU env are supported'

    # set DDP
    args['world_size'] = len(args['gpu_ids'])
    if evaluation_mode(args) == 'controlled_oc' and args['gpu_ids'] != ['0','1']:
        raise ValueError('Controlled evaluation requires CUDA_VISIBLE_DEVICES=0,1')
    args['batch_size'] = args['batch_size'] // args['world_size']


    # Validate before spawning workers or constructing the destructive logger.
    if args['epoch'] > args['T_0']:
        raise ValueError('This native-length implementation keeps XLS-R frozen; epoch must be <= T_0')
    if not 0 <= args['short_oc_weight'] <= 1 or args['padding_loss_weight'] < 0:
        raise ValueError('Invalid auxiliary loss weights')
    if not args['native_length'] and (args['short_oc_weight'] or args['padding_loss_weight']):
        raise ValueError('Legacy E0 requires both auxiliary weights set to zero')
    result_path = os.path.join('/', args['path_log'], args['project'], '/'.join(args['tags']), args['name'])
    if os.path.exists(result_path):
        raise FileExistsError('Result directory already exists: '+result_path)

    # start
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.multiprocessing.spawn(
        run, 
        nprocs=args['world_size'], 
        args=(args, experiment_args)
    )
