"""Matched BPL pair sampling; variable OC and BPL inputs; original OC loss/centroid optimization stay intact."""
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
import hashlib
import json
from pathlib import Path
import random
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from data_processing.augmentation.raw_boost import (
    ISD_additive_noise, LnL_convolutive_noise, SSI_additive_noise, normWav,
)


def evaluation_mode(args):
    mode = args.get('evaluation_mode', 'controlled_oc')
    if mode not in ('original_oc', 'controlled_oc'):
        raise ValueError('Both models must score with their trained OC center; mean-center evaluation is removed')
    return mode


def validate_evaluation_options(args):
    if evaluation_mode(args) != 'controlled_oc':
        return
    if (not args.get('native_length', True) or args.get('num_test_frames', 404) != 404
            or args.get('num_seg', 1) != 1 or args.get('eval_batch_size', 8) != 8):
        raise ValueError('Controlled OC evaluation fixes native masks, front64600, one segment, batch8')
    if not isinstance(args.get('evaluation_seed', 1729), int) or args.get('evaluation_seed', 1729) < 0:
        raise ValueError('evaluation_seed must be a nonnegative integer')


def validate_options(args):
    if args.get('training_mode', 'oc_bpl') != 'oc_bpl':
        raise ValueError('BPL-only training was removed: retain original OC+BPL and centroid updates')
    pairing = args.get('bpl_pairing', 'legacy_long_short')
    if pairing not in ('legacy_long_short', 'matched_same_source', 'matched_independent'):
        raise ValueError('Unknown bpl_pairing')
    source_sampling=args.get('matched_source_sampling','replacement')
    if source_sampling not in ('replacement','coverage'):
        raise ValueError('Unknown matched source sampling')
    if source_sampling=='coverage' and pairing=='legacy_long_short':
        raise ValueError('Coverage source sampling requires a matched BPL pairing')
    activity=args.get('activity_policy_path')
    if activity and (pairing=='legacy_long_short' or not Path(activity).is_file()):
        raise ValueError('Activity policy requires an existing JSON and a matched BPL pairing')
    if pairing == 'legacy_long_short':
        return
    if args.get('bpl_pair_size') != 12 or args.get('bpl_steps_per_epoch') != 108:
        raise ValueError('Matched BPL retains 12x12 pairs and the original108-step schedule')
    if not args.get('native_length', True) or args.get('short_oc_weight', 0.) or args.get('padding_loss_weight', 0.):
        raise ValueError('Matched BPL retains the native OC path with auxiliary losses disabled')
    if not args.get('duration_policy_path') or not Path(args['duration_policy_path']).is_file():
        raise ValueError('Matched BPL requires an external Train duration policy')


@contextmanager
def evaluation_rng(args, stream):
    """Keep original ASP noise, but share its draws and preserve training RNG."""
    device = torch.device(args['device'])
    devices = [device.index] if device.type == 'cuda' else []
    py_state, np_state = random.getstate(), np.random.get_state()
    seed = args.get('evaluation_seed', 1729) + stream + args.get('rank', 0)
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed)
            torch.random.default_generator.manual_seed(seed)
            if devices:
                torch.cuda.manual_seed(seed)
            yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


def rawboost(audio, kind):
    """Exactly the original TrainSet recipe; do not substitute the pilot recipe."""
    if kind == 1:
        audio = LnL_convolutive_noise(audio)
    elif kind == 2:
        audio = ISD_additive_noise(audio)
    elif kind == 3:
        audio = SSI_additive_noise(audio)
    elif kind == 4:
        audio = LnL_convolutive_noise(audio)
        audio = ISD_additive_noise(audio)
        audio = SSI_additive_noise(audio)
    elif kind == 5:
        audio = LnL_convolutive_noise(audio)
        audio = ISD_additive_noise(audio)
    elif kind == 6:
        audio = LnL_convolutive_noise(audio)
        audio = SSI_additive_noise(audio)
    elif kind == 7:
        audio = ISD_additive_noise(audio)
        audio = SSI_additive_noise(audio)
    elif kind == 8:
        audio1 = LnL_convolutive_noise(audio)
        audio2 = ISD_additive_noise(audio)
        audio = normWav(audio1 + audio2, 0)
    elif kind != 0:
        raise ValueError('RawBoost kind must be 0..8')
    return audio


@dataclass(frozen=True)
class CropSpec:
    index: int
    start: int
    stop: int
    augmentation_seed: int


class MatchedTrainSet(Dataset):
    """Class-conditional source weights and crop tables for both OC classes."""
    def __init__(self, args, items):
        self.items = list(items)
        paths = [str(Path(item.path)) for item in self.items]
        if not paths or len(set(paths)) != len(paths):
            raise ValueError('Empty or duplicate Train originals')
        raw = Path(args['duration_policy_path']).read_bytes()
        self.policy = json.loads(raw)
        self.policy_text = raw.decode('utf-8')
        self.policy_sha256 = hashlib.sha256(raw).hexdigest()
        if self.policy.get('sample_rate') != 16000 or self.policy.get('step_samples') != 8000:
            raise ValueError('Expected a mono16k policy with0.5s steps')
        self.targets = np.asarray(self.policy['target_samples'], dtype='int64')
        self.q = np.asarray(self.policy['target_probabilities'], dtype='float64')
        if not np.array_equal(self.targets, 8000*np.arange(1,len(self.targets)+1)):
            raise ValueError('Targets must be consecutive positive0.5s multiples')
        if self.q.shape != self.targets.shape or not np.isfinite(self.q).all() or (self.q<=0).any() or not np.isclose(self.q.sum(),1):
            raise ValueError('Invalid target probabilities')
        labels = np.array([item.label for item in self.items])
        if set(labels) != {0,1}:
            raise ValueError('OC training requires both bona fide and spoof originals')
        self.keys = [(item.label, item.speaker, item.type) for item in self.items]
        profiles = self.policy['conditional_profiles']
        self.profiles = {(p['label'],p['speaker'],p['attack']):p for p in profiles}
        weights = self.policy['original_relative_weights']
        relative = {(p['label'],p['speaker']):p for p in weights}
        if len(self.profiles)!=len(profiles) or len(relative)!=len(weights):
            raise ValueError('Duplicate policy profile/weight')
        if set(self.keys)!=set(self.profiles) or set((y,s) for y,s,a in self.keys)!=set(relative):
            raise ValueError('Policy class/speaker/attack groups differ from Train')
        if any(y==1 and attack!='bonafide' for y,s,attack in self.keys):
            raise ValueError('Bona fide attack metadata must be bonafide')
        info = [sf.info(item.path) for item in self.items]
        if any(v.samplerate!=16000 or v.channels!=1 or v.frames<8000 for v in info):
            raise ValueError('Policy requires mono16k originals of at least0.5s')
        self.lengths = np.array([v.frames for v in info],dtype='int64')
        self.buckets = np.minimum(self.lengths//8000,len(self.targets))
        self.tables = {}
        self.source_probability = np.zeros(len(self.items))
        for key,profile in self.profiles.items():
            y,speaker,attack = key
            ids = np.array([i for i,k in enumerate(self.keys) if k==key],dtype='int64')
            counts = np.bincount(self.buckets[ids],minlength=len(self.targets)+1)[1:]
            if len(ids)!=profile['original_count'] or not np.array_equal(counts,profile['bucket_counts']):
                raise ValueError('Policy original count/length buckets differ from Train: '+str(key))
            table = np.asarray(profile['conditional_probability'],dtype='float64')
            size = len(self.targets)
            if table.shape!=(size,size) or not np.isfinite(table).all() or (table<0).any():
                raise ValueError('Invalid conditional probability table')
            if not np.allclose(table.sum(1),1,atol=1e-10) or np.any(table[np.triu_indices(size,1)]>0):
                raise ValueError('Invalid or impossible conditional crop lengths')
            if not np.allclose(counts@table/len(ids),self.q,atol=1e-9,rtol=0):
                raise ValueError('Policy does not reproduce the target within group')
            self.tables[key] = table
        self.class_indices,self.class_probability = {},{}
        for y in [1,0]:
            class_ids = np.flatnonzero(labels==y)
            for (label,speaker),entry in relative.items():
                if label!=y:
                    continue
                ids = np.array([i for i in class_ids if self.items[i].speaker==speaker],dtype='int64')
                share = len(ids)/len(class_ids)
                ratio,target = entry['original_weight_multiplier'],entry['common_target_share']
                if not np.isclose(entry['original_share'],share,atol=1e-10):
                    raise ValueError('Policy original speaker share differs from Train')
                if not np.isfinite([ratio,target]).all() or min(ratio,target)<=0 or not np.isclose(target,ratio*share):
                    raise ValueError('Invalid source weight')
                self.source_probability[ids] = target/len(ids)
            probabilities = self.source_probability[class_ids]
            if not np.isclose(probabilities.sum(),1,atol=1e-10):
                raise ValueError('Source probabilities must sum to one within each class')
            self.class_indices[y] = class_ids
            self.class_probability[y] = probabilities/probabilities.sum()
        fingerprint = '\n'.join(f'{item.label}|{Path(item.path).stem}|{item.speaker}|{item.type}|{int(n)}' for item,n in zip(self.items,self.lengths))
        self.source_fingerprint = hashlib.sha256(fingerprint.encode()).hexdigest()
        self.activity_policy = None
        if args.get('activity_policy_path'):
            from activity_policy import ActivityPolicy
            self.activity_policy = ActivityPolicy(args['activity_policy_path'],self)

    def __len__(self):
        return len(self.items)

    def crop_spec(self, index, seed_parts):
        duration_seed, position_seed, augmentation_seed = np.random.SeedSequence(seed_parts).generate_state(3)
        duration_rng = np.random.default_rng(duration_seed)
        position_rng = np.random.default_rng(position_seed)
        if self.activity_policy is not None:
            length,start = self.activity_policy.choose(index,duration_rng,position_rng,self.targets)
        else:
            p = self.tables[self.keys[index]][self.buckets[index]-1]
            length = int(duration_rng.choice(self.targets, p=p))
            tile = int(position_rng.integers(self.lengths[index]//length))
            back = bool(position_rng.integers(2))
            start = int(self.lengths[index]-(tile+1)*length) if back else tile*length
        return CropSpec(int(index), start, start+length, int(augmentation_seed))

    def read_crop(self, spec, augment=True):
        if not (0 <= spec.start < spec.stop <= self.lengths[spec.index]):
            raise ValueError('Crop is outside its original')
        # Decode only the requested interval; preserve the original float64
        # samples and validate the full-file header without reading its audio.
        with sf.SoundFile(self.items[spec.index].path, mode='r') as source:
            if (source.samplerate != 16000 or source.channels != 1
                    or source.frames != self.lengths[spec.index]):
                raise ValueError('Original audio changed after policy validation')
            if source.seek(spec.start) != spec.start:
                raise ValueError('Could not seek to the requested crop')
            audio = source.read(frames=spec.stop-spec.start, dtype='float64', always_2d=False)
        if audio.ndim != 1 or len(audio) != spec.stop-spec.start:
            raise ValueError('Incomplete crop read')
        if augment:
            py_state, np_state = random.getstate(), np.random.get_state()
            try:
                random.seed(spec.augmentation_seed)
                np.random.seed(spec.augmentation_seed)
                audio = rawboost(audio, random.randint(0, 8))
            finally:
                random.setstate(py_state)
                np.random.set_state(np_state)
        if len(audio) != spec.stop-spec.start or not np.isfinite(audio).all():
            raise ValueError('Augmentation changed length or produced non-finite samples')
        return torch.as_tensor(audio, dtype=torch.float32)

    def __getitem__(self, spec):
        return self.read_crop(spec), self.items[spec.index].label, spec.index


class CoverageSourcePlan:
    """One visit per original plus residual draws, preserving source marginals.

    For T slots and original probability p_i, the residual expected count is
    T*p_i-1. Shuffling the complete schedule makes every slot marginally p_i.
    Feasibility requires T*p_i >= 1 for every original; fail before training
    rather than silently changing a prior, quota or epoch length.
    """
    def __init__(self,indices,probabilities,total):
        self.indices=np.asarray(indices,dtype=np.int64)
        self.probabilities=np.asarray(probabilities,dtype=np.float64)
        self.total=int(total)
        n=len(self.indices)
        if (not n or len(np.unique(self.indices))!=n or self.probabilities.shape!=(n,)
                or not np.isfinite(self.probabilities).all() or np.any(self.probabilities<=0)
                or not np.isclose(self.probabilities.sum(),1.,atol=1e-12,rtol=0)):
            raise ValueError('Invalid coverage source indices/probabilities')
        expected=self.total*self.probabilities
        if self.total<n or expected.min()<1.-1e-10:
            minimum=int(np.ceil(np.max(1/self.probabilities)-1e-10))
            raise ValueError(f'Coverage cannot retain source priors in {self.total} slots; need at least {minimum}')
        self.extra=self.total-n
        if self.extra:
            residual=np.maximum(expected-1.,0.)
            self.residual=residual/residual.sum()
        else:
            self.residual=np.zeros(n,dtype=np.float64)
        self.expected_probability=(1.+self.extra*self.residual)/self.total
        if not np.allclose(self.expected_probability,self.probabilities,atol=1e-12,rtol=0):
            raise ValueError('Coverage correction would change original source priors')
        self.summary={'originals':n,'slots':self.total,'extra_draws':self.extra,
            'minimum_expected_visits':float(expected.min()),
            'max_expected_probability_error':float(np.max(np.abs(self.expected_probability-self.probabilities)))}

    def draw(self,rng):
        base=rng.permutation(self.indices)
        if self.extra:
            base=np.concatenate([base,rng.choice(self.indices,self.extra,p=self.residual)])
        return rng.permutation(base)


def coverage_source_plans(dataset,totals):
    return {y:CoverageSourcePlan(dataset.class_indices[y],dataset.class_probability[y],totals[y])
            for y in [1,0]}


class MatchedBatchSampler(Sampler):
    def __init__(self,dataset,args):
        self.dataset = dataset
        self.rank,self.world = args['rank'],args['world_size']
        self.seed,self.epoch = args['rand_seed'],0
        self.size,self.steps = args['bpl_pair_size'],args['bpl_steps_per_epoch']
        self.pairing = args['bpl_pairing']
        self.source_sampling = args.get('matched_source_sampling','replacement')
        if self.source_sampling not in ('replacement','coverage'):
            raise ValueError('Unknown matched source sampling')
        self.bona = int(args['batch_size']*args['ratio_bona'])
        self.spoof = int(args['batch_size']*args['ratio_spoof'])
        if (self.bona,self.spoof)!=(12,108):
            raise ValueError('Keep the per-GPU OC batch12 bona+108 spoof')
        self.coverage_plans=None
        if self.source_sampling=='coverage':
            self.coverage_plans=coverage_source_plans(dataset,
                {1:self.steps*self.bona*self.world,0:self.steps*self.spoof*self.world})

    def set_epoch(self,epoch):
        self.epoch = epoch

    def __len__(self):
        return self.steps

    def __iter__(self):
        ds = self.dataset
        rngs = [np.random.default_rng([self.seed,self.epoch,side]) for side in range(3)]
        schedules=None
        if self.coverage_plans is not None:
            schedules=[self.coverage_plans[y].draw(rng).reshape(self.steps,-1)
                       for y,rng in zip([1,0,1],rngs)]
        for step in range(self.steps):
            if schedules is None:
                bona = rngs[0].choice(ds.class_indices[1],self.bona*self.world,p=ds.class_probability[1])
                spoof = rngs[1].choice(ds.class_indices[0],self.spoof*self.world,p=ds.class_probability[0])
                right = rngs[2].choice(ds.class_indices[1],self.bona*self.world,p=ds.class_probability[1])
            else:
                bona,spoof,right=[schedule[step] for schedule in schedules]
            if self.pairing=='matched_same_source':
                right = bona.copy()
            groups = [bona[self.rank*self.bona:(self.rank+1)*self.bona],
                      spoof[self.rank*self.spoof:(self.rank+1)*self.spoof],
                      right[self.rank*self.bona:(self.rank+1)*self.bona]]
            yield [ds.crop_spec(int(index),[self.seed,self.epoch,step,self.rank,side,slot])
                   for side,ids in enumerate(groups) for slot,index in enumerate(ids)]


def collate_matched(rows):
    if len(rows)!=132:
        raise ValueError('Expected120 OC inputs and12 additional BPL bona fide inputs')
    waves,labels,indices = zip(*rows)
    if list(labels)!=[1]*12+[0]*108+[1]*12:
        raise ValueError('OC/BPL input class quotas or ordering changed')
    return {'long':list(waves[:120]),'labels':torch.tensor(labels[:120]),
            'indices':torch.tensor(indices[:120]),'bpl_right':list(waves[120:]),
            'bpl_right_indices':torch.tensor(indices[120:])}


def get_loaders(args,train_db,val_db,eval_db):
    from data_loaders import EnrollmentSet,ExactDistributedEvalSampler,collate_eval
    validate_options(args)
    if args.get('mild_transform_cache'):
        from mild_duration import MildTrainSet
        ds = MildTrainSet(args, train_db)
    else:
        ds = MatchedTrainSet(args,train_db)
    sampler = MatchedBatchSampler(ds,args)
    train = DataLoader(ds,batch_sampler=sampler,num_workers=args['num_workers'],pin_memory=True,
                       collate_fn=collate_matched,
                       generator=torch.Generator().manual_seed(args['rand_seed']+args['rank']))
    def evaluation(items):
        dataset = EnrollmentSet(dict(args,cap_long=True),items)
        return DataLoader(dataset,batch_size=args.get('eval_batch_size',8),
                          sampler=ExactDistributedEvalSampler(dataset,args),num_workers=args['num_workers'],
                          collate_fn=collate_eval,pin_memory=False)
    return train,sampler,evaluation(val_db),{name:evaluation(db.trials) for name,db in eval_db.items()}


def oc_evaluation_protocol(args, frontend_revision=None):
    return {'version': 'controlled_oc_v1', 'score': 'original cosine to trained OC weight',
            'centroid': 'original two-stage OC optimization; no post-hoc replacement',
            'evaluation_input': 'front min(original_samples,64600); no repeat; masks',
            'sample_rate': 16000, 'batch_size_per_rank': 8, 'world_size': args['world_size'],
            'evaluation_seed': args.get('evaluation_seed', 1729),
            'ASP_noise': 'original, fixed RNG stream', 'frontend_revision': frontend_revision,
            'metric': 'original full-ROC linear-crossing EER percent; bona=1; higher=bona',
            'selection': 'dev every5 epochs and final; no target-based checkpoint selection'}
