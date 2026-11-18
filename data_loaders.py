import numpy as np
import soundfile as sf
import random


from data_processing.augmentation.raw_boost import ISD_additive_noise, LnL_convolutive_noise, SSI_additive_noise, normWav

from data_processing import CodecAugmentationLight
from torch.utils.data import Dataset, DataLoader, Sampler
from functools import partial
import math
import torch

def separate_bona_spoof(train_db):
    bona_db = []
    spoof_db = []
    for item in train_db:
        if item.label == 1:
            bona_db.append(item)
        else: 
            spoof_db.append(item)
    return bona_db, spoof_db

class BalancedLengthBatchSampler(Sampler):
    """Shared class-wise sortish schedule; each rank gets the same class quotas."""
    def __init__(self, dataset, args):
        self.rank = args['rank']
        self.world = args['world_size']
        self.seed = args['rand_seed']
        self.epoch = 0
        self.quota = {1: int(args['batch_size']*args['ratio_bona']),
                      0: int(args['batch_size']*args['ratio_spoof'])}
        if min(self.quota.values()) < 1 or sum(self.quota.values()) != args['batch_size']:
            raise ValueError('batch_size must support the requested integer class quotas')
        self.ids = {y: np.array([i for i,item in enumerate(dataset.items) if item.label == y]) for y in [1,0]}
        if any(len(v)==0 for v in self.ids.values()):
            raise ValueError('Both training classes are required')
        self.lengths = np.array([sf.info(item.path).frames for item in dataset.items])
        self.steps = max(math.ceil(len(self.ids[y])/(self.quota[y]*self.world)) for y in [1,0])

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.steps

    def __iter__(self):
        rng = np.random.default_rng(self.seed+self.epoch)
        groups = {}
        for y in [1,0]:
            width = self.quota[y]*self.world
            ids = rng.permutation(self.ids[y])
            needed = self.steps*width
            if needed > len(ids):
                ids = np.concatenate([ids, np.resize(rng.permutation(ids), needed-len(ids))])
            pool = 32*width
            ids = np.concatenate([q[np.argsort(self.lengths[q], kind='stable')]
                                  for q in [ids[i:i+pool] for i in range(0,len(ids),pool)]])
            groups[y] = ids.reshape(self.steps,width)
        for step in rng.permutation(self.steps):
            yield np.concatenate([groups[y][step,self.rank*self.quota[y]:(self.rank+1)*self.quota[y]]
                                  for y in [1,0]]).tolist()


class ExactDistributedEvalSampler(Sampler):
    """Shard without DistributedSampler's duplicated trailing evaluation items."""
    def __init__(self, dataset, args):
        self.n, self.rank, self.world = len(dataset), args['rank'], args['world_size']

    def __iter__(self):
        return iter(range(self.rank,self.n,self.world))

    def __len__(self):
        return max(0,(self.n+self.world-1-self.rank)//self.world)


def collate_train(rows, short_size, include_wrapped=True):
    long, short, labels, indices = zip(*rows)
    batch = {'long':[torch.as_tensor(a,dtype=torch.float32) for a in long],
             'short':[torch.as_tensor(a,dtype=torch.float32) for a in short],
             'labels':torch.tensor(labels,dtype=torch.long), 'indices':torch.tensor(indices)}
    # Auxiliary experiments can opt in; the B control creates no repeated view.
    if include_wrapped:
        wrapped = [np.pad(s,(0,short_size-len(s)),mode='wrap') if len(s)<short_size else s.copy() for s in short]
        batch['wrapped'] = [torch.as_tensor(a,dtype=torch.float32) for a in wrapped]
        batch['pair_active'] = torch.tensor([len(a)<short_size for a in short])
    return batch


def collate_eval(rows):
    waves, attacks, labels, indices = zip(*rows)
    return {'waves':[torch.as_tensor(a,dtype=torch.float32) for a in waves],
            'labels':torch.tensor(labels,dtype=torch.long),'indices':torch.tensor(indices),
            'attack_types':list(attacks)}


def get_loaders(args, train_db, val_db, eval_db):
    if args.get('bpl_pairing', 'legacy_long_short') != 'legacy_long_short':
        from positive_bpl import get_loaders as matched_loaders
        return matched_loaders(args, train_db, val_db, eval_db)
    train_set = TrainSet(args, train_db)
    sampler = BalancedLengthBatchSampler(train_set,args)
    train_loader = DataLoader(train_set, batch_sampler=sampler,
        num_workers=args['num_workers'], pin_memory=True,
        collate_fn=partial(collate_train,short_size=args['num_train_frames_short']*160-40,
            include_wrapped=bool(args.get('short_oc_weight',0.) or args.get('padding_loss_weight',0.))))
    def evaluation_loader(items):
        # Fix dev/target input to B's cap so only training long context changes.
        dataset = EnrollmentSet(dict(args,cap_long=True),items)
        return DataLoader(dataset,batch_size=args.get('eval_batch_size',8),
            sampler=ExactDistributedEvalSampler(dataset,args),num_workers=args['num_workers'],
            collate_fn=collate_eval,pin_memory=False)
    val_loader = evaluation_loader(val_db)
    eval_loader = {name:evaluation_loader(db.trials) for name,db in eval_db.items()}
    return train_loader,sampler,val_loader,eval_loader


class TrainSet(Dataset):
    def __init__(self, args, items):
        self.items = items
        self.native_length = args.get('native_length', True)
        self.cap_long = args.get('cap_long', False)
        self.crop_size = args['num_train_frames'] * 160 - 40
        self.crop_size_short = args['num_train_frames_short'] * 160 - 40
        self.codec_aug = CodecAugmentationLight()
        self.p_codec_aug = 0.8
        
    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        # sample
        item = self.items[index]
        
        # read wav
        audio, sr = sf.read(item.path)

        # if random.random() < self.p_codec_aug:
            # codec_crop_size = int(self.crop_size * 1.2)
            # if not self.native_length and codec_crop_size < audio.shape[0]:
                # audio = audio[:codec_crop_size]
            # audio = self.codec_aug.convert(audio, sr)

        if self.native_length and self.cap_long:
            audio = audio[:self.crop_size]
        elif not self.native_length:
            if audio.shape[0] < self.crop_size:
                shortage = self.crop_size - audio.shape[0]
                audio = np.pad(audio, (0, shortage), 'wrap')
            else:
                audio = audio[:self.crop_size]


        augtype = random.randint(0,8)
        if augtype == 1:
            audio = LnL_convolutive_noise(audio) # Convolutive noise 
        elif augtype == 2:
            audio = ISD_additive_noise(audio) # Impulsive noise
        elif augtype == 3:
            audio = SSI_additive_noise(audio) # coloured additive noise
        elif augtype == 4:
            audio = LnL_convolutive_noise(audio) # Convolutive noise 
            audio = ISD_additive_noise(audio) # Impulsive noise
            audio = SSI_additive_noise(audio) # coloured additive noise
        elif augtype == 5:
            audio = LnL_convolutive_noise(audio) # Convolutive noise 
            audio = ISD_additive_noise(audio) # Impulsive noise
        elif augtype == 6:
            audio = LnL_convolutive_noise(audio) # Convolutive noise 
            audio = SSI_additive_noise(audio) # coloured additive noise
        elif augtype == 7:
            audio = ISD_additive_noise(audio) # Impulsive noise
            audio = SSI_additive_noise(audio) # coloured additive noise
        elif augtype == 8:
            audio1 = LnL_convolutive_noise(audio) # Convolutive noise 
            audio2 = ISD_additive_noise(audio) # Impulsive noise
            audio = audio1 + audio2
            audio=normWav(audio, 0)


        # read short wav
        audio_short, _ = sf.read(item.path)

        # if random.random() < self.p_codec_aug:
            # codec_crop_size_short = int(self.crop_size_short * 1.2)
            # if not self.native_length and codec_crop_size_short < audio_short.shape[0]:
                # audio_short = audio_short[:codec_crop_size_short]
            # audio_short = self.codec_aug.convert(audio_short, sr)

        if audio_short.shape[0] < self.crop_size_short:
            shortage = self.crop_size_short - audio_short.shape[0]
            if not self.native_length:
                audio_short = np.pad(audio_short, (0, shortage), 'wrap')
        else:
            start = random.randint(0, audio_short.shape[0] - self.crop_size_short) 
            audio_short = audio_short[start: start + self.crop_size_short]


        augtype = random.randint(0,8)
        if augtype == 1:
            audio_short = LnL_convolutive_noise(audio_short) # Convolutive noise 
        elif augtype == 2:
            audio_short = ISD_additive_noise(audio_short) # Impulsive noise
        elif augtype == 3:
            audio_short = SSI_additive_noise(audio_short) # coloured additive noise
        elif augtype == 4:
            audio_short = LnL_convolutive_noise(audio_short) # Convolutive noise 
            audio_short = ISD_additive_noise(audio_short) # Impulsive noise
            audio_short = SSI_additive_noise(audio_short) # coloured additive noise
        elif augtype == 5:
            audio_short = LnL_convolutive_noise(audio_short) # Convolutive noise 
            audio_short = ISD_additive_noise(audio_short) # Impulsive noise
        elif augtype == 6:
            audio_short = LnL_convolutive_noise(audio_short) # Convolutive noise 
            audio_short = SSI_additive_noise(audio_short) # coloured additive noise
        elif augtype == 7:
            audio_short = ISD_additive_noise(audio_short) # Impulsive noise
            audio_short = SSI_additive_noise(audio_short) # coloured additive noise
        elif augtype == 8:
            audio1 = LnL_convolutive_noise(audio_short) # Convolutive noise 
            audio2 = ISD_additive_noise(audio_short) # Impulsive noise
            audio_short = audio1 + audio2
            audio_short=normWav(audio_short, 0)


        if sr != 16000 or audio.ndim != 1 or min(len(audio),len(audio_short)) < 720:
            raise ValueError("Expected mono16k audio with at least two valid SSL frames")
        return audio, audio_short, item.label, index


class EnrollmentSet(Dataset):
    def __init__(self, args, items):
        self.items = items    
        self.native_length = args.get('native_length', True)
        self.cap_long = args.get('cap_long', False)
        self.crop_size = args['num_test_frames'] * 160 - 40
        self.num_seg = args['num_seg']
    
    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):        
        # sample
        item = self.items[index]

        # read wav
        audio, sr = sf.read(item.path)
        if sr != 16000 or audio.ndim != 1 or len(audio) < 720:
            raise ValueError("Expected mono16k audio with at least two valid SSL frames")
        if self.native_length and self.cap_long:
            audio = audio[:self.crop_size]
        elif not self.native_length:
            if audio.shape[0] < self.crop_size:
                shortage = self.crop_size - audio.shape[0]
                audio = np.pad(audio, (0, shortage), 'wrap')
            else:
                audio = audio[:self.crop_size]

        return audio, item.type, item.label, index
