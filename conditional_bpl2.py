"""Conditional same-source5/4 or4/3 pairs; all classes' original OC views stay intact."""
from contextlib import contextmanager
import hashlib
import random
import numpy as np

LONG=64600
SHORT=48440
FIVE=80000

def targets(original_samples):
    if original_samples<720:raise ValueError('Too few audio samples')
    return (FIVE,LONG) if original_samples>LONG else (LONG,SHORT)

def wrap_prefix(wave,size):
    if wave.ndim!=1 or len(wave)<720 or not np.isfinite(wave).all():raise ValueError('Invalid mono audio')
    return np.pad(wave,(0,max(0,size-len(wave))),mode='wrap')[:size].copy()

def short_window(parent,original_samples,rng):
    """Stay inside unique content when possible; otherwise use the cyclic parent."""
    long_n,short_n=targets(original_samples)
    if len(parent)!=long_n:raise ValueError('Wrong reference length')
    available=min(original_samples,long_n)
    last=available-short_n if available>=short_n else long_n-short_n
    start=int(rng.integers(last+1))
    return parent[start:start+short_n].copy(),start

def pair_seed(seed,rank,epoch,index,visit,worker_seed):
    text=f'{seed}|{rank}|{epoch}|{index}|{visit}|{worker_seed}|conditionalBPL2'
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4],'little')

@contextmanager
def independent_random(seed):
    py,npstate=random.getstate(),np.random.get_state()
    try:
        random.seed(seed);np.random.seed(seed);yield
    finally:random.setstate(py);np.random.set_state(npstate)

class ConditionalTrainSet:
    """Duck-typed Torch Dataset, importable under its real module name by spawned workers."""
    def __init__(self,args,items,original_class):
        self.args=dict(args);self.items=list(items);self.original=original_class(args,self.items)
        self.epoch=0;self.visits={}
    def __len__(self):return len(self.items)
    def __getitem__(self,index):
        import soundfile as sf
        import torch
        from positive_bpl import rawboost
        item=self.items[index]
        # The archived method consumes the exact original augmentation RNG first.
        long,short,label=self.original[index]
        wave,sr=sf.read(item.path)
        assert sr==16000 and wave.ndim==1 and len(long)==LONG and len(short)==SHORT
        visit=self.visits.get(index,0);self.visits[index]=visit+1
        seed=pair_seed(self.args['rand_seed'],self.args['rank'],self.epoch,index,visit,torch.initial_seed())
        if label:
            with independent_random(seed):
                parent=rawboost(wrap_prefix(wave,FIVE),random.randint(0,8)) if len(wave)>LONG else long.copy()
            student,start=short_window(parent,len(wave),np.random.default_rng(seed))
            parent=parent.astype('float32');student=student.astype('float32')
        else:parent=np.empty(0,dtype='float32');student=parent;start=-1
        return {'long':np.asarray(long,dtype='float32'),'short':np.asarray(short,dtype='float32'),
                'label':int(label),'parent':parent,'student':student,'uid':__import__('pathlib').Path(item.path).stem,
                'original_samples':len(wave),'pair_start':start,'pair_seed':seed}

def collate_train(rows):return rows

def merge_train(bona,spoof):
    import torch
    rows=bona+spoof;n=len(bona)
    assert n>0 and all(r['label']==1 for r in bona) and all(r['label']==0 for r in spoof)
    return {'long':[torch.from_numpy(r['long']) for r in rows],
            'short':[torch.from_numpy(r['short']) for r in rows],
            'student':[torch.from_numpy(r['student']) for r in bona],
            'teacher5':[torch.from_numpy(r['parent']) for r in bona if r['original_samples']>LONG],
            'teacher5_indices':[i for i,r in enumerate(bona) if r['original_samples']>LONG],
            'labels':torch.tensor([r['label'] for r in rows]),'rows':rows,'bona':n}
