"""Standalone original-first, marginal-duration-balanced crop policy.

The training loop is deliberately not changed. Choose an original uniformly
within its class first, then call sample_crop. The fitted conditional length
probabilities retain that original marginal while matching a shared duration
target across classes. Audio stays in its original time order.
"""
from dataclasses import dataclass
from operator import index
import json
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class SampledCrop:
    start_sample: int
    stop_sample: int
    target_samples: int
    anchor: str
    tile: int


class OriginalFirstDurationPolicy:
    def __init__(self, policy):
        self.policy = policy
        self.sample_rate = index(policy['sample_rate'])
        self.step_samples = index(policy['step_samples'])
        self.targets = np.asarray(policy['target_samples'], dtype='int64')
        self.maximum_bucket = int(self.targets[-1]//self.step_samples)
        self.tables = {int(y):np.asarray(v['conditional_probability'], dtype='float64')
                       for y,v in policy['classes'].items()}
        if (self.targets<=0).any() or not np.array_equal(self.targets,np.arange(1,len(self.targets)+1)*self.step_samples):
            raise ValueError('Targets must be contiguous positive step multiples')
        for table in self.tables.values():
            expected = (self.maximum_bucket,len(self.targets))
            if table.shape!=expected or not np.isfinite(table).all() or (table<0).any():
                raise ValueError('Invalid conditional probability table')
            if not np.allclose(table.sum(axis=1),1,atol=1e-10):
                raise ValueError('Each original-length row must sum to one')
            if np.any(table[np.triu_indices(len(self.targets),k=1)]>0):
                raise ValueError('A crop cannot exceed original length')

    @classmethod
    def load(cls,path):
        return cls(json.loads(Path(path).read_text(encoding='utf-8')))

    def length_probabilities(self,num_samples,label):
        num_samples = index(num_samples)
        if num_samples<self.step_samples:
            raise ValueError('This Train-only policy requires >=0.5 s; keep shorter originals under a separately defined policy')
        if int(label) not in self.tables:
            raise ValueError('Unknown class label')
        bucket=min(num_samples//self.step_samples,self.maximum_bucket)
        return self.tables[int(label)][bucket-1].copy()

    def sample_crop(self,num_samples,label,rng):
        """Original identity has already been sampled; never reject/reselect it."""
        num_samples = index(num_samples)
        p=self.length_probabilities(num_samples,label)
        target=int(self.targets[int(rng.choice(len(self.targets),p=p))])
        tile=int(rng.integers(num_samples//target))
        back=bool(rng.integers(2))
        if back:
            start=num_samples-(tile+1)*target
        else:
            start=tile*target
        return SampledCrop(start,start+target,target,'back' if back else 'front',tile)
