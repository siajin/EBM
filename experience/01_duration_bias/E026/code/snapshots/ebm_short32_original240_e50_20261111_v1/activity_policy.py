"""Validated external raw-speech activity policy; no VAD runs during training."""
import hashlib
import json
from pathlib import Path
import numpy as np


def activity_bins(speech_frames,total_frames):
    # zero is its own cell; positive intervals are(0,.2],...,(.8,1].
    return ((np.asarray(speech_frames,dtype=np.int64)*5+total_frames-1)//total_frames).astype(np.int64)


class ActivityPolicy:
    def __init__(self,path,dataset):
        path=Path(path);raw=path.read_bytes();meta=json.loads(raw)
        self.sha256=hashlib.sha256(raw).hexdigest();self.metadata=meta
        if meta.get('schema_version')!=1 or meta.get('reference')!='raw_before_RawBoost':
            raise ValueError('Unknown activity policy schema/reference')
        if meta.get('activity_matching_scope')!='class_global_with_source_marginals':
            raise ValueError('Unexpected activity matching scope')
        if meta.get('duration_policy_sha256')!=dataset.policy_sha256 or meta.get('source_fingerprint')!=dataset.source_fingerprint:
            raise ValueError('Activity policy differs from Train/duration policy')
        if meta.get('activity_bins')!=['zero','(0,.2]','(.2,.4]','(.4,.6]','(.6,.8]','(.8,1]']:
            raise ValueError('Unexpected activity bin definition')
        filename=meta['arrays_file']
        if Path(filename).name!=filename:
            raise ValueError('Activity arrays must be beside their policy JSON')
        array_path=path.parent/filename
        if hashlib.sha256(array_path.read_bytes()).hexdigest()!=meta['arrays_sha256']:
            raise ValueError('Activity array SHA mismatch')
        with np.load(array_path,allow_pickle=False) as data:
            arrays={k:data[k] for k in data.files}
        n=len(dataset);k=len(dataset.targets)
        self.offsets=np.asarray(arrays['context_offsets'],dtype=np.int64)
        self.starts=np.asarray(arrays['starts'],dtype=np.int64)
        self.voice=np.asarray(arrays['speech_frames'],dtype=np.int64)
        self.positions=np.asarray(arrays['position_probabilities'],dtype=np.float64)
        self.lengths=np.asarray(arrays['length_probabilities'],dtype=np.float64)
        if self.offsets.shape!=(n*k+1,) or self.offsets[0]!=0 or np.any(np.diff(self.offsets)<0):
            raise ValueError('Invalid activity context offsets')
        m=int(self.offsets[-1])
        if any(v.shape!=(m,) for v in [self.starts,self.voice,self.positions]) or self.lengths.shape!=(n,k):
            raise ValueError('Invalid activity array dimensions')
        if (not np.isfinite(self.positions).all() or not np.isfinite(self.lengths).all()
                or (self.positions<0).any() or (self.lengths<0).any()):
            raise ValueError('Invalid activity probabilities')
        if not np.allclose(self.lengths.sum(1),1,atol=1e-10,rtol=0):
            raise ValueError('Activity lengths must normalize for every original')
        uids=np.asarray([Path(item.path).stem for item in dataset.items])
        if not np.array_equal(arrays['source_uids'],uids):
            raise ValueError('Activity UID order differs from Train')
        stats=[Path(item.path).stat() for item in dataset.items]
        if (not np.array_equal(arrays['source_sizes'],[v.st_size for v in stats])
                or not np.array_equal(arrays['source_mtime_ns'],[v.st_mtime_ns for v in stats])):
            raise ValueError('Original files changed since activity measurement')
        context=np.repeat(np.arange(n*k),np.diff(self.offsets));orig=context//k;length=context%k
        frames=dataset.targets[length]//320
        if ((self.starts<0).any() or np.any(self.starts+dataset.targets[length]>dataset.lengths[orig])
                or (self.voice<0).any() or np.any(self.voice>frames)):
            raise ValueError('Activity candidate lies outside its original/voice range')
        legal_front=self.starts%dataset.targets[length]==0
        legal_back=(dataset.lengths[orig]-self.starts-dataset.targets[length])%dataset.targets[length]==0
        if not np.all(legal_front|legal_back):
            raise ValueError('Activity candidate is outside the existing front/back tile grid')
        if np.any((context[1:]==context[:-1]) & (self.starts[1:]<=self.starts[:-1])):
            raise ValueError('Activity candidates must be unique and sorted within context')
        sums=np.bincount(context,weights=self.positions,minlength=n*k).reshape(n,k)
        active=self.lengths>0
        if np.any(active & (np.diff(self.offsets).reshape(n,k)==0)) or not np.allclose(sums[active],1,atol=1e-9,rtol=0):
            raise ValueError('Activity position probabilities do not normalize')
        for key in dataset.profiles:
            ids=np.array([i for i,v in enumerate(dataset.keys) if v==key])
            if not np.allclose(self.lengths[ids].mean(0),dataset.q,atol=2e-6,rtol=0):
                raise ValueError('Activity policy changed speaker/attack duration targets')
        labels=np.asarray([v.label for v in dataset.items])[orig]
        speakers=sorted({v.speaker for v in dataset.items});mapping={v:i for i,v in enumerate(speakers)}
        source_speaker=np.asarray([mapping[v.speaker] for v in dataset.items]);speaker=source_speaker[orig]
        bins=activity_bins(self.voice,frames)
        mass=dataset.source_probability[orig]*self.lengths[orig,length]*self.positions
        code=((speaker*2+labels)*k+length)*6+bins
        joint=np.bincount(code,weights=mass,minlength=len(speakers)*2*k*6).reshape(len(speakers),2,k,6)
        code_mean=(speaker*2+labels)*k+length
        means=np.bincount(code_mean,weights=mass*self.voice/frames,minlength=len(speakers)*2*k).reshape(len(speakers),2,k)
        max_bin_error=max_mean_error=0.
        for j,speaker_name in enumerate(speakers):
            shares=[dataset.source_probability[[i for i,v in enumerate(dataset.items) if v.speaker==speaker_name and v.label==y]].sum() for y in [0,1]]
            if not np.isclose(shares[0],shares[1],atol=1e-10,rtol=0):
                raise ValueError('Activity matching requires the existing common speaker priors')
            scale=shares[0]*dataset.q
            max_bin_error=max(max_bin_error,float(np.max(np.abs(joint[j,0]-joint[j,1])/scale[:,None])))
            max_mean_error=max(max_mean_error,float(np.max(np.abs(means[j,0]-means[j,1])/scale)))
        global_joint=joint.sum(0)
        global_means=means.sum(0)
        global_bin_error=float(np.max(np.abs(global_joint[0]-global_joint[1])/dataset.q[:,None]))
        global_mean_error=float(np.max(np.abs(global_means[0]-global_means[1])/dataset.q))
        if global_bin_error>2e-6 or global_mean_error>2e-6:
            raise ValueError('Activity class distributions/means do not match within length')
        self.statistics={'max_conditional_bin_error':global_bin_error,'max_conditional_mean_error':global_mean_error,
                         'max_within_speaker_bin_difference':max_bin_error,
                         'max_within_speaker_mean_difference':max_mean_error,
                         'expected_class_joint':joint.sum(0).tolist(),'candidates':m,'originals':n}

    def choose(self,index,duration_rng,position_rng,targets):
        k=int(duration_rng.choice(len(targets),p=self.lengths[index]))
        a,b=self.offsets[index*len(targets)+k:index*len(targets)+k+2]
        j=int(position_rng.choice(int(b-a),p=self.positions[a:b]))+int(a)
        return int(targets[k]),int(self.starts[j])
