"""Train-only mild duration matching, cached before any optimizer runs."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import subprocess

import numpy as np
import soundfile as sf
import torch


def transform_parameters(samples, target):
    """Round crop inward: never remove more than 20% of the original."""
    if samples < 720 or not np.isfinite(target) or target <= 0:
        raise ValueError('Invalid original/target length')
    rate = float(np.clip(samples / target, .8, 1.2))
    kept = min(samples, max((4*samples+4)//5, int(round(target*rate))))
    rate = float(np.clip(kept/target, .8, 1.2))
    output = int(np.floor(kept/rate+.5))
    if abs(output-target) > 2 or output < 720:
        raise ValueError('Target outside mild transform bounds')
    return kept, rate, output


def tempo(wave, rate):
    """SoX speech WSOLA: pitch preserved, no explicit pad/repeat/length repair."""
    wave = np.asarray(wave, dtype='<f4')
    if wave.ndim != 1 or not np.isfinite(wave).all() or not .8 <= rate <= 1.2:
        raise ValueError('Invalid tempo input')
    if rate == 1.:
        return wave.copy()
    result = subprocess.run(['sox', '-D', '-t', 'f32', '-r', '16000', '-c', '1',
        '-', '-t', 'f32', '-', 'tempo', '-s', format(rate, '.17g')],
        input=wave.tobytes(), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=True, timeout=30)
    out = np.frombuffer(result.stdout, dtype='<f4').copy()
    expected = int(np.floor(len(wave)/rate+.5))
    if len(out) != expected or not np.isfinite(out).all():
        raise ValueError('SoX output length/finiteness invariant failed')
    return out


@dataclass(frozen=True)
class MildSpec:
    index: int
    variant: int
    augmentation_seed: int


class MildTrainSet(torch.utils.data.Dataset):
    def __init__(self, args, items):
        self.items = list(items)
        path = Path(args['duration_policy_path'])
        raw = path.read_bytes()
        self.policy = json.loads(raw)
        self.policy_text = raw.decode()
        self.policy_sha256 = hashlib.sha256(raw).hexdigest()
        if self.policy['version'] != 'mild_train_uniform_v1' or self.policy['dataset'] != '19LA_train':
            raise ValueError('Only the Train mild policy is allowed')
        self.cache = Path(args['mild_transform_cache'])
        ready = json.loads((self.cache.parent/'cache_complete.json').read_text())
        if ready['policy_sha256'] != self.policy_sha256:
            raise ValueError('Waveform cache belongs to another policy')
        index_raw = (self.cache.parent/'cache_index.json').read_bytes()
        if hashlib.sha256(index_raw).hexdigest() != ready['cache_index_sha256']:
            raise ValueError('Waveform cache index changed')
        self.wave_hashes = json.loads(index_raw)
        records = self.policy['originals']
        by_uid = {row['uid']: row for row in records}
        if len(by_uid) != len(records) or set(by_uid) != {Path(i.path).stem for i in self.items}:
            raise ValueError('Train UID mismatch')
        self.records = [by_uid[Path(i.path).stem] for i in self.items]
        self.lengths = np.array([r['samples'] for r in self.records], dtype=np.int64)
        labels = np.array([i.label for i in self.items])
        self.class_indices, self.class_probability = {}, {}
        self.source_probability = np.zeros(len(self.items))
        self.activity_policy = None
        for item, row in zip(self.items, self.records):
            info = sf.info(item.path)
            if info.frames != row['samples'] or info.channels != 1 or info.samplerate != 16000 or item.label != row['label']:
                raise ValueError('Original header/label changed')
            p = np.array([v['probability'] for v in row['variants']])
            if not np.isfinite(p).all() or min(p) <= 0 or not np.isclose(p.sum(), 1., atol=1e-12):
                raise ValueError('Invalid conditional probabilities')
        for y, expected in [(1, 2580), (0, 22800)]:
            ids = np.flatnonzero(labels == y)
            if len(ids) != expected:
                raise ValueError('Unexpected Train class count')
            self.class_indices[y] = ids
            self.class_probability[y] = np.full(len(ids), 1./len(ids))
            self.source_probability[ids] = 1./len(ids)
        fingerprint = '\n'.join(f'{i.label}|{Path(i.path).stem}|{i.speaker}|{i.type}|{n}' for i,n in zip(self.items,self.lengths))
        self.source_fingerprint = hashlib.sha256(fingerprint.encode()).hexdigest()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, spec):
        return self.read_crop(spec), self.items[spec.index].label, spec.index

    def crop_spec(self, index, seed_parts):
        duration_seed, augmentation_seed = np.random.SeedSequence(seed_parts).generate_state(2)
        row = self.records[index]
        variant = int(np.random.default_rng(duration_seed).choice(len(row['variants']),
            p=[v['probability'] for v in row['variants']]))
        return MildSpec(int(index), variant, int(augmentation_seed))

    def read_crop(self, spec, augment=True):
        variant = self.records[spec.index]['variants'][spec.variant]
        audio = np.load(self.cache/variant['file'], allow_pickle=False)
        if audio.dtype != np.float32 or audio.shape != (variant['output_samples'],):
            raise ValueError('Invalid cached waveform')
        if hashlib.sha256(audio.tobytes()).hexdigest() != self.wave_hashes[variant['file']]:
            raise ValueError('Cached waveform changed')
        audio = audio.astype(np.float64)
        if augment:
            from positive_bpl import rawboost
            py_state, np_state = random.getstate(), np.random.get_state()
            try:
                random.seed(spec.augmentation_seed); np.random.seed(spec.augmentation_seed)
                audio = rawboost(audio, random.randint(0, 8))
            finally:
                random.setstate(py_state); np.random.set_state(np_state)
        if len(audio) != variant['output_samples'] or not np.isfinite(audio).all():
            raise ValueError('RawBoost changed duration/finiteness')
        return torch.as_tensor(audio, dtype=torch.float32)
