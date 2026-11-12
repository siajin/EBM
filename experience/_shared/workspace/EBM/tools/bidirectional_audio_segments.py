"""Lossless temporal coverage using front- and back-anchored crop views.

Each side tiles without overlap at each 0.5 s multiple. The two sides can
overlap, and identical crops are intentionally retained with distinct anchors.
Audio stays in its original time order. Short sources are retained once.
This module does not read audio, alter datasets, or start training.
"""
from dataclasses import dataclass
from operator import index


@dataclass(frozen=True)
class Segment:
    start_sample: int
    stop_sample: int
    target_samples: int
    anchor: str
    ordinal: int


def iter_segments(num_samples, step_samples=8000):
    """Yield half-open source intervals; default step is 0.5 s at 16 kHz."""
    num_samples, step_samples = index(num_samples), index(step_samples)
    if num_samples <= 0 or step_samples <= 0:
        raise ValueError('Source and step lengths must be positive integers')
    if num_samples < step_samples:
        yield Segment(0, num_samples, num_samples, 'original', 0)
        return
    for target in range(step_samples, num_samples + 1, step_samples):
        for ordinal in range(num_samples // target):
            yield Segment(ordinal * target, (ordinal + 1) * target,
                          target, 'front', ordinal)
        for ordinal in range(num_samples // target):
            yield Segment(num_samples - (ordinal + 1) * target,
                          num_samples - ordinal * target,
                          target, 'back', ordinal)


def iter_waveform_segments(waveform, step_samples=8000):
    """Yield (interval, crop) pairs for a mono, sample-indexable waveform.

No time reversal, padding, resampling, deduplication, or mutation is applied.
The caller keeps the original label and split for every crop.
"""
    for segment in iter_segments(len(waveform), step_samples):
        yield segment, waveform[segment.start_sample:segment.stop_sample]
