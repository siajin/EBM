from .augmentation.wav_read_functions import *

__all__ = [name for name in globals() if not name.startswith('_')] + [
    'WaveformAugmetation', 'CodecAugmentationLight']


def __getattr__(name):
    # Loading RawBoost-only datasets must not eagerly import unused GPU audio
    # augmentation libraries. Explicit imports retain the original classes.
    if name == 'WaveformAugmetation':
        from .augmentation.augmentation import WaveformAugmetation
        return WaveformAugmetation
    if name == 'CodecAugmentationLight':
        from .augmentation.codec_aug import CodecAugmentationLight
        return CodecAugmentationLight
    raise AttributeError(name)
