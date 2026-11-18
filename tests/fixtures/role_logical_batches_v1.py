"""Frozen iterator oracle from the active 2026-11-12 v1 source; never used for training."""
import torch

def logical_batches(loader_bona, loader_spoof):
    """Same class order, large-loader order and small-loader cycling as original.

    Create the two initial iterators before selecting the large loader, as in
    original240. Failures from a worker are errors, never mistaken for exhaustion.
    """
    if not len(loader_bona) or not len(loader_spoof):
        raise ValueError('Both original class loaders must be nonempty')
    bona, spoof = iter(loader_bona), iter(loader_spoof)
    if len(loader_bona) < len(loader_spoof):
        big, small, small_iter, big_label = loader_spoof, loader_bona, bona, 0
    else:
        big, small, small_iter, big_label = loader_bona, loader_spoof, spoof, 1
    for a in big:
        try:
            b = next(small_iter)
        except StopIteration:
            small_iter = iter(small)
            b = next(small_iter)
        positive, negative = (a, b) if big_label == 1 else (b, a)
        if len(positive) != 3 or len(negative) != 3:
            raise ValueError('Require original (long, short, label) dataset outputs')
        if not bool((positive[2] == 1).all()) or not bool((negative[2] == 0).all()):
            raise ValueError('Original class loaders contain incorrect protocol labels')
        yield tuple(torch.cat((p, n), dim=0) for p, n in zip(positive, negative))
