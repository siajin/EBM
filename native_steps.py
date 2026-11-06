"""Memory-bounded native-length EBM steps; original local all-pairs BPL stays intact."""
from contextlib import nullcontext
import math
import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence


def unwrap(module):
    return module.module if hasattr(module, 'module') else module


def features(plm, waves, device):
    lengths = torch.tensor([len(w) for w in waves], device=device)
    x = pad_sequence(waves, batch_first=True).to(device=device, dtype=torch.float32)
    sample_mask = torch.arange(x.shape[1], device=device)[None, :] < lengths[:, None]
    base = unwrap(plm)
    with torch.no_grad():
        states = base(x, attention_mask=sample_mask.long(), output_hidden_states=True).hidden_states
        stacked = torch.stack(states, dim=1)
    mask = base._get_feature_vector_attention_mask(stacked.shape[2], sample_mask)
    return stacked, mask


def _freeze(model, centroid_only):
    for parameter in model.parameters():
        parameter.requires_grad_(parameter is model.loss.weight if centroid_only else parameter is not model.loss.weight)


def _objective(model, embeddings, batch, args):
    labels = batch['labels'].to(embeddings['long'].device)
    bona = int(labels.sum())
    if bona == 0 or not torch.all(labels[:bona] == 1) or not torch.all(labels[bona:] == 0):
        raise ValueError('A logical batch must contain bonafide first, then spoof')
    n = len(labels)
    rho = args.get('short_oc_weight', 0.25)
    weight = args.get('padding_loss_weight', 0.1)
    oc = {name: model.loss(z, labels, logical_batch_size=n)[0] for name, z in embeddings.items()}
    bpl = model.loss_bpl(embeddings['long'][:bona], embeddings['short'][:bona])
    total = (1-rho)*oc['long'] + bpl if rho else oc['long'] + bpl
    if rho:
        total = total + rho/2 * (oc['short'] + oc['wrapped'])
    pair = total.new_zeros(())
    active_count = total.new_zeros(())
    if weight:
        active = batch['pair_active'].to(labels.device) & (labels == 1)
        cosine = torch.nn.functional.cosine_similarity(embeddings['short'], embeddings['wrapped'], dim=1).clamp(-1, 1)
        local_sum = ((torch.nn.functional.softplus(30*(1-cosine))-math.log(2)) * active).sum()
        active_count = active.sum().to(total.dtype)
        pair_sum = local_sum.detach().clone()
        world = 1
        if dist.is_initialized():
            dist.all_reduce(active_count)
            dist.all_reduce(pair_sum)
            world = dist.get_world_size()
        pair = world * local_sum / active_count.clamp_min(1)
        total = total + weight*pair
        reported_pair = pair_sum / active_count.clamp_min(1)
    else:
        reported_pair = pair
    stats = {f'oc_{name}': float(v.detach()) for name, v in oc.items()}
    stats.update(bpl=float(bpl.detach()), padding=float(reported_pair), active_pairs=int(active_count), loss1=float(total.detach()))
    return total, bona, stats


def train_batch(trainer, batch):
    """Two Adam updates per logical batch, with exact embedding-gradient replay.

    Stochastic backend RNG and physical padding shapes are identical between
    graph-free collection and replay; no cross-microbatch BPL pairs are dropped.
    """
    classifier = trainer.classifier
    model = unwrap(classifier)
    args = trainer.args
    device = args['device']
    micro = args.get('micro_batch_size', 8)
    if micro < 1:
        raise ValueError('micro_batch_size must be positive')
    names = ['long', 'short']
    if args.get('short_oc_weight', 0.25) or args.get('padding_loss_weight', 0.1):
        names.append('wrapped')
    _freeze(model, centroid_only=False)
    trainer.optimizer.zero_grad(set_to_none=True)
    records, collected = [], {}
    n = len(batch['labels'])
    for name in names:
        chunks = []
        for start in range(0, n, micro):
            stop = min(start+micro, n)
            x, mask = features(trainer.plm, batch[name][start:stop], device)
            cpu_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state(device)
            with torch.no_grad():
                chunks.append(model(x, mask=mask, return_embedding=True).detach())
            records.append((name, start, stop, cpu_rng, cuda_rng))
            del x, mask
        collected[name] = torch.cat(chunks).requires_grad_(True)
    loss, bona, stats = _objective(model, collected, batch, args)
    derivatives = torch.autograd.grad(loss, tuple(collected.values()))
    gradients = dict(zip(collected, derivatives))
    del loss, collected
    for number, (name, start, stop, cpu_rng, cuda_rng) in enumerate(records):
        sync = classifier.no_sync() if hasattr(classifier, 'no_sync') and number+1 < len(records) else nullcontext()
        with sync, torch.random.fork_rng(devices=[torch.device(device).index]):
            x, mask = features(trainer.plm, batch[name][start:stop], device)
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(cuda_rng, device)
            z = classifier(x, mask=mask, return_embedding=True)
            z.backward(gradients[name][start:stop])
        del x, mask, z
    trainer.optimizer.step()
    del gradients

    # Recompute with updated theta. The centroid must not see stale embeddings.
    _freeze(model, centroid_only=True)
    trainer.optimizer.zero_grad(set_to_none=True)
    loss2 = 0.
    for start in range(0, bona, micro):
        stop = min(start+micro, bona)
        sync = classifier.no_sync() if hasattr(classifier, 'no_sync') and stop < bona else nullcontext()
        with sync:
            x, mask = features(trainer.plm, batch['long'][start:stop], device)
            labels = batch['labels'][start:stop].to(device)
            value, _ = classifier(x, mask=mask, label=labels, logical_batch_size=bona)
            scaled = value * ((stop-start)/bona)
            scaled.backward()
            loss2 += float(scaled.detach())
        del x, mask, value, scaled
    trainer.optimizer.step()
    stats['loss2'] = loss2
    return stats
