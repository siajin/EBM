"""Locked T-only duration-centroid experiment; no torch, SSH, optimizer or audio.

See docs/experiments/ebm_duration_conditioned_centroids_plan_20261106.md.
Large embeddings and random-control scores are read in bounded chunks.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
from pathlib import Path
from datetime import datetime, timezone
import argparse
import gc
import hashlib
import json
import platform
import shutil
import sys
import time

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.metrics import roc_auc_score, roc_curve
from threadpoolctl import threadpool_limits

from geometry_cache import DEFAULT_ROOT, resolve_run, validate_run
from fast_analyze import eer
from audit_cached_core_results import independent_eer, independent_auc

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / 'docs/experiments/ebm_duration_conditioned_centroids_plan_20261106.md'
CHECKPOINT = '104873703a57cad98b686d770e0860e031a2cf9abef79b72668780c1660e0b24'
EDGES = np.array([2., 3., 64600 / 16000])
METHODS = ['N', 'G', 'D', 'N-OFF', 'G-OFF', 'A', 'S']
BASELINES = ['N', 'G', 'N-OFF', 'G-OFF']
R_COUNT, DRAWS, STABILITY_DRAWS = 199, 10000, 500
SEED, STABILITY_SEED, RANDOM_SEED = 2026110601, 2026110602, 202611060100
EXPECTED = {'T': (2580, 22800), 'V': (1351, 13632), 'DF_pilot': (2000, 18000),
            'DF_full': (14869, 519059), '19LA_eval': (7355, 63882),
            '21LA_eval': (14816, 133360), 'Q': (1197, 8664)}
OUT = None
START = time.monotonic()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def save(name, obj):
    def convert(v):
        if isinstance(v, np.ndarray): return v.tolist()
        if isinstance(v, np.generic): return v.item()
        if isinstance(v, Path): return str(v)
        raise TypeError(type(v).__name__)
    p = OUT / name
    temp = p.with_suffix(p.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=convert,
                               allow_nan=False) + '\n', encoding='utf8')
    temp.replace(p)


def progress(stage, **kw):
    record = dict(stage=stage, seconds=round(time.monotonic() - START, 2), **kw)
    save('progress.json', record)
    print(json.dumps(record, ensure_ascii=True), flush=True)


def unit(x):
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    assert np.isfinite(x).all() and np.all(norms > 1e-12)
    return x / norms


def bins(d):
    d = np.asarray(d)
    assert np.isfinite(d).all() and np.all(d > 0)
    return np.searchsorted(EDGES, d, side='right')


def canonical_speaker(rows):
    # Existing IDs are qualified by split; strip qualifiers for overlap audits.
    return rows.speaker_group.astype(str).str.split('::').str[-1]


class Panel:
    def __init__(self, name, aliases=None):
        self.name = name
        self.maps, self.entries, frames = [], [], []
        aliases = aliases or [name]
        centers = []
        for alias in aliases:
            folder, entry, split = resolve_run(alias)
            assert entry['identity']['checkpoint_sha256'] == CHECKPOINT
            frame = pd.read_parquet(folder / 'extract/scores.parquet')
            frame = frame.loc[frame.split == split].copy()
            map_indices = {}
            for rank in [0, 1]:
                map_indices[rank] = len(self.maps)
                self.maps.append(np.load(folder / f'extract/rank{rank}_z.npy', mmap_mode='r'))
                centers.append(np.load(folder / f'extract/rank{rank}_centroid.npy'))
            frame['_map'] = frame['rank'].map(map_indices)
            frames.append(frame)
            self.entries.append(dict(alias=alias, fingerprint=entry['cache_fingerprint'], folder=str(folder)))
        self.rows = pd.concat(frames, ignore_index=True).sort_values('utterance_id', kind='stable').reset_index(drop=True)
        assert not self.rows.utterance_id.duplicated().any()
        self.c0 = centers[0]
        assert self.c0.shape == (160,) and abs(np.linalg.norm(self.c0) - 1) < 1e-10
        assert all(np.max(np.abs(c - self.c0)) < 1e-12 for c in centers)
        f = self.rows
        assert (int((f.y == 1).sum()), int((f.y == 0).sum())) == EXPECTED[name]
        assert np.isin(f.y, [0, 1]).all()
        assert (f.crop_samples == 64600).all() and (f.crop_start == 0).all()
        self.b = bins(f.duration.to_numpy())
        padding_error = np.max(np.abs(f.padding_fraction - np.maximum(0., 1 - f.duration / EDGES[-1])))
        assert padding_error < 2 / 64600
        self.validation = dict(name=name, n=len(f), n_bona=int(f.y.sum()), n_spoof=int((f.y == 0).sum()),
                               padding_max_error=float(padding_error), entries=self.entries)

    def read(self, ix):
        part = self.rows.iloc[ix]
        values = np.empty((len(part), 160), dtype=np.float64)
        map_codes = part['_map'].to_numpy()
        for mi in np.unique(map_codes):
            mask = map_codes == mi
            values[mask] = self.maps[int(mi)][part.embedding_row.to_numpy()[mask]]
        return unit(values)

    def close(self):
        self.maps.clear()


def fit(ub, bb, c0):
    # Only bona fide unit embeddings and durations reach this function.
    n = np.bincount(bb, minlength=4)
    assert n.tolist() == [109, 877, 1018, 576] and np.all(n >= 30)
    mu = ub.mean(0)
    mk = np.stack([ub[bb == k].mean(0) for k in range(4)])
    assert np.linalg.norm(mu) > 1e-8 and (np.linalg.norm(mk, axis=1) > 1e-8).all()
    g, gk = unit(mu), unit(mk)
    difference = mk - mu
    delta = (n / (n + 100))[:, None] * (difference - (difference @ c0)[:, None] * c0)
    anchor = unit(c0 + delta)
    shrink = unit((n[:, None] * mk + 100 * mu) / (n[:, None] + 100))
    ns, gs = ub @ c0, ub @ g
    model = dict(c0=c0, mu=mu, mu_k=mk, n_k=n, g=g, g_k=gk, A_k=anchor, S_k=shrink,
                 off_N=np.array([ns[bb == k].mean() for k in range(4)]) - ns.mean(),
                 off_G=np.array([gs[bb == k].mean() for k in range(4)]) - gs.mean())
    random_centers = []
    for r in range(R_COUNT):
        assigned = np.random.default_rng(np.random.SeedSequence([RANDOM_SEED + r, 0])).permutation(bb)
        assert np.array_equal(np.bincount(assigned, minlength=4), n)
        random_centers.append(unit(np.stack([ub[assigned == k].mean(0) for k in range(4)])))
    model['R_k'] = np.stack(random_centers)
    return model


def score_unit(u, b, model):
    native, global_s = u @ model['c0'], u @ model['g']
    return {'N': native, 'G': global_s,
            'D': np.einsum('nd,nd->n', u, model['g_k'][b]),
            'N-OFF': native - model['off_N'][b],
            'G-OFF': global_s - model['off_G'][b],
            'A': np.einsum('nd,nd->n', u, model['A_k'][b]),
            'S': np.einsum('nd,nd->n', u, model['S_k'][b])}


def far_threshold(y, s, target=.01):
    spoof = np.sort(s[y == 0])
    allowed = int(np.floor(target * len(spoof)))
    t = float(np.nextafter(spoof[len(spoof) - allowed - 1], np.inf))
    assert np.mean(spoof >= t) <= target
    return t


def metrics(y, s, t=None):
    out = dict(n=len(y), n_bona=int((y == 1).sum()), n_spoof=int((y == 0).sum()))
    both = out['n_bona'] > 0 and out['n_spoof'] > 0
    out.update(eer_percent=eer(y, s) if both else None,
               auroc=float(roc_auc_score(y, s)) if both else None)
    if t is not None:
        eb, es = int(np.sum((y == 1) & (s < t))), int(np.sum((y == 0) & (s >= t)))
        out.update(threshold=float(t), false_rejected_bona=eb, false_accepted_spoof=es,
                   frr_percent=100 * eb / out['n_bona'] if out['n_bona'] else None,
                   far_percent=100 * es / out['n_spoof'] if out['n_spoof'] else None)
    return out


def math_checks(model, u, b, y):
    checks = []
    edge_values = np.ravel(np.array([[np.nextafter(e, -np.inf), e, np.nextafter(e, np.inf)] for e in EDGES]))
    assert bins(edge_values).tolist() == [0, 1, 1, 1, 2, 2, 2, 3, 3]
    checks.append('exact duration boundary routing')
    equal = dict(model, g_k=np.tile(model['g'], (4, 1)))
    assert np.max(np.abs(score_unit(u, b, equal)['D'] - score_unit(u, b, equal)['G'])) < 1e-14
    difference = np.zeros((4, 160))
    assert np.max(np.abs(unit(model['c0'] + difference) - model['c0'])) < 1e-14
    checks.append('equal centers and zero anchored residual reduce to single/native center')
    assert set(score_unit(u, b, model)) == set(METHODS)
    # Scorer accepts no labels; reverse input order and restore it.
    s = score_unit(u, b, model)
    reverse = score_unit(u[::-1], b[::-1], model)
    assert max(np.max(np.abs(s[k] - reverse[k][::-1])) for k in METHODS) < 1e-12
    checks.append('label-free scoring and row-order invariance')
    for k in range(4):
        mask = b == k
        if len(np.unique(y[mask])) == 2:
            for before, after in [('N', 'N-OFF'), ('G', 'G-OFF')]:
                assert abs(roc_auc_score(y[mask], s[before][mask]) - roc_auc_score(y[mask], s[after][mask])) < 1e-12
                assert abs(eer(y[mask], s[before][mask]) - eer(y[mask], s[after][mask])) < 1e-10
    checks.append('scalar offset preserves within-bin AUROC/EER')
    ty = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    ts = np.array([-.5, -.5, 0., .2, .2, .2, .9, .9])
    assert abs(eer(ty, ts) - independent_eer(ty, ts)) < 1e-10
    roc = GroupedROC(ty, ts, np.arange(len(ty)) % 3, 3)
    assert abs(roc.eers(np.ones((1, 3)))[0] - eer(ty, ts)) < 1e-10
    rng = np.random.default_rng(19)
    for weights in rng.multinomial(3, np.ones(3) / 3, size=15):
        ix = np.repeat(np.arange(len(ty)), weights[np.arange(len(ty)) % 3])
        if len(np.unique(ty[ix])) == 2:
            assert abs(roc.eers(weights[None, :])[0] - eer(ty[ix], ts[ix])) < 1e-9
    checks.append('tied-score and weighted EER independently reproduced')
    t = far_threshold(ty, ts, .25)
    assert np.mean(ts[ty == 0] >= t) <= .25
    checks.append('spoof FAR threshold handles ties conservatively')
    return checks


class GroupedROC:
    """Exact tied ROC crossing by threshold search over fixed score knots.

    O(groups * draws * log(knots)) vector searches, not a draws-by-N matrix.
    """
    def __init__(self, y, score, codes, group_count):
        self.knots = np.r_[-np.inf, np.unique(score), np.inf]
        self.arrays = [[], []]
        self.counts = np.zeros((2, group_count), dtype=np.float64)
        for label in [0, 1]:
            for g in range(group_count):
                values = np.sort(score[(y == label) & (codes == g)])
                if len(values):
                    self.arrays[label].append((g, values))
                    self.counts[label, g] = len(values)

    def below(self, label, thresholds, weights):
        result = np.zeros(len(thresholds))
        for g, values in self.arrays[label]:
            result += np.searchsorted(values, thresholds, side='left') * weights[:, g]
        return result

    def eers(self, weights):
        ns, nb = weights @ self.counts[0], weights @ self.counts[1]
        good = (ns > 0) & (nb > 0)
        ns, nb = np.maximum(ns, 1), np.maximum(nb, 1)
        lo, hi = np.zeros(len(weights), dtype=int), np.full(len(weights), len(self.knots) - 1, dtype=int)
        for _ in range(int(np.ceil(np.log2(len(self.knots)))) + 1):
            mid = (lo + hi) // 2
            thresholds = self.knots[mid]
            diff = 1 - self.below(0, thresholds, weights) / ns - self.below(1, thresholds, weights) / nb
            move = diff >= 0
            lo, hi = np.where(move, mid, lo), np.where(move, hi, mid)
        assert np.all(hi - lo <= 1)
        farlo, frlo = 1 - self.below(0, self.knots[lo], weights) / ns, self.below(1, self.knots[lo], weights) / nb
        farhi, frhi = 1 - self.below(0, self.knots[hi], weights) / ns, self.below(1, self.knots[hi], weights) / nb
        dl, dh = farlo - frlo, farhi - frhi
        w = np.divide(dl, dl - dh, out=np.zeros_like(dl), where=(dl != dh))
        answer = 100 * (farlo + w * (farhi - farlo))
        answer[~good] = np.nan
        return answer


def make_draws(f, seed):
    groups, codes = np.unique(f.dependency_group.astype(str), return_inverse=True)
    sources = f.source_corpus.astype(str).to_numpy() if 'source_corpus' in f else np.full(len(f), 'asvspoof')
    group_source = []
    for g in range(len(groups)):
        values = np.unique(sources[codes == g])
        assert len(values) == 1
        group_source.append(values[0])
    group_source = np.array(group_source)
    rng = np.random.default_rng(seed)
    weights = np.zeros((DRAWS, len(groups)), dtype=np.float64)
    for source in sorted(set(group_source)):
        ix = np.flatnonzero(group_source == source)
        weights[:, ix] = rng.multinomial(len(ix), np.full(len(ix), 1 / len(ix)), size=DRAWS)
    return groups, codes, weights


def weighted_auc(y, score, codes, weights):
    ng = weights.shape[1]
    bona = [score[(y == 1) & (codes == g)] for g in range(ng)]
    spoof = [np.sort(score[(y == 0) & (codes == g)]) for g in range(ng)]
    concordance = np.zeros((ng, ng))
    for i, a in enumerate(bona):
        if not len(a): continue
        for j, b in enumerate(spoof):
            if len(b):
                concordance[i, j] = .5 * (np.searchsorted(b, a, side='left').sum() + np.searchsorted(b, a, side='right').sum())
    nb, ns = weights @ np.array([len(a) for a in bona]), weights @ np.array([len(a) for a in spoof])
    denominator = nb * ns
    value = np.einsum('ij,ij->i', weights @ concordance, weights)
    return np.divide(value, denominator, out=np.full(len(weights), np.nan), where=denominator > 0)


def score_panel(panel, model, stream_id, thresholds=None):
    f, n = panel.rows, len(panel.rows)
    scores = {m: np.empty(n) for m in METHODS}
    random_path = OUT / f'random_scores_{panel.name}.npy'
    random_scores = np.lib.format.open_memmap(random_path, mode='w+', dtype='float64', shape=(R_COUNT, n))
    # Separate stream IDs ensure independent T/V/evaluation permutations.
    routing = np.stack([np.random.default_rng(np.random.SeedSequence([RANDOM_SEED + r, stream_id])).permutation(panel.b).astype(np.uint8)
                        for r in range(R_COUNT)])
    random_matrix = model['R_k'].reshape(-1, 160).T
    max_error = 0.
    for start in range(0, n, 2048):
        stop = min(start + 2048, n)
        u = panel.read(np.arange(start, stop))
        part = score_unit(u, panel.b[start:stop], model)
        for m in METHODS: scores[m][start:stop] = part[m]
        max_error = max(max_error, float(np.max(np.abs(part['N'] - f.cosine_s.to_numpy()[start:stop]))))
        all_random = (u @ random_matrix).reshape(stop - start, R_COUNT, 4)
        selected = np.take_along_axis(all_random, routing[:, start:stop].T[:, :, None], axis=2)[:, :, 0]
        random_scores[:, start:stop] = selected.T
    assert max_error < 1e-10
    random_scores.flush()
    del routing
    panel.validation['native_max_error'] = max_error
    output = f.drop(columns=['_map']).copy()
    output['duration_bin'] = panel.b
    for m in METHODS: output[m] = scores[m]
    output.to_parquet(OUT / f'scores_{panel.name}.parquet', index=False)
    records = []
    for r in range(R_COUNT):
        rs = np.array(random_scores[r])
        records.append(dict(dataset=panel.name, repeat=r, seed=RANDOM_SEED + r, eer_percent=eer(f.y.to_numpy(), rs)))
    del random_scores
    save(f'panel_{panel.name}_validation.json', panel.validation)
    return output, records


def summarize(f, name, thresholds):
    y, rows, transitions = f.y.to_numpy(), [], []
    cuts = [('all', 'all', np.ones(len(f), dtype=bool))]
    cuts.extend(('duration', f'B{k}', f.duration_bin.to_numpy() == k) for k in range(4))
    if 'source_corpus' in f:
        for source in sorted(f.source_corpus.unique()):
            source_mask = f.source_corpus.to_numpy() == source
            cuts.append(('source', source, source_mask))
            cuts.extend(('source_duration', f'{source}::B{k}', source_mask & (f.duration_bin.to_numpy() == k)) for k in range(4))
    if 'codec' in f:
        cuts.extend(('codec', str(codec), f.codec.to_numpy() == codec) for codec in sorted(f.codec.dropna().unique()))
    cuts.extend([('tail', 'under1p5', f.duration.to_numpy() < 1.5),
                 ('tail', '1p5to2', (f.duration.to_numpy() >= 1.5) & (f.duration.to_numpy() < 2))])
    for dimension, value, mask in cuts:
        if not mask.any(): continue
        gb = f.loc[mask & (y == 1), 'dependency_group'].nunique()
        gs = f.loc[mask & (y == 0), 'dependency_group'].nunique()
        small = min(int(np.sum(mask & (y == 1))), int(np.sum(mask & (y == 0)))) < 50 or min(gb, gs) < 5
        for m in METHODS:
            rows.append(dict(dataset=name, dimension=dimension, value=value, method=m,
                             bona_groups=gb, spoof_groups=gs, small_sample=small,
                             **metrics(y[mask], f[m].to_numpy()[mask], thresholds[m])))
    base_error = np.where(y == 1, f.N.to_numpy() < thresholds['N'], f.N.to_numpy() >= thresholds['N'])
    for m in METHODS:
        candidate_error = np.where(y == 1, f[m].to_numpy() < thresholds[m], f[m].to_numpy() >= thresholds[m])
        for label in [0, 1]:
            for k in ['all', 0, 1, 2, 3]:
                chosen = (y == label) & (True if k == 'all' else f.duration_bin.to_numpy() == k)
                transitions.append(dict(dataset=name, method=m, y=label, duration_bin=k, n=int(chosen.sum()),
                                        recovered=int(np.sum(chosen & base_error & ~candidate_error)),
                                        newly_wrong=int(np.sum(chosen & ~base_error & candidate_error)),
                                        both_wrong=int(np.sum(chosen & base_error & candidate_error)),
                                        both_correct=int(np.sum(chosen & ~base_error & ~candidate_error))))
    return rows, transitions


def stability(ub, bb, speaker, model):
    unique = np.unique(speaker)
    rng = np.random.default_rng(STABILITY_SEED)
    results = []
    for r in range(STABILITY_DRAWS):
        selected = rng.choice(unique, size=len(unique), replace=True)
        ix = np.concatenate([np.flatnonzero(speaker == s) for s in selected])
        mu = ub[ix].mean(0)
        g = unit(mu)
        for k in range(4):
            ki = ix[bb[ix] == k]
            if not len(ki):
                results.append(dict(repeat=r, duration_bin=k, n=0, angle_to_original_deg=None, angle_to_global_deg=None, resultant_norm=None))
                continue
            mk = ub[ki].mean(0)
            center = unit(mk)
            results.append(dict(repeat=r, duration_bin=k, n=len(ki), resultant_norm=float(np.linalg.norm(mk)),
                                angle_to_original_deg=float(np.degrees(np.arccos(np.clip(center @ model['g_k'][k], -1, 1)))),
                                angle_to_global_deg=float(np.degrees(np.arccos(np.clip(center @ g, -1, 1))))))
    summaries = []
    for k in range(4):
        items = [r for r in results if r['duration_bin'] == k and r['n'] > 0]
        summaries.append(dict(duration_bin=k, valid_draws=len(items), missing_draws=STABILITY_DRAWS - len(items),
                              angle_to_original_quantiles=np.quantile([r['angle_to_original_deg'] for r in items], [.025, .5, .975]).tolist(),
                              angle_to_global_quantiles=np.quantile([r['angle_to_global_deg'] for r in items], [.025, .5, .975]).tolist()))
    save('centroid_stability.json', dict(seed=STABILITY_SEED, draws=STABILITY_DRAWS, summaries=summaries, records=results,
                                         performance_fit_uncertainty_included=False))


def bootstrap_dataset(f, name, seed):
    groups, codes, weights = make_draws(f, seed)
    y = f.y.to_numpy()
    values, absolute = {}, []
    for m in METHODS:
        progress('paired_bootstrap', dataset=name, method=m, draws=DRAWS, groups=len(groups))
        s = f[m].to_numpy()
        roc = GroupedROC(y, s, codes, len(groups))
        all_one = roc.eers(np.ones((1, len(groups))))[0]
        assert abs(all_one - eer(y, s)) < 1e-8
        assert abs(independent_eer(y, s) - eer(y, s)) < 1e-8
        assert abs(independent_auc(y, s) - roc_auc_score(y, s)) < 1e-10
        values[m] = roc.eers(weights)
        valid = np.isfinite(values[m])
        absolute.append(dict(method=m, eer_percent=eer(y, s), valid_draws=int(valid.sum()),
                             CI95=np.quantile(values[m][valid], [.025, .975]).tolist()))
        del roc
    paired = []
    for m in METHODS:
        delta = values[m] - values['D']
        valid = np.isfinite(delta)
        paired.append(dict(baseline=m, candidate='D', improvement_pp=eer(y, f[m].to_numpy()) - eer(y, f.D.to_numpy()),
                           valid_fraction=float(valid.mean()), CI95=np.quantile(delta[valid], [.025, .975]).tolist(),
                           CI98p75=np.quantile(delta[valid], [.00625, .99375]).tolist(), primary=m in BASELINES))
    np.savez_compressed(OUT / f'bootstrap_draws_{name}.npz', group_names=groups, group_weights=weights, **values)
    # Same fixed group draws quantify short-bin directional ranking, exploratorily.
    auc_records = []
    if name == 'DF_full':
        short = f.duration_bin.to_numpy() == 0
        selections = [('B0_all', short)]
        for source in sorted(f.source_corpus.unique()):
            selections.append((f'B0_{source}', short & (f.source_corpus.to_numpy() == source)))
        for cell, mask in selections:
            if len(np.unique(y[mask])) < 2: continue
            a = weighted_auc(y[mask], f.D.to_numpy()[mask], codes[mask], weights)
            b = weighted_auc(y[mask], f.G.to_numpy()[mask], codes[mask], weights)
            # Validate the pair-count matrix against the unweighted AUROC.
            ones = np.ones((1, len(groups)))
            assert abs(weighted_auc(y[mask], f.D.to_numpy()[mask], codes[mask], ones)[0] - roc_auc_score(y[mask], f.D.to_numpy()[mask])) < 1e-10
            good = np.isfinite(a - b)
            auc_records.append(dict(cell=cell, delta_AUROC_D_minus_G=float(roc_auc_score(y[mask], f.D.to_numpy()[mask]) - roc_auc_score(y[mask], f.G.to_numpy()[mask])),
                                    CI95=np.quantile((a - b)[good], [.025, .975]).tolist(), valid_fraction=float(good.mean()), exploratory=True))
    return dict(dataset=name, seed=seed, draws=DRAWS, groups=len(groups), source_stratified=True,
                absolute=absolute, paired=paired, short_bin_AUROC=auc_records,
                fit_and_threshold_uncertainty_included=False, source_family_dependence_resolved=False)


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=DEFAULT_ROOT / 'local_analysis/duration_centroids_20261106_v1')
    args = parser.parse_args()
    OUT = args.out.resolve()
    OUT.mkdir(parents=True, exist_ok=False)
    assert shutil.disk_usage(OUT).free > 3 * 1024**3
    lock = dict(plan_sha256=digest(PLAN), runner_sha256=digest(__file__), created_utc=datetime.now(timezone.utc).isoformat(),
                checkpoint_sha256=CHECKPOINT, duration_edges=EDGES, methods=METHODS, primary='D', primary_comparisons=BASELINES,
                source_fit='T bona fide only', validation_role='V FAR threshold only; no hyperparameter selection',
                random_controls=R_COUNT, random_seed_start=RANDOM_SEED, draws=DRAWS, seed=SEED,
                stability_draws=STABILITY_DRAWS, stability_seed=STABILITY_SEED, shrink_constant=100, anchor_coefficient=1,
                threads=2, chunk=2048, no_optimizer=True, no_audio_forward=True, no_EBM_parameter_update=True,
                no_DF_CV=True, independent_confirmation=False, git_revision=None,
                versions=dict(python=sys.version, platform=platform.platform(), numpy=np.__version__,
                              pandas=pd.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__),
                clarification='Bounded mmap IO via resolve_run avoids loading unused h; random scores stored on disk, not in RAM.')
    save('protocol_lock.json', lock)
    (OUT / 'runner_at_lock.py').write_bytes(Path(__file__).read_bytes())
    (OUT / 'plan_at_lock.md').write_bytes(PLAN.read_bytes())
    try:
        validations, fingerprints = [], {}
        for alias in ['T', 'V', 'DF_pilot', 'DF_remaining', '19LA_eval', '21LA_eval', 'Q']:
            folder, entry, _ = resolve_run(alias)
            if entry['cache_fingerprint'] not in fingerprints:
                progress('validate_cache', dataset=alias)
                validations.append(validate_run(alias, verify_hashes=True))
                fingerprints[entry['cache_fingerprint']] = {'path': str(folder), 'files': entry['files']}
        save('input_validation.json', dict(status='pass', checks=validations, input_catalog_sha256=digest(DEFAULT_ROOT / 'catalog.json')))
        t = Panel('T')
        bi = np.flatnonzero(t.rows.y.to_numpy() == 1)
        ub, bb = t.read(bi), t.b[bi]
        model = fit(ub, bb, t.c0)
        model_hash_path = OUT / 'centroids.npz'
        np.savez_compressed(model_hash_path, **model)
        fit_rows = []
        for k in range(4):
            mask = bb == k
            fit_rows.append(dict(duration_bin=k, n=int(mask.sum()), speakers=int(canonical_speaker(t.rows.iloc[bi[mask]]).nunique()),
                                 resultant_norm=float(np.linalg.norm(model['mu_k'][k])),
                                 angle_to_global_deg=float(np.degrees(np.arccos(np.clip(model['g_k'][k] @ model['g'], -1, 1)))),
                                 angle_to_native_deg=float(np.degrees(np.arccos(np.clip(model['g_k'][k] @ model['c0'], -1, 1))))))
        save('fit_summary.json', dict(fit_alias='T', fit_labels='bona only', rows=fit_rows, model_sha256=digest(model_hash_path)))
        checks = math_checks(model, t.read(np.arange(len(t.rows))), t.b, t.rows.y.to_numpy())
        stability(ub, bb, canonical_speaker(t.rows.iloc[bi]).to_numpy(), model)
        train_speakers, train_ids = set(canonical_speaker(t.rows)), set(t.rows.utterance_id)
        del ub
        t.close()
        del t
        progress('score_validation')
        v = Panel('V')
        assert np.bincount(v.b[v.rows.y.to_numpy() == 1], minlength=4).tolist() == [46, 397, 506, 402]
        vf, random_records = score_panel(v, model, 1)
        thresholds = {m: far_threshold(vf.y.to_numpy(), vf[m].to_numpy()) for m in METHODS}
        threshold_records = {m: metrics(vf.y.to_numpy(), vf[m].to_numpy(), thresholds[m]) for m in METHODS}
        save('V_thresholds.json', dict(target_spoof_far=.01, selection='lowest admissible nextafter-spoof-score threshold', methods=threshold_records))
        assert all(r['far_percent'] <= 1 + 1e-12 for r in threshold_records.values())
        save('scoring_lock.json', dict(model_sha256=digest(model_hash_path), thresholds_sha256=digest(OUT / 'V_thresholds.json'), primary='D',
                                       evaluation_scores_opened=False, no_method_selection=True))
        val_speakers, val_ids = set(canonical_speaker(vf)), set(vf.utterance_id)
        assert not train_speakers & val_speakers and not train_ids & val_ids
        all_metrics, transitions = summarize(vf, 'V', thresholds)
        overlap = []
        seen_ids = {'T': train_ids, 'V': val_ids}
        validation_records = [v.validation]
        v.close()
        del v, vf
        panels = [('DF_pilot', None, 2), ('DF_full', ['DF_pilot', 'DF_remaining'], 3),
                  ('19LA_eval', None, 4), ('21LA_eval', None, 5), ('Q', None, 6)]
        for name, aliases, stream_id in panels:
            progress('score_evaluation', dataset=name)
            panel = Panel(name, aliases)
            assert np.max(np.abs(panel.c0 - model['c0'])) < 1e-12
            ids, speakers = set(panel.rows.utterance_id), set(canonical_speaker(panel.rows))
            assert not ids & train_ids and not ids & val_ids
            # Known cross-corpus speaker overlap is recorded, not hidden.
            for reference, reference_ids in seen_ids.items():
                overlap.append(dict(dataset=name, reference=reference, overlapping_utterance_ids=len(ids & reference_ids)))
            overlap.append(dict(dataset=name, reference='T_speakers', overlapping_speakers=len(speakers & train_speakers)))
            overlap.append(dict(dataset=name, reference='V_speakers', overlapping_speakers=len(speakers & val_speakers)))
            seen_ids[name] = ids
            frame, rr = score_panel(panel, model, stream_id)
            random_records.extend(rr)
            mm, tt = summarize(frame, name, thresholds)
            all_metrics.extend(mm)
            transitions.extend(tt)
            validation_records.append(panel.validation)
            pd.DataFrame(all_metrics).to_csv(OUT / 'metrics_all.csv', index=False)
            pd.DataFrame(random_records).to_csv(OUT / 'random_controls.csv', index=False)
            panel.close()
            del panel, frame
            gc.collect()
        metrics_frame = pd.DataFrame(all_metrics)
        metrics_frame.loc[metrics_frame.dimension == 'all'].to_csv(OUT / 'metrics_overall.csv', index=False)
        metrics_frame.loc[metrics_frame.dimension.isin(['source', 'duration', 'source_duration', 'tail', 'codec'])].to_csv(OUT / 'metrics_source_duration.csv', index=False)
        pd.DataFrame(transitions).to_csv(OUT / 'error_transitions.csv', index=False)
        save('dataset_overlap.json', overlap)
        save('panel_validations.json', validation_records)
        intervals = []
        for i, name in enumerate(['DF_full', '19LA_eval', '21LA_eval']):
            f = pd.read_parquet(OUT / f'scores_{name}.parquet')
            intervals.append(bootstrap_dataset(f, name, SEED + i))
            save('paired_intervals.json', intervals)
            del f
            gc.collect()
        progress('verify_input_immutability')
        final_hash_count = 0
        for record in fingerprints.values():
            for item in record['files']:
                assert digest(Path(record['path']) / item['path']) == item['sha256']
                final_hash_count += 1
        assert digest(PLAN) == lock['plan_sha256'] and digest(__file__) == lock['runner_sha256']
        df_record = intervals[0]
        compare = {r['baseline']: r for r in df_record['paired']}
        valid = all(compare[b]['valid_fraction'] >= .99 for b in BASELINES)
        h1 = valid and compare['N']['CI98p75'][0] > 0
        h2 = h1 and all(compare[b]['CI98p75'][0] > 0 for b in BASELINES)
        random_df = pd.DataFrame(random_records)
        random_summary = []
        overall = metrics_frame.loc[metrics_frame.dimension == 'all']
        for name, part in random_df.groupby('dataset'):
            actual = float(overall.loc[(overall.dataset == name) & (overall.method == 'D'), 'eer_percent'].iloc[0])
            random_summary.append(dict(dataset=name, count=len(part), EER_quantiles=np.quantile(part.eer_percent, [.025, .5, .975]).tolist(),
                                       D_eer_percent=actual, random_EER_at_most_D_fraction=float((part.eer_percent <= actual).mean()), permutation_p_value=False))
        save('random_summary.json', random_summary)
        save('conclusions.json', dict(status='complete_exploratory', H1_supported=bool(h1), H2_supported=bool(h2),
                                      primary_comparisons=[compare[b] for b in BASELINES], random_summary=random_summary,
                                      limits=['T centers and V thresholds fixed in performance intervals', 'source families unresolved',
                                              'DF previously observed', 'No EBM retraining or causal padding intervention']))
        save('verification.json', dict(status='pass', checks=checks, panel_checks=validation_records,
                                       input_files_rehashed_unchanged=final_hash_count, protocol_unchanged=True,
                                       no_audio_forward=True, no_optimizer=True, no_DF_CV=True, all_specified_methods_reported=True))
        (OUT / 'complete').write_text('0\n', encoding='ascii')
        progress('complete', output=str(OUT), H1_supported=bool(h1), H2_supported=bool(h2))
    except BaseException as exc:
        save('failure.json', dict(type=type(exc).__name__, message=str(exc), stage=json.loads((OUT / 'progress.json').read_text()).get('stage') if (OUT / 'progress.json').exists() else 'initialization'))
        raise


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
