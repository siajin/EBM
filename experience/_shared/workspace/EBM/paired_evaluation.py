"""Post-hoc paired input evaluation. Never imported by the active trainer."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

CAP = 64600
PHASES = ['19LA', '21LA', '21DF']
EXPECTED = {'19LA': 71237, '21LA': 148176, '21DF': 533928}
EPOCHS = list(range(5, 51, 5))
COMMON_EPOCHS = list(range(15, 51, 5))
BINS = [0, 16000, 24000, 32000, 40000, 48000, 56000, CAP, np.inf]
META = ['phase', 'index', 'uid', 'speaker', 'label', 'original_samples', 'attack', 'codec']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def views(wave):
    wave = np.asarray(wave, dtype=np.float32)
    assert wave.ndim == 1 and len(wave) >= 720 and np.isfinite(wave).all()
    native = wave[:CAP].copy()
    return native, native if len(native) == CAP else np.resize(native, CAP).copy()


def roc(y, s, weights=None):
    """Tie blocks, higher score=bonafide; independent of sklearn."""
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=np.float64)
    assert len(y) == len(s) == len(w) and set(y) == {0, 1}
    assert np.isfinite(s).all() and np.isfinite(w).all() and (w > 0).all()
    order = np.argsort(-s, kind='stable')
    y, s, w = y[order], s[order], w[order]
    end = np.r_[np.flatnonzero(s[1:] != s[:-1]), len(s)-1]
    tp = np.r_[0., np.cumsum(w*y)[end] / np.sum(w*y)]
    fp = np.r_[0., np.cumsum(w*(1-y))[end] / np.sum(w*(1-y))]
    return fp, tp, np.r_[np.inf, s[end]]


def metric(y, s, weights=None):
    fp, tp, thresholds = roc(y, s, weights)
    difference = fp + tp - 1
    hi = int(np.flatnonzero(difference >= 0)[0]); lo = max(0, hi-1)
    fraction = -difference[lo] / (difference[hi]-difference[lo]) if hi != lo else 0.
    eer = 100 * (fp[lo] + fraction*(fp[hi]-fp[lo]))
    # Finite attainable threshold; within-target descriptive analysis only.
    candidates = np.flatnonzero(np.isfinite(thresholds))
    j = min(candidates, key=lambda i: (abs(difference[i]), fp[i], -thresholds[i]))
    return {'eer_percent': float(eer), 'threshold': float(thresholds[j]),
            'n': len(y), 'bonafide': int(np.sum(y)), 'spoof': int(len(y)-np.sum(y))}


def checkpoint_minima(rows):
    frame = pd.DataFrame(rows)
    result = []
    for (phase, family, condition), group in frame.groupby(['phase', 'family', 'condition']):
        for name, eligible in [('all_available', group), ('common_15_to_50', group[group.epoch.isin(COMMON_EPOCHS)])]:
            assert len(eligible)
            chosen = eligible.sort_values(['eer_percent', 'epoch']).iloc[0].to_dict()
            chosen.update(candidate_scope=name, candidate_count=len(eligible),
                          candidate_epochs=sorted(eligible.epoch.astype(int).tolist()))
            result.append(chosen)
    return result


def common_scores(heads, x, mask, seed, noise_std):
    """Every checkpoint consumes identical pooling RNG for this input batch."""
    import torch
    result = {}; after = None
    device = x.device
    devices = [device.index] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        for key, head in heads.items():
            for module in head.modules():
                if hasattr(module, 'noise_std'):
                    module.noise_std = noise_std
            torch.random.default_generator.manual_seed(seed)
            if device.type == 'cuda': torch.cuda.manual_seed(seed)
            z = head(x, mask=mask, return_embedding=True)
            score = (torch.nn.functional.normalize(z, dim=1) @
                     torch.nn.functional.normalize(head.loss.weight, dim=1).T).flatten()
            assert torch.isfinite(score).all(), key
            current = torch.cuda.get_rng_state(device) if devices else torch.get_rng_state()
            if after is not None: assert torch.equal(current, after), 'Checkpoint RNG consumption differs'
            after = current
            result[key] = score.detach().cpu().numpy().copy()
    return result


from paired_queue_guard import dependency_ready


def length_group(samples):
    return pd.cut(samples, BINS, labels=False, right=True)


def paired_diagnostics(frame, key, family, epoch, phase):
    col = 'score_'+key
    a = frame[frame.condition == 'native'].set_index('uid').sort_index()
    b = frame[frame.condition == 'wrap4'].set_index('uid').sort_index()
    assert a.index.equals(b.index)
    t_native = metric(a.label, a[col])['threshold']
    t_wrap = metric(b.label, b[col])['threshold']
    out = []
    for bin_id in ['all', 'le2', 'gt2'] + sorted(a.length_bin.unique().tolist()):
        aa = (a if bin_id == 'all' else a[a.original_samples <= 32000] if bin_id == 'le2'
              else a[a.original_samples > 32000] if bin_id == 'gt2' else a[a.length_bin == bin_id])
        bb = b.loc[aa.index]
        for policy, tn, tw in [('fixed_native_target_descriptive', t_native, t_native),
                               ('condition_matched_target_descriptive', t_native, t_wrap)]:
            for label in [0, 1]:
                g = aa[aa.label == label]; h = bb.loc[g.index]
                if not len(g): continue
                en = g[col].to_numpy() < tn if label else g[col].to_numpy() >= tn
                ew = h[col].to_numpy() < tw if label else h[col].to_numpy() >= tw
                out.append(dict(phase=phase, family=family, epoch=epoch, length_bin=bin_id,
                    label=label, n=len(g), policy=policy, native_errors=int(en.sum()), wrap_errors=int(ew.sum()),
                    corrected=int((en & ~ew).sum()), newly_wrong=int((~en & ew).sum()),
                    mean_delta=float((h[col]-g[col]).mean()), native_threshold=tn, wrap_threshold=tw))
    return out


def analyze_phase(root, phase):
    """One phase at a time; full UID coverage, score hashes and independent ROC."""
    from sklearn.metrics import roc_curve
    root = Path(root)
    manifest = json.loads((root/'checkpoints.json').read_text())
    parts = []
    for rank in [0, 1]:
        p = root/f'full_{phase}_rank{rank}.csv'
        receipt = json.loads((root/f'full_{phase}_done_rank{rank}.json').read_text())
        assert sha(p) == receipt['sha256']
        f = pd.read_csv(p, float_precision='round_trip', dtype={'speaker': str})
        assert len(f) == receipt['rows']; parts.append(f)
    f = pd.concat(parts, ignore_index=True)
    assert len(f) == EXPECTED[phase]*2 and not f.duplicated(['uid', 'condition']).any()
    assert not f.isna().any().any()
    columns = ['score_'+k for k in manifest]
    assert np.isfinite(f[columns]).all().all()
    for condition, g in f.groupby('condition'):
        assert sorted(g['index'].tolist()) == list(range(EXPECTED[phase]))
    a = f[f.condition == 'native'].set_index('uid').sort_index()
    b = f[f.condition == 'wrap4'].set_index('uid').sort_index()
    assert a.index.equals(b.index)
    for name in META:
        if name != 'uid': assert a[name].equals(b[name]), name
    assert a.used_samples.eq(a.original_samples.clip(upper=CAP)).all()
    assert b.used_samples.eq(CAP).all()
    noop = a.original_samples >= CAP
    assert a.loc[noop, 'wave_sha256'].equals(b.loc[noop, 'wave_sha256'])
    assert np.array_equal(a.loc[noop, columns], b.loc[noop, columns])
    f['length_bin'] = length_group(f.original_samples)
    rows = []; length_rows = []; max_error = 0.
    for key, ck in manifest.items():
        for condition, g in f.groupby('condition'):
            m = metric(g.label, g['score_'+key])
            fp, tp, _ = roc_curve(g.label, g['score_'+key], drop_intermediate=False)
            d = fp+tp-1; hi = int(np.flatnonzero(d >= 0)[0]); lo = max(0, hi-1)
            z = -d[lo]/(d[hi]-d[lo]) if hi != lo else 0.
            other = float(100*(fp[lo]+z*(fp[hi]-fp[lo])))
            max_error = max(max_error, abs(other-m['eer_percent']))
            assert abs(other-m['eer_percent']) < 1e-8
            rows.append(dict(phase=phase, key=key, family=ck['family'], epoch=ck['epoch'], condition=condition, **m))
            groups = list(g.groupby('length_bin')) + [('le2', g[g.original_samples <= 32000]),
                                                       ('gt2', g[g.original_samples > 32000])]
            for bin_id, h in groups:
                base = dict(phase=phase, family=ck['family'], epoch=ck['epoch'], condition=condition,
                            length_bin=bin_id if isinstance(bin_id, str) else int(bin_id), n=len(h),
                            bonafide=int(h.label.sum()), spoof=int((h.label == 0).sum()))
                if h.label.nunique() == 2: base['eer_percent'] = metric(h.label, h['score_'+key])['eer_percent']
                else: base.update(eer_percent=None, reason='only_one_class')
                length_rows.append(base)
    minima = checkpoint_minima(rows)
    diagnostic_keys = {r['key'] for r in minima} | {k for k, v in manifest.items() if v['epoch'] == 50}
    transitions = []
    for key in sorted(diagnostic_keys):
        ck = manifest[key]
        transitions += paired_diagnostics(f, key, ck['family'], ck['epoch'], phase)
    attack_rows = []; weighted = []
    # Preserve the same original-sample duration bins, with unsupported bins explicit.
    if phase == '21LA':
        from tools.review_repetition_results import ipf_weights
        counts = pd.crosstab(a.original_samples.pipe(length_group), a.label).reindex(columns=[0, 1], fill_value=0)
        support = counts.index[(counts > 0).all(axis=1)]
        target = counts.loc[support, 1]/counts.loc[support, 1].sum()
        for condition, g in f.groupby('condition'):
            common = g[g.length_bin.isin(support)].copy().rename(columns={'length_bin': 'bin'})
            weights, residual = ipf_weights(common, target)
            for key in sorted(diagnostic_keys):
                ck = manifest[key]; col = 'score_'+key
                for name, w in [('common_support_observed', None), ('duration_matched_attack_codec_preserved', weights)]:
                    weighted.append(dict(family=ck['family'], epoch=ck['epoch'], condition=condition, policy=name,
                        n=len(common), excluded=len(g)-len(common), ipf_residual=residual,
                        eer_percent=metric(common.label, common[col], w)['eer_percent']))
                t = metric(g.label, g[col])['threshold']
                for (attack, codec), h in g[g.label == 0].groupby(['attack', 'codec']):
                    attack_rows.append(dict(family=ck['family'], epoch=ck['epoch'], condition=condition,
                        attack=attack, codec=codec, n=len(h), misses=int((h[col] >= t).sum()), threshold=t,
                        threshold_source='same_target_descriptive_only'))
    result = dict(status='complete', phase=phase, full_metrics=rows, minima=minima,
                  length_eer=length_rows, transitions=transitions, attack_codec_errors=attack_rows,
                  duration_weighting=weighted, validation=dict(rows=len(f), unique_sources=EXPECTED[phase],
                  long_noop_sources=int(noop.sum()), independent_roc_max_error_pp=max_error,
                  threshold_scope='target-derived descriptive only; no deployment claims; no dev inference'))
    save(root/f'analysis_{phase}.json', result)
    for name in ['full_metrics', 'minima', 'length_eer', 'transitions', 'attack_codec_errors', 'duration_weighting']:
        pd.DataFrame(result[name]).to_csv(root/f'{name}_{phase}.csv', index=False)
    return result
