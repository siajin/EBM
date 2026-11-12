"""Metadata-only feasibility audit; no audio/embeddings or model fitting."""
import hashlib
import json
from pathlib import Path
import math
import numpy as np
import pandas as pd
from geometry_cache import DEFAULT_ROOT, resolve_run

OUT = DEFAULT_ROOT / 'local_analysis/variable_length_audit_20261106_v1'


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def read(alias):
    folder, entry, split = resolve_run(alias)
    frame = pd.read_parquet(folder/'extract/scores.parquet', columns=['split', 'utterance_id', 'y', 'duration'])
    frame = frame[frame.split == split].reset_index(drop=True)
    assert not frame.utterance_id.duplicated().any() and frame.duration.gt(0).all()
    frame['samples'] = (frame.duration * 16000).round().astype('int64')
    return frame, entry['cache_fingerprint']


def frames(samples):
    n = np.asarray(samples)
    for kernel, stride in zip([10,3,3,3,3,2,2], [5,2,2,2,2,2,2]):
        n = (n-kernel)//stride+1
    return n


def metrics(data, batches):
    lengths = data.samples.to_numpy()
    true_frames = padded_frames = padded_attention = true_attention = 0
    ratios = []
    seen = []
    for ids in batches:
        t = frames(lengths[ids]); assert np.min(t) >= 2
        true_frames += int(t.sum()); padded_frames += int(t.max()*len(t))
        true_attention += int((t*t).sum()); padded_attention += int(t.max()**2*len(t))
        ratios.append(int(t.max())/float(t.mean())); seen.extend(ids)
    counts = np.bincount(np.asarray(seen), minlength=len(data))
    return dict(local_batches=len(batches), rows_seen=len(seen), unique_rows=int((counts>0).sum()),
        omitted_rows=int((counts==0).sum()), duplicated_occurrences=int(np.maximum(counts-1,0).sum()),
        useful_frame_fraction=true_frames/padded_frames,
        useful_attention_pair_fraction=true_attention/padded_attention,
        max_batch_max_to_mean_frame_ratio=max(ratios),
        allocated_frames=padded_frames, attention_pairs=padded_attention)


def schedule(data, mode, seed=20261106):
    """Global16 bona+144 spoof, then rank-local8+72. All original rows covered.

    This is a packing simulation/reference, not a production loader.
    Sortish uses randomly formed pools of32 steps; step order is randomized.
    """
    rng = np.random.default_rng(seed); nsteps = max(math.ceil((data.y==1).sum()/16), math.ceil((data.y==0).sum()/144))
    groups = {}
    for y, quota in [(1,16),(0,144)]:
        ids = np.flatnonzero(data.y.to_numpy()==y); ids = rng.permutation(ids)
        required = nsteps*quota
        if required > len(ids): ids = np.concatenate([ids, np.resize(rng.permutation(ids), required-len(ids))])
        if mode == 'sorted':
            ids = ids[np.argsort(data.samples.to_numpy()[ids], kind='stable')]
        elif mode == 'sortish32':
            ids = np.concatenate([q[np.argsort(data.samples.to_numpy()[q],kind='stable')]
                                  for q in [ids[i:i+32*quota] for i in range(0,len(ids),32*quota)]])
        groups[y] = ids.reshape(nsteps,quota)
    global_steps = []
    for step in rng.permutation(nsteps):
        global_steps.append([np.concatenate([groups[1][step,rank*8:(rank+1)*8],groups[0][step,rank*72:(rank+1)*72]]) for rank in [0,1]])
    assert all(all(data.iloc[q].y.tolist()==[1]*8+[0]*72 for q in step) for step in global_steps)
    return global_steps


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    sets = {}; identities = {}
    for name in ['T','V','19LA_eval','21LA_eval','DF_pilot','DF_remaining']:
        sets[name],identities[name] = read(name)
    sets['DF_full'] = pd.concat([sets['DF_pilot'],sets['DF_remaining']],ignore_index=True)
    assert len(sets['DF_full'])==533928 and not sets['DF_full'].utterance_id.duplicated().any()
    stats = {}
    for name,d in sets.items():
        stats[name] = dict(N=len(d), distinct_lengths=int(d.samples.nunique()),
            quantiles_seconds={str(k):float(v) for k,v in d.duration.quantile([0,.01,.05,.5,.9,.95,.99,1]).items()},
            below_4_0375=int(d.duration.lt(4.0375).sum()),
            class_stats={str(y):dict(N=len(q),below_4_0375=int(q.duration.lt(4.0375).sum())) for y,q in d.groupby('y')})
    d=sets['T']; rng=np.random.default_rng(20261106)
    schemes={}
    for mode in ['random','sortish32','sorted']:
        steps=schedule(d,mode); schemes[mode]=metrics(d,[q for step in steps for q in step])
        schemes[mode]['global_steps']=len(steps)
    f=frames(d.samples.to_numpy()); counts=d.groupby('samples').size()
    exact=dict(singleton_utterances=int(counts[counts==1].sum()),
               utterances_in_exact_length_groups_ge80=int(counts[counts>=80].sum()),
               max_same_length_group=int(counts.max()),
               groups_with_both_labels=int(d.groupby('samples').y.nunique().eq(2).sum()))
    # A fixed4.0375s batch has201 frontend frames. Compare the same scheduled
    # occurrences, without interpreting FLOP proxies as measured GPU speed.
    steps=schedule(d,'sortish32'); occurrence=np.concatenate([q for step in steps for q in step]); original=frames(d.samples.to_numpy()[occurrence])
    proxy=dict(full_length_unpadded_frame_ratio_to_fixed201=float(original.mean()/201),
               full_length_unpadded_attention_ratio_to_fixed201=float(np.mean(original**2)/(201**2)),
               sortish_padded_frame_ratio_to_fixed201=schemes['sortish32']['allocated_frames']/(len(occurrence)*201),
               sortish_padded_attention_ratio_to_fixed201=schemes['sortish32']['attention_pairs']/(len(occurrence)*201**2),
               train_max_frames=int(f.max()),fixed_frames=201,
               max_train_attention_ratio=float((f.max()/201)**2),
               warning='Frame/attention-operation proxies only; no wall-clock or capacity claim')
    bins=(np.floor(d.duration*2)/2).rename('bin_lower_seconds')
    ctab=pd.crosstab(bins,d.y).rename(columns={0:'spoof',1:'bonafide'}).reset_index()
    ctab.to_csv(OUT/'train_half_second_class_counts.csv',index=False)
    save('metadata_audit.json',dict(status='pass',source_fingerprints=identities,datasets=stats,
        exact_length_groups=exact,packing_simulation=schemes,cost_proxies=proxy,
        label= '1=bonafide, 0=spoof',coverage='T/V are existing diagnostic cache splits, not newly audited complete official train/dev inventory'))
    # Score-blind bounded GPU probe selections: class-specific min/median/max
    # train clips and the shortest/longest full-DF clips.
    selected=[]
    for y in [1,0]:
        q=d[d.y==y].sort_values(['samples','utterance_id']);
        for pos,tag in [(0,'minimum'),(len(q)//2,'median'),(len(q)-1,'maximum')]:
            row=q.iloc[pos].to_dict();row.update(cache_alias='T',selection=f'train_y{y}_{tag}');selected.append(row)
    for alias in ['DF_pilot','DF_remaining']:
        q=sets[alias].sort_values(['samples','utterance_id'])
        for pos,tag in [(0,'minimum'),(len(q)-1,'maximum')]:
            row=q.iloc[pos].to_dict();row.update(cache_alias=alias,selection=f'{alias}_{tag}');selected.append(row)
    pd.DataFrame(selected).to_parquet(OUT/'probe_selection.parquet',index=False)
    save('probe_selection_lock.json',dict(N=len(selected),selection='Per-class train min/median/max and DF-part min/max duration; ties by UID; no score or embedding selection',
        sha256=hashlib.sha256((OUT/'probe_selection.parquet').read_bytes()).hexdigest(),
        tests=['actual waveform/header duration','single variable T classifier equivalence','padding mask on every26 temporal pooling','padded batch vs original single','backpropagation without optimizer update','bounded original maximum-duration forward'],
        production_source_changes=False,training_launched=False,frontend_outputs_saved=False))
    print(json.dumps(dict(status='pass',datasets=stats,exact=exact,packing=schemes,proxies=proxy),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
