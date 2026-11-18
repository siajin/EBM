"""Convert the previously audited randomized TRAIN plan to a UID keyed policy."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd


def main(workspace, destination):
    workspace, destination = Path(workspace), Path(destination)
    source = workspace/'artifacts/duration_matching_20261109'
    plan_path = source/'single_transform_plan.parquet'
    originals = pd.read_parquet(plan_path)
    originals = originals[originals.dataset == '19LA_train']
    cached = workspace/'artifacts/geometry_cache/runs/20261105_fast_p0_01/686e593443178bb8/extract/scores.parquet'
    known = json.loads((workspace/'artifacts/class_distributions_20261108/validation.json').read_text())
    expected_hash = next(x['sha256'] for x in known['source_files'] if Path(x['path']) == cached.relative_to(workspace))
    assert hashlib.sha256(cached.read_bytes()).hexdigest() == expected_hash
    meta = pd.read_parquet(cached, columns=['split','utterance_id','y','duration'])
    meta = meta[meta.split == 'T'].set_index('utterance_id')
    assert set(originals.utterance_id) == set(meta.index) and len(originals) == 25380
    assert np.array_equal(originals.y, meta.loc[originals.utterance_id].y)
    assert np.allclose(originals.original_duration_s, meta.loc[originals.utterance_id].duration, atol=1e-12, rtol=0)
    result, provenance, weighted = [], {}, {}
    for y, n in [(1,2580),(0,22800)]:
        frame = originals[originals.y == y].sort_values(['original_duration_s','utterance_id']).reset_index(drop=True)
        assert len(frame) == n
        path = source/f'19LA_train_{y}_randomized_segments.parquet'
        provenance[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        segments = pd.read_parquet(path)
        durations, weights = [], []
        for index, group in segments.groupby('source_sorted_index', sort=True):
            row = frame.iloc[int(index)]
            samples = int(round(row.original_duration_s*16000))
            variants = []
            # Merge duplicate targets for the same original; retain their mass.
            for target, records in group.groupby('target_duration_s', sort=True):
                rate = float(np.clip(samples/(target*16000), .8,1.2))
                kept = min(samples,max((4*samples+4)//5,int(round(target*16000*rate))))
                rate = float(np.clip(kept/(target*16000),.8,1.2))
                output = int(np.floor(kept/rate+.5))
                assert abs(output-target*16000) <= 2 and kept >= .8*samples
                seed = int.from_bytes(hashlib.sha256(f'{row.utterance_id}|{target:.17g}|1'.encode()).digest()[:8], 'little')
                start = int(np.random.default_rng(seed).integers(samples-kept+1))
                probability = float(records.conditional_probability.sum())
                variants.append({'file':f'{row.utterance_id}_{len(variants):02d}.npy',
                    'target_samples':float(target*16000),'kept_samples':kept,'start':start,
                    'rate':rate,'output_samples':output,'probability':probability})
                durations.append(output/16000); weights.append(probability/n)
            assert np.isclose(sum(v['probability'] for v in variants),1.,atol=1e-12,rtol=0)
            result.append({'uid':row.utterance_id,'label':y,'samples':samples,'variants':variants})
        weighted[y] = (np.array(durations),np.array(weights))
    hist = {}
    for width in [.05,.1,.25]:
        edges = np.arange(0,31+width/2,width)
        h = [np.histogram(weighted[y][0],bins=edges,weights=weighted[y][1])[0] for y in [1,0]]
        hist[str(width)] = float(100*np.minimum(h[0],h[1]).sum())
        assert hist[str(width)] > 99.9
    policy = {'version':'mild_train_uniform_v1','dataset':'19LA_train','sample_rate':16000,
        'speed_bounds':[.8,1.2],'maximum_crop_fraction':.2,'transform':'SoX tempo -s (speech WSOLA)',
        'crop_position':'fixed seeded uniform position per UID/target variant; target random each visit',
        'source_sampling':'uniform within class; every original once per epoch plus uniform residual',
        'source_metadata_sha256':expected_hash,'randomized_plan_sha256':provenance,
        'nominal_rounded_histogram_overlap_percent':hist,'originals':result}
    destination.parent.mkdir(parents=True,exist_ok=True)
    assert not destination.exists()
    destination.write_text(json.dumps(policy,separators=(',',':'))+'\n',encoding='utf-8')
    count = sum(len(r['variants']) for r in result)
    print(json.dumps({'originals':len(result),'variants':count,'overlap':hist,
        'waveform_bytes':sum(v['output_samples']*4+128 for r in result for v in r['variants']),
        'policy_sha256':hashlib.sha256(destination.read_bytes()).hexdigest()},indent=2))


if __name__ == '__main__':
    main(*sys.argv[1:])
