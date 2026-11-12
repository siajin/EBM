"""Metadata-only nonoverlapping segmentation census. No audio/model/server access."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from bidirectional_audio_segments import iter_segments, iter_waveform_segments

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--bidirectional', action='store_true',
                    help='Tile from both ends, retain duplicate crops, and keep <0.5 s sources once')
args = parser.parse_args()
MULTIPLIER = 2 if args.bidirectional else 1

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / ('artifacts/segment_length_histograms_bidirectional_20261109'
              if args.bidirectional else 'artifacts/segment_length_histograms_20261109')
OUT.mkdir(parents=True, exist_ok=True)
CACHE = ROOT / 'artifacts/geometry_cache/runs'
SOURCE_DIRS = [
    '20261105_fast_p0_01/686e593443178bb8',
    '20261105_frozen_all_01/Q/c288fb7a7ab48dd3',
    '20261105_frozen_all_01/19LA_eval/4b8955da49f12316',
    '20261105_21la_full_03/4110118cb92e35c1',
    '20261105_df_fast_01/1b0a7488e72f4ec2',
    '20261105_frozen_all_01/DF_remaining/096bc3b970f33181',
]
prior = json.loads((ROOT / 'artifacts/class_distributions_20261108/validation.json').read_text())
known = {str(Path(x['path'])): x['sha256'] for x in prior['source_files']}
frames, provenance = [], []
for directory in SOURCE_DIRS:
    path = CACHE / directory / 'extract/scores.parquet'
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha == known[str(path.relative_to(ROOT))]
    frames.append(pd.read_parquet(path, columns=['split', 'utterance_id', 'y', 'duration']))
    provenance.append({'path': str(path.relative_to(ROOT)), 'sha256': sha})

datasets = [
    ('19LA_train', '2019 LA · train', frames[0].loc[frames[0].split == 'T'], (2580, 22800)),
    ('19LA_dev', '2019 LA · dev', pd.concat([frames[0].loc[frames[0].split == 'V'], frames[1]]), (2548, 22296)),
    ('19LA_eval', '2019 LA · eval', frames[2], (7355, 63882)),
    ('21LA_eval', '2021 LA · eval', frames[3], (14816, 133360)),
    ('21DF_eval', '2021 DF · eval', pd.concat(frames[4:]), (14869, 519059)),
]
rows, summaries, dataset_info = [], [], []
for key, title, data, expected in datasets:
    assert data.utterance_id.is_unique and not data.isna().any().any()
    assert set(data.y.unique()) == {0, 1}
    lengths = np.rint(data.duration.to_numpy() * 16000).astype('int64')
    assert (lengths > 0).all()
    assert np.max(np.abs(lengths / 16000 - data.duration.to_numpy())) < 1e-9
    targets = np.arange(8000, int(lengths.max()) + 1, 8000, dtype='int64')
    has_short_bin = bool(args.bidirectional and np.any(lengths < 8000))
    bin_lengths = ([None] if has_short_bin else []) + (targets/16000).tolist()
    bin_labels = (['<0.5 (원음)'] if has_short_bin else []) + [f'{t/16000:g}' for t in targets]
    classes = {}
    for y, name, expected_n in [(1, 'bonafide', expected[0]), (0, 'spoof', expected[1])]:
        n = lengths[data.y.to_numpy() == y]
        assert len(n) == expected_n
        one_side = np.array([np.sum(n // target) for target in targets], dtype='int64')
        # Independently, count the source utterances reaching every successive multiple.
        independent = [sum(int(np.count_nonzero(n >= multiple)) for multiple in range(int(t), int(n.max()) + 1, int(t))) for t in targets]
        assert np.array_equal(one_side, independent)
        counts = MULTIPLIER * one_side
        assert np.all(np.diff(counts) <= 0)
        assert np.all((n[:, None] // targets) * targets <= n[:, None])
        short = n[n < 8000]
        short_count = int(len(short)) if args.bidirectional else 0
        if args.bidirectional:
            # Each side covers a contiguous prefix/suffix of q*d samples.
            # They meet or overlap at every eligible target: q*d >= L-q*d.
            for target in targets:
                eligible = n[n >= target]
                covered_per_side = (eligible // target)*target
                assert np.all(covered_per_side >= eligible-covered_per_side)
        under2 = int(counts[targets < 32000].sum()) + short_count
        weighted_samples = int(np.sum(counts*targets)) + (int(short.sum()) if args.bidirectional else 0)
        if has_short_bin:
            counts = np.r_[short_count, counts]
        total = int(counts.sum())
        assert np.isclose((counts / total).sum(), 1)
        if key == '19LA_train':
            audit = json.loads((ROOT / 'artifacts/ebm_C_20261108/metadata_validation.json').read_text())['class_length_counts'][str(y)]
            assert len(n) == audit['N'] and int(n.sum()) == audit['total_original_samples']
        classes[name] = {'original_utterances': len(n), 'original_max_seconds': float(n.max()/16000),
                         'all_segment_count': total, 'counts': counts.tolist(),
                         'fractions': (counts/total).tolist(), 'short_originals_kept': short_count,
                         'forward_only_segment_count':int(one_side.sum()),
                         'uncovered_original_samples':0 if args.bidirectional else int(np.sum(n % 8000))}
        summaries.append({'dataset': key, 'class': name, 'original_utterances': len(n),
                          'all_segment_count': total, 'segments_per_original': total/len(n),
                          'under2_fraction': under2/total, 'short_originals_kept':short_count,
                          'under2_original_fraction': float(np.mean(n < 32000)),
                          'generated_mean_seconds': weighted_samples/16000/total,
                          'uncovered_original_samples':0 if args.bidirectional else int(np.sum(n % 8000))})
        for label, length, count in zip(bin_labels, bin_lengths, counts):
            eligible_count = int(len(short)) if length is None else int(np.count_nonzero(n >= round(length*16000)))
            rows.append({'dataset': key, 'class': name, 'segment_seconds': length, 'length_bin':label,
                         'segment_count': int(count), 'within_class_fraction': float(count/total),
                         'eligible_originals': eligible_count,
                         'original_utterances': len(n), 'segments_per_original': float(count/len(n))})
    b = np.array(classes['bonafide']['fractions'])
    s = np.array(classes['spoof']['fractions'])
    dataset_info.append({'key': key, 'title': title, 'lengths': bin_lengths, 'length_labels':bin_labels,
                         'overlap_percent':float(100*np.minimum(b,s).sum()), 'classes': classes})

# User's example and exact sample boundaries, including <0.5 sec sources.
assert [64000//t for t in [8000,16000,24000,32000]] == [8,4,2,2]
assert [7999//8000, 8000//8000, 15999//8000, 16000//8000] == [0,1,1,2]
if args.bidirectional:
    # Check actual emitted intervals and time order, including the user's 4.2 s case.
    for num_samples, step in [(n, 8) for n in range(1, 161)] + [(67200,8000),(64000,8000),(7999,8000)]:
        crops = list(iter_segments(num_samples, step))
        source = np.arange(num_samples)
        for interval, view in iter_waveform_segments(source, step):
            assert np.array_equal(view, np.arange(interval.start_sample, interval.stop_sample))
        for target in {c.target_samples for c in crops}:
            intervals = sorted((c.start_sample,c.stop_sample) for c in crops if c.target_samples==target)
            last = 0
            for start, stop in intervals:
                assert 0 <= start <= last and stop <= num_samples and stop-start==target
                last = max(last,stop)
            assert last==num_samples
            expected_count = 1 if num_samples < step else 2*(num_samples//target)
            assert len(intervals)==expected_count
    example = [c.__dict__ | {'start_seconds':c.start_sample/16000,'stop_seconds':c.stop_sample/16000}
               for c in iter_segments(67200)]
    pd.DataFrame(example).to_csv(OUT/'example_4p2_seconds.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(rows).to_csv(OUT/'segment_histograms.csv', index=False, encoding='utf-8-sig')
pd.DataFrame(summaries).to_csv(OUT/'summary.csv', index=False, encoding='utf-8-sig')
result = {'method': {'sample_rate':16000, 'target_step_seconds':.5,
                    'direction_multiplier':MULTIPLIER, 'anchors':['front','back'] if args.bidirectional else ['front'],
                    'segments_per_original':('2*floor(original_samples / target_samples); <8000-sample source kept once' if args.bidirectional else 'floor(original_samples / target_samples)'),
                    'overlap':('none within one anchor/target; overlap allowed between anchors' if args.bidirectional else False),
                    'remainder':('covered by the opposite anchor; short sources retained whole' if args.bidirectional else 'discard separately for each target length'),
                    'duplicate_policy':'keep all, including identical front/back intervals', 'back_anchor_reverses_audio':False,
                    'audio_files_created':False,
                    'normalization':'all generated segment counts over all target lengths, separately per dataset and class',
                    'scope':'uncapped original durations; before augmentation; splits kept separate; not sampler weighted',
                    'dependence':'different target lengths reuse the same source; generated segments are not independent new utterances'},
          'sources': provenance, 'datasets': dataset_info, 'summaries':summaries,
          'validation': {'source_sha_match':True, 'source_uid_unique_within_split':True, 'expected_class_counts_match':True,
                         'counts_independently_checked_all_lengths':True, 'sample_precision_checked':True,
                         'bidirectional_coverage_checked_all_source_lengths_and_eligible_targets':bool(args.bidirectional),
                         'actual_interval_and_waveform_checks':163 if args.bidirectional else 0,
                         'uncovered_original_samples':sum(c['uncovered_original_samples'] for d in dataset_info for c in d['classes'].values()),
                         'original_total':sum(len(x[2]) for x in datasets)}}
(OUT/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n', encoding='utf-8')

# Complete fixed chart configurations; <=30 categories per panel, full tail retained.
configs = {}
for d in dataset_info:
    windows = [(0,len(d['lengths']))] if len(d['lengths']) <= 40 else [(0,30),(30,len(d['lengths']))]
    for wi, (lo,hi) in enumerate(windows):
        charts = []
        for metric, label in [('counts','생성 조각 수'), ('fractions','클래스 내부 비율')]:
            part = [[label, d['classes']['bonafide'][metric][i], d['classes']['spoof'][metric][i]] for i,label in enumerate(d['length_labels']) if lo<=i<hi]
            spec = {'type':'bar','x':'길이','y':['Bona fide','Spoof'],'xLabel':'조각 길이 (초)',
                    'yLabel':label, 'sortOrder':'original','startAtZero':True,
                    'valueFormat':{'style':'percent' if metric=='fractions' else 'decimal','maximumFractionDigits':2 if metric=='fractions' else 0}}
            if metric=='fractions':
                spec['yDomain']=[0, max(max(c['fractions']) for c in d['classes'].values())*1.08]
            else:
                spec['yDomain']=[0, max(max(c['counts']) for c in d['classes'].values())*1.08]
            charts.append({'id':'segment-'+d['key']+'-'+metric,'title':label,'height':300,
                           'spec':spec,'columns':['길이','Bona fide','Spoof'],'rows':part,
                           'description': '모든 길이를 합친 클래스별 조각 수가 분모입니다.' if metric=='fractions' else None})
        configs[d['key']+'-'+str(wi)]={'charts':charts}
(OUT/'chart-configs.json').write_text(json.dumps(configs,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(pd.DataFrame(summaries).to_string(index=False))
print('Validated configs:',len(configs))
