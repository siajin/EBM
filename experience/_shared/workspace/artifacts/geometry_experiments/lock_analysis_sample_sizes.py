"""Lock score-independent metadata-only views for the cached experiment plan.

Does not load h/z, predictions or scores, contact the server, fit a model, or
change the raw cache catalog. Full-cache row positions bind each view to its
source fingerprint. Existing locks are verified, never overwritten.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from geometry_cache import DEFAULT_ROOT, resolve_run


OUT = DEFAULT_ROOT / 'analysis_manifests/frozen_data_sizes_v1'
PREFIXES = {
    '21LA_pilot': '21LA-pilot-20261105-v1|',
    '21LA_small': '21LA-small-20261105-v1|',
    'DF_small': 'DF-small-20261105-v1|',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select(frame, prefix, counts):
    assert not frame.utterance_id.duplicated().any()
    ranked = frame.assign(selection_priority=frame.utterance_id.map(
        lambda uid: hashlib.sha256((prefix + uid).encode()).hexdigest()))
    parts = []
    for label, n in counts.items():
        pool = ranked.loc[ranked.y == label].sort_values(['selection_priority', 'utterance_id'], kind='stable')
        assert len(pool) >= n
        parts.append(pool.head(n))
    return pd.concat(parts).sort_values('utterance_id', kind='stable').reset_index(drop=True)


def source(alias):
    folder, entry, split = resolve_run(alias)
    path = folder / 'extract/scores.parquet'
    wanted = ['dataset', 'split', 'utterance_id', 'y', 'speaker_group', 'dependency_group',
              'attack_id', 'codec', 'source_corpus', 'transmission', 'trim']
    columns = [name for name in wanted if name in pq.ParquetFile(path).schema.names]
    rows = pd.read_parquet(path, columns=columns)
    rows['cache_row'] = np.arange(len(rows), dtype=np.int64)
    rows = rows.loc[rows.split == split].copy()
    assert not {'raw_score', 'cosine_s', 'raw_norm', 'tangent_norm'} & set(rows.columns)
    return rows, entry


def describe(frame, full):
    result = {'rows': len(frame), 'class_counts': {str(k): int(v) for k, v in frame.groupby('y').size().items()},
              'unique_UIDs': int(frame.utterance_id.nunique()), 'speaker_groups': int(frame.speaker_group.nunique()),
              'speaker_groups_per_class': {str(k): int(v) for k, v in frame.groupby('y').speaker_group.nunique().items()}}
    for name in ['attack_id', 'codec', 'transmission']:
        if name in frame:
            result[name + '_values'] = sorted(frame[name].dropna().unique().tolist())
    cells = [name for name in ['y', 'attack_id', 'codec', 'transmission'] if name in frame]
    reference = set(map(tuple, full[cells].drop_duplicates().to_numpy()))
    chosen = set(map(tuple, frame[cells].drop_duplicates().to_numpy()))
    result['observed_condition_cells'] = {'columns': cells, 'source_cells': len(reference),
                                          'selected_cells': len(chosen), 'missing_cells': len(reference - chosen)}
    return result


def main():
    la, la_entry = source('21LA_eval')
    df, df_entry = source('DF_pilot')
    pilot = select(la, PREFIXES['21LA_pilot'], {1: 2000, 0: 18000})
    small = select(pilot, PREFIXES['21LA_small'], {1: 200, 0: 1800})
    df_small = select(df, PREFIXES['DF_small'], {1: 200, 0: 1800})
    assert set(small.utterance_id) <= set(pilot.utterance_id)
    assert pilot.utterance_id.tolist() == select(la.sample(frac=1, random_state=17), PREFIXES['21LA_pilot'],
                                               {1: 2000, 0: 18000}).utterance_id.tolist()
    frames = {'21LA_pilot': pilot, '21LA_small': small, 'DF_small': df_small}
    entries = {'21LA_pilot': la_entry, '21LA_small': la_entry, 'DF_small': df_entry}
    sources = {'21LA_pilot': la, '21LA_small': la, 'DF_small': df}
    if OUT.exists():
        locked = json.loads((OUT / 'lock.json').read_text(encoding='utf-8'))
        for name, frame in frames.items():
            item = locked['views'][name]
            assert item['source_cache_fingerprint'] == entries[name]['cache_fingerprint']
            path = OUT / item['manifest_file']
            assert digest(path) == item['manifest_sha256']
            pd.testing.assert_frame_equal(pd.read_parquet(path), frame)
        print(json.dumps({'status': 'verified_existing_lock', 'views': list(frames), 'overwritten': False}))
        return
    OUT.mkdir(parents=True, exist_ok=False)
    report = {'version': 'frozen_data_sizes_v1', 'UTC': datetime.now(timezone.utc).isoformat(),
              'scope': 'metadata-only sample lock; no new performance experiment',
              'uses_scores_or_embeddings': False, 'score_independent_selection': True,
              'script_sha256': digest(Path(__file__)), 'views': {}}
    for name, frame in frames.items():
        path = OUT / (name + '.parquet')
        frame.to_parquet(path, index=False)
        entry = entries[name]
        row = {'source_alias': 'DF_pilot' if name == 'DF_small' else '21LA_eval',
               'parent_view': '21LA_pilot' if name == '21LA_small' else None,
               'selection_hash_prefix': PREFIXES[name], 'manifest_file': path.name,
               'manifest_sha256': digest(path), 'source_cache_fingerprint': entry['cache_fingerprint'],
               'checkpoint_sha256': entry['identity']['checkpoint_sha256'],
               **describe(frame, sources[name])}
        original = sources[name].set_index('cache_row')
        assert frame.utterance_id.tolist() == original.loc[frame.cache_row, 'utterance_id'].tolist()
        row['class_inclusion_probabilities_from_source'] = {
            str(k): float((frame.y == k).sum() / (sources[name].y == k).sum()) for k in [0, 1]}
        report['views'][name] = row
    (OUT / 'lock.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'locked', 'views': report['views'], 'out': str(OUT)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
