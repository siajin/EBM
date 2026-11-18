"""Independent tie-aware score audit for the completed fixed checkpoint evaluation."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def independent_eer(labels, scores):
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    assert np.isfinite(scores).all() and set(np.unique(labels)) == {0, 1}
    order = np.argsort(-scores, kind='stable')
    labels, scores = labels[order], scores[order]
    boundaries = np.r_[0, np.flatnonzero(scores[1:] != scores[:-1]) + 1]
    positives = np.add.reduceat(labels, boundaries)
    groups = np.diff(np.r_[boundaries, len(labels)])
    tp = np.r_[0, np.cumsum(positives)]
    fp = np.r_[0, np.cumsum(groups - positives)]
    fpr = fp / int((labels == 0).sum())
    fnr = 1 - tp / int((labels == 1).sum())
    difference = fpr - fnr
    k = int(np.flatnonzero(difference >= 0)[0])
    if k == 0:
        return float(100 * fpr[0])
    fraction = -difference[k - 1] / (difference[k] - difference[k - 1])
    return float(100 * (fpr[k - 1] + fraction * (fpr[k] - fpr[k - 1])))


def main(root):
    config = json.loads((root / 'evaluation_config.json').read_text())
    complete = json.loads((root / 'complete.json').read_text())
    assert complete['status'] == 'complete' and complete['epoch'] == 25
    summary = {'epoch': 25, 'checkpoint_sha256': config['checkpoint_sha256'],
               'optimizer_updates': 0, 'phases': {}}
    for phase, expected in config['phases'].items():
        path = root / 'evaluation' / f'scores_{phase}.parquet'
        report = json.loads((root / f'metrics_{phase}.json').read_text())
        scores = pd.read_parquet(path)
        assert len(scores) == expected
        assert scores['index'].tolist() == list(range(expected))
        assert scores.uid.is_unique and not scores.isna().any().any()
        assert np.isfinite(scores.score.to_numpy()).all()
        assert np.array_equal(scores.used_samples, np.minimum(scores.original_samples, 64600))
        counts = [int((scores.label == label).sum()) for label in (1, 0)]
        assert counts == config['class_counts'][phase]
        value = independent_eer(scores.label, scores.score)
        assert abs(value - report['EER_percent']) < 1e-10, (phase, value, report['EER_percent'])
        summary['phases'][phase] = {'n': expected, 'class_counts': counts,
                                    'EER_percent': value, 'reported_EER_percent': report['EER_percent'],
                                    'scores_sha256': sha(path), 'duplicate_uids': 0,
                                    'missing_indices': 0, 'nonfinite_scores': 0}
    target = root / 'independent_audit.json'
    assert not target.exists()
    target.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
