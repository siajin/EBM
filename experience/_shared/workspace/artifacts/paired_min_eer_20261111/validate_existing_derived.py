"""Exercise the new analysis path on completed derived scores; no inference/training."""
import json
from pathlib import Path
import sys

import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT/'EBM'))
from paired_evaluation import analyze_phase, sha, save

old = PROJECT/'artifacts/repetition_shortcut_20261111'
destination = HERE/'derived_analysis_integration'
destination.mkdir(exist_ok=True)
metadata = pd.read_csv(old/'reviewed_analysis/codec_metadata.csv')
manifest = {family+'_e50': dict(family=family, epoch=50) for family in ['legacy240', 'B']}
save(destination/'checkpoints.json', manifest)
for rank in [0, 1]:
    frame = pd.read_csv(old/f'new_results/full_rank{rank}.csv', float_precision='round_trip', dtype={'speaker': str})
    frame = frame[frame.phase == '21LA'].merge(metadata[['uid', 'codec']], on='uid', validate='many_to_one')
    frame = frame.rename(columns={'score_legacy240': 'score_legacy240_e50', 'score_B': 'score_B_e50'})
    path = destination/f'full_21LA_rank{rank}.csv'
    frame.to_csv(path, index=False)
    save(destination/f'full_21LA_done_rank{rank}.json', dict(rows=len(frame), sha256=sha(path)))
result = analyze_phase(destination, '21LA')
reference = pd.read_csv(old/'reviewed_analysis/independent_full_eer.csv')
errors = []
for row in result['full_metrics']:
    ref = reference[(reference.model == row['family']) & (reference.phase == '21LA') & (reference.condition == row['condition'])]
    assert len(ref) == 1
    errors.append(abs(row['eer_percent']-float(ref.iloc[0].eer)))
assert max(errors) < 1e-9
save(HERE/'derived_analysis_validation.json', dict(status='passed',
    rows=result['validation']['rows'], scope='completed noise0 legacy240/B50 derived results; not new benchmark inference',
    validated=['full UID/index coverage', 'identity long scores', 'independent ROC', 'IPF with actual metadata',
               'length EER', 'paired error reconciliation', 'JSON/CSV serialization'],
    maximum_error_previous_independent_eer_pp=max(errors), optimizer_updates=0))
print(json.dumps(json.loads((HERE/'derived_analysis_validation.json').read_text())))
