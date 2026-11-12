from pathlib import Path
import json
import re

import pandas as pd

root = Path(__file__).resolve().parent
text = (root / 'repetition-error-rates.html').read_text(encoding='utf8')
config = json.loads(re.search(r'<script type="application/json" data-viz-chart>(.*?)</script>', text, re.S).group(1))
rates = pd.read_csv(root / 'repeat_by_class.csv')
for i, label in enumerate(['bonafide', 'spoof']):
    for j, repeat in enumerate([True, False]):
        row = rates[(rates['class'] == label) & (rates['repeat'] == repeat)].iloc[0]
        assert abs(config['rows'][i][j + 1] - row['errors'] / row['n']) < 1e-14
assert len(config['rows']) == 2 and len(config['columns']) == 3
assert '\\"' not in text
verification = {'json_payload': 'pass', 'exact_count_ratios': 'pass',
                'rates_checked': 4, 'literal_markup': 'pass',
                'rendering': 'not inspected',
                'standard_checker': 'Unavailable: bundled Node invocation returned Python and exit 1.'}
(root / 'chart_verification.json').write_text(json.dumps(verification, indent=2) + '\n', encoding='utf8')
print(json.dumps(verification))
