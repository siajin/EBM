"""Documentation-only corrections and receipt capture; no experiment entrypoint."""
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
p = OUT / 'method_records.py'
t = p.read_text(encoding='utf-8')
t = t.replace('長さ調整ではなく内容を保った速度変形の固定モデル診断', '고정 모델에서 길이 정규화와 속도 변형 비교')
t = t.replace('artifacts/train_crop_io_activity_20261110/analysis.json', 'artifacts/train_crop_io_activity_20261110/results/analysis.json')
p.write_text(t, encoding='utf-8')

external = Path('F:/IRLab/XLSR/19LA_train_last_20261105_01')
receipt = {
    'observed_at_local': datetime.now().astimezone().isoformat(),
    'source_directory': str(external),
    'read_only_metadata_capture': True,
    'new_feature_extraction': False,
    'raw_array_rehashed_now': False,
    'status': json.loads((external / 'status.json').read_text(encoding='utf-8')),
    'summary': json.loads((external / 'summary.json').read_text(encoding='utf-8')),
    'completion_markers': {
        str(x.relative_to(external)): x.read_text(encoding='utf-8').strip()
        for x in external.rglob('*')
        if x.is_file() and x.name in ('complete', 'complete.txt', 'run.exit', 'complete.exit')
    },
}
(OUT / 'last_hidden_export_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print('Corrected title/source; captured external-drive completion metadata.')
