"""Standard-library-only guard for the host watcher."""
import json
from pathlib import Path

EPOCHS = list(range(5, 51, 5))

def dependency_ready(config):
    """A committed epoch50 alone does not release the training container."""
    root = Path(config['wait_for_root'])
    terminal = root/'run.exit'
    if not terminal.exists(): return False
    code = int(terminal.read_text())
    assert code in (0, 2), f'Prior pipeline failed/stopped ({code}); no automatic launch'
    stages = Path(config['current_result'])/'short32_stages'
    for epoch in EPOCHS:
        p = stages/f'checkpoint_{epoch}.complete.json'
        if not p.exists(): raise AssertionError(f'Uncommitted checkpoint {epoch}; no automatic launch')
        marker = json.loads(p.read_text())
        assert marker['status'] == 'committed' and marker['epoch'] == epoch
    summary = json.loads((stages/'complete.json').read_text())
    assert summary['status'] in ('complete', 'evaluation_incomplete')
    return True

