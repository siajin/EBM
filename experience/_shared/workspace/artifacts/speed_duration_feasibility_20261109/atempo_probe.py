"""Independent algorithm check, metadata-only sub-selection from locked probe."""
import json
from pathlib import Path
import sys

import pandas as pd
import prepare_probe as base

MAIN=base.OUT
base.OUT=base.HERE/'atempo_probe'
base.REMOTE='/home/mentee/analysis/ebm_speed_duration_atempo_20261109_v1'

def prepare():
    base.OUT.mkdir(exist_ok=False)
    f=pd.read_csv(MAIN/'selected.csv')
    f=f.groupby(['phase','label','length_bin'],sort=False,group_keys=False).head(2).reset_index(drop=True)
    f.selection_order=range(len(f))
    assert len(f)==64 and not f.duplicated(['phase','uid']).any()
    f.to_csv(base.OUT/'selected.csv',index=False)
    protocol=json.loads((MAIN/'protocol.json').read_text())
    protocol.update(N=64,quota_per_phase_class_length_bin=2,selected_sha256=base.sha(base.OUT/'selected.csv'),remote_root=base.REMOTE,
        algorithm='FFmpeg atempo, pitch preserving; nominal target length without repair',
        selection='Metadata-only first2 of each phase/class/length stratum in prior locked selection. No outcome selection.')
    protocol['limitations'].append('atempo output lengths are approximate; actual lengths and discrepancies recorded, never crop/pad to repair')
    base.save(base.OUT/'protocol.json',protocol)
    c=base.connect()
    try:
        base.run(c,"""
import json,subprocess
from pathlib import Path
root=Path(ROOT)
assert not root.exists()
assert Path('/home/mentee/analysis/ebm_speed_duration_probe_20261109_v1/run.exit').read_text().strip()=='0'
assert Path('/proc/352729/root/.dockerenv').exists()
assert subprocess.check_output(['ps','-p','352729','-o','pgid=,tpgid='],text=True).split()==['352729','352729']
root.mkdir()
print(json.dumps({'prepared':True}))
""".replace('ROOT',repr(base.REMOTE)))
        with c.open_sftp() as s:
            for name in ['selected.csv','protocol.json']:
                s.put(str(base.OUT/name),base.REMOTE+'/'+name)
            with s.open('/home/mentee/analysis/ebm_speed_duration_probe_20261109_v1/dependencies.json') as f:deps=f.read()
            with s.open(base.REMOTE+'/dependencies.json','wb') as f:f.write(deps)
    finally:c.close()
    print(json.dumps({'prepared':64}))

if __name__=='__main__':
    action=sys.argv[1]
    if action=='prepare':prepare()
    elif action=='launch':base.launch('diagnose_speed_atempo.py')
    elif action in ['status','sync']:base.status(action=='sync')
    else:raise ValueError(action)
