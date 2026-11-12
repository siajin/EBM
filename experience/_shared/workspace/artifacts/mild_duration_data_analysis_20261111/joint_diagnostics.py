"""Separate marginal priors, joint duration distributions, and input-time shares."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
bank=pd.read_parquet(HERE/'variant_metadata.parquet')
original=bank.drop_duplicates('index').copy()
edges=np.arange(0,31.05,.1)


def joint_overlap(frame,duration,weight):
    groups=sorted(frame.speaker.unique())
    h=[]
    for label in [1,0]:
        f=frame[frame.label==label];total=weight(f).sum()
        hist=[]
        for speaker in groups:
            g=f[f.speaker==speaker]
            hist.extend(np.histogram(g[duration],bins=edges,weights=weight(g)/total)[0])
        assert np.isclose(sum(hist),1)
        h.append(np.array(hist))
    return 100*float(np.minimum(h[0],h[1]).sum())


before=joint_overlap(original,'original_seconds',lambda x:np.ones(len(x)))
after=joint_overlap(bank,'target_seconds',lambda x:x.class_weight.to_numpy())
rows=[]
for label in [1,0]:
    f=bank[bank.label==label];o=original[original.label==label]
    for dim in ['speaker','attack']:
        old=o.groupby(dim).original_seconds.sum();old=old/old.sum()
        new=(f.target_seconds*f.probability).groupby(f[dim]).sum();new=new/new.sum()
        raw=((f.stop-f.start)*f.probability).groupby(f[dim]).sum();raw=raw/raw.sum()
        for group in old.index:
            rows.append({'label':label,'dimension':dim,'group':group,
                'before_audio_time_percent':100*old[group],
                'after_input_time_percent':100*new[group],
                'retained_original_time_percent':100*raw[group],
                'input_time_delta_pp':100*(new[group]-old[group]),
                'retained_original_time_delta_pp':100*(raw[group]-old[group])})
pd.DataFrame(rows).to_csv(HERE/'speaker_attack_time_shares.csv',index=False)
result={'speaker_by_duration_joint_overlap_percent':{'before':before,'after':after,'bin_seconds':.1},
    'maximum_speaker_input_time_share_change_pp':max(abs(r['input_time_delta_pp']) for r in rows if r['dimension']=='speaker'),
    'maximum_attack_input_time_share_change_pp':max(abs(r['input_time_delta_pp']) for r in rows if r['dimension']=='attack'),
    'maximum_speaker_retained_original_time_share_change_pp':max(abs(r['retained_original_time_delta_pp']) for r in rows if r['dimension']=='speaker'),
    'maximum_attack_retained_original_time_share_change_pp':max(abs(r['retained_original_time_delta_pp']) for r in rows if r['dimension']=='attack'),
    'interpretation':'Expected count shares unchanged. Input-time shares can change through tempo. These are not OC objective sample weights.'}
(HERE/'joint_diagnostics.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
