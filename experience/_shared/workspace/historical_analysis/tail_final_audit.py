from pathlib import Path
import json
import numpy as np
assert Path('/.dockerenv').exists()
root=Path('/results/analysis')
out={}
for name,file in [('EBM80_21DF_tail_interventions','intervention_scores.npz'),('EBM80_21DF_tail_silence','silence_scores.npz')]:
    p=root/name; d=np.load(p/file); s=json.loads((p/'summary.json').read_text())
    y=d['labels']; assert len(y)==2000 and y.sum()==1000
    assert np.isfinite(d['scores']).all()
    out[name]={}
    for j,method in enumerate(d['names']):
        a=d['scores'][:,j]; ordered=np.sort(a)
        assert ordered[999]<ordered[1000]
        threshold=float((float(ordered[999])+float(ordered[1000]))/2)
        fn=int(((y==1)&(a<threshold)).sum()); fp=int(((y==0)&(a>=threshold)).sum())
        assert fn==fp
        direct=fn/10
        assert abs(direct-s['overall'][method]['sample_eer_pct'])<1e-6
        out[name][str(method)]={'equal_error_count_per_class':fn,'sample_eer_pct':direct}
print(json.dumps(out,indent=2))
(root/'EBM80_21DF_tail_silence/independent_eer_check.json').write_text(json.dumps(out,indent=2))
