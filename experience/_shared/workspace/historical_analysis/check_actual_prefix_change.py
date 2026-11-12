from pathlib import Path
import json
import numpy as np
assert Path('/.dockerenv').exists()
r=Path('/results/analysis/EBM80_21DF_leading_quiet_paired')
p=r/'actual_prefix_change.json';assert not p.exists()
d=np.load(r/'paired_scores.npz');s=d['scores'].astype(float);y=d['labels'];lead=d['checks'][:,0].astype(int)
j=json.loads((r/'summary.json').read_text());eps=j['meaningful_delta_epsilon']
out={'target_leading_s':1.02,'epsilon':eps,'by_actual_change':{},'identity_check':{},'notes':[
 'Original-to-1.02s comparisons also alter the common-body selection, padding and quiet waveform. These subgroups do not isolate a pure duration effect.',
 'Every originally shorter leading region requires dropping some later samples to fit the fixed common body; crop count is checked explicitly.',
 'The controlled 0.06s-to-1.02s pair holds that selected body and total filler content fixed, but changes front/back allocation.']}
for fam,col in [('zeros',4),('ownquiet',7)]:
    result={}
    for label,word in [(1,'bonafide'),(0,'spoof')]:
        result[word]={}
        for name,m in [('actually_lengthened',lead<16320),('unchanged_length',lead==16320),('actually_shortened',lead>16320)]:
            mask=m&(y==label);a=s[mask,col]-s[mask,0]
            if not mask.any():
                result[word][name]={'n':0}
                continue
            result[word][name]={'n':int(mask.sum()),'original_prefix_median_s':float(np.median(lead[mask])/16000),
              'toward_center_n':int((a>eps).sum()),'away_from_center_n':int((a<-eps).sum()),'within_epsilon_n':int((np.abs(a)<=eps).sum()),
              'mean_delta_cosine':float(a.mean()),'median_delta_cosine':float(np.median(a)),
              'right_crop_n':int((d['checks'][mask,3]>0).sum()),
              'toward_ge01_n':int((a>=.1).sum()),'away_ge01_n':int((a<=-.1).sum())}
    out['by_actual_change'][fam]=result
m=lead==16320
err=float(np.max(np.abs(s[m,7]-s[m,0])))
assert err<eps
out['identity_check']={'original_prefix_already_102_n':int(m.sum()),'ownquiet_102_vs_original_max_delta':err,'all_within_epsilon':True}
p.write_text(json.dumps(out,indent=2,allow_nan=False))
print(json.dumps(out,indent=2))
