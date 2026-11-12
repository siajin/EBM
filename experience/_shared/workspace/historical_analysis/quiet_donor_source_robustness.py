from pathlib import Path
import json,numpy as np
assert Path('/.dockerenv').exists()
root=Path('/results/analysis/EBM80_quiet_donor_audit');r=json.loads((root/'analysis/summary.json').read_text());dm=json.loads((root/'donor_manifest_private.json').read_text());tr=json.loads((root/'recipient_manifest_private.json').read_text());pairs=np.load(root/'assignment_private.npz')['pairs'];y=np.array([t['label'] for t in tr])
ss=[np.load(root/f'inference/shard_{i}.npz') for i in range(2)];order=np.argsort(np.concatenate([a['indices'] for a in ss]));s=np.concatenate([a['scores'] for a in ss])[order,5:].reshape(2000,3,2,2);g=np.concatenate([a['gains'] for a in ss])[order,5:].reshape(2000,3,2,2)
mask=np.all(np.isfinite(s),axis=(1,2,3))&(np.max(g,axis=(1,2,3))<=10);result={}
for word,lab in [('bonafide',1),('spoof',0)]:
 m=mask&(y==lab);result[word]={}
 for a,b in [(0,1),(0,2),(1,2)]:
  delta=s[:,a,:,0]-s[:,b,:,0];vals=[]
  for rep in [0,1]:
   for sourceindex in [a,b]:
    sp=np.array([dm[int(j)]['speaker'] for j in pairs[:,sourceindex,rep,0]])
    for donor in np.unique(sp[m]):
     kept=m&(sp!=donor)
     if kept.any():vals.append(float(.5*(delta[kept,rep].mean()+delta[m,1-rep].mean())))
  key=r['plan']['sources'][a]+'_minus_'+r['plan']['sources'][b]
  result[word][key]={'n':int(m.sum()),'mean':float(delta[m].mean()),'leave_one_donor_speaker_mean_range':[min(vals),max(vals)],'removals':len(vals)}
out={'source_comparison_bonafide_donors_only':result,'limitations':'Finite-bank leave-one sensitivity only. Each bank remains weight0.5 but recipient composition can change within a bank after removing a donor. Ranges are not confidence intervals or donor-population guarantees.'}
(root/'analysis/source_robustness.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
