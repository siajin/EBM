from pathlib import Path
import json,subprocess,time
root=Path('/results/analysis/EBM80_quiet_donor_audit');assert Path('/.dockerenv').exists();steps=[]
for name in ['analyze_quiet_donor_audit.py','verify_quiet_donor_inputs.py']:
 start=time.time()
 with (root/(name+'.log')).open('w') as f:r=subprocess.run(['python','/home/mentee/analysis/'+name],stdout=f,stderr=subprocess.STDOUT)
 steps.append(dict(script=name,exit_code=r.returncode,seconds=time.time()-start));(root/'post_steps.json').write_text(json.dumps(steps,indent=2));print(json.dumps(steps[-1]),flush=True)
 if r.returncode:raise SystemExit(r.returncode)
print('DONOR_POSTCHECKS_COMPLETE',flush=True)
