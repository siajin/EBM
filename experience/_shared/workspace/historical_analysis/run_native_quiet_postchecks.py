from pathlib import Path
import subprocess,json,time
root=Path('/results/analysis/EBM80_native_quiet_strict')
assert Path('/.dockerenv').exists() and (root/'inference/COMPLETED.json').exists()
assert not (root/'postcheck_steps.json').exists()
steps=[]
for name,script in [('independent_validation','verify_native_quiet_strict.py'),('statistics','analyze_native_quiet_strict.py'),
                    ('robustness','check_native_quiet_robustness.py'),('repeat_inference','native_quiet_repeat_check.py'),
                    ('evaluation_view','evalview_quiet_strict.py')]:
    start=time.time()
    with (root/(name+'.log')).open('w') as f:
        result=subprocess.run(['python','/home/mentee/analysis/'+script],stdout=f,stderr=subprocess.STDOUT)
    steps.append({'step':name,'exit_code':result.returncode,'seconds':time.time()-start})
    (root/'postcheck_steps.json').write_text(json.dumps(steps,indent=2))
    print(json.dumps(steps[-1]),flush=True)
    if result.returncode:raise SystemExit(result.returncode)
print('ALL_POSTCHECKS_COMPLETE',flush=True)
