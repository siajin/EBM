"""Prepare an explicit, inspected source-only commit on the server."""
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

code=Path('/home/mentee/code/EBM')
backup=Path(sys.argv[1])
backup.mkdir(exist_ok=False)

def git(*args):
    return subprocess.check_output(['git',*args],cwd=code,text=True).strip()

assert git('branch','--show-current')=='main'
assert not git('diff','--cached','--name-only'),'Pre-existing staged changes; leave untouched'
assert git('rev-parse','HEAD')=='16f9cba7ced226fbe31fb3f217cfafbc6e801075'
previous=json.loads(Path('/home/mentee/analysis/ebm_control_B_20261106_2118/change.json').read_text())
for relative,digest in previous['all_python_hashes'].items():
    assert hashlib.sha256((code/relative).read_bytes()).hexdigest()==digest,relative
for name in ['.gitignore','.git/config']:
    shutil.copy2(code/name,backup/name.replace('/','_'))

ignore='''
# Private experiment workspace: source only; never commit data or credentials.
*.bak.*
*.pt
*.pth
*.ckpt
*.safetensors
*.onnx
*.flac
*.mp3
*.ogg
*.m4a
*.npy
*.npz
*.parquet
*.h5
*.hdf5
*.bundle
data/
datasets/
results/
checkpoints/
artifacts/
.env.*
*.pem
*.key
id_rsa*
id_ed25519*
*password*.txt
*token*.txt
'''
with (code/'.gitignore').open('a') as f:f.write(ignore)
(code/'docs').mkdir(exist_ok=True)
assert not (code/'docs/CONTROL_B.md').exists()
assert not (code/'docs/WORKFLOW.md').exists()
shutil.copy2(Path(sys.argv[2]),code/'docs/CONTROL_B.md')
(code/'docs/WORKFLOW.md').write_text('''# Private EBM workspace

This repository contains the current B control condition. See CONTROL_B.md.
The original project and its license are retained; upstream is kimho1wq/EBM.

- origin is the private siajin/EBM repository. Work on branches in this repository.
- The server training checkout is /home/mentee/code/EBM.
- Keep data, model weights, generated results, local backups, and credentials out of Git.
- Never stop or restart a running experiment while updating source without the user's request.
- GPU work uses the existing container and CUDA_VISIBLE_DEVICES=0,1.
- Check the active container, experiment, and GPU state before launching anything.
- Use a new experiment name; do not overwrite an existing results directory.
- Pull only with a clean working tree, and use git pull --ff-only.
- B has not completed full training. The existing tests are bounded correctness checks.

Code changes should be committed and pushed to origin. Then update the other checkout
with git pull --ff-only before starting a new, explicitly requested experiment.
''')

changed=git('diff','--name-only').splitlines()
expected={'.gitignore','arguments.py','data_loaders.py','dataset/ASVspoof2019_la_train.py',
          'dataset/ASVspoof2021_df_eval.py','logger/local.py','main.py','model/b_linear.py',
          'model/loss.py','train.py'}
assert set(changed)==expected,changed
selected=sorted(expected|{'native_steps.py','docs/CONTROL_B.md','docs/WORKFLOW.md'})
tracked=git('ls-files').splitlines()
files=sorted(set(tracked)|set(selected))
findings=[]
patterns=[r'gh[pousr]_[A-Za-z0-9]{20,}',r'github_pat_[A-Za-z0-9_]{20,}',
          r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',r'AKIA[A-Z0-9]{16}']
for relative in files:
    p=code/relative
    assert p.is_file() and not p.is_symlink(),relative
    assert p.stat().st_size<1_000_000,relative
    text=p.read_text()
    if any(re.search(pattern,text) for pattern in patterns):findings.append(relative)
    if p.suffix=='.py':
        tree=ast.parse(text,filename=relative)
        for node in ast.walk(tree):
            pairs=[]
            if isinstance(node,ast.Dict):
                pairs=[(k.value,v) for k,v in zip(node.keys,node.values)
                       if isinstance(k,ast.Constant) and isinstance(k.value,str)]
            elif isinstance(node,ast.Assign):
                pairs=[(target.id,node.value) for target in node.targets if isinstance(target,ast.Name)]
            for name,value in pairs:
                if re.search(r'(?:token|password|secret|api_key)',name,re.I) and isinstance(value,ast.Constant) and isinstance(value.value,str) and value.value:
                    findings.append(relative+':'+str(value.lineno))
assert not findings,{'credential_candidates_locations_only':findings}
git('config','user.name','sia_02')
git('config','user.email','53578618+siajin@users.noreply.github.com')
git('add','--',*selected)
staged=git('diff','--cached','--name-only').splitlines()
assert set(staged)==set(selected),staged
git('commit','-m','Implement controlled zero-padding B condition with original local BPL')
assert not git('status','--porcelain'),git('status','--porcelain')
sha=git('rev-parse','HEAD')
git('bundle','create',str(backup/'EBM.bundle'),'main')
data={'commit':sha,'source_files':len(files),'committed_files':selected,'credential_scan':'passed',
      'B_python_hashes_unchanged':True,'private_repo':'siajin/EBM','training_started':False}
(backup/'prepared.json').write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps(data))
