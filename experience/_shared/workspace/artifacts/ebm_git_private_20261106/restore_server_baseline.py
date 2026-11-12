"""Restore the archived batch240 source on its own server branch; never train."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

code=Path('/home/mentee/code/EBM')
archive=Path('/home/mentee/results/CM/LA/EBM_e50_bs240_lr1e5/backup/scripts')
result=archive.parent.parent
root=Path(sys.argv[1])
branch='baseline/e50-bs240-lr1e5'
base='0b860798c87daef8ea47cd24f15c8aaecb9a8c7e'

def git(*args):return subprocess.check_output(['git',*args],cwd=code,text=True).strip()
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def manifest(folder):
    return {str(p.relative_to(folder)):digest(p) for p in sorted(folder.rglob('*'))
            if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts}

assert git('branch','--show-current')=='main'
assert git('rev-parse','HEAD')==base
assert not git('status','--porcelain'),'Do not overwrite uncommitted work'
assert subprocess.check_output(['ps','-p','352729','-o','pgid=,tpgid='],text=True).split()==['352729','352729']
gpu=subprocess.check_output(['nvidia-smi','-i','6,7','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
assert all(int(row.split(',')[0])<100 and int(row.split(',')[1])==0 for row in gpu.splitlines())
root.mkdir(exist_ok=False)
(root/'before').mkdir()
for name in git('ls-files').splitlines():
    dst=root/'before'/name
    dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(code/name,dst)
archive_before=manifest(archive)
results_before=manifest(result)
# The upstream tree enumerates exactly the original source files, not backups.
source=git('ls-tree','-r','--name-only','16f9cba7ced226fbe31fb3f217cfafbc6e801075').splitlines()
assert len(source)==34 and all((archive/p).is_file() for p in source)
for name in source:
    if name.endswith('.py'):ast.parse((archive/name).read_text(),filename=name)

# Preserve the user's earlier explicit codec-off instruction while reverting
# every training/data/model path to the archived baseline implementation.
loader=(archive/'data_loaders.py').read_text()
tree=ast.parse(loader)
blocks=[n for n in ast.walk(tree) if isinstance(n,ast.If) and
        any(isinstance(v,ast.Attribute) and v.attr=='p_codec_aug' for v in ast.walk(n.test))]
assert len(blocks)==2
lines=loader.splitlines(keepends=True)
for block in sorted(blocks,key=lambda n:n.lineno,reverse=True):
    for i in range(block.lineno-1,block.end_lineno):
        stripped=lines[i].lstrip()
        indent=lines[i][:-len(stripped)]
        lines[i]=indent+'# '+stripped
loader=''.join(lines)
assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='convert'
               for n in ast.walk(ast.parse(loader)))

git('switch','-c',branch)
for name in source:
    if name=='.gitignore':continue  # Keep exclusions for data, secrets and backups.
    target=code/name
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(archive/name,target)
(code/'data_loaders.py').write_text(loader)
git('rm','--','native_steps.py')
(code/'docs/SERVER_BASELINE.md').write_text('''# Server baseline checkout

The server is on baseline/e50-bs240-lr1e5. main retains the B zero-padding experiment.
Source was restored from results/CM/LA/EBM_e50_bs240_lr1e5/backup/scripts.

Training/data/model code matches that archive except for the two online-codec
conditional blocks, which remain commented to honor the user's codec-off instruction.
The original fixed long/short repeat+crop path, class-specific loaders, original
two-stage update, local BPL, original evaluation and logger are restored.
No native microbatch helper, masks, new auxiliary losses, or length sampler is used.

Epoch50, global batch240, lr1e-5 and original experiment name are restored.
Source code restoration is not a new training run and does not restore/reload weights.
Existing result files and checkpoints have not been changed.

Important: the archived logger deletes an existing results directory of the same name.
Do not launch this restored configuration over the existing baseline results.
Use a new name for any explicitly requested rerun.

docs/CONTROL_B.md describes main, not this server branch. Do not pull main onto
this branch or restart an existing experiment automatically.
''')
workflow=code/'docs/WORKFLOW.md'
text=workflow.read_text()
text=text.replace('This repository contains the current B control condition. See CONTROL_B.md.',
    'This server branch restores the archived batch240 baseline. See SERVER_BASELINE.md.\nThe B control condition remains on main; CONTROL_B.md describes main.')
workflow.write_text(text)

different=[p for p in source if digest(code/p)!=digest(archive/p)]
assert different==['.gitignore','data_loaders.py'],different
for p in code.rglob('*.py'):
    if '.git' not in p.parts and '.bak.' not in p.name:ast.parse(p.read_text(),filename=str(p))
git('add','--',*source,'docs/SERVER_BASELINE.md','docs/WORKFLOW.md')
git('commit','-m','Restore archived batch240 server baseline while keeping codec disabled')
assert not git('status','--porcelain')
assert manifest(archive)==archive_before
assert manifest(result)==results_before,'Historical results must remain untouched'
data={'status':'restored_not_trained','branch':branch,'commit':git('rev-parse','HEAD'),
      'B_main_preserved':git('rev-parse','main')==base,'source_archive':str(archive),
      'source_files_compared':len(source),'archive_identical_files':len(source)-len(different),
      'archive_exceptions':different,'codec_off_preserved':True,'results_and_checkpoints_unchanged':True,
      'native_helper_removed':not (code/'native_steps.py').exists(),'syntax_pass':True,
      'root':str(root),'archive_hashes':archive_before,'original_result_hashes':results_before}
(root/'restore.json').write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps({k:v for k,v in data.items() if k not in ['archive_hashes','original_result_hashes']}))
