"""Fail before launch unless the isolated C condition and protected B are intact."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(sys.argv[1]);CODE=Path('/home/mentee/code/EBM')
sys.path.insert(0,str(CODE))
import arguments
import torch
from transformers import AutoConfig

launch=json.loads((ROOT/'launch.json').read_text())
def git(*a):return subprocess.check_output(['git',*a],cwd=CODE,text=True).strip()
assert Path('/.dockerenv').exists() and os.environ['CUDA_VISIBLE_DEVICES']=='0,1' and torch.cuda.device_count()==2
assert git('rev-parse','HEAD')==launch['commit'] and git('branch','--show-current')==launch['branch']
assert not git('status','--porcelain')
changed=git('diff','--name-only',launch['B_parent'],'--','*.py').splitlines()
assert changed==['arguments.py','data_loaders.py'],changed
assert int((ROOT/'verification.exit').read_text())==0
assert json.loads((ROOT/'verification_complete.json').read_text())['status']=='passed'
for rank in [0,1]:assert json.loads((ROOT/f'verification_rank{rank}.json').read_text())['status']=='passed'
args,_,_=arguments.get_args()
expected={'name':launch['experiment_name'],'epoch':50,'batch_size':240,'rand_seed':1,'lr':1e-5,'lr_min':1e-5,
          'native_length':True,'cap_long':False,'short_oc_weight':0.,'padding_loss_weight':0.,
          'micro_batch_size':8,'eval_batch_size':8,'num_train_frames':404,'num_train_frames_short':303,
          'num_test_frames':404,'num_workers':4,'T_0':150,'use_class_weight':True}
assert all(args[k]==v for k,v in expected.items())
base_source=subprocess.check_output(['git','show',launch['B_parent']+':arguments.py'],cwd=CODE,text=True)
base={'__file__':str(CODE/'arguments.py')};exec(compile(base_source,'B_arguments.py','exec'),base)
b=base['get_args']()[0]
differences={k:[b[k],args[k]] for k in b if b[k]!=args[k]}
assert set(differences)=={'name','cap_long'},differences
assert not [n for n in ast.walk(ast.parse((CODE/'data_loaders.py').read_text())) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='convert']
result=Path(args['path_log'])/args['project']/Path(*args['tags'])/args['name'];assert not result.exists()
audit=json.loads(Path('/home/mentee/analysis/ebm_B_final_eval_20261107_215451/training_audit.json').read_text())
protected={k:hashlib.sha256(Path(v['path']).read_bytes()).hexdigest()==v['sha256'] for k,v in audit['checkpoints'].items()}
assert all(protected.values())
frontend=AutoConfig.from_pretrained(args['PLM_name'],revision='main')
assert frontend._commit_hash=='1a640f32ac3e39899438a2931f9924c02f080a54'
hashes={n:hashlib.sha256((CODE/n).read_bytes()).hexdigest() for n in git('ls-files','*.py').splitlines()}
info={'status':'passed','condition':expected,'only_config_differences_from_B':differences,
      'evaluation_long_cap_samples':64600,'training_long_cap':None,'B_checkpoints_unchanged':protected,
      'commit':launch['commit'],'branch':launch['branch'],'result_path':str(result),'python_hashes':hashes,
      'frontend_commit':frontend._commit_hash,'torch_version':torch.__version__,'CUDA_VISIBLE_DEVICES':'0,1',
      'source_runtime_changed_files':changed,'two_GPU_bounded_verification_passed':True}
print(json.dumps(info,indent=2))
