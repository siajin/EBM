"""Preflight only: validate the B condition before the single training launch."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

root=Path(sys.argv[1])
launch=json.loads((root/'launch.json').read_text())
code=Path('/home/mentee/code/EBM')
sys.path.insert(0,str(code))
import arguments
import torch

def git(*a):return subprocess.check_output(['git',*a],cwd=code,text=True).strip()
assert Path('/.dockerenv').exists()
assert os.environ['CUDA_VISIBLE_DEVICES']=='0,1'
assert torch.cuda.device_count()==2
assert git('rev-parse','HEAD')==launch['commit']
assert git('branch','--show-current')==launch['branch']
assert not git('status','--porcelain')
args,_,_=arguments.get_args()
expected={'name':'EBM_e50_bs240_lr1e5_B_zeropad_short303','epoch':50,'batch_size':240,
          'rand_seed':1,'lr':1e-5,'native_length':True,'cap_long':True,
          'short_oc_weight':0.,'padding_loss_weight':0.,'micro_batch_size':8,
          'num_train_frames':404,'num_train_frames_short':303,'eval_batch_size':8}
assert all(args[k]==v for k,v in expected.items())
assert args['usable_gpu'] is None
assert (code/'native_steps.py').is_file()
assert not subprocess.check_output(['git','diff','main','--','*.py'],cwd=code)
calls=[n for n in ast.walk(ast.parse((code/'data_loaders.py').read_text()))
       if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='convert']
assert not calls
out=Path(args['path_log'])/args['project']/Path(*args['tags'])/args['name']
assert not out.exists(),'Refuse to overwrite any previous result directory'
for k in ['path_19LA','path_21LA','path_21DF']:assert Path(args[k]).is_dir(),k
python_files=git('ls-files','*.py').splitlines()
hashes={name:hashlib.sha256((code/name).read_bytes()).hexdigest() for name in python_files}
info={'status':'passed','condition':expected,'commit':launch['commit'],'branch':launch['branch'],
      'result_path':str(out),'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES'],
      'devices':[torch.cuda.get_device_name(i) for i in range(2)],
      'active_codec_convert_calls':0,'Python_identical_to_verified_B_main':True,
      'python_hashes':hashes,'new_loss_disabled':True}
print(json.dumps(info,indent=2))
