"""Import/input checks only, with no logger, optimizer, or training launch."""
import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0,'/home/mentee/code/EBM')
import arguments,data_loaders,main,model,train

code=Path('/home/mentee/code/EBM')
root=Path(sys.argv[1])
args,_,_=arguments.get_args()
assert args['name']=='EBM_e50_bs240_lr1e5'
assert args['epoch']==50 and args['batch_size']==240 and args['lr']==1e-5
assert args['num_train_frames']==404 and args['num_train_frames_short']==303
assert 'native_length' not in args
original=ast.parse(Path('/home/mentee/results/CM/LA/EBM_e50_bs240_lr1e5/backup/scripts/data_loaders.py').read_text())
class StripCodec(ast.NodeTransformer):
    def visit_If(self,node):
        if any(isinstance(v,ast.Attribute) and v.attr=='p_codec_aug' for v in ast.walk(node.test)):
            return None
        return self.generic_visit(node)
assert ast.dump(StripCodec().visit(original))==ast.dump(ast.parse((code/'data_loaders.py').read_text()))
probe=pd.read_parquet('/home/mentee/analysis/embedding_geometry/20261106_variable_length_01/manifest.parquet').set_index('selection')
cases=[]
for key in ['train_y1_minimum','train_y0_minimum','train_y1_maximum','train_y0_maximum']:
    r=probe.loc[key]
    item=SimpleNamespace(path=r.path,label=int(r.y),type=r.attack_id)
    dataset=data_loaders.TrainSet(args,[item])
    def forbidden(*a,**kw):raise AssertionError('Codec must stay off')
    dataset.codec_aug.convert=forbidden
    with patch.object(data_loaders.random,'random',return_value=0.),patch.object(data_loaders.random,'randint',return_value=0):
        row=dataset[0]
    audio,_=sf.read(item.path)
    expected=lambda n:np.pad(audio,(0,max(0,n-len(audio))),mode='wrap')[:n]
    assert np.array_equal(np.asarray(row[0]).reshape(-1),expected(64600))
    assert np.array_equal(np.asarray(row[1]).reshape(-1),expected(48440))
    cases.append({'selection':key,'original_samples':len(audio),'long_samples':len(row[0]),'short_samples':len(row[1])})
result={'status':'passed','imports':'arguments,data_loaders,main,model,train',
        'config':'e50_bs240_lr1e5','dataset_checks':cases,'codec_off':True,
        'dataloader_AST_only_codec_blocks_removed':True,'training_started':False}
(root/'verify.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
