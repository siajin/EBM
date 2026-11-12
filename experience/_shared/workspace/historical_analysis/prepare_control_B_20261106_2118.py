"""Apply a small server-only patch; retain every original source on the server."""
import ast
import hashlib
import json
from pathlib import Path
import sys

code = Path('/home/mentee/code/EBM')
root = Path(sys.argv[1])
root.mkdir(exist_ok=False)
(root/'before').mkdir()

def hashes():
    return {str(p.relative_to(code)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(code.rglob('*.py')) if '.git' not in p.parts}

before = hashes()
args = (code/'arguments.py').read_text()
loader = (code/'data_loaders.py').read_text()

def replace_once(text, old, new):
    assert text.count(old) == 1, old[:90]
    return text.replace(old, new)

args = replace_once(args, 'EBM_e50_bs240_lr1e5_native_short303_E1',
                    'EBM_e50_bs240_lr1e5_B_zeropad_short303')
args = replace_once(args, '# Native long / 3-second short. Codec settings remain in TrainSet.',
                    '# B: cap long at 404 frames and short at 303; never repeat audio.')
args = replace_once(args, "'native_length'       : True,",
                    "'native_length'       : True,\n        'cap_long'            : True,")
assert "'short_oc_weight'     : 0.0" in args and "'padding_loss_weight' : 0.0" in args

old = '''def collate_train(rows, short_size):
    long, short, labels, indices = zip(*rows)
    wrapped = [np.pad(s,(0,short_size-len(s)),mode='wrap') if len(s)<short_size else s.copy() for s in short]
    return {'long':[torch.as_tensor(a,dtype=torch.float32) for a in long],
            'short':[torch.as_tensor(a,dtype=torch.float32) for a in short],
            'wrapped':[torch.as_tensor(a,dtype=torch.float32) for a in wrapped],
            'pair_active':torch.tensor([len(a)<short_size for a in short]),
            'labels':torch.tensor(labels,dtype=torch.long), 'indices':torch.tensor(indices)}'''
new = '''def collate_train(rows, short_size, include_wrapped=True):
    long, short, labels, indices = zip(*rows)
    batch = {'long':[torch.as_tensor(a,dtype=torch.float32) for a in long],
             'short':[torch.as_tensor(a,dtype=torch.float32) for a in short],
             'labels':torch.tensor(labels,dtype=torch.long), 'indices':torch.tensor(indices)}
    # Auxiliary experiments can opt in; the B control creates no repeated view.
    if include_wrapped:
        wrapped = [np.pad(s,(0,short_size-len(s)),mode='wrap') if len(s)<short_size else s.copy() for s in short]
        batch['wrapped'] = [torch.as_tensor(a,dtype=torch.float32) for a in wrapped]
        batch['pair_active'] = torch.tensor([len(a)<short_size for a in short])
    return batch'''
loader = replace_once(loader, old, new)
loader = replace_once(loader,
    "collate_fn=partial(collate_train,short_size=args['num_train_frames_short']*160-40))",
    "collate_fn=partial(collate_train,short_size=args['num_train_frames_short']*160-40,\n            include_wrapped=bool(args.get('short_oc_weight',0.) or args.get('padding_loss_weight',0.))))")
assert loader.count("self.native_length = args.get('native_length', True)") == 2
loader = loader.replace("self.native_length = args.get('native_length', True)",
    "self.native_length = args.get('native_length', True)\n        self.cap_long = args.get('cap_long', False)")
old = "        if not self.native_length:\n            if audio.shape[0] < self.crop_size:"
assert loader.count(old) == 2
loader = loader.replace(old,
    "        if self.native_length and self.cap_long:\n            audio = audio[:self.crop_size]\n        elif not self.native_length:\n            if audio.shape[0] < self.crop_size:")

for name, text in [('arguments.py',args),('data_loaders.py',loader)]:
    ast.parse(text,filename=name)
    (root/'before'/name).write_bytes((code/name).read_bytes())
    (root/(name+'.new')).write_text(text)
# Both files are syntactically checked before either is installed.
for name in ['arguments.py','data_loaders.py']:
    (root/(name+'.new')).replace(code/name)
after = hashes()
changed = [k for k in before if before[k] != after[k]]
assert changed == ['arguments.py','data_loaders.py'], changed
for p in code.rglob('*.py'):
    if '.git' not in p.parts:
        ast.parse(p.read_text(),filename=str(p))
info = {'status':'installed_not_trained','root':str(root),'changed_files':changed,
        'before':{k:before[k] for k in changed},'after':{k:after[k] for k in changed},
        'all_python_hashes':after,'all_python_syntax_pass':True,
        'condition':{'name':'B','native_length':True,'cap_long':True,
                     'long_max_samples':64600,'short_max_samples':48440,
                     'short_oc_weight':0.,'padding_loss_weight':0.,'global_batch':240}}
(root/'change.json').write_text(json.dumps(info,indent=2)+'\n')
print(json.dumps({k:v for k,v in info.items() if k!='all_python_hashes'}))
