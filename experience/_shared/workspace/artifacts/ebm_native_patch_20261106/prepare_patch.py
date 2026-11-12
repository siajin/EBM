"""Server-only staged patch. Backups never leave the research server."""
import ast
import hashlib
import json
from pathlib import Path
import py_compile
import re
import shutil
import sys

ROOT = Path(sys.argv[1])
SOURCE = Path('/home/mentee/code/EBM')
FILES = ['arguments.py','data_loaders.py','model/b_linear.py','model/loss.py','main.py','train.py','logger/local.py']


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def replace(text,old,new):
    if text.count(old)!=1:
        raise RuntimeError('Expected one source anchor: '+old[:90])
    return text.replace(old,new,1)


def main():
    stage=ROOT/'stage';backup=ROOT/'before'
    stage.mkdir();backup.mkdir()
    original={str(p.relative_to(SOURCE)):sha(p) for p in SOURCE.rglob('*.py') if '.git' not in p.parts}
    for name in original:
        dest=stage/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(SOURCE/name,dest)
    for name in FILES:
        target=backup/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(SOURCE/name,target)
    def read(name):return (backup/name).read_text()
    def write(name,text):
        dest=stage/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(text);py_compile.compile(str(dest),doraise=True)

    t=read('arguments.py')
    match=re.search(r"('name'\s*:\s*)('.*?')",t)
    assert match
    oldname=ast.literal_eval(match.group(2))
    t=t[:match.start(2)]+repr(oldname+'_native_short303_pair')+t[match.end(2):]
    insertion="""        # Native long / 3-second short. Codec settings remain in TrainSet.
        'native_length'       : True,
        'micro_batch_size'    : 8,
        'eval_batch_size'     : 8,
        'short_oc_weight'     : 0.25,
        'padding_loss_weight' : 0.10,

"""
    at=t.index("        'num_train_frames'")
    t=t[:at]+insertion+t[at:];write('arguments.py',t)

    t=read('data_loaders.py')
    t=t.replace('from torch.utils.data import Dataset, DataLoader, DistributedSampler',
                'from torch.utils.data import Dataset, DataLoader, Sampler\nfrom functools import partial\nimport math\nimport torch')
    begin=t.index('# Separate bonafide and spoofed')
    end=t.index('class TrainSet(Dataset):')
    t=t[:begin]+(ROOT/'loader_support.py.txt').read_text()+t[end:]
    t=replace(t,"        self.crop_size = args['num_train_frames'] * 160 - 40", "        self.native_length = args.get('native_length', True)\n        self.crop_size = args['num_train_frames'] * 160 - 40")
    t=replace(t,"        self.crop_size = args['num_test_frames'] * 160 - 40", "        self.native_length = args.get('native_length', True)\n        self.crop_size = args['num_test_frames'] * 160 - 40")
    t=replace(t,'if codec_crop_size < audio.shape[0]:','if not self.native_length and codec_crop_size < audio.shape[0]:')
    t=replace(t,'if codec_crop_size_short < audio_short.shape[0]:','if not self.native_length and codec_crop_size_short < audio_short.shape[0]:')
    block="""        if audio.shape[0] < self.crop_size:
            shortage = self.crop_size - audio.shape[0]
            audio = np.pad(audio, (0, shortage), 'wrap')
        else:
            audio = audio[:self.crop_size]"""
    assert t.count(block)==2
    t=t.replace(block,'        if not self.native_length:\n'+'\n'.join('    '+line for line in block.splitlines()))
    t=replace(t,"            audio_short = np.pad(audio_short, (0, shortage), 'wrap')", "            if not self.native_length:\n                audio_short = np.pad(audio_short, (0, shortage), 'wrap')")
    t=replace(t,'        return audio, audio_short, item.label','        if sr != 16000 or audio.ndim != 1 or min(len(audio),len(audio_short)) < 720:\n            raise ValueError("Expected mono16k audio with at least two valid SSL frames")\n        return audio, audio_short, item.label, index')
    t=replace(t,'        audio, _ = sf.read(item.path)','        audio, sr = sf.read(item.path)\n        if sr != 16000 or audio.ndim != 1 or len(audio) < 720:\n            raise ValueError("Expected mono16k audio with at least two valid SSL frames")')
    t=replace(t,'        return audio, item.type, item.label','        return audio, item.type, item.label, index')
    # Preserve every codec probability assignment and conversion call.
    old_codec=[line.strip() for line in read('data_loaders.py').splitlines() if 'p_codec_aug' in line or '.codec_aug.convert(' in line]
    new_codec=[line.strip() for line in t.splitlines() if 'p_codec_aug' in line or '.codec_aug.convert(' in line]
    assert old_codec==new_codec
    write('data_loaders.py',t)

    t=read('model/b_linear.py')
    t=replace(t,'def forward(self, x, x_short=None, label=None, bona_size=None):',
              'def forward(self, x, x_short=None, label=None, bona_size=None, mask=None, mask_short=None, return_embedding=False, logical_batch_size=None):')
    t=replace(t,'x = self.bridge_module(x)','x = self.bridge_module(x, mask=mask)')
    t=replace(t,'x = self.AHP_ASP(x)','x = self.AHP_ASP(x, mask=mask)')
    t=replace(t,'        if x_short != None:', '        if return_embedding:\n            return x\n\n        if x_short is not None:')
    t=replace(t,'x_short = self.bridge_module(x_short)','x_short = self.bridge_module(x_short, mask=mask_short)')
    t=replace(t,'x_short = self.AHP_ASP(x_short)','x_short = self.AHP_ASP(x_short, mask=mask_short)')
    t=replace(t,'self.loss(x, label)','self.loss(x, label, logical_batch_size=logical_batch_size)')
    t=t.replace('x_short != None','x_short is not None')
    # Only Bridge_module has the one-argument forward signature.
    t=replace(t,'    def forward(self, x):\n        B, L, T, H', '    def forward(self, x, mask=None):\n        B, L, T, H')
    t=replace(t,'self.AHP_ASP[i](x_i_ouput)','self.AHP_ASP[i](x_i_ouput, mask=mask)')
    t=replace(t,'def forward(self, inputs, get_w=False, tanh=True):','def forward(self, inputs, get_w=False, tanh=True, mask=None):')
    t=replace(t,'        batch_size = inputs.size(0)', '''        if mask is not None:
            mask = mask.to(device=inputs.device, dtype=torch.bool)
            if mask.shape != inputs.shape[:2] or not torch.all(mask.sum(1) >= (2 if self.asp else 1)):
                raise ValueError('Invalid temporal mask or too few valid frames')
            inputs = inputs.masked_fill(~mask.unsqueeze(-1), 0.)
        batch_size = inputs.size(0)''')
    t=replace(t,'        # attention (batchsize, length, num_head)', '''        if tanh:
            weights = torch.tanh(weights)
        if mask is not None:
            weights = weights.masked_fill(~mask.unsqueeze(-1), -torch.inf)
        # attention (batchsize, length, num_head)''')
    t=replace(t,'attentions = nn.functional.softmax(torch.tanh(weights), dim=1)','attentions = nn.functional.softmax(weights, dim=1)')
    t=replace(t,'            avg_repr, std_repr = weighted.sum(1), (weighted+noise).std(1)', '''            avg_repr = weighted.sum(1)
            if mask is None:
                std_repr = (weighted+noise).std(1)
            else:
                valid = mask.unsqueeze(-1)
                count = valid.sum(1)
                noisy = weighted + noise
                mean = (noisy*valid).sum(1)/count
                variance = (((noisy-mean.unsqueeze(1))**2)*valid).sum(1)/(count-1)
                std_repr = variance.clamp_min(1e-30).sqrt()''')
    write('model/b_linear.py',t)

    t=read('model/loss.py')
    upstream=(ROOT/'upstream_loss.py').read_text()
    upstream_node=next(n for n in ast.parse(upstream).body if isinstance(n,ast.ClassDef) and n.name=='BPL')
    node=next(n for n in ast.parse(t).body if isinstance(n,ast.ClassDef) and n.name=='BPL')
    bpl_was_original=ast.dump(node)==ast.dump(upstream_node)
    if not bpl_was_original:
        lines=t.splitlines(keepends=True)
        t=''.join(lines[:node.lineno-1])+ast.get_source_segment(upstream,upstream_node)+'\n\n'+''.join(lines[node.end_lineno:])
        t='\n'.join(line for line in t.splitlines() if 'torch.distributed' not in line)+'\n'
    t=replace(t,'def forward(self, x, label=None):','def forward(self, x, label=None, logical_batch_size=None):')
    t=replace(t,'if x.size(0) > 4:','if (x.size(0) if logical_batch_size is None else logical_batch_size) > 4:')
    write('model/loss.py',t)

    t=read('main.py')
    t=replace(t,'''    args['device'] = f'cuda:{args["process_id"]}'\n''', '''    args['device'] = f'cuda:{args["process_id"]}'\n    torch.cuda.set_device(args['process_id'])\n''')
    old="""        trainer.train_loader_bona, trainer.sampler_bona, trainer.train_loader_spoof, trainer.sampler_spoof, \\
            trainer.val_loader, trainer.eval_loader = data_loaders.get_loaders(args, train_set, train_db_19la.dev_set, test_set)"""
    t=replace(t,old,"""        trainer.train_loader, trainer.train_sampler, trainer.val_loader, trainer.eval_loader = \\
            data_loaders.get_loaders(args, train_set, train_db_19la.dev_set, test_set)""")
    old="""    trainer.plm = nn.SyncBatchNorm.convert_sync_batchnorm(trainer.plm)
    trainer.plm = nn.parallel.DistributedDataParallel(
        trainer.plm, device_ids=[args['device']], find_unused_parameters=True
    )"""
    t=replace(t,old,"    trainer.plm.eval()\n    trainer.plm.requires_grad_(False)")
    anchor="    # start\n"
    guards="""    # Validate before spawning workers or constructing the destructive logger.
    if args['epoch'] > args['T_0']:
        raise ValueError('This native-length implementation keeps XLS-R frozen; epoch must be <= T_0')
    if not 0 <= args['short_oc_weight'] <= 1 or args['padding_loss_weight'] < 0:
        raise ValueError('Invalid auxiliary loss weights')
    if not args['native_length'] and (args['short_oc_weight'] or args['padding_loss_weight']):
        raise ValueError('Legacy E0 requires both auxiliary weights set to zero')
    result_path = os.path.join('/', args['path_log'], args['project'], '/'.join(args['tags']), args['name'])
    if os.path.exists(result_path):
        raise FileExistsError('Result directory already exists: '+result_path)

"""
    t=replace(t,anchor,guards+anchor);write('main.py',t)
    write('train.py',(ROOT/'trainer.py.txt').read_text())
    t=read('logger/local.py')
    t=replace(t,'\t\t\tshutil.rmtree(path)','\t\t\traise FileExistsError("Refusing to overwrite experiment results: " + path)')
    write('logger/local.py',t)
    write('native_steps.py',(ROOT/'native_steps.py').read_text())
    report={'source_before':original,'files':FILES+['native_steps.py'],
        'stage_hashes':{name:sha(stage/name) for name in FILES+['native_steps.py']},
        'bpl_already_matches_github':bpl_was_original,'github_loss_sha256':sha(ROOT/'upstream_loss.py'),
        'codec_lines_preserved':old_codec,'batch_size_preserved':240,'installed':False}
    (ROOT/'prepared.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['source_before','codec_lines_preserved']},indent=2))


if __name__=='__main__':main()
