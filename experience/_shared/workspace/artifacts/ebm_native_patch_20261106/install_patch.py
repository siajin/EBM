"""Apply verified staged files, refusing concurrent source changes or old results."""
import ast
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    root=Path(sys.argv[1]);source=Path('/home/mentee/code/EBM');stage=root/'stage'
    before=json.loads((root/'prepared.json').read_text())
    assert not before['installed'] and (root/'stage_test.exit').read_text().strip()=='0'
    complete=json.loads((root/'stage_complete.json').read_text());assert complete['status']=='pass'
    for name,digest in complete['files'].items():assert sha(root/name)==digest
    for name,digest in before['source_before'].items():
        assert sha(source/name)==digest,'Concurrent source change: '+name
    for name,digest in before['stage_hashes'].items():assert sha(stage/name)==digest
    assert not (source/'native_steps.py').exists(),'Do not overwrite an unrelated helper'
    changed=[]
    try:
        for name in before['files']:
            target=source/name;pending=target.with_name(target.name+'.native-pending')
            assert not pending.exists()
            shutil.copy2(stage/name,pending);os.replace(pending,target);changed.append(name)
    except BaseException:
        for name in changed:
            if (root/'before'/name).exists():shutil.copy2(root/'before'/name,source/name)
            else:(source/name).unlink()
        raise
    unmodified={n:sha(source/n) for n in before['source_before'] if n not in before['files']}
    assert all(d==before['source_before'][n] for n,d in unmodified.items())
    public=ast.parse((root/'upstream_loss.py').read_text());current=ast.parse((source/'model/loss.py').read_text())
    get=lambda tree:next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='BPL')
    assert ast.dump(get(public))==ast.dump(get(current))
    codec=lambda path:[line.strip() for line in path.read_text().splitlines() if 'p_codec_aug' in line or '.codec_aug.convert(' in line]
    assert codec(root/'before/data_loaders.py')==codec(source/'data_loaders.py')
    summaries={}
    for name in changed:
        old=(root/'before'/name).read_text().splitlines() if (root/'before'/name).exists() else []
        new=(source/name).read_text().splitlines();plus=minus=0
        for tag,a,b,c,d in difflib.SequenceMatcher(a=old,b=new).get_opcodes():
            if tag!='equal':minus+=b-a;plus+=d-c
        summaries[name]={'before_lines':len(old),'after_lines':len(new),'added_lines':plus,'removed_lines':minus,'sha256':sha(source/name)}
    report={'status':'installed','root':str(source),'backup_root':str(root/'before'),
        'files':summaries,'unrelated_python_files_preserved':len(unmodified),
        'bpl_matches_github_AST':True,'codec_probability_and_calls_preserved':True,
        'existing_data_paths_and_user_dataset_edits_preserved':True,'full_training_launched':False}
    (root/'installed.json').write_text(json.dumps(report,indent=2)+'\n')
    before['installed']=True;(root/'prepared.json').write_text(json.dumps(before,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
