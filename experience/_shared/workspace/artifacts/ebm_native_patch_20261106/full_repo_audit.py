"""Server-only full-tree comparison; export metadata, never private source/secrets."""
import ast
import difflib
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import tarfile

ROOT=Path(sys.argv[1]);SOURCE=Path('/home/mentee/code/EBM')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def functions(text):
    try:tree=ast.parse(text)
    except SyntaxError:return {}
    result={}
    def visit(body,prefix=''):
        for node in body:
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
                name=prefix+node.name
                result[name]=dict(line=node.lineno,end_line=node.end_lineno,ast=hashlib.sha256(ast.dump(node).encode()).hexdigest())
                if isinstance(node,ast.ClassDef):visit(node.body,name+'.')
    visit(tree.body)
    return result


def codec_info(path):
    text=path.read_text();tree=ast.parse(text)
    calls=[n.lineno for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='convert']
    return dict(active_convert_call_lines=calls,probability_assignment_lines=[i for i,l in enumerate(text.splitlines(),1) if 'self.p_codec_aug =' in l],sha256=sha(path))


def main():
    upstream=ROOT/'upstream';upstream.mkdir()
    with tarfile.open(ROOT/'upstream.tar.gz') as tar:
        for member in tar:
            parts=Path(member.name).parts[1:]
            if not parts or member.isdir():continue
            assert member.isfile() and '..' not in parts
            dest=upstream.joinpath(*parts);dest.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(member) as f:dest.write_bytes(f.read())
    tree=json.loads((ROOT/'upstream_tree.json').read_text())
    names=sorted(n['path'] for n in tree['tree'] if n['type']=='blob')
    # Verify the public archive against GitHub's git blob identities.
    for n in tree['tree']:
        if n['type']!='blob':continue
        b=(upstream/n['path']).read_bytes();assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==n['sha']
    before={str(p.relative_to(SOURCE)):sha(p) for p in SOURCE.rglob('*') if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts}
    codec_before=codec_info(SOURCE/'data_loaders.py')
    backup=ROOT/'before_codec_fix';backup.mkdir()
    shutil.copy2(SOURCE/'data_loaders.py',backup/'data_loaders.py')
    # Honor the user's persistent codec-off instruction without changing p.
    text=(SOURCE/'data_loaders.py').read_text();lines=text.splitlines(keepends=True)
    node=next(n for n in ast.walk(ast.parse(text)) if isinstance(n,ast.ClassDef) and n.name=='TrainSet')
    method=next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name=='__getitem__')
    blocks=[n for n in method.body if isinstance(n,ast.If) and 'self.p_codec_aug' in ast.get_source_segment(text,n.test)]
    assert len(blocks)==2 and len(codec_before['active_convert_call_lines'])==2
    for block in blocks:
        for i in range(block.lineno-1,block.end_lineno):
            indent=len(lines[i])-len(lines[i].lstrip())
            lines[i]=lines[i][:indent]+'# '+lines[i][indent:]
    revised=''.join(lines);ast.parse(revised)
    probability=lambda s:[l.strip() for l in s.splitlines() if 'self.p_codec_aug =' in l]
    assert probability(text)==probability(revised)
    pending=SOURCE/'data_loaders.py.codec-off-pending';pending.write_text(revised);pending.replace(SOURCE/'data_loaders.py')
    codec_after=codec_info(SOURCE/'data_loaders.py');assert codec_after['active_convert_call_lines']==[]
    records=[]
    for name in names:
        old=upstream/name;new=SOURCE/name
        if not new.exists():records.append(dict(path=name,status='missing'));continue
        same=sha(old)==sha(new)
        record=dict(path=name,status='identical' if same else 'changed',upstream_sha256=sha(old),current_sha256=sha(new))
        if not same:
            a=old.read_text();b=new.read_text();fa=functions(a);fb=functions(b)
            record['changed_symbols']={k:{'upstream_line':fa.get(k,{}).get('line'),'current_line':fb.get(k,{}).get('line')} for k in sorted(set(fa)|set(fb)) if fa.get(k,{}).get('ast')!=fb.get(k,{}).get('ast')}
            record['upstream_lines']=len(a.splitlines());record['current_lines']=len(b.splitlines())
            record['python_AST_equal']=ast.dump(ast.parse(a))==ast.dump(ast.parse(b)) if name.endswith('.py') else None
        records.append(record)
    extras=[name for name in before if name not in names]
    whitelist={'name','epoch','batch_size','lr','lr_min','weight_decay','T_0','T_mult','num_workers','num_train_frames','num_train_frames_short','num_test_frames','PLM_name','hidden_size','agg_size','output_size','alpha','alpha_cl','r_real','r_fake','native_length','micro_batch_size','eval_batch_size','short_oc_weight','padding_loss_weight','rand_seed','use_class_weight'}
    def dictionary(path):
        return {k.value:v for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.Dict) for k,v in zip(n.keys,n.values) if isinstance(k,ast.Constant) and isinstance(k.value,str)}
    orig=dictionary(upstream/'arguments.py');current=dictionary(SOURCE/'arguments.py');config=[]
    for k in sorted(set(orig)|set(current)):
        if k in orig and k in current and ast.dump(orig[k])==ast.dump(current[k]):continue
        row={'key':k,'status':'added' if k not in orig else 'removed' if k not in current else 'changed'}
        if k in whitelist:
            for label,d in [('upstream',orig),('current',current)]:
                if k in d:
                    try:row[label]=ast.literal_eval(d[k])
                    except (ValueError,TypeError):row[label]='expression changed'
        else:row['values']='omitted (environment/authentication settings)'
        config.append(row)
    # Only safe non-argument implementation diffs stay on the server.
    for name in ['dataset/ASVspoof2019_la_train.py','dataset/ASVspoof2021_df_eval.py','main.py']:
        diff=''.join(difflib.unified_diff((upstream/name).read_text().splitlines(True),(SOURCE/name).read_text().splitlines(True),fromfile='upstream/'+name,tofile='current/'+name))
        safe='\n'.join(line for line in diff.splitlines() if not re.search(r'password|api[_-]?key|token|secret',line,re.I))
        (ROOT/(name.replace('/','_')+'.diff.txt')).write_text(safe+'\n')
    result=dict(status='complete',revision='16f9cba7ced226fbe31fb3f217cfafbc6e801075',upstream_file_count=len(names),
        identical=sum(r['status']=='identical' for r in records),changed=sum(r['status']=='changed' for r in records),missing=sum(r['status']=='missing' for r in records),
        files=records,extra_local_files=extras,config_differences=config,codec_before=codec_before,codec_after=codec_after,
        codec_fix='Commented both codec conditional blocks; probability and codec implementation untouched',
        codec_before_sha256=before['data_loaders.py'],final_current_hashes={n:sha(SOURCE/n) for n in before},
        non_codec_files_unchanged_during_audit=all(sha(SOURCE/n)==s for n,s in before.items() if n!='data_loaders.py'))
    (ROOT/'repo_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['files','final_current_hashes']},indent=2))


if __name__=='__main__':main()
