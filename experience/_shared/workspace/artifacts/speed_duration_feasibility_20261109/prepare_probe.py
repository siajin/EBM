"""Metadata lock and one isolated inference probe; no training/source changes."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
OUT = HERE/'probe'
REMOTE = '/home/mentee/analysis/ebm_speed_duration_probe_20261109_v1'
sys.path.insert(0, str(PROJECT/'artifacts/geometry_experiments'))
from run_padding_causal import connect

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path, obj): Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
def run(c, source):
    _, o, e = c.exec_command("python - <<'PY'\n"+source+"\nPY", timeout=50)
    text=o.read().decode(); error=e.read().decode()
    assert o.channel.recv_exit_status()==0, error[-2000:]
    return json.loads(text)

def prepare():
    OUT.mkdir(exist_ok=False)
    rows=[]
    for phase in ['dev','19LA','21LA','21DF']:
        f=pd.read_parquet(PROJECT/f'artifacts/geometry_cache/local_analysis/B_final_analysis_20261108_v1/paired_scores_{phase}.parquet')
        assert f.uid.is_unique and not f[['uid','label','original_samples','speaker']].isna().any().any()
        f['length_bin']=pd.cut(f.original_samples,[0,32000,48440,64600,np.inf],right=False,labels=['under2','2to3.0275','3.0275to4.0375','atleast4.0375']).astype(str)
        for (label, binname), g in f.groupby(['label','length_bin'],sort=True):
            g=g.copy()
            g['priority']=[hashlib.sha256(('speed-duration-probe-v1|'+phase+'|'+u).encode()).hexdigest() for u in g.uid]
            g=g.sort_values('priority');g['speaker_order']=g.groupby('speaker').cumcount()
            chosen=g.sort_values(['speaker_order','priority']).head(16)
            # Metadata-only extreme sentinels included before any new scores exist.
            sentinel=g.loc[[g.original_samples.idxmin() if binname=='under2' else g.original_samples.idxmax()]] if binname in ['under2','atleast4.0375'] else g.iloc[:0]
            chosen=pd.concat([sentinel,chosen]).drop_duplicates('uid').head(16)
            assert len(chosen)==16
            for row in chosen.to_dict('records'):
                rows.append({k:row[k] for k in ['index','uid','speaker','label','original_samples','attack','length_bin']}|{'phase':phase})
    selected=pd.DataFrame(rows);selected['selection_order']=range(len(selected))
    assert len(selected)==512 and not selected.duplicated(['phase','uid']).any()
    selected.to_csv(OUT/'selected.csv',index=False)
    old=json.loads((PROJECT/'artifacts/ebm_B_C_checkpoints_20261109/provenance.json').read_text())
    checkpoints={'legacy240':old['protected_checkpoints']['legacy240_CLS50'], 'B':old['checkpoints']['B_e50'], 'C':old['checkpoints']['C_e50']}
    previous=json.loads((PROJECT/'artifacts/geometry_cache/local_analysis/B_final_analysis_20261108_v1/analysis/operating_points.json').read_text())
    metrics=json.loads((PROJECT/'artifacts/ebm_B_C_checkpoints_20261109/metrics.json').read_text())
    protocol={'status':'locked_before_forward','N':512,'quota_per_phase_class_length_bin':16,'phases':['dev','19LA','21LA','21DF'],
        'selection':'UID hash, speaker round robin, metadata minimum/maximum sentinel; no score selection',
        'selected_sha256':sha(OUT/'selected.csv'),'checkpoints':checkpoints,'target_samples':64600,
        'conditions':['native_cap','full_native','slow_cap','fast_cap','norm_full4','roundtrip_cap'],
        'fixed_rates':{'slow_cap':.8,'fast_cap':1.25},'noise_std':0.,
        'dev_thresholds':{'legacy240':previous['legacy240']['dev_threshold'],'B':metrics['dev']['B_e50']['dev_threshold'],'C':metrics['dev']['C_e50']['dev_threshold']},
        'limitations':['Balanced diagnostic sample is not representative benchmark EER','Fixed dev thresholds come from full native-cap dev with pooling noise1e-5; probe noise0','Slow/fast share exact cap content; full normalization uses full content','Time-stretch changes DSP artifacts as well as timing','No training, optimizer, raw audio export, hidden-state export or checkpoint export'],
        'remote_root':REMOTE}
    save(OUT/'protocol.json',protocol)
    c=connect()
    try:
        observation=run(c,"""
import json,subprocess
from pathlib import Path
root=Path(ROOT)
assert not root.exists()
assert Path('/home/mentee/analysis/ebm_B_C_checkpoints_20261109/run.exit').read_text().strip()=='0'
assert Path('/proc/352729/root/.dockerenv').exists()
assert subprocess.check_output(['ps','-p','352729','-o','pgid=,tpgid='],text=True).split()==['352729','352729']
gpu=subprocess.check_output(['nvidia-smi','-i','6,7','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
assert all(int(x.split(',')[0])<100 and int(x.split(',')[1])==0 for x in gpu.splitlines())
code=Path('/home/mentee/code/EBM')
assert not subprocess.check_output(['git','status','--porcelain'],cwd=code,text=True).strip()
root.mkdir()
print(json.dumps({'gpu':gpu,'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip()}))
""".replace('ROOT',repr(REMOTE)))
        deps="""import importlib.metadata,json,shutil
from pathlib import Path
result={}
for name in ['torch','torchaudio','librosa','numpy','scipy','transformers','soundfile']:
 try: result[name]=importlib.metadata.version(name)
 except importlib.metadata.PackageNotFoundError: result[name]=None
result['ffmpeg']=shutil.which('ffmpeg')
try:
 import librosa
 result['librosa_import']='ok'
except Exception as e:result['librosa_import']=repr(e)
Path(__file__).with_name('dependencies.json').write_text(json.dumps(result,indent=2))
"""
        with c.open_sftp() as s:
            for name in ['selected.csv','protocol.json']:s.put(str(OUT/name),REMOTE+'/'+name)
            with s.open(REMOTE+'/dependencies.py','w') as f:f.write(deps)
        command='python '+REMOTE+'/dependencies.py > '+REMOTE+'/dependencies.log 2>&1\n'
        run(c,"import subprocess,json\nsubprocess.check_call(['screen','-S','352153.exp','-p','0','-X','stuff',"+repr(command)+"])\nprint(json.dumps({'dependency_probe_sent':True}))")
        save(OUT/'prepared.json',observation)
    finally:c.close()
    print(json.dumps({'selected':len(selected),'root':REMOTE,'status':'dependency_probe_sent'}))

def launch(evaluator_name='diagnose_speed_views.py'):
    assert evaluator_name in ['diagnose_speed_views.py','diagnose_speed_atempo.py']
    evaluator=PROJECT/'EBM/tools'/evaluator_name
    assert evaluator.exists()
    script='''#!/bin/bash
set -u
ROOT="'''+REMOTE+'''"
test -f /.dockerenv || exit 90
mkdir "$ROOT/launch.claim" || exit 95
export CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd /home/mentee/code/EBM || exit 91
date -Iseconds > "$ROOT/started.txt"
python -u "$ROOT/diagnose_speed_views.py" "$ROOT" --smoke > "$ROOT/smoke.log" 2>&1
rc=$?; echo "$rc" > "$ROOT/smoke.exit"
if [ "$rc" -eq 0 ]; then
 python -u "$ROOT/diagnose_speed_views.py" "$ROOT" > "$ROOT/eval.log" 2>&1
 rc=$?
fi
echo "$rc" > "$ROOT/run.exit"
date -Iseconds > "$ROOT/finished.txt"
'''
    script=script.replace('diagnose_speed_views.py',evaluator_name)
    c=connect()
    try:
        with c.open_sftp() as s:
            with s.open(REMOTE+'/dependencies.json') as f:deps=json.loads(f.read())
            assert deps['librosa_import']=='ok',deps
            s.put(str(evaluator),REMOTE+'/'+evaluator_name)
            with s.open(REMOTE+'/run.sh','w') as f:f.write(script)
        outcome=run(c,"""
import ast,hashlib,json,subprocess
from pathlib import Path
root=Path(ROOT);code=Path('/home/mentee/code/EBM')
assert not (root/'launch.claim').exists()
assert subprocess.check_output(['ps','-p','352729','-o','pgid=,tpgid='],text=True).split()==['352729','352729']
gpu=subprocess.check_output(['nvidia-smi','-i','6,7','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
assert all(int(x.split(',')[0])<100 and int(x.split(',')[1])==0 for x in gpu.splitlines())
ast.parse((root/'diagnose_speed_views.py').read_text())
subprocess.check_call(['bash','-n',str(root/'run.sh')])
assert not subprocess.check_output(['git','status','--porcelain'],cwd=code,text=True).strip()
protocol=json.loads((root/'protocol.json').read_text())
for record in protocol['checkpoints'].values():assert hashlib.sha256(Path(record['path']).read_bytes()).hexdigest()==record['sha256']
source_hashes={str(p.relative_to(code)):hashlib.sha256(p.read_bytes()).hexdigest() for p in code.rglob('*.py') if '.git' not in p.parts}
(root/'source_hashes.json').write_text(json.dumps(source_hashes,indent=2))
subprocess.check_call(['screen','-S','352153.exp','-p','0','-X','stuff','bash '+str(root/'run.sh')+'\\n'])
print(json.dumps({'status':'inference_launched','evaluator_sha256':hashlib.sha256((root/'diagnose_speed_views.py').read_bytes()).hexdigest()}))
""".replace('ROOT',repr(REMOTE)).replace('diagnose_speed_views.py',evaluator_name))
        save(OUT/'launch.json',outcome);print(json.dumps(outcome))
    finally:c.close()

def status(sync=False):
    c=connect();obs={}
    try:
        with c.open_sftp() as s:
            for name in ['dependencies.json','started.txt','smoke.exit','run.exit','finished.txt','progress_rank0.json','progress_rank1.json','complete.json','complete_smoke.json']:
                try:
                    with s.open(REMOTE+'/'+name) as f:t=f.read().decode()
                    obs[name]=json.loads(t) if name.endswith('.json') else t.strip()
                except FileNotFoundError:pass
            for name in ['dependencies.log','smoke.log','eval.log']:
                try:
                    with s.open(REMOTE+'/'+name) as f:t=f.read().decode()
                    obs[name+'_tail']=t.splitlines()[-8:]
                except FileNotFoundError:pass
            if sync:
                assert obs.get('run.exit')=='0' and 'complete.json' in obs
                for record in s.listdir_attr(REMOTE):
                    if record.filename.endswith(('.csv','.json')):
                        s.get(REMOTE+'/'+record.filename,str(OUT/record.filename))
                code='/home/mentee/code/EBM'
                obs['preservation']=run(c,"""
import hashlib,json,subprocess
from pathlib import Path
root=Path(ROOT);code=Path('/home/mentee/code/EBM')
old=json.loads((root/'source_hashes.json').read_text())
assert all(hashlib.sha256((code/name).read_bytes()).hexdigest()==sha for name,sha in old.items())
p=json.loads((root/'protocol.json').read_text())
assert all(hashlib.sha256(Path(v['path']).read_bytes()).hexdigest()==v['sha256'] for v in p['checkpoints'].values())
print(json.dumps({'all_source_hashes_unchanged':True,'checkpoints_unchanged':True,'git_clean':not subprocess.check_output(['git','status','--porcelain'],cwd=code,text=True).strip()}))
""".replace('ROOT',repr(REMOTE)))
        save(OUT/'last_check.json',obs);print(json.dumps(obs,ensure_ascii=False,indent=2))
    finally:c.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','launch','status','sync']);a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='launch':launch()
    else:status(a.action=='sync')
