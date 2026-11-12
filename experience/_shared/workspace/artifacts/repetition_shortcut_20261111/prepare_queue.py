"""Lock derived-only cohorts, isolate code on server, and queue behind active pipeline."""
import hashlib
import json
from pathlib import Path
import shlex
import sys
import pandas as pd

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
REMOTE='/home/mentee/analysis/ebm_repetition_shortcut_20261111_v1'
DEPENDENCY='/home/mentee/analysis/ebm_baseline240_mild_e20_20261111_v2'
sys.path.insert(0,str(PROJECT/'artifacts/geometry_experiments'))
from run_padding_causal import connect

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def run(c,source):
    _,o,e=c.exec_command("python - <<'PY'\n"+source+"\nPY",timeout=60)
    result=o.read().decode();err=e.read().decode();assert o.channel.recv_exit_status()==0,err[-2500:]
    return json.loads(result)

def prepare():
    assert not (HERE/'prepared.json').exists(),'Already prepared; use status.py'
    cache=PROJECT/'artifacts/geometry_cache/local_analysis/B_final_analysis_20261108_v1'
    sources={};frames=[]
    for phase in ['dev','21LA']:
        file=cache/f'paired_scores_{phase}.parquet';f=pd.read_parquet(file);f['phase']=phase
        f=f[['phase','index','uid','speaker','label','original_samples','attack']]
        assert f.uid.is_unique;frames.append(f);sources[str(file.relative_to(PROJECT))]=sha(file)
    full=pd.concat(frames,ignore_index=True);assert len(full)==173020
    full.to_csv(HERE/'selected.csv',index=False)
    f=full[(full.phase=='21LA')&(full.original_samples>64600)].copy()
    meta=PROJECT/'artifacts/coverage20_length_shortcut_20261111/trial_metadata_21LA.txt'
    m=pd.read_csv(meta,sep=r'\s+',header=None,names=['speaker_m','uid','codec','transmission','attack_m','class_m','trim','subset'])
    f=f.merge(m[['uid','codec']],on='uid',validate='one_to_one')
    chosen=[]
    for attack,g in f.groupby('attack',sort=True):
        quota=128 if attack=='bonafide' else 16
        g=g.copy();g['priority']=[hashlib.sha256(('repeat-shortcut-v1|'+u).encode()).hexdigest() for u in g.uid]
        g=g.sort_values('priority');g['speaker_order']=g.groupby('speaker').cumcount();g['codec_order']=g.groupby('codec').cumcount()
        g=g.sort_values(['speaker_order','codec_order','priority']).head(quota);assert len(g)==quota
        chosen.append(g)
    probe=pd.concat(chosen,ignore_index=True);assert len(probe)==336
    probe=probe[full.columns];probe.to_csv(HERE/'probe_selected.csv',index=False)
    provenance=json.loads((PROJECT/'artifacts/ebm_B_C_checkpoints_20261109/provenance.json').read_text())
    checkpoints={'legacy240':provenance['protected_checkpoints']['legacy240_CLS50'],'B':provenance['checkpoints']['B_e50']}
    protocol={'status':'locked_before_new_inference','selection_hashes':{file:sha(HERE/file) for file in ['selected.csv','probe_selected.csv']},'checkpoints':checkpoints,'full_n':173020,'probe_n':336,'probe_views_per_uid':31,'selection':'All dev/21LA for native-wrap; probe long21LA128bona+16perattack, metadata UID SHA256 speaker round-robin then codec balance. No score/error selection.','sources':sources,'metadata_sha256':sha(meta),'noise_std':0,'physical_batch':8,'output':'Only derived scores, waveform hashes and metadata; no audio, SSL states, models or new training','limits':json.loads((HERE/'plan.json').read_text())['statistical_limits']+['Previously inspected targets: exploratory','Cached older native score reweighting and new noise0 evaluation are distinct runs','Conditional lengths are coarse bins; uniform weighting gives large weight to23normals1-1.5s','Legacy240 historical codec usage unconfirmed; B differs in sampler/training graph too, not a clean training ablation']}
    save(HERE/'protocol.json',protocol)
    # Isolated snapshot is made internally on server, without downloading lab source.
    c=connect()
    try:
        observation=run(c,"""
import ast,hashlib,json,subprocess
from pathlib import Path
root=Path(ROOT);code=Path('/home/mentee/code/EBM')
assert not root.exists(),'Unique root already exists'
assert Path('/proc/352729/root/.dockerenv').is_file()
assert not subprocess.check_output(['git','status','--porcelain'],cwd=code,text=True).strip()
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip();assert commit=='cf199af03044c40837c08ce31d633e2c048bde08'
root.mkdir();snapshot=root/'code';snapshot.mkdir()
archive=subprocess.Popen(['git','archive','HEAD'],cwd=code,stdout=subprocess.PIPE)
extract=subprocess.run(['tar','-x','-C',str(snapshot)],stdin=archive.stdout,check=True);archive.stdout.close();assert archive.wait()==0
hashes={str(p.relative_to(snapshot)):hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.rglob('*.py')}
(root/'source_hashes.json').write_text(json.dumps(hashes,indent=2))
print(json.dumps({'status':'snapshot_prepared_no_gpu_launch','commit':commit,'source_files':len(hashes),'dependency_exit':(Path(DEPENDENCY)/'pipeline.exit').read_text().strip() if (Path(DEPENDENCY)/'pipeline.exit').exists() else None}))
""".replace('ROOT',repr(REMOTE)).replace('DEPENDENCY',repr(DEPENDENCY)))
        mappings={'probe_repetition_shortcut.py':PROJECT/'EBM/tools/probe_repetition_shortcut.py','probe_common.py':PROJECT/'EBM/tools/diagnose_speed_views.py','summarize_repetition_shortcut.py':PROJECT/'EBM/tools/summarize_repetition_shortcut.py','analyze_repetition_shortcut.py':PROJECT/'EBM/tools/analyze_repetition_shortcut.py','trial_metadata_21LA.txt':meta}
        for name in ['selected.csv','probe_selected.csv','protocol.json','plan.json']:mappings[name]=HERE/name
        with c.open_sftp() as s:
            for name,file in mappings.items():s.put(str(file),REMOTE+'/'+name)
        manifest={name:sha(file) for name,file in mappings.items()};save(HERE/'payload_hashes.json',manifest)
        pipeline='''#!/bin/bash
set -u
ROOT=ROOT_VALUE
test -f /.dockerenv || exit 90
mkdir "$ROOT/launch.claim" || exit 95
export CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$ROOT/code" || exit 91
step() {
 stage="$1"; shift
 echo "$stage" > "$ROOT/stage.txt"
 python -u "$ROOT/$1" "${@:2}" > "$ROOT/$stage.log" 2>&1
 rc=$?; echo "$rc" > "$ROOT/$stage.exit"
 if [ "$rc" -ne 0 ]; then exit "$rc"; fi
}
step full_smoke probe_repetition_shortcut.py "$ROOT" --smoke
step probe_smoke probe_repetition_shortcut.py "$ROOT" --probe --smoke
step full probe_repetition_shortcut.py "$ROOT"
step probe probe_repetition_shortcut.py "$ROOT" --probe
step analysis summarize_repetition_shortcut.py "$ROOT"
echo complete > "$ROOT/stage.txt"
'''.replace('ROOT_VALUE',shlex.quote(REMOTE))
        wrapper='''#!/bin/bash
ROOT=ROOT_VALUE
date -Iseconds > "$ROOT/started.txt"
timeout --signal=TERM --kill-after=60s 10h bash "$ROOT/pipeline.sh" > "$ROOT/pipeline.log" 2>&1
rc=$?; echo "$rc" > "$ROOT/run.exit"
date -Iseconds > "$ROOT/finished.txt"
'''.replace('ROOT_VALUE',shlex.quote(REMOTE))
        queue='''import hashlib,json,subprocess,time
from pathlib import Path
root=Path(ROOT_VALUE);dependency=Path(DEPENDENCY_VALUE);deadline=time.time()+48*3600
def state(status,**kw):
 d={'status':status,'updated_unix':time.time(),'dependency':str(dependency),'deadline':deadline,**kw};p=root/'queue_state.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2));tmp.replace(p)
def verify():
 p=json.loads((root/'protocol.json').read_text())
 for file,h in json.loads((root/'payload_hashes.json').read_text()).items():assert hashlib.sha256((root/file).read_bytes()).hexdigest()==h,file
 for file,h in json.loads((root/'source_hashes.json').read_text()).items():assert hashlib.sha256((root/'code'/file).read_bytes()).hexdigest()==h,file
 for c in p['checkpoints'].values():assert hashlib.sha256(Path(c['path']).read_bytes()).hexdigest()==c['sha256']
 assert Path('/proc/352729/root/.dockerenv').is_file()
 assert not (root/'launch.claim').exists()
 assert not subprocess.check_output(['git','status','--porcelain'],cwd='/home/mentee/code/EBM',text=True).strip()
def idle():
 if subprocess.check_output(['ps','-p','352729','-o','pgid=,tpgid='],text=True).split()!=['352729','352729']:return False
 gpu=subprocess.check_output(['nvidia-smi','-i','6,7','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 return len(gpu.splitlines())==2 and all(int(x.split(',')[0])<100 and int(x.split(',')[1])==0 for x in gpu.splitlines())
try:
 stable=0;verify();state('waiting_dependency_success_and_idle')
 while time.time()<deadline:
  exitfile=dependency/'pipeline.exit'
  if exitfile.exists():
   rc=exitfile.read_text().strip()
   if rc!='0':state('aborted_dependency_failed',dependency_exit=rc);break
   stable=stable+1 if idle() else 0
   if stable>=3:
    verify();assert idle();subprocess.check_call(['screen','-S','352153.exp','-p','0','-X','stuff','bash '+str(root/'run.sh')+'\\n'])
    state('command_sent_once');break
  time.sleep(20)
 else:state('expired_no_launch')
except Exception as e:
 state('failed_no_retry',error=repr(e));raise
'''.replace('ROOT_VALUE',repr(REMOTE)).replace('DEPENDENCY_VALUE',repr(DEPENDENCY))
        for name,content in [('pipeline.sh',pipeline),('run.sh',wrapper),('wait_and_launch.py',queue)]:
            (HERE/name).write_text(content,encoding='utf-8',newline='\n')
        manifest.update({name:sha(HERE/name) for name in ['pipeline.sh','run.sh']});save(HERE/'payload_hashes.json',manifest)
        with c.open_sftp() as s:
            for name in ['pipeline.sh','run.sh','wait_and_launch.py','payload_hashes.json']:s.put(str(HERE/name),REMOTE+'/'+name)
        result=run(c,"""
import ast,hashlib,json,subprocess
from pathlib import Path
root=Path(ROOT)
for f in root.glob('*.py'):ast.parse(f.read_text())
for f in ['pipeline.sh','run.sh']:subprocess.check_call(['bash','-n',str(root/f)])
for f,h in json.loads((root/'payload_hashes.json').read_text()).items():assert hashlib.sha256((root/f).read_bytes()).hexdigest()==h
p=json.loads((root/'protocol.json').read_text())
for r in p['checkpoints'].values():assert hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256']
assert not (root/'queue.pid').exists()
with (root/'queue.log').open('x') as log:
 child=subprocess.Popen(['python','-u',str(root/'wait_and_launch.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=root)
(root/'queue.pid').write_text(str(child.pid)+'\\n')
print(json.dumps({'status':'queue_registered','pid':child.pid,'root':str(root),'full_utterances':p['full_n'],'probe_utterances':p['probe_n'],'automatic_retries':0,'spending_ceiling_usd':0}))
""".replace('ROOT',repr(REMOTE)))
        save(HERE/'prepared.json',observation);save(HERE/'state.json',result);print(json.dumps(result))
    finally:c.close()

if __name__=='__main__':prepare()
