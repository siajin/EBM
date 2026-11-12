"""Lock, launch, observe, and retrieve only derived paired padding results."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import time
import paramiko
import pandas as pd
from geometry_cache import DEFAULT_ROOT,resolve_run

OUT=DEFAULT_ROOT/'local_analysis/padding_causal_20261106_v1'
REMOTE='/home/mentee/analysis/embedding_geometry/20261106_padding_causal_01'

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def connect():
    c=paramiko.SSHClient();c.load_system_host_keys();c.connect('172.16.162.222',username='mentee',password=(Path.home()/'.ssh/mentee-password.txt').read_text().splitlines()[0],look_for_keys=False,allow_agent=False,timeout=15);return c
def run(c,cmd):
    _,o,e=c.exec_command(cmd,timeout=30);s=o.read().decode();err=e.read().decode();rc=o.channel.recv_exit_status();assert rc==0,(rc,err[-1500:]);return s

def setup():
    OUT.mkdir(parents=True,exist_ok=False)
    frames=[];sources={}
    for alias in ['DF_pilot','DF_remaining']:
        folder,entry,split=resolve_run(alias);f=pd.read_parquet(folder/'extract/scores.parquet');f=f[(f.split==split)&(f.codec=='nocodec')].copy();f['cache_alias']=alias;frames.append(f);sources[alias]=entry['cache_fingerprint']
    f=pd.concat(frames,ignore_index=True);assert not f.utterance_id.duplicated().any()
    f['duration_bin']=pd.cut(f.duration,[0,2,4.0375,float('inf')],right=False,labels=['under2','2to4','long']).astype(str)
    chosen=[]
    for (source,y,binname),g in f.groupby(['source_corpus','y','duration_bin'],sort=True):
        n=32 if y==1 and binname=='long' else 16 if y==0 and binname=='long' else 40 if y==1 else 20
        g=g.copy();g['uid_priority']=[hashlib.sha256(('padding-causal-v1|'+u).encode()).hexdigest() for u in g.utterance_id]
        g=g.sort_values('uid_priority');g['within_speaker_order']=g.groupby('speaker_group').cumcount();g=g.sort_values(['within_speaker_order','uid_priority'])
        chosen.append(g.head(n))
    selected=pd.concat(chosen,ignore_index=True);selected['selection_order']=range(len(selected))
    keep=['utterance_id','y','source_corpus','speaker_group','duration_bin','cache_alias','selection_order']
    selected[keep].to_parquet(OUT/'selected.parquet',index=False)
    counts=selected.groupby(['source_corpus','y','duration_bin'],sort=True).size().reset_index(name='n');counts.to_csv(OUT/'selection_counts.csv',index=False)
    protocol={'status':'locked_before_new_forward','created_epoch':time.time(),'reference_chat':'01a1abcd-37ab-73b3-9e02-3b493a9315ea',
        'question':'Whether identical-content padding intervention changes the adverse normal movement, and whether artificial shortening reproduces source-specific failure',
        'N_selected':len(selected),'counts':counts.to_dict('records'),'source_fingerprints':sources,'selected_sha256':digest(OUT/'selected.parquet'),
        'selection':'nocodec only, metadata source x class x original duration, UID hash round-robin across speakers; no score or embedding based selection',
        'conditions':['native_wrap','native_reflect','native_original','for long originals only: first/last1.5s and2s each wrap/reflect/original'],
        'content_control':'same exact waveform within each padding comparison; no gain normalization, added silence, silence trimming or speech-rate change',
        'primary_family':{'comparisons':12,'targets':['short normals original-minus-wrap angle, each3sources','short normals reflect-minus-wrap angle, each3sources','long normals cropped2s-wrap-minus-native angle averaged head/tail, each3sources','long normals cropped2s-original-minus-cropped2s-wrap angle averaged head/tail, each3sources'],
            'CI':'source x speaker cluster paired bootstrap5000; individual95% and Bonferroni family95%; no independent source-family assumption'},
        'secondary':'1.5s dose; spoof score/error changes; original 160D tangent directions; derived frontend temporal-mean cosine movement only',
        'diagnostic_error_threshold':-0.9145589682336335,'threshold_scope':'retrospective full-DF native EER point, fixed across conditions; not deployment threshold',
        'limits':['Previously inspected DF; new controlled forward is exploratory, not unseen confirmatory benchmark','unpadding also changes total context length; cannot uniquely attribute to periodicity alone','reflection reverses repeated audio; mechanistic sensitivity control, not recommended deployment','cropping removes phonetic content; two fixed windows averaged per utterance, never independent utterances','selected strata are not population proportions; no full-DF performance claim','no retraining, no silence-distribution hypothesis, no canceled DF4x3 CV'],
        'source_code_sha256':digest(Path(__file__).with_name('padding_causal_sidecar.py'))}
    save(OUT/'protocol.json',protocol)
    c=connect()
    try:
        idle=run(c,'ps -p 352729 -o pgid=,tpgid=').split()==['352729','352729']
        assert idle,'Existing shell is busy; no launch'
        assert run(c,'test -f /proc/352729/root/.dockerenv && echo yes').strip()=='yes'
        gpu=run(c,'nvidia-smi -i 6,7 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits')
        assert all(int(a.split(',')[0])<100 and int(a.split(',')[1])==0 for a in gpu.splitlines()),'Assigned GPUs busy'
        run(c,'mkdir '+shlex.quote(REMOTE))
        with c.open_sftp() as s:
            for name in ['selected.parquet','protocol.json']:s.put(str(OUT/name),REMOTE+'/'+name)
            s.put(str(Path(__file__).with_name('padding_causal_sidecar.py')),REMOTE+'/padding_causal_sidecar_v1.py')
            launch='''#!/bin/bash
set -u
test -f /.dockerenv || exit 90
export CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
ROOT="'''+REMOTE+'''"
cd /home/mentee/code/EBM || exit 91
for stage in prepare extract; do
 python -u "$ROOT/padding_causal_sidecar_v1.py" "$stage" --root "$ROOT" > "$ROOT/$stage.log" 2>&1
 rc=$?; echo "$rc" > "$ROOT/$stage.exit"
 if [ "$rc" -ne 0 ]; then echo "$rc" > "$ROOT/run.exit"; exit "$rc"; fi
done
echo 0 > "$ROOT/run.exit"
'''
            with s.open(REMOTE+'/run.sh','w') as q:q.write(launch)
        # Recheck foreground immediately before delivering the one new command.
        assert run(c,'ps -p 352729 -o pgid=,tpgid=').split()==['352729','352729']
        run(c,'screen -S 352153.exp -p 0 -X stuff '+shlex.quote('bash '+REMOTE+'/run.sh\n'))
        save(OUT/'launched.json',{'epoch':time.time(),'remote':REMOTE,'assigned_gpu_readback':gpu,'existing_container_reused':True})
    finally:c.close()
    print(json.dumps({'status':'launched','N':len(selected),'counts':counts.to_dict('records')},ensure_ascii=False),flush=True)

def status(sync=False):
    c=connect();result={}
    try:
        with c.open_sftp() as s:
            for name in ['prepare.exit','extract.exit','run.exit','rank0_progress.json','rank1_progress.json','rank0_QC.json','rank1_QC.json','complete.json']:
                try:
                    with s.open(REMOTE+'/'+name) as q:t=q.read().decode()
                    if name.endswith('.json'):
                        value=json.loads(t)
                        if name.endswith('_QC.json'):value={k:v for k,v in value.items() if k!='rows'}
                        result[name]=value
                    else:result[name]=t.strip()
                except FileNotFoundError:pass
            if result.get('run.exit') not in [None,'0']:
                for name in ['prepare.log','extract.log']:
                    try:
                        with s.open(REMOTE+'/'+name) as q:result[name+'_tail']=q.read().decode()[-4000:]
                    except FileNotFoundError:pass
            if sync:
                assert result.get('run.exit')=='0' and result['complete.json']['status']=='pass'
                targets=result['complete.json']['files']
                for name,sha in targets.items():
                    dest=OUT/name
                    if not dest.exists():s.get(REMOTE+'/'+name,str(dest))
                    assert digest(dest)==sha
                for name in ['complete.json','prepare_complete.json']:
                    if not (OUT/name).exists():s.get(REMOTE+'/'+name,str(OUT/name))
                result['sync']='hash-verified derived results only'
        save(OUT/'last_status.json',result)
        print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    finally:c.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['setup','status','sync']);a=p.parse_args()
    setup() if a.action=='setup' else status(a.action=='sync')
