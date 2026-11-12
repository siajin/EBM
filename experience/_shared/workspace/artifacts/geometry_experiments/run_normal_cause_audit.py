"""Prepare/retrieve a bounded CPU-only acoustic audit on the original server."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import time
import paramiko
import pandas as pd
from geometry_cache import resolve_run

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/geometry_cache/local_analysis/normal_cause_audit_v1'
REMOTE='/home/mentee/analysis/embedding_geometry/20261105_normal_cause_01'


def connect():
    c=paramiko.SSHClient();c.load_system_host_keys()
    c.connect('172.16.162.222',username='mentee',password=(Path.home()/'.ssh/mentee-password.txt').read_text().splitlines()[0],look_for_keys=False,allow_agent=False,timeout=15)
    return c


def load(name):
    p,e,s=resolve_run(name);f=pd.read_parquet(p/'extract/scores.parquet');return f[f.split==s].reset_index(drop=True)


def ordered(f,n,prefix):
    order=sorted(range(len(f)),key=lambda i:hashlib.sha256((prefix+str(f.iloc[i].utterance_id)).encode()).digest())
    return f.iloc[order[:n]]


def setup():
    OUT.mkdir(parents=True,exist_ok=False)
    pilot=load('DF_pilot');full=pd.concat([pilot,load('DF_remaining')],ignore_index=True)
    assert len(full)==533928 and not full.utterance_id.duplicated().any()
    subsets=[]
    normal=full[(full.y==1)&(full.codec=='nocodec')].copy();normal['purpose']='nocodec_normal';subsets.append(normal)
    for source,g in pilot[pilot.y==1].groupby('source_corpus'):
        q=ordered(g,200,'normal-acoustic-paired-v1|').copy();q['purpose']='paired_pilot';subsets.append(q)
    for source,g in full[(full.y==0)&(full.codec=='nocodec')].groupby('source_corpus'):
        q=ordered(g,400,'normal-acoustic-spoof-reference-v1|').copy();q['purpose']='spoof_reference';subsets.append(q)
    df=pd.concat(subsets,ignore_index=True).drop_duplicates('utterance_id').copy();df['audio_dataset']='DF'
    t=load('T');t=t[t.y==1].copy();t['source_corpus']='T';t['codec']='clean';t['purpose']='training_normal_reference';t['audio_dataset']='T'
    t=pd.concat([ordered(g,26,'normal-acoustic-T-v1|') for _,g in t.groupby('speaker_group')],ignore_index=True)
    selected=pd.concat([df,t],ignore_index=True);selected['case_id']=selected.audio_dataset+'|'+selected.utterance_id
    assert not selected.case_id.duplicated().any()
    selected=selected.sort_values('case_id').reset_index(drop=True)
    selected.to_parquet(OUT/'analysis_rows.parquet',index=False)
    manifest=selected[['case_id','audio_dataset','utterance_id']]
    manifest.to_csv(OUT/'selected_ids.csv',index=False)
    protocol=dict(status='posthoc_exploratory',created_epoch=time.time(),question='Which acoustic properties track normal-domain failures?',
        original_audio_stays_on_server=True,CPU_only=True,no_silence_distribution_features=True,
        sample_selection='All DF nocodec normals; 200 normal pilot IDs per corpus; 400 nocodec spoof IDs per corpus; 26 T normal IDs per speaker, all metadata-only SHA256 order',
        rows=len(selected),counts=selected.groupby(['audio_dataset','source_corpus','y']).size().to_dict().__str__(),
        selected_ids_sha256=hashlib.sha256((OUT/'selected_ids.csv').read_bytes()).hexdigest(),
        feature_blocks=['level','spectrum','cepstrum','periodicity_pitch','spectral_time_variation'],
        analyses=['Within-source and within-speaker feature/score association','Grouped OOF acoustic blocks beyond source+duration','Existing paired R3 noise/Opus score changes by source','Original-family/codec pairing if official mapping resolves'],
        causal_limits='Observational feature associations cannot uniquely distinguish channel, phonetic content and vocal production. Existing paired R3 estimates only its actual transformations.')
    c=connect()
    try:
        _,o,e=c.exec_command('readlink /proc/352729/ns/mnt');protocol['container_mount_namespace']=o.read().decode().strip();assert o.channel.recv_exit_status()==0
        _,o,e=c.exec_command('mkdir '+shlex.quote(REMOTE));assert o.channel.recv_exit_status()==0,'Existing remote result; no overwrite'
        with c.open_sftp() as s:
            s.put(str(OUT/'selected_ids.csv'),REMOTE+'/selected_ids.csv')
            s.put(str(Path(__file__).with_name('normal_acoustic_features.py')),REMOTE+'/normal_acoustic_features_v1.py')
            with s.open(REMOTE+'/protocol.json','w') as f:f.write(json.dumps(protocol,indent=2))
        launch="import subprocess,pathlib;root=pathlib.Path("+repr(REMOTE)+");log=(root/'features.log').open('xb');p=subprocess.Popen(['python3','-u',str(root/'normal_acoustic_features_v1.py'),'--root',str(root)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);print(p.pid)"
        _,o,e=c.exec_command('python3 -c '+shlex.quote(launch));pid=int(o.read().decode().strip());assert o.channel.recv_exit_status()==0
        protocol['server_PID']=pid
    finally:c.close()
    (OUT/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(dict(rows=len(selected),server_PID=pid,remote=REMOTE)),flush=True)


def status(sync=False):
    c=connect()
    try:
        with c.open_sftp() as s:
            for name in ['progress.json','complete.json','synthetic_QC.json']:
                try:
                    with s.open(REMOTE+'/'+name) as f:text=f.read().decode()
                    print(name,text,flush=True)
                except FileNotFoundError:pass
            if sync:
                with s.open(REMOTE+'/complete.json') as f:complete=json.loads(f.read())
                assert complete['status']=='pass'
                for name in ['features.csv','complete.json','synthetic_QC.json']:
                    target=OUT/name
                    if not target.exists():s.get(REMOTE+'/'+name,str(target))
                assert hashlib.sha256((OUT/'features.csv').read_bytes()).hexdigest()==complete['features_sha256']
                print('Completed derived acoustic descriptors downloaded; no audio transferred',flush=True)
            else:
                try:
                    with s.open(REMOTE+'/features.log') as f:print('log_tail',f.read().decode()[-1000:])
                except FileNotFoundError:pass
    finally:c.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['setup','status','sync']);a=p.parse_args()
    setup() if a.action=='setup' else status(a.action=='sync')
