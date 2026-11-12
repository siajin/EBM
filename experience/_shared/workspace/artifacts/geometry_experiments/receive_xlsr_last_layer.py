"""One-shot laptop receiver for XLS-R features, without a server disk spool."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import struct
import threading
import time

import numpy as np
import pandas as pd
from queue_xlsr_last_layer import connect

ROOT='/home/mentee/analysis/embedding_geometry/20261105_19la_train_xlsr_stream_01'
UPSTREAM='/home/mentee/analysis/embedding_geometry/20261105_frozen_all_01/R23'
LOCAL=Path('F:/IRLab/XLSR/19LA_train_last_20261105_01')
COUNT, FRAMES, DIM = 25380, 201, 1024
ROW_BYTES=FRAMES*DIM*4
STOP=threading.Event()
LOCK=threading.Lock()
PROGRESS={}


def save(path, value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2),encoding='utf-8');temp.replace(path)


def command(client, text):
    _,out,err=client.exec_command(text,timeout=30)
    result=out.read().decode();error=err.read().decode();rc=out.channel.recv_exit_status()
    if rc:raise RuntimeError(f'Remote command failed ({rc}): {error[:500]}')
    return result.strip()


def read_remote(client, path):
    with client.open_sftp() as s:
        try:
            with s.open(path) as f:return f.read()
        except FileNotFoundError:return None


def status_file(status, **extra):
    with LOCK:progress=dict(PROGRESS)
    save(LOCAL/'status.json',dict(status=status,checked_epoch=time.time(),
        remote_root=ROOT,local_folder=str(LOCAL),server_feature_files=False,ranks=progress,**extra))


def setup():
    assert shutil.disk_usage(LOCAL.anchor).free > COUNT*ROW_BYTES+5*1024**3, 'F: lacks space plus reserve'
    LOCAL.mkdir(parents=True,exist_ok=False)
    client=connect()
    try:
        namespace=command(client,'readlink /proc/352729/ns/mnt')
        assert command(client,'test -f /proc/352729/root/.dockerenv && echo yes')=='yes'
        command(client,'mkdir '+shlex.quote(ROOT))
        with client.open_sftp() as s:
            s.put(str(Path(__file__).with_name('xlsr_last_layer_extract.py')),ROOT+'/xlsr_last_layer_stream_v1.py')
            launcher='''#!/bin/bash
set -u
ROOT="'''+ROOT+'''"
test -f /.dockerenv || { echo 90 > "$ROOT/run.exit"; exit 90; }
export CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
cd /home/mentee/code/EBM || { echo 91 > "$ROOT/run.exit"; exit 91; }
for stage in prepare qc extract; do
 python -u "$ROOT/xlsr_last_layer_stream_v1.py" "$stage" --root "$ROOT" > "$ROOT/$stage.log" 2>&1
 rc=$?
 echo "$rc" > "$ROOT/$stage.exit"
 if [ "$rc" -ne 0 ]; then echo "$rc" > "$ROOT/run.exit"; exit "$rc"; fi
done
echo 0 > "$ROOT/run.exit"
'''
            with s.open(ROOT+'/run_stream.sh','w') as f:f.write(launcher)
        payload='import os,pathlib;root=pathlib.Path('+repr(ROOT)+');os.mkfifo(str(root/"rank0.fifo"),0o600);os.mkfifo(str(root/"rank1.fifo"),0o600)'
        command(client,'python3 -c '+shlex.quote(payload))
        request=dict(dataset='19LA_train',count=COUNT,shape=[COUNT,FRAMES,DIM],dtype='float32',
            storage='laptop_only_FIFO_SSH_stream',local_folder=str(LOCAL),remote_root=ROOT,
            upstream=UPSTREAM,expected_container_namespace=namespace,user_requested=True,requested_epoch=time.time())
        save(LOCAL/'request.json',request)
        with client.open_sftp() as s:
            with s.open(ROOT+'/request.json','w') as f:f.write(json.dumps(request,indent=2))
        status_file('ready_for_local_receiver_queue')
        print(json.dumps(request,indent=2),flush=True)
    finally:client.close()


def read_exact(channel, count):
    result=bytearray();last=time.monotonic()
    while len(result)<count:
        if STOP.is_set():raise RuntimeError('Receiver stopped after a transfer failure')
        try:block=channel.recv(min(count-len(result),1024*1024))
        except socket.timeout:
            if time.monotonic()-last>240:raise TimeoutError('No stream data for 240 seconds')
            continue
        if not block:raise EOFError('SSH stream ended before the final validated record')
        result.extend(block);last=time.monotonic()
    return bytes(result)


def read_header(channel):
    size=struct.unpack('<I',read_exact(channel,4))[0]
    if not 0<size<65536:raise ValueError('Invalid stream header size')
    return json.loads(read_exact(channel,size))


def receive_rank(rank,index):
    client=None
    try:
        client=connect();transport=client.get_transport();channel=transport.open_session(timeout=20)
        channel.exec_command('exec cat -- '+shlex.quote(ROOT+f'/rank{rank}.fifo'));channel.settimeout(5)
        begin=read_header(channel);rows=index[index['rank']==rank].reset_index(drop=True)
        assert begin==dict(kind='begin',schema=1,rank=rank,count=len(rows),shape=[len(rows),FRAMES,DIM],dtype='<f4')
        target=LOCAL/f'rank{rank}_last_hidden.partial.npy'
        assert not target.exists()
        array=np.lib.format.open_memmap(target,mode='w+',dtype='<f4',shape=(len(rows),FRAMES,DIM))
        raw_hash=hashlib.sha256();start=time.monotonic()
        for i in range(len(rows)):
            header=read_header(channel);row=rows.iloc[i]
            assert header['kind']=='data' and header['rank']==rank and header['row']==i
            assert header['global_row']==int(row.global_row) and header['utterance_id']==row.utterance_id
            assert header['bytes']==ROW_BYTES
            payload=read_exact(channel,ROW_BYTES)
            if hashlib.sha256(payload).hexdigest()!=header['sha256']:raise ValueError('Feature block hash mismatch')
            values=np.frombuffer(payload,dtype='<f4').reshape(FRAMES,DIM)
            assert np.isfinite(values).all()
            raw_hash.update(payload);array[i]=values
            if (i+1)%128==0 or i+1==len(rows):
                array.flush()
                with LOCK:PROGRESS[str(rank)]=dict(completed=i+1,total=len(rows),elapsed_seconds=time.monotonic()-start)
        end=read_header(channel)
        assert end==dict(kind='end',rank=rank,count=len(rows),raw_sha256=raw_hash.hexdigest())
        array.flush();del array
        assert channel.recv_exit_status()==0
        report=dict(status='received',rank=rank,count=len(rows),file=target.name,
            raw_sha256=raw_hash.hexdigest(),elapsed_seconds=time.monotonic()-start)
        save(LOCAL/f'rank{rank}_received.json',report)
        with LOCK:PROGRESS[str(rank)]=dict(completed=len(rows),total=len(rows),status='received')
    except Exception as error:
        with LOCK:PROGRESS[str(rank)]=dict(status='failed',error=str(error))
        STOP.set()
    finally:
        if client is not None:client.close()


def fetch_metadata(client):
    names=['request.json','input_audit.json','index.parquet','prepare.exit','qc.exit','qc/summary.json',
        'extract.exit','run.exit','extract/summary.json','extract/rank0_complete.json','extract/rank1_complete.json']
    for name in names:
        body=read_remote(client,ROOT+'/'+name)
        if body is not None:
            target=LOCAL/'metadata'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(body)


def verify_local_files(index, server_summary):
    reports=[]
    for rank in range(2):
        status_file('verifying_local_file',verifying_rank=rank)
        received=json.loads((LOCAL/f'rank{rank}_received.json').read_text())
        expected=server_summary['rank_reports'][rank]
        assert received['raw_sha256']==expected['raw_sha256']
        path=LOCAL/received['file'];file_hash=hashlib.sha256();raw_hash=hashlib.sha256()
        with path.open('rb') as stream:
            version=np.lib.format.read_magic(stream)
            if version==(1,0):shape,order,dtype=np.lib.format.read_array_header_1_0(stream)
            elif version==(2,0):shape,order,dtype=np.lib.format.read_array_header_2_0(stream)
            else:raise ValueError('Unsupported numpy header')
            assert shape==(COUNT//2,FRAMES,DIM) and not order and dtype==np.dtype('<f4')
            offset=stream.tell();stream.seek(0);file_hash.update(stream.read(offset));nbytes=0
            while block:=stream.read(8*1024*1024):
                file_hash.update(block);raw_hash.update(block);nbytes+=len(block)
                assert np.isfinite(np.frombuffer(block,dtype='<f4')).all()
        assert nbytes==COUNT//2*ROW_BYTES and raw_hash.hexdigest()==expected['raw_sha256']
        rows=index[index['rank']==rank]
        assert rows.feature_row.tolist()==list(range(len(rows))) and len(rows)==shape[0]
        reports.append(dict(rank=rank,file=f'rank{rank}_last_hidden.npy',shape=list(shape),dtype='float32',
            file_bytes=path.stat().st_size,sha256=file_hash.hexdigest(),raw_sha256=raw_hash.hexdigest()))
    for rank in range(2):
        source=LOCAL/f'rank{rank}_last_hidden.partial.npy';destination=LOCAL/f'rank{rank}_last_hidden.npy'
        assert not destination.exists();source.replace(destination)
    return reports


def watch():
    with (LOCAL/'receiver.lock').open('x') as f:f.write(str(os.getpid()))
    request=json.loads((LOCAL/'request.json').read_text());client=None;threads=[]
    try:
        if os.name=='nt':
            import ctypes
            # Keep the system awake only while this receiver is alive; the
            # display may sleep and no persistent power setting is changed.
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        status_file('waiting_for_existing_R23')
        while True:
            if shutil.disk_usage(LOCAL).free < COUNT*ROW_BYTES+5*1024**3:raise OSError('F: lacks space plus reserve')
            try:
                client=connect();rc=read_remote(client,UPSTREAM+'/run.exit')
                if rc is None:
                    status_file('waiting_for_existing_R23');client.close();client=None;time.sleep(15);continue
                assert rc.strip()==b'0', 'Existing R23 failed; no GPU submission'
                assert read_remote(client,UPSTREAM+'/21LA_pilot.exit').strip()==b'0'
                assert command(client,'readlink /proc/352729/ns/mnt')==request['expected_container_namespace']
                assert command(client,'test -f /proc/352729/root/.dockerenv && echo yes')=='yes'
                idle=command(client,'ps -p 352729 -o pgid=,tpgid=').split()==['352729','352729']
                lines=command(client,'nvidia-smi -i 6,7 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits').splitlines()
                available=len(lines)==2 and all(int(x.split(',')[0])<100 and int(x.split(',')[1])==0 for x in lines)
                if idle and available:break
                status_file('waiting_for_idle_container_and_both_GPUs');client.close();client=None;time.sleep(15)
            except (OSError,EOFError) as error:
                if client is not None:client.close();client=None
                status_file('waiting_for_server_connection',detail=str(error));time.sleep(15)
        assert read_remote(client,ROOT+'/run.exit') is None and not (LOCAL/'launched.json').exists()
        assert command(client,'ps -p 352729 -o pgid=,tpgid=').split()==['352729','352729']
        save(LOCAL/'launched.json',dict(submitted_epoch=time.time(),screen='352153.exp:0'))
        command(client,'screen -S 352153.exp -p 0 -X stuff '+shlex.quote('bash '+ROOT+'/run_stream.sh\n'))
        status_file('preparing_server_metadata_and_QC')
        deadline=time.monotonic()+300
        while (rc:=read_remote(client,ROOT+'/prepare.exit')) is None:
            assert time.monotonic()<deadline,'Prepare timeout';time.sleep(3)
        assert rc.strip()==b'0','Prepare failed; inspect server prepare.log'
        fetch_metadata(client)
        audit=json.loads((LOCAL/'metadata/input_audit.json').read_text())
        index_path=LOCAL/'metadata/index.parquet'
        assert hashlib.sha256(index_path.read_bytes()).hexdigest()==audit['index_sha256']
        index=pd.read_parquet(index_path)
        assert len(index)==COUNT and not index.utterance_id.duplicated().any() and 'path' not in index.columns
        for rank in range(2):
            thread=threading.Thread(target=receive_rank,args=(rank,index),daemon=True);thread.start();threads.append(thread)
        while any(thread.is_alive() for thread in threads):
            if STOP.is_set():raise RuntimeError('Stream receiver failed; see rank status')
            rc=read_remote(client,ROOT+'/run.exit')
            if rc is not None and rc.strip()!=b'0':raise RuntimeError('Server extraction failed; no automatic restart')
            status_file('receiving_or_waiting_for_QC');time.sleep(10)
        if STOP.is_set():raise RuntimeError('Stream receiver failed; see rank status')
        status_file('waiting_for_server_final_verification')
        deadline=time.monotonic()+180
        while (rc:=read_remote(client,ROOT+'/run.exit')) is None:
            assert time.monotonic()<deadline,'Server finalization timeout';time.sleep(3)
        assert rc.strip()==b'0'
        fetch_metadata(client)
        summary=json.loads((LOCAL/'metadata/extract/summary.json').read_text())
        assert summary['status']=='pass' and summary['sample_count']==COUNT and summary['server_feature_files'] is False
        reports=verify_local_files(index,summary)
        save(LOCAL/'summary.json',dict(status='complete',dataset='19LA_train',shape=[COUNT,FRAMES,DIM],dtype='float32',
            storage='laptop_only',server_feature_files=False,files=reports,index_file='metadata/index.parquet',
            frontend_state_sha256=summary['frontend_state_sha256'],finished_epoch=time.time()))
        (LOCAL/'complete').write_text('0\n');status_file('complete')
        print('Complete:',LOCAL,flush=True)
    except Exception as error:
        STOP.set();status_file('failed_no_automatic_restart',error=str(error));raise
    finally:
        if client is not None:client.close()
        if os.name=='nt':
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['setup','watch','status']);options=parser.parse_args()
    if options.action=='setup':setup()
    elif options.action=='watch':watch()
    else:print((LOCAL/'status.json').read_text())
