"""Allowlisted transfer and offline access to completed R2/R3 derived vectors."""
from pathlib import Path
import json
import paramiko
import numpy as np
import pandas as pd
from frozen_common import DEFAULT_ROOT,CachedEmbeddings,CHECKPOINT,save,digest

REMOTE='/home/mentee/analysis/embedding_geometry/20261105_frozen_all_01/R23'
CATALOG=DEFAULT_ROOT/'r23_catalog.json'
INVENTORY=r'''
from pathlib import Path
import json,hashlib
root=Path('/home/mentee/analysis/embedding_geometry/20261105_frozen_all_01/R23')
conditions=['R2-00','R2-10','R2-01','R2-11','R3-noise20','R3-noise10','R3-Opus32k_recompression']
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8388608),b''):h.update(chunk)
 return h.hexdigest()
entries=[]
for name in ['DF_pilot','21LA_pilot']:
 end=root/(name+'.exit')
 if not end.exists() or end.read_text().strip()!='0':continue
 common=['protocol_lock.json','preprocessing_QC.json',name+'/rank0_complete.json',name+'/rank1_complete.json']
 for cond in conditions:
  folder=root/name/cond
  assert (folder/'complete').read_text().strip()=='0'
  summary=json.loads((folder/'summary.json').read_text());assert summary['status']=='pass' and summary['rows']==20000
  names=common+[name+'/'+cond+'/'+n for n in ['summary.json','scores.parquet']+[f'rank{r}_{s}' for r in [0,1] for s in ['z.npy','h.npy','centroid.npy','rows.parquet']]]
  files=[]
  for n in names:
   p=root/n;assert p.is_file() and not p.is_symlink();p.resolve().relative_to(root.resolve())
   files.append(dict(path=n,bytes=p.stat().st_size,sha256=digest(p)))
  entries.append(dict(dataset=name,condition=cond,fingerprint=summary['fingerprint'],files=files,remote_root=str(root)))
print(json.dumps(entries))
'''

def sync():
    client=paramiko.SSHClient();client.load_system_host_keys();downloads=0
    try:
        client.connect('172.16.162.222',username='mentee',password=Path('C:/Users/sihwa/.ssh/mentee-password.txt').read_text().splitlines()[0],look_for_keys=False,allow_agent=False,timeout=15)
        stdin,out,err=client.exec_command('python3 -u -',timeout=30);stdin.write(INVENTORY);stdin.channel.shutdown_write();payload=out.read().decode();error=err.read().decode();assert out.channel.recv_exit_status()==0,error
        entries=json.loads(payload);catalog=json.loads(CATALOG.read_text()) if CATALOG.exists() else {'entries':{}}
        with client.open_sftp() as s:
            for entry in entries:
                key=entry['dataset']+'::'+entry['condition'];folder=DEFAULT_ROOT/'runs/20261105_frozen_all_01/R23'/entry['fingerprint'][:16];folder.mkdir(parents=True,exist_ok=True)
                for record in entry['files']:
                    relative=Path(record['path']);assert not relative.is_absolute() and '..' not in relative.parts;target=folder/relative;target.resolve().relative_to(DEFAULT_ROOT.resolve())
                    if target.exists():assert digest(target)==record['sha256'] and target.stat().st_size==record['bytes'];continue
                    target.parent.mkdir(parents=True,exist_ok=True);temp=target.with_suffix(target.suffix+'.part');s.get(REMOTE+'/'+record['path'],str(temp));assert digest(temp)==record['sha256'] and temp.stat().st_size==record['bytes'];temp.replace(target);downloads+=1
                catalog['entries'][key]=dict(entry,local_folder=str(folder.relative_to(DEFAULT_ROOT)).replace('\\','/'))
        save(CATALOG,catalog)
    finally:client.close()
    print('R23 downloaded files',downloads,flush=True)
    return catalog

def load(name,condition):
    catalog=json.loads(CATALOG.read_text());entry=catalog['entries'][name+'::'+condition];folder=DEFAULT_ROOT/entry['local_folder'];path=folder/name/condition
    rows=pd.read_parquet(path/'scores.parquet');z=np.concatenate([np.load(path/f'rank{r}_z.npy') for r in [0,1]]);h=np.concatenate([np.load(path/f'rank{r}_h.npy') for r in [0,1]]);c=np.load(path/'rank0_centroid.npy')
    summary=json.loads((path/'summary.json').read_text());assert summary['identity']['checkpoint_sha256']==CHECKPOINT
    assert len(rows)==len(z)==len(h)==20000 and not rows.utterance_id.duplicated().any()
    score=z.astype(float)/np.linalg.norm(z.astype(float),axis=1,keepdims=True)@c;assert np.max(np.abs(score-rows.cosine_s))<1e-10
    return CachedEmbeddings(z,h,rows,c,entry)

if __name__=='__main__':sync()
