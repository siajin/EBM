"""Copy only aggregate historical analysis JSON receipts; no audio/features/models."""
import json
import stat
from pathlib import Path
import paramiko

OUT=Path(__file__).parent
client=paramiko.SSHClient();client.load_system_host_keys()
password=Path('C:/Users/sihwa/.ssh/mentee-password.txt').read_text().splitlines()[0]
client.connect('172.16.162.222',username='mentee',password=password,look_for_keys=False,allow_agent=False,timeout=15)
del password
sftp=client.open_sftp()
base='/home/mentee/results/analysis'
result={}
try:
    for parent_item in sftp.listdir_attr(base):
        if not stat.S_ISDIR(parent_item.st_mode):
            continue
        directory=parent_item.filename
        path=base+'/'+directory
        if not directory.startswith(('EBM80','ASV_quiet','EBM_padding')):
            continue
        item={'remote_root':path,'file_inventory':[],'aggregate_json':{},'method_documents':{}}
        def visit(current, prefix='', depth=0):
            for f in sftp.listdir_attr(current):
                name=prefix+f.filename
                item['file_inventory'].append({'name':name,'bytes':f.st_size,'directory':stat.S_ISDIR(f.st_mode)})
                if 'private' in name.lower():
                    continue
                if stat.S_ISDIR(f.st_mode):
                    if depth<2:
                        visit(current+'/'+f.filename,name+'/',depth+1)
                    continue
                if f.filename.endswith('.json') and 'manifest' not in f.filename.lower() and f.st_size<1000000:
                    with sftp.open(current+'/'+f.filename,'r') as fh:
                        raw=fh.read().decode('utf-8')
                    item['aggregate_json'][name]=json.loads(raw)
                elif f.filename.endswith(('.md','.txt')) and f.st_size<30000 and any(x in name.lower() for x in ['report','interpretation','readme','exit']):
                    with sftp.open(current+'/'+f.filename,'r') as fh:
                        item['method_documents'][name]=fh.read().decode('utf-8')
        visit(path)
        result[directory]=item
finally:
    sftp.close();client.close()
(OUT/'historical_aggregate_receipts.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
for name,item in sorted(result.items()):
    print(name, {k:list(v)[:14] if isinstance(v,dict) else type(v).__name__ for k,v in item['aggregate_json'].items()})
