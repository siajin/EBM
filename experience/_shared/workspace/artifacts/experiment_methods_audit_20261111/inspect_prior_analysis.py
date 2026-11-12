"""Read-only discovery of historical analysis folders; no remote execution."""
import json
import stat
from datetime import datetime
from pathlib import Path
import paramiko

OUT=Path(__file__).parent
client=paramiko.SSHClient()
client.load_system_host_keys()
password=Path('C:/Users/sihwa/.ssh/mentee-password.txt').read_text().splitlines()[0]
client.connect('172.16.162.222',username='mentee',password=password,look_for_keys=False,allow_agent=False,timeout=15)
del password
sftp=client.open_sftp()
result={'observed_at':datetime.now().astimezone().isoformat(),'remote_mutations':False,'directories':{},'historical_summary_receipts':{}}
try:
    for parent in ['/home/mentee/results/analysis','/home/mentee/analysis']:
        result['directories'][parent]=sorted(x.filename for x in sftp.listdir_attr(parent) if stat.S_ISDIR(x.st_mode))
    for parent in ['/home/mentee/results/analysis/ASV_quiet_BPL_distribution_20261030_v1','/home/mentee/results/analysis/EBM_padding_length_20261030_v1']:
        try:
            names=sftp.listdir(parent)
            item={'files':sorted(names),'summaries':{}}
            for name in names:
                if name.endswith('.json') and any(x in name.lower() for x in ['summary','complete','config','validation','verification']):
                    path=parent+'/'+name
                    if sftp.stat(path).st_size < 300000:
                        with sftp.open(path,'r') as f:
                            raw=f.read().decode('utf-8')
                        item['summaries'][name]=json.loads(raw)
            result['historical_summary_receipts'][parent]=item
        except FileNotFoundError:
            result['historical_summary_receipts'][parent]={'status':'not_found'}
finally:
    sftp.close()
    client.close()
(OUT/'historical_server_analysis_inventory.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'analysis_directories':result['directories'],'historical_summary_files':{k:v.get('files',[]) for k,v in result['historical_summary_receipts'].items()}},ensure_ascii=False))
