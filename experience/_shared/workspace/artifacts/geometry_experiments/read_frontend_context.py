"""Read/sync a completed one-off context test; never starts or restarts it."""
import argparse
import json
from run_padding_causal import OUT as BASE,connect,save,digest
OUT=BASE/'frontend_context'
REMOTE='/home/mentee/analysis/embedding_geometry/20261106_frontend_context_01'

def read(sync=False):
    c=connect();out={}
    try:
        with c.open_sftp() as s:
            for n in ['run.exit','rank0_progress.json','rank1_progress.json','complete.json']:
                try:
                    with s.open(REMOTE+'/'+n) as f:v=f.read().decode()
                    out[n]=json.loads(v) if n.endswith('.json') else v.strip()
                except FileNotFoundError:pass
            if out.get('run.exit') not in [None,'0']:
                with s.open(REMOTE+'/run.log') as f:out['error_tail']=f.read().decode()[-4500:]
            if sync:
                assert out.get('run.exit')=='0' and out['complete.json']['status']=='pass'
                for n,sha in out['complete.json']['files'].items():
                    if not (OUT/n).exists():s.get(REMOTE+'/'+n,str(OUT/n))
                    assert digest(OUT/n)==sha
                if not (OUT/'complete.json').exists():s.get(REMOTE+'/complete.json',str(OUT/'complete.json'))
        save(OUT/'last_status.json',out);print(json.dumps(out,ensure_ascii=False,indent=2),flush=True)
    finally:c.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sync',action='store_true');a=p.parse_args();read(a.sync)
