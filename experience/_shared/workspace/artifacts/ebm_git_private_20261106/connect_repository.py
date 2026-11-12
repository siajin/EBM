"""Connect server and local checkouts without exporting any private credential."""
import json
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'geometry_experiments'))
from run_padding_causal import connect

REPO='siajin/EBM'
URL='https://github.com/'+REPO+'.git'
CODE='/home/mentee/code/EBM'
REMOTE='/home/mentee/analysis/ebm_private_git_20261106_2125'
SSH='/home/mentee/.ssh'
KEY=SSH+'/ebm_siajin_ed25519'
KNOWN=SSH+'/known_hosts_ebm_siajin'

def gh(*args,stdin=None):
    p=subprocess.run(['gh',*args],input=stdin,text=True,encoding='utf-8',capture_output=True)
    assert p.returncode==0,p.stderr[-1500:]
    return p.stdout

def local_git(*args,cwd=None):
    p=subprocess.run(['git',*args],cwd=cwd,text=True,encoding='utf-8',capture_output=True)
    assert p.returncode==0,p.stderr[-1500:]
    return p.stdout.strip()

def remote(c,command,timeout=40):
    _,out,err=c.exec_command(command,timeout=timeout)
    text=out.read().decode();error=err.read().decode()
    assert out.channel.recv_exit_status()==0,error[-1500:]
    return text.strip()

def python(c,source):return remote(c,"python - <<'PY'\n"+source+"\nPY")

def main():
    info=json.loads(gh('api','repos/'+REPO))
    assert info['private'] and info['owner']['login']=='siajin'
    sshkeys=json.loads(gh('api','meta'))['ssh_keys']
    assert sshkeys and all(k.startswith(('ssh-','ecdsa-')) for k in sshkeys)
    c=connect()
    try:
        # Create a repository-scoped private key on the server; never read it.
        python(c,"""from pathlib import Path
import os,subprocess
folder=Path('/home/mentee/.ssh');folder.mkdir(mode=0o700,exist_ok=True)
key=folder/'ebm_siajin_ed25519'
if not key.exists():
 subprocess.run(['ssh-keygen','-q','-t','ed25519','-N','','-C','siajin EBM repository','-f',str(key)],check=True)
os.chmod(key,0o600)
print('Dedicated key generated on server')
""")
        with c.open_sftp() as s:
            with s.open(KEY+'.pub') as f:public=f.read().decode().strip()
            with s.open(KNOWN,'w') as f:
                f.write('\n'.join(host+' '+key for host in ['github.com','[ssh.github.com]:443'] for key in sshkeys)+'\n')
            s.chmod(KNOWN,0o600)
        existing=json.loads(gh('api','repos/'+REPO+'/keys'))
        matching=[k for k in existing if k['key'].split()[:2]==public.split()[:2]]
        keyinfo=matching[0] if matching else json.loads(gh('api','--method','POST','repos/'+REPO+'/keys','--input','-',
            stdin=json.dumps({'title':'EBM server checkout','key':public,'read_only':False})))
        assert keyinfo['read_only'] is False
        # Existing public history and the user's current code stay in place.
        config="""import json,subprocess
from pathlib import Path
code=Path('/home/mentee/code/EBM')
def git(*args):return subprocess.check_output(['git',*args],cwd=code,text=True).strip()
assert not git('status','--porcelain')
origin=git('remote','get-url','origin')
if origin=='https://github.com/kimho1wq/EBM.git':
 git('remote','rename','origin','upstream')
 git('remote','add','origin','git@github.com:siajin/EBM.git')
else:assert origin=='git@github.com:siajin/EBM.git'
assert git('remote','get-url','upstream')=='https://github.com/kimho1wq/EBM.git'
git('config','core.sshCommand','ssh -p 443 -o HostName=ssh.github.com -i /home/mentee/.ssh/ebm_siajin_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=/home/mentee/.ssh/known_hosts_ebm_siajin -o ConnectTimeout=10')
print(git('rev-parse','HEAD'))
"""
        commit=python(c,config)
        print(json.dumps({'private_repo_verified':True,'server_commit':commit,'deploy_key_id':keyinfo['id']}),flush=True)
        result=remote(c,'git -C '+CODE+' push -u origin main',timeout=50)
        print('Server commit pushed to private origin.',flush=True)
        info=json.loads(gh('api','repos/'+REPO))
        assert info['private'] and info['default_branch']=='main'
        head=json.loads(gh('api','repos/'+REPO+'/commits/main'))['sha']
        assert head==commit
        local=Path.cwd()/'EBM'
        helpers=['-c','credential.helper=','-c','credential.helper=!gh auth git-credential']
        assert not local.exists(),'Local EBM folder already exists; inspect before cloning'
        local_git(*helpers,'clone','--branch','main',URL,str(local))
        local_git('remote','add','upstream','https://github.com/kimho1wq/EBM.git',cwd=local)
        local_git('config','--local','--add','credential.https://github.com.helper','',cwd=local)
        local_git('config','--local','--add','credential.https://github.com.helper','!gh auth git-credential',cwd=local)
        local_git('config','user.name','sia_02',cwd=local)
        local_git('config','user.email','53578618+siajin@users.noreply.github.com',cwd=local)
        assert local_git('rev-parse','HEAD',cwd=local)==commit
        assert not local_git('status','--porcelain',cwd=local)
        assert local_git('rev-parse','@{upstream}',cwd=local)==commit
        # Confirm the server can authenticate and read its own private remote.
        assert remote(c,'git -C '+CODE+' ls-remote origin refs/heads/main').split()[0]==commit
        assert not remote(c,'git -C '+CODE+' status --porcelain')
        report={'repository':'https://github.com/'+REPO,'visibility':'private','commit':commit,
                'local_checkout':str(local),'server_checkout':CODE,'branch':'main',
                'upstream':'https://github.com/kimho1wq/EBM','deploy_key_id':keyinfo['id'],
                'server_private_key_exported':False,'both_worktrees_clean':True,
                'training_started':False,'original_license_and_history_retained':True}
        path=Path(__file__).with_name('connection.json')
        path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        with c.open_sftp() as s:
            with s.open(REMOTE+'/connection.json','w') as f:f.write(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)
    finally:c.close()

if __name__=='__main__':main()
