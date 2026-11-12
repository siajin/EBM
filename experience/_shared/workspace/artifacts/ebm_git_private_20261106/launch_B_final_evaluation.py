"""Prepare a provenance record, then smoke-test and evaluate once on the server."""
import json
from pathlib import Path
import shlex
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'geometry_experiments'))
from run_padding_causal import connect

folder=Path(__file__).resolve().parent
statefile=folder/'B_final_evaluation_state.json'
assert not statefile.exists(),'An evaluation state already exists; inspect instead of relaunching'
def run(c,cmd):
    _,o,e=c.exec_command(cmd,timeout=45);s=o.read().decode(errors='replace');err=e.read().decode(errors='replace')
    assert o.channel.recv_exit_status()==0,err[-1500:]
    return s.strip()

c=connect()
try:
    assert run(c,'ps -p 352729 -o pgid=,tpgid=').split()==['352729','352729']
    assert run(c,'test -f /proc/352729/root/.dockerenv && echo yes')=='yes'
    assert run(c,'cat /home/mentee/analysis/ebm_train_B_20261106_213627/run.exit')=='0'
    gpu=run(c,'nvidia-smi -i 6,7 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits')
    assert all(int(x.split(',')[0])<100 and int(x.split(',')[1])==0 for x in gpu.splitlines())
    assert not run(c,'git -C /home/mentee/code/EBM status --porcelain')
    assert not run(c,"find /home/mentee/analysis -maxdepth 1 -type d -name 'ebm_B_final_eval_*'")
    root='/home/mentee/analysis/ebm_B_final_eval_'+run(c,'date +%Y%m%d_%H%M%S')
    run(c,'mkdir '+shlex.quote(root))
    audit=r'''
import datetime,hashlib,json,re,subprocess
from pathlib import Path
root=Path(ROOT)
train=Path('/home/mentee/analysis/ebm_train_B_20261106_213627')
b=Path('/home/mentee/results/CM/LA/EBM_e50_bs240_lr1e5_B_zeropad_short303')
old=Path('/home/mentee/results/CM/LA/EBM_e50_bs240_lr1e5')
text=(train/'train.log').read_text(errors='replace')
records={}
pattern=r'epoch([0-9]+): loss1=([-0-9.]+) loss2=([-0-9.]+).*?([0-9]+)/108'
for match in re.finditer(pattern,text):
 e,a,c,n=match.groups();e,n=int(e),int(n)
 if 1<=e<=50 and 1<=n<=108:records.setdefault((e,n),{'epoch':e,'batch':n,'loss1_rounded':float(a),'loss2_rounded':float(c)})
assert set(records)=={(e,n) for e in range(1,51) for n in range(1,109)},len(records)
checkpoint_paths={'B_CLS50':b/'model/CLS_50.pt','B_TrainState50':b/'model/TrainState_50.pt','legacy240_CLS50':old/'model/BestCLS_50_21DF.pt'}
checkpoints={k:{'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for k,p in checkpoint_paths.items()}
metric_counts={}
for p in (b/'metric').rglob('*.txt'):metric_counts[str(p.relative_to(b/'metric'))]=len(p.read_text().splitlines())
start=datetime.datetime.fromisoformat((train/'started.txt').read_text().strip())
end=datetime.datetime.fromisoformat((train/'finished.txt').read_text().strip())
info={'status':'training_complete_evaluation_preparing','training_run_exit':int((train/'run.exit').read_text()),'epochs':50,'logical_batches':len(records),'start':start.isoformat(),'finish':end.isoformat(),'training_elapsed_seconds':(end-start).total_seconds(),'checkpoints':checkpoints,'metric_line_counts':metric_counts,'dev_EER_logged':(b/'metric/V/EER.txt').read_text().splitlines(),'missing_final_dev_log_recovered_by_independent_eval':True,'historical_legacy_EER':{phase:(old/'metric'/phase/'EER.txt').read_text().splitlines() for phase in ['19LA','21LA','21DF']},'evaluation_protocol':{'input':'B max64600, no wrap, masks','models':'both fixed50 checkpoints','pooling_noise_std':1e-5,'paired_common_pooling_RNG':True,'batch_per_rank':8,'threshold':'dev only, minimum absFPR-FNR then lowerFPR then higherthreshold'}}
(root/'training_audit.json').write_text(json.dumps(info,indent=2)+'\n')
(root/'training_history.json').write_text(json.dumps([records[k] for k in sorted(records)])+'\n')
print(json.dumps({'epochs':50,'verified_logical_batches':len(records),'training_elapsed_seconds':info['training_elapsed_seconds'],'checkpoints':{k:v['sha256'] for k,v in checkpoints.items()},'metric_line_counts':metric_counts,'logged_dev_epochs':len(info['dev_EER_logged'])}))
'''.replace('ROOT',repr(root))
    print(run(c,"python - <<'PY'\n"+audit+"\nPY"),flush=True)
    script='''#!/bin/bash
set -u
ROOT="'''+root+'''"
mkdir "$ROOT/launch.claim" || exit 95
test -f /.dockerenv || { echo 90 > "$ROOT/run.exit"; exit 90; }
export CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd /home/mentee/code/EBM || exit 91
date -Iseconds > "$ROOT/started.txt"
python -u "$ROOT/evaluate_B_final.py" "$ROOT" --smoke > "$ROOT/smoke.log" 2>&1
rc=$?; echo "$rc" > "$ROOT/smoke.exit"
if [ "$rc" -ne 0 ]; then echo "$rc" > "$ROOT/run.exit"; exit "$rc"; fi
python -u "$ROOT/evaluate_B_final.py" "$ROOT" > "$ROOT/eval.log" 2>&1 &
pid=$!; echo "$pid" > "$ROOT/parent_pid.txt"
wait "$pid"
rc=$?; echo "$rc" > "$ROOT/run.exit"
date -Iseconds > "$ROOT/finished.txt"
exit "$rc"
'''
    with c.open_sftp() as s:
        s.put(str(folder/'evaluate_B_final.py'),root+'/evaluate_B_final.py')
        with s.open(root+'/run.sh','w') as f:f.write(script)
        for name in ['training_audit.json','training_history.json']:
            s.get(root+'/'+name,str(folder/('B_final_'+name)))
    state={'stage':'smoke_then_paired_final_evaluation','root':root,'models':['B_CLS50','legacy240_CLS50'],'phases':['dev','19LA','21LA','21DF'],'training_complete':True,'evaluation_complete':False,'analysis_complete':False,'prevent_duplicate_launch':True}
    statefile.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    with c.open_sftp() as s:
        with s.open(root+'/evaluation_state.json','w') as f:f.write(json.dumps(state,indent=2)+'\n')
    assert run(c,'ps -p 352729 -o pgid=,tpgid=').split()==['352729','352729']
    run(c,'screen -S 352153.exp -p 0 -X stuff '+shlex.quote('bash '+root+'/run.sh\n'))
    print(json.dumps(state),flush=True)
finally:c.close()
