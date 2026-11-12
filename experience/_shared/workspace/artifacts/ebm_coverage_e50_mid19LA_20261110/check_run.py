"""Read-only progress check and derived-result receipt for the coverage pipeline."""
import csv,datetime,json,pathlib,re
import paramiko
HERE=pathlib.Path(__file__).resolve().parent
state=json.loads((HERE/'state.json').read_text());root=state['root']
client=paramiko.SSHClient();client.load_system_host_keys()
client.connect('172.16.162.222',username='mentee',
    password=pathlib.Path('C:/Users/sihwa/.ssh/mentee-password.txt').read_text().splitlines()[0],
    look_for_keys=False,allow_agent=False,timeout=12)
transfer=client.open_sftp()


def read(name):
    try:return transfer.open(root+'/'+name).read().decode()
    except IOError:return None


try:
    snapshot={'stage':read('stage.txt'),'exits':{},'train_progress':[],
        'checked_KST':datetime.datetime.now(datetime.timezone.utc).astimezone(datetime.timezone(datetime.timedelta(hours=9))).isoformat()}
    for stage in ['unit','preflight','train_gpu_smoke','eval_smoke','train','training_audit','evaluation','pipeline']:
        text=read(stage+'.exit')
        if text is not None:snapshot['exits'][stage]=int(text)
    for rank in range(2):
        text=read(f'train_progress_rank{rank}.json')
        if text:snapshot['train_progress'].append(json.loads(text))
        text=read(f'evaluation/progress_rank{rank}.json')
        if text:snapshot.setdefault('evaluation_progress',[]).append(json.loads(text))
    completed=[]
    names=set(transfer.listdir(root))
    for epoch in range(1,51):
        if all(f'coverage_epoch{epoch}_rank{rank}.json' in names for rank in [0,1]):
            completed.append(epoch)
    snapshot['training_epochs_completed']=max(completed,default=0)
    if completed:
        snapshot['latest_completed_epoch_coverage']=json.loads(read(f'coverage_epoch{max(completed)}_rank0.json'))
    for name in ['training_started.txt','training_finished.txt','finished.txt','failed.txt']:
        text=read(name)
        if text:snapshot[name]=text.strip()
    stage=(snapshot['stage'] or '').strip()
    if stage:
        try:
            with transfer.open(root+'/'+stage+'.log') as stream:
                size=stream.stat().st_size;stream.seek(max(0,size-12000));tail=stream.read().decode(errors='replace')
            snapshot['current_log_error_markers']=re.findall(r'Traceback|CUDA out of memory|RuntimeError|AssertionError|NCCL error',tail)
            if stage=='train':
                rendered=re.findall(r'epoch(\d+):[^\r\n]*?\|\s*(\d+)/108',tail)
                if rendered:
                    epoch,batch=map(int,rendered[-1])
                    snapshot['latest_rendered_training_progress']={'epoch':epoch,'batch':batch,'batches':108,
                        'total_logical_batches_completed':(epoch-1)*108+batch,
                        'training_percent':100*((epoch-1)*108+batch)/5400}
            (HERE/'latest_log_tail.txt').write_text(tail,encoding='utf-8')
        except IOError:pass
    _,out,err=client.exec_command('nvidia-smi -i 6,7 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits')
    snapshot['gpu']=out.read().decode().strip()
    available=['preflight.json','train_gpu_smoke_rank0.json','train_gpu_smoke_rank1.json','eval_smoke_complete.json',
        'final_training_audit.json','train_done_rank0.json','train_done_rank1.json',
        'comparison.json','comparison.csv','comparison_by_length.csv','comparison_complete.json']
    for name in available:
        text=read(name)
        if text:(HERE/name).write_text(text,encoding='utf-8')
    if (HERE/'comparison.json').exists():
        complete=json.loads((HERE/'comparison.json').read_text())
        phases=state.get('comparison_primary_phases',['19LA','21LA','21DF'])
        primary={**complete,
            'metrics':[row for row in complete['metrics'] if row['phase'] in phases],
            'EER_changes':[row for row in complete['EER_changes'] if row['phase'] in phases],
            'phase_trial_counts':{phase:complete['phase_trial_counts'][phase] for phase in phases},
            'dev_role':'Auxiliary monitoring/operating threshold only; excluded from primary comparison.'}
        (HERE/'primary_comparison.json').write_text(json.dumps(primary,indent=2))
        with (HERE/'primary_comparison.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(primary['EER_changes'][0]))
            writer.writeheader();writer.writerows(primary['EER_changes'])
    if state.get('user_stopped_to_change_intermediate_evaluation'):
        state['status']='superseded_by_user_authorized_19LA_restart'
    elif state.get('user_requested_immediate_stop'):
        state['status']='user_stopped_due_to_high_EER'
    elif any(value for value in snapshot['exits'].values()):state['status']='failed'
    elif snapshot['exits'].get('pipeline')==0:state['status']='complete'
    else:state['status']=stage+'_running'
    state.update(training_complete=snapshot['exits'].get('train')==0,
        evaluation_complete=snapshot['exits'].get('evaluation')==0,
        comparison_complete=(HERE/'comparison_complete.json').exists(),last_check=snapshot)
    (HERE/'last_check.json').write_text(json.dumps(snapshot,indent=2))
    (HERE/'state.json').write_text(json.dumps(state,indent=2))
    print(json.dumps(snapshot,indent=2))
finally:transfer.close();client.close()
