"""Verify completed evaluation records and compare matched existing EERs."""
import argparse,csv,hashlib,json,math,shutil
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def independent_eer(labels,scores):
    rows=sorted(zip(scores,labels),reverse=True);P=sum(labels);N=len(labels)-P
    assert P>0 and N>0;tp=fp=0;i=0;x0=0.;d0=-1.
    while i<len(rows):
        j=i
        while j<len(rows) and rows[j][0]==rows[i][0]:tp+=rows[j][1];fp+=1-rows[j][1];j+=1
        x=fp/N;d=x+tp/P-1
        if d>=0:return 100*(x0-d0/(d-d0)*(x-x0))
        x0,d0=x,d;i=j
    raise AssertionError('ROC crossing absent')

def prepare_shared(root,result):
    root=Path(root);result=Path(result)
    training=json.loads((result/'stages/training_complete.json').read_text())
    assert training['status']=='complete' and training['saved_epochs']==list(range(5,51,5))
    assert all(r['phase']=='19LA' and r['status']=='complete' for r in training['results']) and len(training['results'])==10
    dest=root/'shared_eval';dest.mkdir(exist_ok=False)
    shutil.copyfile(root/'shared_eval.py',dest/'shared_eval.py')
    experiment=json.loads((root/'experiment.json').read_text());preserved=[];hashes={};role=None
    for epoch in range(5,51,5):
        cp=json.loads((result/'stages'/f'checkpoint_{epoch}.complete.json').read_text());assert cp['status']=='committed'
        for name,record in cp['files'].items():assert sha(result/'model'/name)==record['sha256']
        hashes[str(epoch)]=cp['files'][f'CLS_{epoch}.pt']['sha256']
        current=json.loads((result/'model'/f'Model_{epoch}.json').read_text())['role']
        assert current['schema_version']=='ebm_role_v4' and current['bona_bpl_pairing']=='same_uid'
        if role is None:role=current
        else:assert role==current
        record=result/'stages'/f'eval_{epoch}_19LA.json';scores=result/'stages'/f'eval_{epoch}_19LA_scores.csv'
        m=json.loads(record.read_text());assert m['status']=='complete' and m['rows']==71237 and m['scope']=='full'
        preserved.append(dict(phase='19LA',epoch=epoch,record_path=str(record),record_sha256=sha(record),score_path=str(scores),score_sha256=sha(scores)))
    cfg=dict(source=experiment['source_sha256'],source_root=str(root/'code'),result_root=str(result),expected_role=role,
             checkpoint_sha256=hashes,preserved=preserved,pending={p:['e'+str(e) for e in range(5,51,5)] for p in ['21LA','21DF']},
             frontend_revision='1a640f32ac3e39899438a2931f9924c02f080a54',controls={'shared_eval.py':sha(dest/'shared_eval.py')},
             spending_ceiling_usd=0,automatic_retries=0)
    (dest/'manifest.json').write_text(json.dumps(cfg,indent=2))
    controls=dict(cfg['controls'],**{'manifest.json':sha(dest/'manifest.json')})
    (dest/'control_manifest.json').write_text(json.dumps(controls,indent=2))
    print(json.dumps(dict(status='prepared_shared_evaluation',models=10,preserved_19LA=10,root=str(dest))))

def complete(root,result):
    root=Path(root);result=Path(result);base=json.loads((root/'baseline_metrics.json').read_text())
    summary=json.loads((root/'shared_eval/full/complete.json').read_text());assert summary['status']=='complete' and len(summary['results'])==30
    assert (root/'shared_eval/run.exit').read_text().strip()=='0'
    prior={(r['phase'],r['epoch']):r for r in base['rows']};rows=[];checks=[]
    counts={'19LA':71237,'21LA':148176,'21DF':533928}
    for phase,count in counts.items():
        for epoch in range(5,51,5):
            if phase=='19LA':
                name=f'eval_{epoch}_{phase}';meta=json.loads((result/'stages'/f'{name}.json').read_text())
                assert meta['status']=='complete' and meta['scope']=='full' and meta['rows']==count
                path=result/'stages'/f'{name}_scores.csv'
                with path.open() as stream:table=list(csv.DictReader(stream))
                assert len(table)==count and [int(r['index']) for r in table]==list(range(count)) and len({r['uid'] for r in table})==count
                labels=[int(r['label']) for r in table];scores=[float(r['score_bona']) for r in table]
                assert all(math.isfinite(s) for s in scores)
                error=abs(independent_eer(labels,scores)-meta['eer_percent']);assert error<1e-9
                eer=meta['eer_percent'];receipt=dict(score_sha256=sha(path))
            else:
                meta=json.loads((root/'shared_eval/full'/f'metrics_{phase}.json').read_text())
                assert meta['status']=='complete' and meta['scope']=='full' and meta['rows']==count
                cell=meta['metrics']['e'+str(epoch)];eer=cell['EER_percent'];error=cell['independent_error_pp'];assert error<1e-9
                for name,item in meta['score_files'].items():assert sha(root/'shared_eval/full'/name)==item['sha256']
                receipt=dict(score_files=meta['score_files'])
            old=prior[phase,epoch];row=dict(old,bona_uid=eer)
            checks.append(dict(phase=phase,epoch=epoch,independent_error_pp=error,**receipt))
            row['uid_minus_allcross_pp']=row['bona_uid']-row['bona_allcross'];rows.append(row)
    models=['legacy240','spoof_only','bona_allcross','bona_uid']
    minima={}
    for phase in counts:
        minima[phase]={}
        for model in models:
            available=[r for r in rows if r['phase']==phase and r[model] is not None]
            common=[r for r in available if r['epoch']>=15]
            minima[phase][model]=dict(full_available=min(available,key=lambda r:(r[model],r['epoch'])),
                                      common15_50=min(common,key=lambda r:(r[model],r['epoch'])),candidates=len(available),common_candidates=len(common))
    out=dict(status='complete_verified',protocol=base['protocol'],rows=rows,minima=minima,checks=checks,
             limits=['Single seed','Retrospective minima may have different epochs','Use common15..50 grid for minimum comparisons'],
             during_training=['19LA'],after_training=['21LA','21DF'],feature_reuse='One frozen frontend call per batch shared by10 saved heads after training; no feature files')
    (root/'comparison.json').write_text(json.dumps(out,indent=2,ensure_ascii=False))
    text='# 정상 same UID BPL 비교 결과\n\nEER %, 낮을수록 좋음. 동일 epoch·동일 wrap 입력 비교.\n\n| 집합 | epoch | 기존240 | 스푸프만 | 정상 all-cross | 정상 same UID | UID−allcross %p |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for row in rows:
        values=['—' if row[k] is None else f'{row[k]:.6f}' for k in models]
        text+='| '+row['phase']+' | '+str(row['epoch'])+' | '+' | '.join(values)+f" | {row['uid_minus_allcross_pp']:+.6f} |\n"
    (root/'comparison.md').write_text(text)
    final=dict(summary,stage='complete',saved_epochs=list(range(5,51,5)),comparison=str(root/'comparison.json'))
    (result/'stages/complete.json').write_text(json.dumps(final,indent=2))
    print(json.dumps(dict(status='complete_verified',cells=len(rows),max_independent_eer_error=max(r['independent_error_pp'] for r in checks))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare_shared','compare']);p.add_argument('root');p.add_argument('result');args=p.parse_args()
    (prepare_shared if args.mode=='prepare_shared' else complete)(args.root,args.result)
