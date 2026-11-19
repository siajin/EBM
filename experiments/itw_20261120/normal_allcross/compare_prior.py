"""Read completed scores only; compare new family with two previous ITW families."""
import csv,hashlib,json,sys
from pathlib import Path
r=Path(sys.argv[1]);prior=json.loads((r/'prior_preservation.json').read_text());old=Path(prior['root'])
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
for n,h in prior['sha256'].items():assert sha(old/n)==h,n
new=json.loads((r/'full/complete.json').read_text());previous=json.loads((old/'full/complete.json').read_text())
assert new['status']==previous['status']=='complete' and new['rows']==previous['rows']==31779
assert new['models']==10 and previous['models']==18
assert new['protocol']==previous['protocol'] and new['bonafide']==previous['bonafide']==19963
assert new['frontend_calls']==new['batches']==3974
assert all(m['independent_error_pp']<1e-9 for m in new['metrics'].values())
for n,h in new['score_files'].items():assert sha(r/'full'/n)==h,n
for rank in [0,1]:
 with (r/'full'/f'ITW_rank{rank}.csv').open() as a,(old/'full'/f'ITW_rank{rank}.csv').open() as b:
  first=list(csv.DictReader(a));second=list(csv.DictReader(b))
 assert len(first)==len(second)
 assert all((x['index'],x['uid'],x['label'])==(y['index'],y['uid'],y['label']) for x,y in zip(first,second))
families=dict(previous['families']);assert not set(families)&set(new['families']);families.update(new['families'])
metrics=dict(previous['metrics']);assert not set(metrics)&set(new['metrics']);metrics.update(new['metrics'])
assert len(families)==3 and len(metrics)==28
names={'EBM_e50_bs240_lr1e5':'기존 240','EBM_e50_bs240_lr1e5_spoofUID_bonaAllcrossBPL_pair1_pull09_wrapeval_20261119_v1':'스푸프 같은 UID + 정상 전체 교차 BPL','EBM_e50_bs240_lr1e5_spoof_dualBPL_pair1_pull09_wrapeval_20261116_v1':'스푸프 전체 교차 + 정상 전체 교차 BPL (1.0/0.9)'}
lines=['# ITW 세 모델 비교','', '31,779개 동일 음성, wrap64600/no-mask/batch8/동결 XLSR/공통 matched_wrap1729/ASP1e-5. EER 단위는 %다.','', '| 모델 | 후보 수 | 최저 EER | epoch | 50epoch EER |','|---|---:|---:|---:|---:|']
for k in names:
 v=families[k];lines.append(f"| {names[k]} | {v['candidates']} | {v['minimum']['eer_percent']:.4f}% | {v['minimum']['epoch']} | {v['epoch50']['eer_percent']:.4f}% |")
lines+=['','최저값은 ITW에서 사후 선택한 관측 최소로 독립적인 checkpoint 선택 성능이 아니다. 기존240은15~50epoch8개, 나머지는5~50epoch10개다.','', '## 동일 epoch 전체 비교','', '| epoch | 기존 240 | 스푸프 같은 UID + 정상 전체 교차 | 스푸프 전체 교차 + 정상 전체 교차 |','|---|---:|---:|---:|']
for e in range(5,51,5):
 cells=[]
 for k in names:
  v=next((v['eer_percent'] for v in metrics.values() if v['family']==k and v['epoch']==e),None);cells.append('미보존' if v is None else f'{v:.4f}%')
 lines.append('| '+str(e)+' | '+' | '.join(cells)+' |')
lines+=['','공통15~50epoch에서의 최소값/epoch도 comparison.json에 포함했다. 이전 결과와 UID·index·label 순서를 전수 대조했고 이전 보호파일 SHA 불변을 확인했다.','']
(r/'comparison.md').write_text('\n'.join(lines))
(r/'comparison.json').write_text(json.dumps(dict(status='complete_verified',families=families,metrics=metrics,rows=31779,models=28,prior_files_unchanged=True,UID_index_label_exact_match=True,protocol=new['protocol'],new_spending_usd=0,automatic_retries=0),indent=2))
print(json.dumps(dict(status='complete_verified',families=3,models=28,added_models=10)))
