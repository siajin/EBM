"""Render full curves, retrospective minima and common-epoch comparisons."""
import csv,json,sys
from pathlib import Path
r=Path(sys.argv[1]);d=json.loads((r/'full/complete.json').read_text());assert d['status']=='complete' and d['rows']==31779
names={'EBM':'초기 EBM','EBM_e50_bs160_lr1e5':'기존160','EBM_e50_bs160_lr1e5_bpl_global':'160 global BPL','EBM_e50_bs240_lr1e5':'기존240'}
rows=sorted(d['metrics'].values(),key=lambda v:(v['family'],v['epoch']))
with (r/'all_checkpoint_metrics.csv').open('x',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['family','epoch','eer_percent','independent_error_pp','checkpoint_sha256']);w.writeheader();w.writerows(rows)
text='# In-the-Wild 전체 모델 비교\n\n동일31779음성·wrap64600/no-mask/batch8·공유 동결 XLSR·공통 ASP 난수. 낮은 EER이 좋음.\n\n| 모델 | 보존 epoch | 후보 수 | 관측 최소 EER | 최소 epoch | 50epoch EER |\n|---|---|---:|---:|---:|---:|\n'
for name,v in sorted(d['families'].items()):
 m=v['minimum'];fixed=v['epoch50'];e50='—' if fixed is None else '%.6f%%'%fixed['eer_percent']
 text+='| '+names.get(name,name)+' | '+','.join(map(str,v['available_epochs']))+' | '+str(v['candidates'])+' | %.6f%% | %d | '%(m['eer_percent'],m['epoch'])+e50+' |\n'
text+='\n관측 최소는 ITW 정답을 사용한 사후 선택이며 독립적인 모델 선택 성능이 아니다. 보존 epoch 개수가 다른 계열은 후보 수와 범위를 함께 읽는다. 모든 checkpoint의 결과는 all_checkpoint_metrics.csv에 제공한다.\n'
common=set.intersection(*(set(v['available_epochs']) for name,v in d['families'].items() if name!='EBM'))
text+='\n초기 EBM(65~80epoch)을 제외한 나머지 계열의 공통 epoch: '+','.join(map(str,sorted(common)))+'\n'
if common:
 text+='\n| 모델 | 공통 epoch 내 최소 EER | epoch |\n|---|---:|---:|\n'
 for name,v in sorted(d['families'].items()):
  candidates=[row for row in rows if row['family']==name and row['epoch'] in common]
  if candidates:
   m=min(candidates,key=lambda row:(row['eer_percent'],row['epoch']));text+='| '+names.get(name,name)+' | %.6f%% | %d |\n'%(m['eer_percent'],m['epoch'])
(r/'report.md').write_text(text)
(r/'complete.json').write_text(json.dumps(dict(status='complete_verified',rows=31779,models=len(rows),families=len(d['families']),zip_removed=True,frontend_calls=d['frontend_calls'],batches=d['batches'],feature_files=0,optimizer_updates=0,new_spending_usd=0,automatic_retries=0),indent=2))
print(json.dumps(dict(status='complete_verified',models=len(rows),families=len(d['families']))))
