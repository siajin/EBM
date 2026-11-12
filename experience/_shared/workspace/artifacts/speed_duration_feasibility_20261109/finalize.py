"""Reconcile both algorithms, append second-method evidence and record final state."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import prepare_probe as base

HERE=Path(__file__).resolve().parent
report=base.PROJECT/'docs/experiments/ebm_B_C_all_checkpoint_evaluation_20261109.md'
figdir=report.parent/'figures/speed_duration_feasibility_20261109'
a=pd.read_csv(HERE/'atempo_probe/all_scores.csv')
f=pd.read_csv(HERE/'probe/all_scores.csv')
keys=a[['phase','index']].drop_duplicates()
f=f.merge(keys,on=['phase','index'],validate='many_to_one')
protocol=json.loads((HERE/'probe/protocol.json').read_text())
x=f[f.condition=='native_cap'].set_index(['phase','index']).sort_index()
y=a[a.condition=='native_cap'].set_index(['phase','index']).sort_index()
for head in ['legacy240','B','C']:
    assert np.max(np.abs(x['score_'+head]-y['score_'+head]))==0
rows=[]
for algorithm,frame in [('phase_vocoder',f),('atempo',a)]:
    for head in ['legacy240','B','C']:
        threshold=protocol['dev_thresholds'][head]
        for condition,q in frame.groupby('condition'):
            normal=q[q.label==1];spoof=q[q.label==0]
            rows.append({'algorithm':algorithm,'head':head,'condition':condition,
                         'normal_rejected':int((normal['score_'+head]<threshold).sum()),'normal_N':len(normal),
                         'spoof_accepted':int((spoof['score_'+head]>=threshold).sum()),'spoof_N':len(spoof)})
table=pd.DataFrame(rows);table.to_csv(HERE/'matched_algorithm_comparison.csv',index=False)
conditions=['native_cap','slow_cap','fast_cap','roundtrip_cap','norm_full4']
labels=['Original cap','0.8x slow','1.25x fast','Roundtrip','Full to ~4 s']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(10,4.7));z=np.arange(5);width=.34
for offset,algorithm,name,color in [(-.5,'phase_vocoder','Phase vocoder','#2878a2'),(.5,'atempo','FFmpeg atempo','#d17928')]:
    values=[100*float(table.loc[(table['head']=='B')&(table.algorithm==algorithm)&(table.condition==c),'normal_rejected'].iloc[0])/32 for c in conditions]
    bars=ax.bar(z+offset*width,values,width=width,label=name,color=color);ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=10)
ax.set(xticks=z,xticklabels=labels,ylabel='Genuine rejection (%)',ylim=(0,112),title='Same frozen B, same 32 genuine utterances: algorithm matters')
ax.legend(frameon=False,loc='upper left');fig.text(.1,.01,'Matched 64-utterance diagnostic subset (32 genuine / 32 spoof); fixed full-dev threshold.',fontsize=10)
fig.tight_layout(rect=(0,.04,1,1));fig.savefig(figdir/'matched_algorithm_normal_rejection.png',dpi=170);plt.close(fig)
text=report.read_text(encoding='utf-8')
marker='### 다른 변환 방식으로 재검사한 결과'
assert marker not in text
text=text.replace('이는 재학습 불가능의 증거가 아니라 전처리·손실을 분리해야 한다는 근거다.',
                  '이는 재학습 불가능의 증거가 아니라 전처리·손실을 분리해야 한다는 근거다. 후속 FFmpeg atempo 검사에서는 같은 표본의 완만한 감속이 훨씬 덜 해로웠다. 따라서 phase-vocoder 실패를 속도 변형 자체의 실패로 일반화하지 않는다.')
section='''

### 다른 변환 방식으로 재검사한 결과

첫 512개 선택 중 phase×class×길이구간마다 metadata 순서의2개씩64개를 재선택했다. 기존 세 모델50epoch·noise0·두GPU·6조건을 유지하고 FFmpeg atempo로만 교체했다. 이미 실행한 phase-vocoder도 정확히 같은64개로 제한해 비교했으며 세 모델의 native-cap 점수는 두 실행 사이 오차0이다. 전체 benchmark가 아니고 정상32/spoof32를 네 집합에 균등 배치한 소표본 진단이다.

| 같은64개에서 B의 입력 | Phase vocoder 정상거부/32 | atempo 정상거부/32 | Phase vocoder spoof수락/32 | atempo spoof수락/32 |
|---|---:|---:|---:|---:|
'''
for c,label in zip(conditions,labels):
    p=table[(table['head']=='B')&(table.algorithm=='phase_vocoder')&(table.condition==c)].iloc[0]
    q=table[(table['head']=='B')&(table.algorithm=='atempo')&(table.condition==c)].iloc[0]
    section+=f"| {label} | {p.normal_rejected} | {q.normal_rejected} | {p.spoof_accepted} | {q.spoof_accepted} |\n"
section+='''
![동일표본에서 알고리즘별 정상 거부 비교](figures/speed_duration_feasibility_20261109/matched_algorithm_normal_rejection.png)

이 소표본에서 atempo0.8배 감속과 roundtrip은 원본 대비 추가 정상거부가 없었다. 반면 phase-vocoder는 거의 모두 spoof 방향으로 이동했다. 따라서 변환 알고리즘이 핵심 변수이며, '속도는 무관하니 어떠한 time-stretch도 안전하다'는 전제는 성립하지 않는다. atempo1.25배에서는 정상거부3개 증가, 전체→약4초에서는5개 증가했고 spoof수락도1→3개로 늘었다. 완만한 양방향 증강을 검증할 근거는 남지만 모든 길이를 강하게 정규화하는 안이 지지된 것은 아니다.

atempo는 출력 길이를 정확히 보장하지 않는다. 길이보정 crop/pad를 하지 않았으며 norm_full4의 실제 길이는3.4331875–4.061875초, 최대 오차0.6043125초였다. 따라서 이 실행은 정확4.0375초 정규화의 동등 재현이 아니라 알고리즘 민감도를 확인하는 근사 대조군이다. 샘플 당 극단 rate와 작은 정상 분모 때문에 전체 성능/통계적 유의성을 주장하지 않는다. 원본과 모두 같은 길이로 두 global-speed view를 만들 수 없다는 수학적 제약도 여전히 남는다.

현 단계 권고: **정확4초 전처리와 속도 불변 학습을 분리하고, 원본 anchor와 완만한 양방향 속도 일관성부터 실험한다.** 우선 더 큰 dev 표본에서 atempo류·고품질 pitch-preserving 알고리즘을 비교하고 정상 margin과 spoof margin을 함께 확인한다. augmentation 단독 대비 consistency 추가 이득, clean 성능 유지, unseen-rate/unseen-algorithm 전이가 나와야 학습 기여로 인정한다. exact-duration 정규화를 추가하려면 극단 배율을 따로 다루고 길이 오차·음소 보존·변환 강도에 대한 class 상관을 검증한다.
'''
report.write_text(text+section,encoding='utf-8')
c=base.connect()
try:
    state=base.run(c,"""
import datetime,json,subprocess
from pathlib import Path
out={'checked_KST':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(timespec='seconds')}
out['gpu6_7']=subprocess.check_output(['nvidia-smi','-i','6,7','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
out['shell_idle']=subprocess.check_output(['ps','-p','352729','-o','pgid=,tpgid='],text=True).split()==['352729','352729']
out['run_exits']={name:(Path('/home/mentee/analysis')/name/'run.exit').read_text().strip() for name in ['ebm_speed_duration_probe_20261109_v1','ebm_speed_duration_atempo_20261109_v1']}
out['git_clean']=not subprocess.check_output(['git','status','--porcelain'],cwd='/home/mentee/code/EBM',text=True).strip()
print(json.dumps(out))
""")
finally:c.close()
assert state['shell_idle'] and state['git_clean'] and set(state['run_exits'].values())=={'0'}
base.save(HERE/'final_server_check.json',state)
base.save(HERE/'state.json',{'status':'inference_and_analysis_complete','phase_vocoder_N':512,'atempo_N':64,
    'models':['legacy240_e50','B_e50','C_e50'],'conditions':conditions+['full_native'],'report':str(report.relative_to(base.PROJECT)),
    'no_training':True,'automation_remains_paused':True,'server':state})
for name in ['MENTEE_SERVER.md','AGENTS.md']:
    path=base.PROJECT/name
    if not path.exists():continue
    previous=path.read_text(encoding='utf-8')
    status='> **2026-11-09 속도 진단 완료:** B/C 전체 체크포인트 평가는20:11:34 KST exit0으로 완료. 별도 속도 진단은 기존 컨테이너/GPU0,1에서 phase-vocoder512개(22:35:08 종료)·FFmpeg atempo64개(22:39:37 종료), 기존240/B/C50epoch·6조건 추론만 실행하고 모두 exit0. 분석·ROC/SHA/metadata 검산 완료, source/checkpoint불변·서버Git clean·GPU유휴 확인. 상태 `artifacts/speed_duration_feasibility_20261109/state.json`, 보고서 `docs/experiments/ebm_B_C_all_checkpoint_evaluation_20261109.md` 속도절. 학습/optimizer갱신/예약재활성화 없음. 아래 실행중 표시는 과거기록이다.\n\n'
    if status not in previous:path.write_text(status+previous,encoding='utf-8')
print(json.dumps(state,ensure_ascii=False))
