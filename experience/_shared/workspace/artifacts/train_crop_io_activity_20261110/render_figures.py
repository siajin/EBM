from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
r=Path(__file__).parent/'results'
font_manager.fontManager.addfont('C:/Windows/Fonts/malgun.ttf')
plt.rcParams.update({'font.family':'Malgun Gothic','font.size':10,'axes.unicode_minus':False,
                     'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white',
                     'savefig.facecolor':'white','axes.titleweight':'normal'})
colors={'bona_OC':'#0072B2','spoof_OC':'#D55E00','bona_OC_plus_BPL':'#009E73'}
names={'bona_OC':'보나 · OC','spoof_OC':'스푸프 · OC','bona_OC_plus_BPL':'보나 · OC+BPL'}
styles={'bona_OC':'-','spoof_OC':'--','bona_OC_plus_BPL':'-.'}
cov=pd.read_csv(r/'epoch_coverage.csv')
fig,axs=plt.subplots(1,2,figsize=(10.8,3.5),layout='constrained')
for key in colors:
 g=cov[cov.population==key]
 axs[0].plot(g.epoch,100*g.unique_fraction,color=colors[key],ls=styles[key],lw=1.5,label=names[key])
 first=g[g.epoch<=15]
 axs[1].plot(first.epoch,100*first.cumulative_unique_fraction,color=colors[key],ls=styles[key],lw=1.6,marker='o',ms=3,label=names[key])
axs[0].axhline(100,color='#777777',lw=1,ls=':')
axs[0].text(2,96,'기존 순회 sampler: 매 epoch 100%',fontsize=9,color='#555555')
axs[0].set(title='한 epoch에서 선택한 고유 원음',xlabel='재현 epoch',ylabel='전체 원음 중 선택 비율 (%)',xlim=(1,50),ylim=(0,105))
axs[1].set(title='누적 고유 원음 · 첫 15epoch',xlabel='재현 epoch',ylabel='한 번 이상 선택된 원음 (%)',xlim=(1,15),ylim=(0,105))
for ax in axs:ax.grid(axis='y',alpha=.2);ax.set_yticks([0,20,40,60,80,100])
axs[1].legend(loc='lower right',frameon=False,fontsize=9)
fig.suptitle('Train 샘플러 50epoch 재현 · 모델 학습 실행 아님',fontsize=12)
fig.savefig(r/'train_source_coverage.png',dpi=180);plt.close(fig)
a=pd.read_csv(r/'OC_activity_by_length.csv');fig,axs=plt.subplots(1,2,figsize=(10.8,3.5),layout='constrained')
x=np.arange(8);width=.36
for label,color,offset,name in [(1,'#0072B2',-width/2,'보나'),(0,'#D55E00',width/2,'스푸프')]:
 g=a[a.label==label].sort_values('crop_seconds')
 axs[0].bar(x+offset,g.vad2_speech_fraction*100,width,color=color,label=name)
 axs[1].bar(x+offset,g.speech_below100ms_fraction*100,width,color=color,label=name)
 for ax in axs:ax.set_xticks(x,[f'{v:g}' for v in g.crop_seconds])
axs[0].set(title='VAD가 음성으로 판정한 프레임 비율',ylabel='프레임 비율 (%)',ylim=(0,100))
axs[1].set(title='VAD 음성활동이 100ms 미만인 조각',ylabel='조각 비율 (%)',ylim=(0,25))
for ax in axs:ax.set_xlabel('조각 길이 (초)');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
axs[0].legend(frameon=False,fontsize=9)
fig.suptitle('Train 1epoch OC 조각 · 증강 전 VAD2 추정 (보나 2,592 / 스푸프 23,328)',fontsize=11)
fig.savefig(r/'train_crop_activity.png',dpi=180);plt.close(fig)
