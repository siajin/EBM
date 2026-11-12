from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT=Path(__file__).resolve().parent
plan=pd.read_parquet(OUT/'single_transform_plan.parquet')
summary=pd.read_csv(OUT/'summary.csv')
names=['2019 LA · train','2019 LA · dev','2019 LA · eval','2021 LA · eval','2021 DF · eval']
plt.rcParams.update({'font.family':'Malgun Gothic','font.size':12,'axes.unicode_minus':False,
                     'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
for end,width,suffix in [(10,.05,'0_10s'),(2,.05,'0_2s')]:
    fig,axes=plt.subplots(2,3,figsize=(18,10),sharex=True,sharey=True)
    ymax=0
    for ax,(_,row),name in zip(axes.flat,summary.iterrows(),names):
        d=plan[plan.dataset==row.dataset]
        edges=np.arange(0,30.05,.05)
        for label,color,ls in [(1,'#2376AB','-'),(0,'#D97813','--')]:
            v=d.loc[d.y==label,'target_duration_s'].to_numpy()
            assert v.min()>=0 and v.max()<30
            h=np.histogram(v,bins=edges)[0]/len(v)*100
            ymax=max(ymax,h[edges[:-1]<end].max())
            ax.stairs(h,edges,color=color,ls=ls,lw=2,baseline=None)
        ax.set_title(name,loc='left',fontweight='bold',pad=15)
        ax.text(.97,.95,f"겹침 {row['overlap_single_after_bin0.05s_pct']:.2f}%\n변형 전 {row['overlap_before_bin0.05s_pct']:.2f}%",
                transform=ax.transAxes,ha='right',va='top',fontsize=11,
                bbox={'facecolor':'white','edgecolor':'none','alpha':.9})
        ax.set_xlim(0,end)
        ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
        ax.tick_params(labelbottom=True)
        ax.set_xlabel('변형 후 길이 (초)')
        ax.set_ylabel('전체 클래스 중 비율 / bin (%)')
    axes.flat[5].set_visible(False)
    for ax in axes.flat[:5]:ax.set_ylim(0,np.ceil(ymax*1.2*2)/2)
    fig.suptitle(f'속도 0.8–1.2배 + 최대 20% 크롭 · 분포 정렬 후 (0–{end}초)',
                 x=.065,y=.975,ha='left',fontsize=23)
    fig.legend(handles=[Line2D([0],[0],color='#2376AB',lw=2,label='Bona fide'),
                        Line2D([0],[0],color='#D97813',ls='--',lw=2,label='Spoof')],
               loc='upper center',bbox_to_anchor=(.5,.92),ncol=2,frameon=False)
    fig.text(.065,.025,'발화당 변형 1개 · 파일 제외·가중치 변경 없음 · 0.05초 공통 bin · 겹침률은 전체 길이 범위에서 계산',
             fontsize=11,color='#555555')
    fig.subplots_adjust(left=.065,right=.985,bottom=.12,top=.82,wspace=.27,hspace=.52)
    fig.savefig(OUT/f'aligned_{suffix}.png',dpi=160,facecolor='white')
    fig.savefig(OUT/f'aligned_{suffix}.svg',facecolor='white')
    plt.close(fig)
print('Compact figures exported')
