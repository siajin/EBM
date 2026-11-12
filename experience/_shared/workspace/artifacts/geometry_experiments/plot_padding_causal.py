"""Scientific figures for paired interventions; no outcomes selected for display."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_padding_causal import OUT

BLUE,ORANGE,GRAY='#276d9d','#c45e26','#4b5562'
SOURCES=['asvspoof','vcc2018','vcc2020']
NAMES={'asvspoof':'ASVspoof','vcc2018':'VCC2018','vcc2020':'VCC2020'}

def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=180,facecolor='white');fig.savefig(OUT/(name+'.svg'),facecolor='white');plt.close(fig)

def plot_inputs(f):
    fig,axes=plt.subplots(2,3,figsize=(14,8),sharex=True,sharey=True)
    threshold=np.degrees(np.arccos(-.9145589682336335))
    for col,source in enumerate(SOURCES):
        for row,y,color,label in [(0,1,BLUE,'정상'),(1,0,ORANGE,'스푸프')]:
            ax=axes[row,col];q=f[(f.source_corpus==source)&(f.y==y)&(f.duration_bin!='long')]
            ww=q.pivot(index='utterance_id',columns='condition',values='theta_deg')[['native_wrap','native_reflect','native_original']]
            for uid,values in ww.iterrows():
                jitter=(int(hashlib.sha256(uid.encode()).hexdigest()[:6],16)/0xffffff-.5)*.10
                ax.plot(np.arange(3)+jitter,values.to_numpy(),color=color,alpha=.12,lw=.6)
                ax.scatter(np.arange(3)+jitter,values.to_numpy(),color=color,s=6,alpha=.18)
            means=ww.mean().to_numpy();ax.plot(range(3),means,color=color,lw=2.1,marker='D',ms=5)
            ax.axhline(threshold,color=GRAY,ls=':',lw=1)
            ax.set_ylim(0,180);ax.set_yticks([0,45,90,135,180]);ax.grid(axis='y',alpha=.14)
            ax.spines[['top','right']].set_visible(False);ax.set_xlim(-.25,2.25)
            ax.set_xticks(range(3),['반복 패딩','반사 패딩','원음 그대로'])
            ax.set_title(f'{NAMES[source]} · {label} {len(ww)}개',fontsize=12,fontweight='bold',pad=12)
            ax.set_ylabel('정상 센트로이드와의 각도 (°)')
    fig.suptitle('같은 짧은 원음: 패딩을 빼면 정상과 스푸프가 함께 이동',fontsize=18,fontweight='bold',y=.99)
    fig.text(.05,.025,'무압축·원음 4.0375초 미만 · 가는 선 하나 = 같은 발화 · 굵은 선 = 평균\n점선은 기존 전체 DF EER의 고정 진단 기준각이며 배포 문턱값이 아닙니다. 정상은 작은 각도, 스푸프는 큰 각도가 유리합니다.',fontsize=10,color=GRAY)
    fig.subplots_adjust(left=.07,right=.98,top=.90,bottom=.15,hspace=.35,wspace=.22)
    save(fig,'01_same_waveform_padding')

def plot_primary():
    pairs=json.loads((OUT/'paired_effects.json').read_text())
    tags=[('native_original minus native_wrap','짧은 정상: 원음 그대로 - 반복 패딩'),('native_reflect minus native_wrap','짧은 정상: 반사 패딩 - 반복 패딩'),('2s crop_wrap minus native','긴 정상: 2초 절단·반복 - 원래 입력'),('2s crop_original minus crop_wrap','같은 2초: 원음 그대로 - 반복 패딩')]
    fig,axes=plt.subplots(2,2,figsize=(13.6,7.5))
    for ax,(tag,title) in zip(axes.flat,tags):
        rows=[next(p for p in pairs if p['primary'] and p['source']==s and p['contrast'].startswith(tag)) for s in SOURCES]
        for i,p in enumerate(rows):
            lo,hi=p['family95_CI_12comparisons'];v=p['mean_effect']
            ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',color=BLUE,capsize=3,ms=6,lw=1.4)
            ax.text(126,i,f'{v:+.1f}°',ha='right',va='center',fontsize=10)
        ax.set_yticks(range(3),[NAMES[s] for s in SOURCES]);ax.invert_yaxis();ax.set_xlim(-140,130)
        ax.axvline(0,color=GRAY,lw=1);ax.grid(axis='x',alpha=.13);ax.spines[['top','right']].set_visible(False)
        ax.set_title(title,fontsize=12,fontweight='bold',pad=13)
        ax.set_xlabel('센트로이드 각도 변화 (°) · 음수는 정상 중심 쪽')
    fig.suptitle('사전에 고정한 12개 비교: 화자 군집·다중 비교 보정',fontsize=17,fontweight='bold',y=.99)
    fig.text(.04,.025,'동일 발화의 짝지은 변화 · 화자 군집 부트스트랩 5,000회 · 막대는 12비교 Bonferroni family 95% 구간\n절단은 원래 4.0375초 창의 앞·뒤 2초를 각각 처리한 뒤 발화별 평균. 구간은 관측한 출처·화자 표본에 조건부입니다.',fontsize=10,color=GRAY)
    fig.subplots_adjust(left=.11,right=.97,top=.88,bottom=.17,hspace=.48,wspace=.38)
    save(fig,'02_primary_contrasts')

def plot_mechanism():
    p=OUT/'pooling_mechanism/results.parquet'
    if not p.exists():return
    f=pd.read_parquet(p);fig,axes=plt.subplots(1,3,figsize=(14,5.8),sharey=True)
    names=['noise0_base_score','dup2_none_score','dup2_both_score']
    for ax,source in zip(axes,SOURCES):
        q=f[(f.source_corpus==source)&(f.y==1)&(f.duration_bin!='long')]
        for _,row in q.iterrows():
            a=np.degrees(np.arccos(np.clip(row[names].to_numpy(float),-1,1)))
            ax.plot(range(3),a,color=BLUE,alpha=.13,lw=.6)
        values=np.degrees(np.arccos(np.clip(q[names].to_numpy(float),-1,1))).mean(0)
        ax.plot(range(3),values,color=BLUE,lw=2,marker='D',ms=5)
        ax.set_title(f'{NAMES[source]} · 짧은 정상 {len(q)}개',fontsize=12,fontweight='bold',pad=15)
        ax.set_xticks(range(3),['원래 특징','같은 프레임\n2회 복제','표준편차\n크기 복원']);ax.set_xlim(-.2,2.2)
        ax.set_ylim(0,180);ax.set_yticks([0,45,90,135,180]);ax.grid(axis='y',alpha=.14);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('정상 센트로이드와의 각도 (°)')
    fig.suptitle('동일 특징의 복제만으로는 큰 점수 이동이 재현되지 않음',fontsize=18,fontweight='bold',y=.99)
    fig.text(.045,.025,'특징 값·학습 가중치 고정 · pooling의 미소 잡음만 끈 계산 대조 · 복제 시 표준편차 분기는 약 절반으로 줄지만 점수 변화는 작음\n표준편차의 단순 길이 배율이 주원인이라는 가설을 지지하지 않습니다. 실제 원음 반복은 frontend 특징도 바꿉니다.',fontsize=10,color=GRAY)
    fig.subplots_adjust(left=.07,right=.98,top=.84,bottom=.23,wspace=.22)
    save(fig,'03_identical_feature_duplication')

def plot_context():
    f=pd.read_parquet(OUT/'frontend_context/results.parquet')
    columns=['mask0_score','mask1_score','mask2_score','mask4_score','mask7_score']
    labels=['음성 먼저\n반복','초기 0-8\n특징 교체','중간 9-16\n특징 교체','후기 17-24\n특징 교체','전체 특징\n뒤에서 반복']
    fig,axes=plt.subplots(1,3,figsize=(16,6.2),sharey=True)
    for ax,source in zip(axes,SOURCES):
        q=f[(f.source_corpus==source)&(f.y==1)&(f.duration_bin!='long')]
        a=np.degrees(np.arccos(np.clip(q[columns].to_numpy(float),-1,1)))
        for line in a:ax.plot(range(5),line,color=BLUE,alpha=.10,lw=.6)
        ax.plot(range(5),a.mean(0),color=BLUE,marker='D',ms=5,lw=2)
        for i,v in enumerate(a.mean(0)):ax.annotate(f'{v:.1f}°',(i,v),xytext=(0,10),textcoords='offset points',ha='center',fontsize=10)
        ax.set_xticks(range(5),labels);ax.set_xlim(-.25,4.25);ax.set_ylim(0,180);ax.set_yticks([0,45,90,135,180]);ax.grid(axis='y',alpha=.14)
        ax.spines[['top','right']].set_visible(False);ax.set_title(f'{NAMES[source]} · 짧은 정상 {len(q)}개',fontsize=13,fontweight='bold',pad=15)
    axes[0].set_ylabel('정상 센트로이드와의 각도 (°)')
    fig.suptitle('backend 입력은 모두 201프레임: 반복을 frontend 앞·뒤 어디에서 하는가',fontsize=18,fontweight='bold',y=.99)
    fig.text(.04,.025,'같은 원음·가중치·backend 길이 · 가는 선 = 같은 발화 · 굵은 선/수치 = 평균 · 무음 처리 없음\n특징 교체는 원음을 반복 없이 통과시킨 XLS-R hidden states를 201프레임으로 반복한 값으로 대체합니다.\n층별 혼합은 내부 함수 민감도 실험이며 실제 음성·배포 개선을 뜻하지 않습니다. hidden state 번호는 0부터입니다.',fontsize=10,color=GRAY)
    fig.subplots_adjust(left=.06,right=.98,top=.85,bottom=.25,wspace=.18)
    save(fig,'04_frontend_context_localization')

if __name__=='__main__':
    plt.rcParams.update({'font.family':'Malgun Gothic','font.size':10,'axes.unicode_minus':False,'svg.fonttype':'none'})
    plot_inputs(pd.read_parquet(OUT/'paired_derived.parquet'));plot_primary();plot_mechanism();plot_context()
