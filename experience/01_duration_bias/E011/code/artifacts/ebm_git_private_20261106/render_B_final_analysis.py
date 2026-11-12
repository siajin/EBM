"""Reproducible static research figures and dev-only threshold sensitivity."""
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'artifacts/geometry_cache/local_analysis/B_final_analysis_20261108_v1'
OUT=ROOT/'docs/experiments/figures/ebm_B_final_20261108'
OUT.mkdir(parents=True,exist_ok=True)
BLUE='#2364AA';GOLD='#B87016';PINK='#A83C72'
MODELS=['legacy240','B'];COLORS=[GOLD,BLUE]
BINS=['under2','2to3.0275','3.0275to4.0375','atleast4.0375']
BINLABEL=['<2','2–3.0275','3.0275–4.0375','≥4.0375']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':13,
                     'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':160})

def read(name):return json.loads((DATA/name).read_text(encoding='utf-8'))
def save(name):
    plt.savefig(OUT/name,bbox_inches='tight',facecolor='white');plt.close()

comparison=read('analysis/comparison.json')
rates=pd.DataFrame(read('analysis/paired_error_rates.json'))
geometry=pd.DataFrame(read('analysis/geometry_statistics.json'))
basis=read('analysis/projection_basis.json')

# Secondary operating rule: accept at most q of dev spoof trials. No target tuning.
dev=pd.read_parquet(DATA/'paired_scores_dev.parquet');sensitivity=[]
operating_points={}
for model in MODELS:
    s=np.sort(dev.loc[dev.label==1,'score_'+model].to_numpy())
    threshold=comparison['dev'][model]['dev_threshold']
    operating_points[model]={'dev_threshold':threshold,'angle_threshold_degrees':float(np.degrees(np.arccos(threshold))),
                             'smallest_normal_scores':s[:4].tolist(),'largest_spoof_score':float(dev.loc[dev.label==0,'score_'+model].max())}
(DATA/'analysis/operating_points.json').write_text(json.dumps(operating_points,indent=2)+'\n',encoding='utf-8')
for q in [.001,.01]:
    for model in MODELS:
        column='score_'+model
        spoof=np.sort(dev.loc[dev.label==0,column].to_numpy())[::-1]
        k=int(np.floor(q*len(spoof)))
        threshold=float(np.nextafter(spoof[k],np.inf))
        assert (dev.loc[dev.label==0,column]>=threshold).mean()<=q
        for phase in ['dev','19LA','21LA','21DF']:
            f=dev if phase=='dev' else pd.read_parquet(DATA/f'paired_scores_{phase}.parquet')
            for group in ['all','under2']:
                g=f if group=='all' else f[f.original_samples<32000]
                for label in [1,0]:
                    x=g[g.label==label]
                    if len(x):
                        err=(x[column]<threshold) if label else (x[column]>=threshold)
                        sensitivity.append(dict(dev_spoof_budget=q,model=model,threshold=threshold,
                            phase=phase,length_bin=group,label=label,N=len(x),errors=int(err.sum()),
                            error_percent=float(100*err.mean())))
pd.DataFrame(sensitivity).to_csv(DATA/'analysis/threshold_sensitivity.csv',index=False)

# Exact units and denominator table is supplied in the Markdown report.
fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
phases=['19LA','21LA','21DF'];x=np.arange(3)
historical=read('historical_legacy_final.json')['final_EER_percent']
for ax,old,title in [(axes[0],{p:comparison[p]['legacy240']['EER_percent'] for p in phases},'Same non-repeat / masked input'),
                      (axes[1],historical,'Historical repeat pipeline vs B pipeline')]:
    for m,vals,offset,color in [('legacy240',[old[p] for p in phases],-.18,GOLD),
                              ('B',[comparison[p]['B']['EER_percent'] for p in phases],.18,BLUE)]:
        bars=ax.bar(x+offset,vals,.34,label=m,color=color,hatch='//' if m=='legacy240' else '')
        ax.bar_label(bars,fmt='%.3f',padding=4)
    ax.set(xticks=x,xticklabels=phases,ylim=(0,10.3),ylabel='EER (%)',title=title)
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
axes[0].legend(frameon=False)
fig.suptitle('Fixed epoch 50: comparison protocol changes the conclusion',fontsize=16)
save('eer_protocols.png')

# Small multiples: each column shares a scale; per-class denominators explicit.
fig,axes=plt.subplots(3,2,figsize=(13,12))
for i,phase in enumerate(phases):
    for j,label in enumerate([1,0]):
        ax=axes[i,j];g=rates[(rates.phase==phase)&(rates.label==label)&(rates.length_bin!='all')].set_index('length_bin').loc[BINS]
        for model,col,offset,color in [('legacy240','legacy_error_percent',-.17,GOLD),('B','B_error_percent',.17,BLUE)]:
            bars=ax.bar(np.arange(4)+offset,g[col],.32,color=color,label=model,hatch='//' if model=='legacy240' else '')
            ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=9)
        ax.set_xticks(np.arange(4),[f'{b}\nN={n:,}' for b,n in zip(BINLABEL,g.N)])
        ymax=rates[rates.label==label][['B_error_percent','legacy_error_percent']].max().max()
        ax.set_ylim(0,5*np.ceil((ymax+3)/5))
        ax.set_title(phase+(' · bona fide rejection' if label else ' · spoof acceptance'))
        ax.set_ylabel('Error (%)');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        if i==2:ax.set_xlabel('Original duration (seconds); N is per class / bin')
axes[0,0].legend(frameon=False)
fig.suptitle('Errors at each model’s dev-selected EER operating threshold',fontsize=16,y=.985)
fig.subplots_adjust(top=.93,hspace=.42,bottom=.07)
save('length_error_rates.png')

# Paired angle changes: intervals are marginal / exploratory, not family adjusted.
fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
for ax,label in zip(axes,[1,0]):
    for model,color,offset,marker in [('legacy240',GOLD,-.11,'s'),('B',BLUE,.11,'o')]:
        g=geometry[(geometry.model==model)&(geometry.label==label)&(geometry.length_bin=='under2')].set_index('phase').loc[phases]
        mid=g.native_minus_wrap_mean_angle.to_numpy();ci=np.array(g.angle_delta_CI95.tolist())
        ax.errorbar(np.arange(3)+offset,mid,yerr=np.stack([mid-ci[:,0],ci[:,1]-mid]),fmt=marker,
                    color=color,capsize=5,label=model,ms=7)
    ax.axhline(0,color='#444444',lw=1)
    limits=np.array(geometry[geometry.length_bin=='under2'].angle_delta_CI95.tolist())
    ax.set(xticks=np.arange(3),xticklabels=phases,ylim=(30*np.floor((limits.min()-5)/30),30*np.ceil((limits.max()+5)/30)),ylabel='Mean native − repeat angle (degrees)',
           title='Bona fide <2 s' if label else 'Spoof <2 s')
    ax.grid(axis='y',alpha=.2)
axes[0].legend(frameon=False)
fig.suptitle('Locked diagnostic cohort: 32 cases per corpus × class × bin; noise = 0',fontsize=14)
save('input_intervention_angles.png')

# Spherical projection: retain own centroid as north pole, tangent axes per model.
selected=pd.read_parquet(DATA/'mechanism_2x2/selected.parquet').sort_values('selection_index')
coords=np.load(DATA/'analysis/diagnostic_projections.npz')['coordinates']
ids=selected[(selected.phase=='21DF')&(selected.length_bin=='under2')].selection_index.to_numpy()
fig=plt.figure(figsize=(12.5,10))
for row,view in enumerate([0,1]):
    for col,m in enumerate([1,0]):
        ax=fig.add_subplot(2,2,row*2+col+1,projection='3d')
        az=np.linspace(0,2*np.pi,24);polar=np.linspace(0,np.pi,14)
        ax.plot_wireframe(np.outer(np.cos(az),np.sin(polar)),np.outer(np.sin(az),np.sin(polar)),
                          np.outer(np.ones_like(az),np.cos(polar)),color='#BBBBBB',alpha=.22,lw=.5)
        for label,color,marker in [(1,BLUE,'o'),(0,PINK,'^')]:
            ix=ids[selected.iloc[ids].label.to_numpy()==label];p=coords[ix,m,view]
            ax.scatter(*p.T,c=color,marker=marker,s=24,alpha=.8,depthshade=False)
        ax.scatter([0],[0],[1],c='black',marker='*',s=120,depthshade=False)
        ax.text(0,0,1.15,'learned c',ha='center',fontsize=10)
        ax.set(xlim=(-1,1),ylim=(-1,1),zlim=(-1,1.2),xlabel='Tangent 1',ylabel='Tangent 2',zlabel='cos(angle to c)')
        ax.set_xticks([-1,0,1]);ax.set_yticks([-1,0,1]);ax.set_zticks([-1,0,1])
        ax.set_box_aspect((1,1,1));ax.view_init(18,42)
        ax.set_title(('legacy240' if m else 'B')+(' · native input' if view==0 else ' · repeated to 4.0375 s'),pad=12)
fig.legend(handles=[Line2D([],[],color=BLUE,marker='o',linestyle='',label='Bona fide (32)'),
                    Line2D([],[],color=PINK,marker='^',linestyle='',label='Spoof (32)'),
                    Line2D([],[],color='black',marker='*',linestyle='',label='Learned centroid')],
           loc='lower center',ncol=3,frameon=False)
fig.suptitle('21DF <2 s: orthogonal 160 → 3 projection; each model has its own tangent basis',fontsize=14,y=.98)
fig.subplots_adjust(top=.91,bottom=.075,wspace=.08,hspace=.23)
save('centroid_projection_3d.png')

# Epoch means, not rounded console values; independent dev50 indicated separately.
loss=pd.DataFrame(read('training_loss_identity_audit.json')['epoch_means']);audit=read('training_audit.json')
fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
for name,color,style in [('loss1',BLUE,'-'),('loss2',GOLD,'--'),('oc_long','#6C5591',':'),('bpl','#555555','-.')]:
    axes[0].plot(np.arange(1,51),loss[name],label=name,color=color,ls=style)
axes[0].set(yscale='log',xlabel='Epoch',ylabel='Epoch mean loss (log scale)',title='Training objectives, all 5,400 batches',xlim=(1,50))
axes[0].legend(frameon=False,ncol=2);axes[0].grid(alpha=.2)
de=[(int(re.search('[0-9]+',s)[0]),float(s.split(':')[-1])) for s in audit['dev_EER_logged']]
axes[1].plot(*np.array(de).T,marker='o',color=BLUE,label='Logged dev')
axes[1].scatter([50],[comparison['dev']['B']['EER_percent']],marker='D',s=70,color=GOLD,label='Independent epoch-50 dev')
axes[1].set(xlabel='Epoch',ylabel='Dev EER (%)',ylim=(0,.12),xlim=(4,51),title='Dev only; no target-based epoch selection')
axes[1].grid(alpha=.2);axes[1].legend(frameon=False)
save('training_dev.png')

print(json.dumps({'figures':sorted(p.name for p in OUT.glob('*.png')),
                  'threshold_sensitivity_DF_under2_normal':[r for r in sensitivity if r['phase']=='21DF' and r['length_bin']=='under2' and r['label']==1]},indent=2))
