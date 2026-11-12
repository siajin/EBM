import os,json
from pathlib import Path
os.environ['MPLCONFIGDIR']='/home/mentee/.cache/matplotlib'
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
R=Path('/home/mentee/results/analysis/ASV_quiet_BPL_distribution_20261030_v1')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':13,'axes.labelsize':11,'figure.facecolor':'white','axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
blue='#2563A6';orange='#D27D24';datasets=['19LA_train','19LA_dev','19LA_eval','21LA_eval','21DF_ASV_eval'];labels=['19LA train','19LA dev','19LA eval*','21LA eval*','21DF ASV*']
def main():
 native=json.loads((R/'native_summary.json').read_text());fig,axs=plt.subplots(2,2,figsize=(14,10),gridspec_kw={'height_ratios':[1,1.3]})
 for col,db in enumerate(['20','40']):
  ax=axs[0,col]
  for k,name in enumerate(['bonafide','spoof']):
   boxes=[]
   for ds in datasets:
    s=native['native'][ds][db][name]['leading_s'];boxes.append({'med':s['median'],'q1':s['q25'],'q3':s['q75'],'whislo':s['q10'],'whishi':s['q90'],'fliers':[]})
   color=[blue,orange][k];pos=np.arange(5)+(k-.5)*.3
   ax.bxp(boxes,positions=pos,widths=.25,showfliers=False,patch_artist=True,boxprops={'facecolor':color,'alpha':.22,'edgecolor':color},medianprops={'color':color,'linewidth':2},whiskerprops={'color':color},capprops={'color':color})
  ax.set_xticks(np.arange(5));ax.set_xticklabels(labels,rotation=15,ha='right');ax.set_ylim(0,2);ax.set_ylabel('Leading low-energy duration (seconds)');ax.set_title(f'Leading duration | threshold -{db} dB');ax.grid(axis='y',alpha=.17)
  if col==0:ax.legend(handles=[Line2D([0],[0],color=blue,lw=3,label='Bonafide'),Line2D([0],[0],color=orange,lw=3,ls='--',label='Spoof')],loc='upper right')
  matrix=[];names=[]
  for ds,label in zip(datasets,labels):
   for group,short in [('bonafide','B'),('spoof','S')]:matrix.append(native['native'][ds][db][group]['mean_position_profile']);names.append(label+' / '+short)
  ax=axs[1,col];im=ax.imshow(np.array(matrix)*100,vmin=0,vmax=100,aspect='auto',cmap='Blues',extent=[0,100,9.5,-.5]);ax.set_yticks(range(10));ax.set_yticklabels(names);ax.set_xticks([0,25,50,75,100]);ax.set_xlabel('Position within each full recording (%)');ax.set_title(f'Low-energy occupancy | threshold -{db} dB');fig.colorbar(im,ax=ax,label='Mean low-energy fraction in bin (%)',shrink=.85)
 fig.suptitle('ASVspoof: low-energy duration and position before training augmentation',fontsize=17)
 fig.text(.5,.025,'20 ms RMS; threshold relative to each recording maximum. Boxes: 25-75%, line: median, whiskers: 10-90%.\nTrain/dev: full sets. *Evaluation: 1,000 files per class; 21DF includes ASVspoof source only. VCC excluded.',fontsize=10,ha='center')
 fig.tight_layout(rect=(0,.10,1,.94))
 fig.savefig(R/'native_duration_position.png',dpi=160);fig.savefig(R/'native_duration_position.svg');plt.close(fig)
 if not (R/'pair_summary.json').exists():return
 d=json.loads((R/'pair_summary.json').read_text());stages=d['plan']['stages'][:3];titles=['Crop only: no codec/noise','Actual codec + crop, before RawBoost','Actual final waveform after RawBoost'];fig,axs=plt.subplots(2,3,figsize=(15,8.5))
 for row,db in enumerate(['20','40']):
  for col,(stage,title) in enumerate(zip(stages,titles)):
   a=d['stages'][stage][db]['bonafide'];ax=axs[row,col]
   for view,duration,color,style in [('long',4.0375,blue,'-'),('short',3.0275,orange,'--')]:
    profile=np.array(a[view]['mean_position_profile'])*100;x=(np.arange(20)+.5)*duration/20;ax.plot(x,profile,color=color,ls=style,lw=2,label=f'{duration:.2f}s view');ax.scatter(x,profile,color=color,s=10)
   ax.set_xlim(0,4.04);ax.set_ylim(0,100);ax.set_xticks([0,1,2,3,4]);ax.set_yticks([0,25,50,75,100]);ax.grid(alpha=.18);ax.set_xlabel('Position within model input (seconds)');ax.set_ylabel(f'Low-energy occupancy (%)\nthreshold -{db} dB');ax.set_title(title)
   both=a['leading_presence_pct']['both'];ax.text(.98,.97,f'Leading quiet in both: {both:.1f}%',ha='right',va='top',transform=ax.transAxes,fontsize=10,bbox={'facecolor':'white','edgecolor':'none','alpha':.85})
   if row==0 and col==0:ax.legend(loc='lower right')
 fig.suptitle('Bonafide BPL inputs: 4.04s versus 3.03s',fontsize=18)
 fig.text(.5,.025,'ASVspoof 19LA train: all 2,580 bonafide files, 2 seeded draws/file (5,160 pairs).\n20 bins per view; own-view maximum RMS threshold. Crop-only short view samples the full original recording. Actual codec/noise draws are independent across views.',fontsize=10,ha='center')
 fig.tight_layout(rect=(0,.10,1,.94))
 fig.savefig(R/'bpl_input_position.png',dpi=160);fig.savefig(R/'bpl_input_position.svg');plt.close(fig)
 print('FIGURES_READY',flush=True)
if __name__=='__main__':main()
