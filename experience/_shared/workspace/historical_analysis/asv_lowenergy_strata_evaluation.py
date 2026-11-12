import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
os.environ['MPLCONFIGDIR']='/home/mentee/.cache/matplotlib'
from pathlib import Path
import json,hashlib,time
from collections import Counter
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_curve,roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
assert Path('/.dockerenv').exists()
root=Path('/results/analysis/EBM80_ASV_quiet_presence_hypothesis_20261030')
out=Path('/results/analysis/EBM80_ASV_lowenergy_le100_ge500_evaluation')
out.mkdir(exist_ok=False)
d=np.load(root/'private/original_presence_features.npz')
y=d['label'].astype(int);score=d['score'].astype(float);lead=d['lead20'].astype(float);speaker=d['speaker'];attack=d['attack'];codec=d['codec']
assert len(y)==1298 and y.sum()==1050 and np.isfinite(score).all()
frame=np.rint(lead/.02).astype(int);assert np.max(abs(frame*.02-lead))<1e-10
masks={'all_asv':np.ones(len(y),bool),'le100ms':frame<=5,'ge500ms':frame>=25}
assert not (masks['le100ms']&masks['ge500ms']).any()
plan={'population':'Existing 21DF ASV-source eval samples only, true B and S in EACH duration stratum. No VCC or training-set performance evaluation.',
 'checkpoint':'/home/mentee/results/CM/LA/EBM/model/BestCLS_80_21DF.pt',
 'input':'Cached whole-native-file inference view 0, no augmentation/cropping/repetition/padding. No new GPU inference.',
 'duration':'Leading consecutive 20ms RMS frames <10% of full-file max RMS (-20dB). Exact integer-frame boundaries: <=5 frames (100ms INCLUDED), >=25 frames (500ms INCLUDED).',
 'eer':'Primary: linear interpolation of ROC at FPR=FNR. A secondary nearest discrete EER approximation is provided because small classes cause large steps. AUC independently verified by ranks.',
 'common_threshold':'One retrospective operating threshold selected on ALL 1298 ASV eval files by minimum |FPR-FNR|; ties by lower mean error. It is then frozen for both strata. This is NOT independent calibration or deployment/test accuracy.',
 'uncertainty':'2000 speaker-cluster bootstrap resamples within each duration stratum, resampling speakers jointly across B/S. Resamples missing a class are excluded and counted. Descriptive percentile intervals; rare-class support cannot be repaired by bootstrapping.',
 'comparison_limits':'Different original files and potentially different speakers/attacks/codecs. Subset EER differences are not causal effects of prefix duration. 21DF was involved in model selection.',
 'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
(out/'plan.json').write_text(json.dumps(plan,indent=2))

def curve_metrics(yy,ss):
    fpr,tpr,th=roc_curve(yy,ss,drop_intermediate=False);fnr=1-tpr
    diff=fpr-fnr;k=int(np.flatnonzero(diff>=0)[0])
    if k==0:eer=float(fpr[0])
    else:
        w=-diff[k-1]/(diff[k]-diff[k-1]);eer=float(fpr[k-1]+w*(fpr[k]-fpr[k-1]))
    near=np.flatnonzero(np.isclose(abs(diff),abs(diff).min(),atol=1e-12,rtol=0))
    j=int(near[np.argmin((fpr+fnr)[near])])
    return {'eer_pct':100*eer,'auc':float(roc_auc_score(yy,ss)),
      'nearest_discrete_eer_pct':float(50*(fpr[j]+fnr[j])),
      'nearest_discrete_frr_pct':float(100*fnr[j]),'nearest_discrete_far_pct':float(100*fpr[j]),
      'nearest_discrete_threshold':float(th[j])},(fpr,tpr,th)

pooled,pooled_curve=curve_metrics(y,score);threshold=pooled['nearest_discrete_threshold'];assert np.isfinite(threshold)

def independent_eer_auc(yy,ss):
    # Ascending thresholds and searchsorted give the DET independently of sklearn ROC.
    b=np.sort(ss[yy==1]);s=np.sort(ss[yy==0]);ts=np.r_[-np.inf,np.nextafter(np.unique(ss),np.inf)]
    fr=np.searchsorted(b,ts,side='left')/len(b);fa=(len(s)-np.searchsorted(s,ts,side='left'))/len(s)
    diff=fr-fa;k=int(np.flatnonzero(diff>=0)[0]);w=-diff[k-1]/(diff[k]-diff[k-1])
    eer=float(fr[k-1]+w*(fr[k]-fr[k-1]))
    ranks=rankdata(ss,method='average');auc=float((ranks[yy==1].sum()-len(b)*(len(b)+1)/2)/(len(b)*len(s)))
    return 100*eer,auc

def fixed_metrics(yy,ss):
    b=yy==1;s=yy==0
    fn=int((ss[b]<threshold).sum());fp=int((ss[s]>=threshold).sum())
    fr=fn/int(b.sum());fa=fp/int(s.sum())
    return {'threshold':threshold,'bona_rejected_n':fn,'bona_n':int(b.sum()),'bona_rejection_pct':100*fr,
      'spoof_accepted_n':fp,'spoof_n':int(s.sum()),'spoof_acceptance_pct':100*fa,
      'balanced_error_pct':50*(fr+fa),'balanced_accuracy_pct':100-50*(fr+fa),
      'raw_accuracy_pct':100*(1-(fn+fp)/len(yy))}

def describe(ss,dd,gg):
    return {'n':len(ss),'speakers':len(set(gg)),'cosine_mean':float(ss.mean()),'cosine_median':float(np.median(ss)),
      'cosine_q10':float(np.quantile(ss,.1)),'cosine_q90':float(np.quantile(ss,.9)),
      'angle_to_OC_median_deg':float(np.median(np.degrees(np.arccos(np.clip(ss,-1,1))))),
      'leading_median_s':float(np.median(dd))}

def bootstrap(mask):
    yy=y[mask];ss=score[mask];gg=speaker[mask]
    groups=[np.flatnonzero(gg==v) for v in np.unique(gg)]
    rng=np.random.default_rng(412026);vals=[];skips=0
    for _ in range(2000):
        j=np.concatenate([groups[k] for k in rng.integers(0,len(groups),len(groups))])
        if len(np.unique(yy[j]))<2:skips+=1;continue
        met,_=curve_metrics(yy[j],ss[j]);fixed=fixed_metrics(yy[j],ss[j])
        vals.append([met['eer_pct'],met['auc'],fixed['bona_rejection_pct'],fixed['spoof_acceptance_pct'],fixed['balanced_error_pct']])
    q=np.quantile(vals,[.025,.975],axis=0)
    return {'replicates_requested':2000,'valid_replicates':len(vals),'missing_class_replicates':skips,
      'ci95':{k:q[:,i].tolist() for i,k in enumerate(['eer_pct','auc','bona_rejection_pct','spoof_acceptance_pct','balanced_error_pct'])}}

summary={'plan':plan,'common_threshold':threshold,'groups':{},'attack_composition':[],
 'boundaries':{'exactly100ms':{'B':int(((frame==5)&(y==1)).sum()),'S':int(((frame==5)&(y==0)).sum())},
               'exactly500ms':{'B':int(((frame==25)&(y==1)).sum()),'S':int(((frame==25)&(y==0)).sum())},
               'excluded_middle':{'B':int(((frame>5)&(frame<25)&(y==1)).sum()),'S':int(((frame>5)&(frame<25)&(y==0)).sum())}},
 'verification':{'integer_boundary_checks':True,'input_sha256':hashlib.sha256((root/'private/original_presence_features.npz').read_bytes()).hexdigest(),
  'independent_metric_checks':[]}}
curves={}
for name,mask in masks.items():
    yy=y[mask];ss=score[mask];met,curve=curve_metrics(yy,ss);curves[name]=curve
    ee,au=independent_eer_auc(yy,ss)
    assert abs(ee-met['eer_pct'])<1e-9 and abs(au-met['auc'])<1e-12
    summary['verification']['independent_metric_checks'].append(name)
    group={'n':int(mask.sum()),'metrics':met,'fixed_common_threshold':fixed_metrics(yy,ss),
      'classes':{},'bootstrap':bootstrap(mask),'codec_counts':{}}
    for lab,label in [(1,'bonafide'),(0,'spoof')]:
        sel=mask&(y==lab);group['classes'][label]=describe(score[sel],lead[sel],speaker[sel]);group['codec_counts'][label]=dict(Counter(codec[sel]))
    summary['groups'][name]=group
    print(name,json.dumps(group['metrics']),'counts',group['classes']['bonafide']['n'],group['classes']['spoof']['n'],flush=True)
for at in sorted(set(attack[y==0])):
    r={'attack':at}
    for name in ['le100ms','ge500ms']:
        m=masks[name]&(y==0)&(attack==at);n=int(m.sum())
        r[name]={'n':n,'fraction_of_spoof_group':n/summary['groups'][name]['classes']['spoof']['n'],
          'accepted_at_common_threshold_n':int((score[m]>=threshold).sum())}
        if n:r[name].update(cosine_median=float(np.median(score[m])),cosine_mean=float(score[m].mean()))
    summary['attack_composition'].append(r)
common=[r for r in summary['attack_composition'] if r['le100ms']['n']>0 and r['ge500ms']['n']>0]
supported=[r for r in common if r['le100ms']['n']>=5 and r['ge500ms']['n']>=5]
summary['attack_support']={'attacks_in_both':[r['attack'] for r in common],
 'shared_attacks_with_at_least5_spoof_per_group':[r['attack'] for r in supported],
 'shared_attack_spoof_n':{name:sum(r[name]['n'] for r in common) for name in ['le100ms','ge500ms']},
 'interpretation':'Attack-adjusted comparison would require overlap. Sparse common attacks do not permit a reliable general attack-matched duration conclusion.'}
summary['groups']['le100ms']['support_warning']='Rare bona fide support: empirical EER/AUC and bootstrap intervals can be highly unstable; bootstrap cannot create unseen bona fide variability.'
(out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,
 'axes.grid':True,'grid.alpha':.15,'figure.facecolor':'white','savefig.facecolor':'white'})
colors={'le100ms':'#34689b','ge500ms':'#b68527'};labels={'le100ms':'Leading low energy ≤100 ms','ge500ms':'Leading low energy ≥500 ms'}
fig,axes=plt.subplots(1,2,figsize=(13,5.3),constrained_layout=True)
for name in ['le100ms','ge500ms']:
    r=summary['groups'][name];nb=r['classes']['bonafide']['n'];ns=r['classes']['spoof']['n'];fp,tp,_=curves[name]
    axes[0].plot(fp,tp,c=colors[name],lw=1.8,label=f'{labels[name]} (B={nb}, S={ns})')
    m=masks[name]&(y==0);edges=np.linspace(-1,1,101);counts,_=np.histogram(score[m],bins=edges)
    axes[1].step(edges[1:],np.cumsum(counts)/m.sum(),where='post',c=colors[name],lw=1.8,label=labels[name])
axes[0].plot([0,1],[0,1],c='#999999',ls='--',lw=.8)
axes[0].set(xlabel='Spoof acceptance rate (FPR)',ylabel='Bonafide acceptance rate (TPR)',title='A. Empirical ROC: different original-file groups',xlim=(0,1),ylim=(0,1.01))
axes[0].legend(fontsize=9,loc='lower right')
axes[1].axvline(threshold,c='#555555',ls='--',lw=1,label='Common retrospective threshold')
axes[1].set(xlabel='Cosine to trained bonafide OC center (higher → B)',ylabel='Cumulative fraction of spoof files',title='B. Spoof score distributions (100 bins)',xlim=(-1,1),ylim=(0,1.01))
axes[1].legend(fontsize=9,loc='upper left')
fig.suptitle('21DF ASV-source only · frozen epoch80 · full native input\n−20 dB leading low energy; duration groups have different B/S and attack composition',fontsize=14)
fig.savefig(out/'strata_evaluation.png',dpi=170);fig.savefig(out/'strata_evaluation.pdf');plt.close(fig)

rows=summary['attack_composition'];xx=np.arange(len(rows));fig,ax=plt.subplots(figsize=(11,7.2),constrained_layout=True)
for name,offset in [('le100ms',-.19),('ge500ms',.19)]:
    values=np.array([100*r[name]['fraction_of_spoof_group'] for r in rows]);ax.barh(xx+offset,values,height=.36,color=colors[name],label=labels[name])
    for i,r in enumerate(rows):
        if r[name]['n']:ax.text(values[i]+.25,i+offset,str(r[name]['n']),va='center',fontsize=9)
ax.set_yticks(xx,[r['attack'] for r in rows]);ax.invert_yaxis();ax.set(xlabel='Share of spoof files within each duration group (%)',title='Spoof attack composition: counts printed beside bars')
ax.legend(loc='lower right',fontsize=10);ax.set_xlim(0,max(100*r[name]['fraction_of_spoof_group'] for r in rows for name in colors)*1.18)
fig.savefig(out/'attack_composition.png',dpi=170);plt.close(fig)

lines=['# 원본 ASV: 앞 저에너지 ≤100ms 및 ≥500ms 평가','',
'동일한 epoch80 모델의 기존 원본 전체 입력 점수를 재사용했다. 각 집단에 bona fide와 spoof를 모두 포함했다. 100ms 이하에는 정확히 100ms도 포함한다. VCC는 포함하지 않았다.',
'저에너지 기준: 앞부분의 연속 20ms 프레임 RMS가 전체 원본 최대 RMS의 10% 미만(−20dB). 파형을 짧게/길게 바꾸지 않고 자연 길이로 파일만 선택했다.','',
'| 집단 | B 수(화자) | S 수(화자) | 보간 EER | EER 95% bootstrap 구간 | AUC |','|---|---:|---:|---:|---:|---:|']
for name,title in [('all_asv','전체 ASV'),('le100ms','≤100ms'),('ge500ms','≥500ms')]:
    r=summary['groups'][name];b=r['classes']['bonafide'];ss=r['classes']['spoof'];ci=r['bootstrap']['ci95']['eer_pct']
    lines.append(f"|{title}|{b['n']} ({b['speakers']})|{ss['n']} ({ss['speakers']})|{r['metrics']['eer_pct']:.3f}%|{ci[0]:.3f}~{ci[1]:.3f}%|{r['metrics']['auc']:.5f}|")
lines+=['','EER은 집단별 ROC에서 FPR=FNR 교차를 선형 보간한 값이다. 작은 B 표본에서는 한 파일 때문에 FNR이 크게 변하므로 보간 EER과 가장 가까운 실제 임계값의 오류율을 구분해야 한다. 화자 단위 bootstrap 2,000회이며 한 클래스가 빠진 반복은 제외하고 summary.json에 횟수를 기록했다. 적은 B/화자 수의 문제는 bootstrap으로 해결되지 않는다.',
'','## 동일 임계값에서의 기술적 비교','',
f'전체 ASV 평가 표본에서 한 번 선택한 코사인 임계값 {threshold:.9f}를 두 집단에 동일하게 적용했다. 평가 표본 자체로 선택한 사후 비교용 임계값이다. 독립 개발셋 보정이나 배포 성능으로 해석하면 안 된다.','',
'| 집단 | B→S 거절 | S→B 수락 | 균형 오류율 |','|---|---:|---:|---:|']
for name,title in [('all_asv','전체 ASV'),('le100ms','≤100ms'),('ge500ms','≥500ms')]:
    r=summary['groups'][name]['fixed_common_threshold'];lines.append(f"|{title}|{r['bona_rejected_n']}/{r['bona_n']} ({r['bona_rejection_pct']:.2f}%)|{r['spoof_accepted_n']}/{r['spoof_n']} ({r['spoof_acceptance_pct']:.2f}%)|{r['balanced_error_pct']:.2f}%|")
lines+=['','## 점수 분포','', '| 집단 | B 코사인 중앙값 | S 코사인 중앙값 |','|---|---:|---:|']
for name,title in [('le100ms','≤100ms'),('ge500ms','≥500ms')]:
    r=summary['groups'][name]['classes'];lines.append(f"|{title}|{r['bonafide']['cosine_median']:+.5f}|{r['spoof']['cosine_median']:+.5f}|")
lines+=['','코사인은 높을수록 학습된 B 중심에 가깝다. 확률이 아니다.','',
'## 공격 구성과 해석 제한','', '| Spoof 공격 | ≤100ms 수 | ≥500ms 수 |','|---|---:|---:|']
for r in rows:lines.append(f"|{r['attack']}|{r['le100ms']['n']}|{r['ge500ms']['n']}|")
lines+=['',f"양쪽에서 spoof가 5개 이상 있는 공통 공격: {summary['attack_support']['shared_attacks_with_at_least5_spoof_per_group']}.",
'집단 간 EER 차이는 앞 구간 길이를 바꾼 효과가 아니다. 서로 다른 파일·화자·공격·코덱 구성의 성능이다. 특히 짧은 집단의 B 지원과 공통 공격 지원이 부족하면 길이 때문에 더 좋아졌거나 나빠졌다고 결론낼 수 없다.',
'이전 고정 crop/repeat 평가와 달리 원본 전체 입력의 점수다. 다른 입력 조건 또는 전체 21DF EER과 직접 동일시하면 안 된다. 21DF는 과거 모델 선택에도 사용되어 완전히 새로운 외부 검증이 아니다.',
'EER은 독립 DET 검색으로, AUC는 순위합으로 각각 검산했다. 종료코드와 산출물을 확인한다.',
'','서버 경로: /results/analysis/EBM80_ASV_lowenergy_le100_ge500_evaluation/']
(out/'report_ko.md').write_text('\n'.join(lines)+'\n')
(out/'COMPLETED.json').write_text(json.dumps({'status':'complete','groups':list(masks),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}))
print(json.dumps(summary,indent=2),flush=True)
