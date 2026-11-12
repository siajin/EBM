from pathlib import Path
import json,hashlib
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_curve
assert Path('/.dockerenv').exists()
root=Path('/results/analysis/EBM80_21DF_leading_quiet_paired')
j=json.loads((root/'summary.json').read_text()); d=np.load(root/'paired_scores.npz')
y=d['labels']; b=y==1; src=d['sources']; scores=d['scores']; names=d['names'].tolist(); n=len(y); th=j['threshold']
assert n==2000 and b.sum()==1000 and scores.shape==(2000,13)
assert all(np.isfinite(d[k]).all() for k in ['scores','checks','measured_leading_s','calibration_scores'])
assert np.abs(scores).max()<=1.00001
assert np.max(np.abs(scores[:,0]-scores[:,1]))<1e-4
cy=d['calibration_labels']; cs=d['calibration_scores'];assert np.sum(cs[cy==1]<th)==np.sum(cs[cy==0]>=th)
old=json.loads(Path('/results/analysis/EBM80_21DF_cue_audit/summary.json').read_text())
for source in np.unique(src):
    for label in [0,1]:
        m=(src==source)&(y==label)
        assert abs(np.median(d['checks'][m,0])/16000-old['feature_summary']['21DF_'+source][str(label)]['leading_s'])<1e-9
def metrics(a,yy):
    pos=yy==1; n1=pos.sum(); n0=(~pos).sum()
    auc=(rankdata(a)[pos].sum()-n1*(n1+1)/2)/(n1*n0)
    fpr,tpr,_=roc_curve(yy,a,drop_intermediate=False); diff=fpr+tpr-1; k=np.flatnonzero(diff>=0)[0]
    if k==0: eer=float(100*fpr[0])
    else:
        w=-diff[k-1]/(diff[k]-diff[k-1]);eer=float(100*(fpr[k-1]+w*(fpr[k]-fpr[k-1])))
    fr=float(100*np.mean(a[pos]<th)); fa=float(100*np.mean(a[~pos]>=th))
    # On balanced sets, independently count errors at the pooled median when no boundary ties occur.
    if n1==n0:
        ss=np.sort(a); mid=len(ss)//2
        if ss[mid-1]<ss[mid]:
            t=(float(ss[mid-1])+float(ss[mid]))/2
            independent=100*np.mean(a[pos]<t)
            assert np.sum(a[pos]<t)==np.sum(a[~pos]>=t) and abs(independent-eer)<1e-8
    return {'eer_pct':eer,'auc':float(auc),'bona_reject_pct':fr,'spoof_accept_pct':fa,'balanced_error_pct':(fr+fa)/2}
for k,name in enumerate(names):
    for key,arr in {'B_S':scores[:,0],'TB_S':np.where(b,scores[:,k],scores[:,0]),'B_TS':np.where(b,scores[:,0],scores[:,k]),'TB_TS':scores[:,k]}.items():
        measured=metrics(arr,y); reported=j['four_conditions'][name][key]
        assert all(abs(measured[v]-reported[v])<1e-8 for v in measured)
    for label,word in [(1,'bonafide'),(0,'spoof')]:
        m=y==label; e0=np.sum((scores[m,0]>=th)!=bool(label)); e1=np.sum((scores[m,k]>=th)!=bool(label))
        pair=j['pairwise_vs_original'][name][word]
        assert e1==e0+pair['correct_to_wrong_n']-pair['wrong_to_correct_n']
out={'all_checks_passed':True,'paired_rows':n,'metric_conditions_independently_checked':len(names)*4,'fixed_threshold':th,
     'prefix_achieved':{},'assignment_experiment':{},'quiet_valid_subset':{},'checkpoint_hash_matches':False}
families={'zero20':[2,3,4],'ownquiet20':[5,6,7],'variable20':[8,9,4],'zero40':[10,11,12]}
for family,ix in families.items():
    out['prefix_achieved'][family]={}
    target=np.array([.06,.48,1.02]); lead=d['measured_leading_s'][:,ix]
    valid=np.max(np.abs(lead-target),axis=1)<.020001
    for label,word in [(1,'bonafide'),(0,'spoof')]:
        mask=y==label;out['prefix_achieved'][family][word]={'all_3_within_20ms_n':int((valid&mask).sum()),'median_actual_leading_s':np.median(lead[mask],axis=0).tolist()}
    out['assignment_experiment'][family]={}
    for source,mask in [('all',np.ones(n,dtype=bool))]+[(s,src==s) for s in np.unique(src)]:
        conditions={'aligned_Blong_Sshort':np.where(b,scores[:,ix[2]],scores[:,ix[0]]),
                    'reversed_Bshort_Slong':np.where(b,scores[:,ix[0]],scores[:,ix[2]]),
                    'both_short':scores[:,ix[0]],'both_medium':scores[:,ix[1]],'both_long':scores[:,ix[2]]}
        out['assignment_experiment'][family][source]={k:metrics(a[mask],y[mask]) for k,a in conditions.items()}
    if family in ['zero20','ownquiet20']:
        m=valid
        if family=='ownquiet20':m=m&(d['checks'][:,0]>0)
        out['quiet_valid_subset'][family]={word:{'n':int((m&(y==label)).sum()),'mean_long_minus_short_spoof':float((scores[:,ix[0]]-scores[:,ix[2]])[m&(y==label)].mean())} for label,word in [(1,'bonafide'),(0,'spoof')]}
h=hashlib.sha256()
with open(j['provenance']['checkpoint'],'rb') as f:
    for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
assert h.hexdigest()==j['provenance']['checkpoint_sha256'];out['checkpoint_hash_matches']=True
out['summary_sha256']=hashlib.sha256((root/'summary.json').read_bytes()).hexdigest()
(root/'verification.json').write_text(json.dumps(out,indent=2,allow_nan=False))
print(json.dumps(out,indent=2),flush=True)
