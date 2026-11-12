import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MPLCONFIGDIR']='/home/mentee/.cache/matplotlib'
from pathlib import Path
import json, hashlib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import roc_curve
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

assert Path('/.dockerenv').exists(), 'Container required'
source=Path('/results/analysis/EBM80_21DF_centroid/embeddings.npz')
out=Path('/results/analysis/EBM80_21DF_multicenter')
out.mkdir(exist_ok=False)
d=np.load(source); x=d['embeddings'].astype(np.float64); y=d['labels']; oc=d['learned_center'].astype(np.float64)
np.testing.assert_allclose(np.linalg.norm(x,axis=1),1,atol=1e-5)
def unit(v): return v/np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-12)
def centers(z,k,seed):
    if k==1: return unit(z.mean(0,keepdims=True))
    km=KMeans(n_clusters=k,n_init=10,random_state=seed).fit(z)
    c=unit(km.cluster_centers_)
    for _ in range(100):
        a=(z@c.T).argmax(1)
        new=np.stack([z[a==j].mean(0) if (a==j).any() else c[j] for j in range(k)])
        new=unit(new)
        if np.max(np.abs(new-c))<1e-8: break
        c=new
    return new
def eer(labels,s):
    fpr,tpr,_=roc_curve(labels,s,drop_intermediate=False); fnr=1-tpr; diff=fpr-fnr
    i=int(np.where(diff>=0)[0][0])
    if i==0: return float((fpr[0]+fnr[0])/2*100)
    w=-diff[i-1]/(diff[i]-diff[i-1])
    return float((fpr[i-1]+w*(fpr[i]-fpr[i-1]))*100)
def threshold(labels,s):
    fpr,tpr,t=roc_curve(labels,s,drop_intermediate=False)
    return float(t[np.argmin(np.abs(fpr-(1-tpr)))])
def errors(labels,s,t):
    frr=float(np.mean(s[labels==1]<t)*100); far=float(np.mean(s[labels==0]>=t)*100)
    return {'bona_reject_pct':frr,'spoof_accept_pct':far,'balanced_error_pct':(frr+far)/2}
ks=[1,2,3,4,8]; names=['Learned OC']+[f'K={k}' for k in ks]
rows=[]; snapshots={}
for seed in range(1,11):
    rng=np.random.default_rng(seed); parts=[]
    for label in (1,0):
        ids=rng.permutation(np.flatnonzero(y==label)); assert len(ids)==1000
        parts.append((ids[:600],ids[600:800],ids[800:]))
    train=parts[0][0]; cal=np.r_[parts[0][1],parts[1][1]]; test=np.r_[parts[0][2],parts[1][2]]
    assert not (set(train)&set(cal) or set(train)&set(test) or set(cal)&set(test))
    allcenters=[unit(oc[None])]+[centers(x[train],k,seed) for k in ks]
    for name,c in zip(names,allcenters):
        sc=(x@c.T).max(1); th=threshold(y[cal],sc[cal]); r={'seed':seed,'method':name,'eer_pct':eer(y[test],sc[test]),'threshold':th}
        r.update(errors(y[test],sc[test],th))
        fixed=errors(y[test],sc[test],.9); r.update({'margin09_'+k:v for k,v in fixed.items()})
        r['bona_mean_cosine']=float(sc[test][y[test]==1].mean()); r['spoof_mean_cosine']=float(sc[test][y[test]==0].mean())
        rows.append(r)
        if seed==1:
            a=(x[test]@c.T).argmax(1); train_a=(x[train]@c.T).argmax(1)
            snapshots[name]={'centers':c.tolist(),'center_to_oc':(c@unit(oc)).tolist(),'train_bona_counts':np.bincount(train_a,minlength=len(c)).tolist(),'test_bona_counts':np.bincount(a[y[test]==1],minlength=len(c)).tolist(),'test_spoof_counts':np.bincount(a[y[test]==0],minlength=len(c)).tolist(),'test_bona_accepted09_by_center':np.bincount(a[(y[test]==1)&(sc[test]>=.9)],minlength=len(c)).tolist(),'test_spoof_accepted09_by_center':np.bincount(a[(y[test]==0)&(sc[test]>=.9)],minlength=len(c)).tolist()}
            if name=='K=3': example=(test,c,sc)
    print(f'Split {seed}/10 complete',flush=True)
summary={}
metrics=['eer_pct','balanced_error_pct','bona_reject_pct','spoof_accept_pct','margin09_bona_reject_pct','margin09_spoof_accept_pct','bona_mean_cosine','spoof_mean_cosine']
baseline=np.array([r['eer_pct'] for r in rows if r['method']=='Learned OC'])
for name in names:
    rr=[r for r in rows if r['method']==name]
    summary[name]={k:{'mean':float(np.mean([r[k] for r in rr])),'min':float(np.min([r[k] for r in rr])),'max':float(np.max([r[k] for r in rr]))} for k in metrics}
    delta=np.array([r['eer_pct'] for r in rr])-baseline
    summary[name]['paired_eer_difference_pp_mean']=float(delta.mean())
    summary[name]['splits_better_than_oc']=int((delta<0).sum())
report={'summary':summary,'splits':rows,'example_split1':snapshots,'methodology':{'checkpoint':'BestCLS_80_21DF.pt','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'fit':'600 bonafide per split, spherical refinement of KMeans; features frozen','calibration':'200 bonafide + 200 spoof per split; threshold closest to equal error','test':'200 bonafide + 200 spoof per split','repeats':10,'score':'max cosine to centers','limits':['All data from exploratory balanced 21DF sample; not official benchmark EER or a retrained model.','Splits overlap; split ranges are not confidence intervals.','Utterance-disjoint only; speaker/source independence is not established.','Learned OC was trained previously; K=1 fitted mean is a separate control.','Do not select K using this test set for a final performance claim.']}}
(out/'summary.json').write_text(json.dumps(report,indent=2))
fig,axes=plt.subplots(2,2,figsize=(14,10),layout='constrained')
xx=np.arange(len(names))
for field,label,color in [('eer_pct','Held-out EER','#2563A6'),('balanced_error_pct','Error at calibration threshold','#D87527')]:
    means=np.array([summary[n][field]['mean'] for n in names]); lo=np.array([summary[n][field]['min'] for n in names]); hi=np.array([summary[n][field]['max'] for n in names])
    axes[0,0].errorbar(xx,means,yerr=[means-lo,hi-means],marker='o',capsize=3,label=label,color=color)
axes[0,0].set_xticks(xx,names); axes[0,0].set(title='Held-out discrimination (lower is better)',ylabel='Percent'); axes[0,0].legend(fontsize=9); axes[0,0].grid(alpha=.15)
for field,label,color in [('margin09_bona_reject_pct','Bonafide below 0.9','#2563A6'),('margin09_spoof_accept_pct','Spoof at or above 0.9','#D87527')]:
    axes[0,1].plot(xx,[summary[n][field]['mean'] for n in names],marker='o',label=label,color=color)
axes[0,1].set_xticks(xx,names); axes[0,1].set(title='Coverage trade-off at fixed cosine 0.9',ylabel='Percent (not calibrated operating point)'); axes[0,1].legend(fontsize=9); axes[0,1].grid(alpha=.15)
test,c,sc=example
pca=PCA(n_components=2).fit(x); xy=pca.transform(x[test]); cp=pca.transform(c)
for label,name,color in [(1,'Bonafide','#2563A6'),(0,'Spoof','#D87527')]:
    axes[1,0].scatter(*xy[y[test]==label].T,s=12,alpha=.4,color=color,marker='o' if label else '^',label=name)
    axes[1,1].hist(sc[test][y[test]==label],bins=np.linspace(-1,1,51),density=True,histtype='step',linewidth=2,color=color,label=name)
axes[1,0].scatter(*cp.T,c='black',marker='*',s=180,label='K=3 centers')
for j in range(3): axes[1,0].annotate(str(j+1),cp[j],xytext=(6,6),textcoords='offset points')
axes[1,0].set(title='Illustrative split 1: K=3 (not selected as best)',xlabel='PC1',ylabel='PC2'); axes[1,0].legend(fontsize=9)
axes[1,1].set(title='Split 1: maximum cosine to K=3 centers',xlabel='Maximum cosine similarity',ylabel='Density'); axes[1,1].legend()
fig.suptitle('Multiple bonafide centers on frozen epoch80 embeddings | exploratory 21DF sample\n10 utterance-disjoint fit/calibration/test splits; error bars = split min/max, not confidence intervals',fontsize=13)
fig.savefig(out/'multicenter.png',dpi=170); fig.savefig(out/'multicenter.pdf'); plt.close(fig)
print(json.dumps(summary,indent=2),flush=True)
