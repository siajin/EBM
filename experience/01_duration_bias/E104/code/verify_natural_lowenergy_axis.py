import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
os.environ['MPLCONFIGDIR']='/home/mentee/.cache/matplotlib'
from pathlib import Path
import json,hashlib
import numpy as np
import soundfile as sf
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
assert Path('/.dockerenv').exists()
root=Path('/results/analysis/EBM80_ASV_bona_natural_lowenergy_axis_db20')
assert (root/'COMPLETED.json').exists()
s=json.loads((root/'summary.json').read_text())
d=np.load(root/'private/axis_arrays.npz')
rows=json.loads((root/'private/original_file_features.json').read_text())
z=d['Z'];y=d['y'];lead=d['lead'];g=d['speaker'];train=d['train'];test=d['test'];u=d['u'];score=d['score']
assert len(rows)==len(z)==1050 and z.shape[1]==160
assert len(set(g[train])&set(g[test]))==0
assert sorted(np.r_[train,test].tolist())==list(range(1050))
assert all(r['source']=='asvspoof' and r['label']==1 for r in rows)
assert all((r['split']=='test')==(i in set(test)) for i,r in enumerate(rows))
model=Ridge(alpha=s['raw_axis']['alpha']).fit(z[train],y[train])
assert np.max(abs(model.predict(z[test])-d['raw_pred']))<1e-12
assert np.max(abs(model.coef_/np.linalg.norm(model.coef_)-u))<1e-12
assert np.max(abs((z-z[train].mean(0))@u-d['P']))<1e-12
assert np.max(abs(z@d['center']-score))<1e-5
checks=[]
for i in np.random.default_rng(20261030).choice(len(rows),20,replace=False):
    r=rows[i];a,sr=sf.read('/data/ASVspoof2021_DF/ASVspoof2021_DF_eval/flac/'+r['id']+'.wav',dtype='float32')
    assert sr==16000 and hashlib.sha256(a.tobytes()).hexdigest()==r['pcm_sha256']
    rms=np.array([np.sqrt(np.mean(a[q:q+320].astype(float)**2)) for q in range(0,len(a),320)])
    for db in [20,30,40,50]:
        value=float(np.flatnonzero(rms>=rms.max()*10**(-db/20))[0])*.02
        assert abs(value-r[f'lead{db}_s'])<1e-10
    checks.append(int(i))

# A separate five-fold speaker partition checks split sensitivity; each fold retunes only internally.
unique=np.unique(g);rng=np.random.default_rng(739391);rng.shuffle(unique)
groups=np.array_split(unique,5);oof=np.empty(len(y));fold_results=[]
alphas=np.logspace(-5,2,15)
for fold,held in enumerate(groups):
    te=np.flatnonzero(np.isin(g,held));tr=np.flatnonzero(~np.isin(g,held))
    cv=np.zeros(len(alphas))
    for ia,ib in GroupKFold(n_splits=4).split(z[tr],y[tr],g[tr]):
        a,b=tr[ia],tr[ib]
        for j,alpha in enumerate(alphas):
            m=Ridge(alpha=float(alpha)).fit(z[a],y[a]);cv[j]+=np.sum((m.predict(z[b])-y[b])**2)
    alpha=float(alphas[np.argmin(cv)]);m=Ridge(alpha=alpha).fit(z[tr],y[tr]);pred=m.predict(z[te]);oof[te]=pred
    v=m.coef_/np.linalg.norm(m.coef_)
    fold_results.append({'fold':fold+1,'n':len(te),'speakers':len(held),'alpha':alpha,
      'spearman':float(spearmanr(lead[te],pred).statistic),'log_duration_r2':float(r2_score(y[te],pred)),
      'coefficient_direction_cosine_to_primary_axis':float(v@u)})
    print('stability fold',fold+1,fold_results[-1],flush=True)

var_total=float(np.var(z[test],axis=0).sum())
cov=np.sum((z[train]-z[train].mean(0))*(y[train]-y[train].mean())[:,None],axis=0)
encoding_axis=cov/np.linalg.norm(cov)
encoding_proj=(z-z[train].mean(0))@encoding_axis
extra={
 'verification':{'passed':True,'raw_audio_feature_rechecks':20,'all_1050_pcm_hashes_verified_in_main':True,
   'ridge_prediction_reproduction_max_error':float(np.max(abs(model.predict(z[test])-d['raw_pred']))),
   'speaker_overlap_n':0,'native_view':0,'inputs_unchanged':True},
 'axis_scale':{'heldout_projection_variance':float(np.var(d['P'][test])),
    'heldout_total_embedding_variance':var_total,
    'heldout_variance_fraction_along_axis':float(np.var(d['P'][test])/var_total),
    'interpretation':'This is the fraction of embedding spread along the chosen axis, NOT the causal fraction explained by low-energy duration.'},
 'speaker_split_stability':{'folds':fold_results,'pooled_out_of_fold_prediction_spearman':float(spearmanr(y,oof).statistic),
    'pooled_out_of_fold_log_duration_r2':float(r2_score(y,oof)),
    'note':'Predictions from five different axes. Not a single spatial coordinate. Test-speaker outcomes were excluded from all fitting and tuning for each fold.'},
 'alternative_mean_trend_axis':{'fit':'Training covariance of embedding and log-duration, normalized (PLS1 direction).',
    'heldout_spearman':float(spearmanr(lead[test],encoding_proj[test]).statistic),
    'cosine_to_ridge_axis':float(encoding_axis@u)},
 'range_sensitivity':[],
 'no_prefix_support':{'all_n':int((lead==0).sum()),'all_speakers':len(set(g[lead==0])),
   'test_n':int((lead[test]==0).sum()),'test_speakers':len(set(g[test][lead[test]==0]))},
 'tail_support':{'heldout_OC_cosine_below_09_n':int((score[test]<.9).sum()),'all_OC_cosine_below_09_n':int((score<.9).sum()),
   'note':'0.9 is an exploratory descriptive cutoff, not a calibrated decision boundary.'}}
for name,mask in [('positive_at_most_1_5_s',(lead[test]>0)&(lead[test]<=1.5)),('at_most_2_s',lead[test]<=2)]:
    te=test[mask];extra['range_sensitivity'].append({'range':name,'n':len(te),
     'spearman_axis':float(spearmanr(lead[te],d['P'][te]).statistic),
     'spearman_OC':float(spearmanr(lead[te],score[te]).statistic)})
(root/'verification_and_stability.json').write_text(json.dumps(extra,indent=2))
np.savez_compressed(root/'private/oof_predictions.npz',predictions=oof)

# Focused views supplement the full-range figure; no hidden outlier removal.
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,
 'axes.grid':True,'grid.alpha':.15,'figure.facecolor':'white','savefig.facecolor':'white'})
norm=matplotlib.colors.PowerNorm(gamma=.5,vmin=0,vmax=float(lead.max()))
fig,axes=plt.subplots(1,2,figsize=(13,5.6),constrained_layout=True)
pc=d['pc'];P=d['P'];Q=d['Q']
for ax,xv,yv,title,xt,yt in [
 (axes[0],pc[:,0],pc[:,1],'A. PCA, central region','PC1','PC2'),
 (axes[1],P,Q,'B. Duration-associated axis, central region','Duration-associated axis (longer direction →)','Orthogonal PC1')]:
    xlo,xhi=np.quantile(xv[train],[.01,.99]);ylo,yhi=np.quantile(yv[train],[.01,.99])
    xp=(xhi-xlo)*.08;yp=(yhi-ylo)*.08
    xlo-=xp;xhi+=xp;ylo-=yp;yhi+=yp
    inside=(xv[test]>=xlo)&(xv[test]<=xhi)&(yv[test]>=ylo)&(yv[test]<=yhi)
    sc=ax.scatter(xv[test],yv[test],c=lead[test],cmap='cividis',norm=norm,s=28,alpha=.8,lw=.2,edgecolor='#333333')
    ax.set(xlim=(xlo,xhi),ylim=(ylo,yhi),xlabel=xt,ylabel=yt,
      title=f'{title}\n{int(inside.sum())}/{len(test)} points visible; full range in companion figure')
    ax.set_aspect('equal',adjustable='box')
fig.colorbar(sc,ax=axes,shrink=.85,label='Original leading low-energy duration (s; nonlinear color scale)')
fig.suptitle('Original ASV bona fide: held-out speakers, focused spatial views\nZoom bounds fixed from training-speaker 1st/99th percentiles; −20 dB / 20 ms',fontsize=14)
fig.savefig(root/'natural_lowenergy_space_zoom.png',dpi=180);fig.savefig(root/'natural_lowenergy_space_zoom.pdf');plt.close(fig)

# A compact display of natural duration bins makes the overlap and unequal group sizes explicit.
binrows=[r for r in s['bins_heldout'] if r['n']]
fig,axes=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
for ax,key,title,ylabel in [(axes[0],'projection_mean','A. Mean coordinate by original duration','Mean duration-axis coordinate'),
                            (axes[1],'cosine_mean','B. Mean cosine by original duration','Mean cosine to OC center')]:
    xx=np.arange(len(binrows));ax.scatter(xx,[r[key] for r in binrows],s=65,c='#34689b')
    ax.set_xticks(xx,[r['bin'].replace(' s','')+'\nn='+str(r['n']) for r in binrows],fontsize=9)
    ax.set(xlabel='Leading low-energy duration (seconds)',ylabel=ylabel,title=title)
    if key=='projection_mean':ax.axhline(0,c='#888888',lw=.8)
fig.suptitle('Original ASV bona fide: descriptive bin means on held-out speakers\nSeparate recordings; unequal speaker/codec mixes; means do not show an intervention effect',fontsize=13)
fig.savefig(root/'natural_lowenergy_bin_means.png',dpi=180);plt.close(fig)
print(json.dumps(extra,indent=2),flush=True)
