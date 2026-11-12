"""Original ASV bona-fide geometry. All per-file data stay on the server."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MPLCONFIGDIR'] = '/home/mentee/.cache/matplotlib'
from pathlib import Path
import json, hashlib, time
from collections import Counter
import numpy as np
import soundfile as sf
from scipy.stats import spearmanr, rankdata
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

assert Path('/.dockerenv').exists(), 'Run only inside the authorized container'
ROOT = Path('/results/analysis/EBM80_native_quiet_strict')
OUT = Path('/results/analysis/EBM80_ASV_bona_natural_lowenergy_axis')
OUT.mkdir(exist_ok=False)
(OUT/'private').mkdir()
SEED = 271828
REPS = 2000
ALPHAS = np.logspace(-5, 2, 15)
plan = {
 'scope': 'ASVspoof2021 DF eval, source=asvspoof, label=bonafide; existing sampled 1050 files, not entire corpus.',
 'checkpoint': '/home/mentee/results/CM/LA/EBM/model/BestCLS_80_21DF.pt',
 'embedding': 'Existing native view 0 only: entire decoded mono 16kHz waveform, no crop, repeat, padding, resampling, gain change or augmentation. L2-normalized 160-D embeddings.',
 'primary_target': 'Leading consecutive 20ms nonoverlapping RMS frames strictly below 1% of the maximum frame RMS (-40 dB); duration in seconds. Zero means no complete leading low-energy frame at this resolution.',
 'secondary_thresholds_db': [-30, -50],
 'axis': 'Ridge predict log1p(leading seconds) from centered embeddings; positive axis is the coefficient direction. No coordinate scaling.',
 'validation': 'Seed 271828 GroupShuffleSplit by speaker: 30% held out. Alpha selected by 4-fold GroupKFold inside training speakers. Test set never used to choose axes, signs, alpha, PCA, or controls.',
 'controls': 'Train-fitted nuisance regressions for target and embeddings: codec, log1p(post-leading duration), squared log1p(post-leading duration), log1p(trailing quiet duration), active-frame RMS dB, its square. Residual-axis alpha chosen with nuisance fits repeated inside each training fold.',
 'sensitivity_controls': 'Replace post-leading duration with total duration, an explicitly secondary part-whole-conditioned check. Also use -30/-50 dB definitions with the SAME primary axis, and positive-duration-only analysis.',
 'uncertainty': '2000 held-out speaker-cluster bootstrap percentile intervals, conditional on training fit; exploratory, no multiplicity-adjusted significance claims.',
 'interpretation': 'Natural cross-file association, not a within-file intervention, not proof of causal shortcut learning; target-informed axis requires held-out replication. Source/label homogeneous. VCC untouched.',
 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
}
(OUT/'plan.json').write_text(json.dumps(plan, indent=2))

manifest = json.loads((ROOT/'sample_manifest.json').read_text())
inf_manifest = json.loads((ROOT/'inference/sample_manifest.json').read_text())
assert manifest == inf_manifest
data = np.load(ROOT/'analysis/merged_private.npz')
assert len(manifest) == len(data['labels']) == 4000
assert np.array_equal(np.array([m['label'] for m in manifest]), data['labels'])
assert np.array_equal(np.array([m['source'] for m in manifest]), data['source'])
assert np.array_equal(np.array([m['speaker'] for m in manifest]), data['speaker'])
idx = np.array([i for i,m in enumerate(manifest) if m['source']=='asvspoof' and m['label']==1])
assert len(idx) == 1050
rows = [manifest[i] for i in idx]
Z = data['embeddings'][idx,0].astype(np.float64)
score = data['scores'][idx,0].astype(np.float64)
center = data['center'].astype(np.float64)
assert Z.shape == (1050,160) and np.isfinite(Z).all()
assert np.max(np.abs(np.linalg.norm(Z, axis=1)-1)) < 1e-5
assert abs(np.linalg.norm(center)-1) < 1e-5
assert np.max(np.abs(Z@center-score)) < 1e-5
speakers = np.array([m['speaker'] for m in rows])
codecs = np.array([m['codec'] for m in rows])
features = []
start = time.time()
for i,m in enumerate(rows):
    audio, sr = sf.read(Path('/data/ASVspoof2021_DF/ASVspoof2021_DF_eval/flac')/(m['id']+'.wav'), dtype='float32')
    assert sr==16000 and audio.ndim==1 and np.isfinite(audio).all()
    assert len(audio)==m['samples']
    assert hashlib.sha256(audio.tobytes()).hexdigest()==m['pcm_sha256']
    # Last short frame uses its actual sample count; no waveform padding.
    sq = audio.astype(np.float64)**2
    starts = np.arange(0,len(audio),320)
    counts = np.minimum(320,len(audio)-starts)
    rms = np.sqrt(np.add.reduceat(sq,starts)/counts)
    peak = float(rms.max()); assert peak > 0
    f = {'total_s':len(audio)/sr, 'peak_frame_dbfs':20*np.log10(peak)}
    for db in [30,40,50]:
        active = np.flatnonzero(rms >= peak*10**(-db/20))
        first,last = int(active[0]),int(active[-1])
        f[f'lead{db}_s'] = min(first*320,len(audio))/sr
        f[f'trail{db}_s'] = max(0,len(audio)-min((last+1)*320,len(audio)))/sr
        if db==40:
            f['active_rms_dbfs'] = 10*np.log10(np.sum(rms[active]**2*counts[active])/np.sum(counts[active]))
    features.append(f)
    if i%250==0: print('raw feature checks',i,len(rows),'elapsed',round(time.time()-start,1),flush=True)
F = {k:np.array([f[k] for f in features]) for k in features[0]}
lead = F['lead40_s']; y = np.log1p(lead)
assert np.all(F['total_s']>lead)
train,test = next(GroupShuffleSplit(n_splits=1,test_size=.3,random_state=SEED).split(Z,y,speakers))
assert not set(speakers[train]) & set(speakers[test])
assert len(set(speakers[train]))>=4
folds = list(GroupKFold(n_splits=4).split(Z[train], y[train], speakers[train]))

def nuisance_matrix(a,b,use_total=False):
    """Fit all scaling and codec vocabulary on a; apply to b."""
    duration = F['total_s'] if use_total else F['total_s']-lead
    base = np.column_stack([np.log1p(duration), np.log1p(F['trail40_s']), F['active_rms_dbfs']])
    mu = base[a].mean(0); sd = np.maximum(base[a].std(0),1e-8)
    standard = (base-mu)/sd
    cat = sorted(set(codecs[a]))
    C = np.column_stack([standard,standard[:,0]**2,standard[:,2]**2]+[(codecs==c).astype(float) for c in cat[1:]])
    return C[a],C[b]

def residualize(a,b,use_total=False):
    ca,cb = nuisance_matrix(a,b,use_total)
    ym = Ridge(alpha=1e-3).fit(ca,y[a]); zm = Ridge(alpha=1e-3).fit(ca,Z[a])
    return Z[a]-zm.predict(ca),Z[b]-zm.predict(cb),y[a]-ym.predict(ca),y[b]-ym.predict(cb),ym.predict(cb),ym,zm

def fit_axis(adjust=False,use_total=False):
    cv = []
    for ia,ib in folds:
        a,b = train[ia],train[ib]
        if adjust:
            xa,xb,ya,yb,*_ = residualize(a,b,use_total)
        else: xa,xb,ya,yb = Z[a],Z[b],y[a],y[b]
        cv.append([np.sum((Ridge(alpha=float(alpha)).fit(xa,ya).predict(xb)-yb)**2) for alpha in ALPHAS])
    mse = np.sum(cv,axis=0)/len(train)
    alpha = float(ALPHAS[int(np.argmin(mse))])
    if adjust:
        xa,xb,ya,yb,base,ym,zm = residualize(train,test,use_total)
    else:
        xa,xb,ya,yb,base,ym,zm = Z[train],Z[test],y[train],y[test],np.zeros(len(test)),None,None
    model = Ridge(alpha=alpha).fit(xa,ya)
    u = model.coef_/np.linalg.norm(model.coef_)
    return dict(alpha=alpha,cv_mse=mse.tolist(),model=model,u=u,train_mean=xa.mean(0),
        train_projection=(xa-xa.mean(0))@u,test_projection=(xb-xa.mean(0))@u,
        test_target=yb,test_pred=model.predict(xb),base=base,ym=ym,zm=zm)

raw = fit_axis()
adjusted = fit_axis(True)
total_adjusted = fit_axis(True,True)
print('axes fitted; heldout N',len(test),'speakers',len(set(speakers[test])),flush=True)

def rho(x,v):
    if len(x)<3 or np.ptp(x)<1e-12 or np.ptp(v)<1e-12:return float('nan')
    return float(spearmanr(x,v).statistic)

def cluster_corr(x,v,g,seed=SEED):
    unique = np.unique(g); groups = [np.flatnonzero(g==s) for s in unique]
    rng = np.random.default_rng(seed); vals=[]
    for _ in range(REPS):
        j=np.concatenate([groups[k] for k in rng.integers(0,len(groups),len(groups))])
        vals.append(rho(x[j],v[j]))
    return {'n':len(x),'speakers':len(unique),'spearman':rho(x,v),
      'ci95':np.nanquantile(vals,[.025,.975]).tolist(),'valid_bootstrap_replicates':int(np.isfinite(vals).sum())}

def demean(x,g):
    out=np.array(x,dtype=float).copy()
    for s in np.unique(g):out[g==s]-=out[g==s].mean()
    return out

def within_speaker_rank_corr(x,v,g):
    # Pooled ranks, followed by speaker fixed effects; singleton groups contribute zero.
    a=demean(rankdata(x),g); b=demean(rankdata(v),g)
    unique=np.unique(g); groups=[np.flatnonzero(g==s) for s in unique]
    rng=np.random.default_rng(SEED+10); vals=[]
    for _ in range(REPS):
        j=np.concatenate([groups[k] for k in rng.integers(0,len(groups),len(groups))])
        vals.append(float(np.dot(a[j],b[j])/np.sqrt(np.dot(a[j],a[j])*np.dot(b[j],b[j]))))
    return {'n':len(x),'speakers':len(unique),
      'partial_rank_correlation':float(np.dot(a,b)/np.sqrt(np.dot(a,a)*np.dot(b,b))),
      'ci95_conditional_on_rank_transform':np.quantile(vals,[.025,.975]).tolist()}

def fit_metrics(fit):
    target=fit['test_target'];pred=fit['test_pred']
    m={'alpha':fit['alpha'],'cv_mse':fit['cv_mse'],
       'heldout_correlation':cluster_corr(target,fit['test_projection'],speakers[test]),
       'heldout_target_r2':float(r2_score(target,pred)),
       'heldout_mse':float(np.mean((target-pred)**2)),
       'axis_cosine_with_OC_center':float(fit['u']@center)}
    if fit['ym'] is not None:
        fullpred=fit['base']+pred
        m['heldout_total_target_r2']=float(r2_score(y[test],fullpred))
        m['nuisance_only_target_r2']=float(r2_score(y[test],fit['base']))
        m['incremental_r2']=m['heldout_total_target_r2']-m['nuisance_only_target_r2']
    return m

P = (Z-raw['train_mean'])@raw['u']
pca = PCA(n_components=8,svd_solver='full').fit(Z[train])
pc = pca.transform(Z)
residual = Z-Z@raw['u'][:,None]*raw['u'][None,:]
orth_pca = PCA(n_components=1,svd_solver='full').fit(residual[train])
v = orth_pca.components_[0]; assert abs(v@raw['u'])<1e-10
Q=(Z-Z[train].mean(0))@v
geometry = []
edges=[0,.1,.5,1,1.5,np.inf]
def bins(mask):
    result=[]
    defs=[('0 s',lead==0),('(0, 0.1] s',(lead>0)&(lead<=.1)),('(0.1, 0.5] s',(lead>.1)&(lead<=.5)),
          ('(0.5, 1.0] s',(lead>.5)&(lead<=1)),('(1.0, 1.5] s',(lead>1)&(lead<=1.5)),('>1.5 s',lead>1.5)]
    for label,sel in defs:
        m=mask&sel;n=int(m.sum())
        r={'bin':label,'n':n,'speakers':len(set(speakers[m]))}
        if n:
            r.update(leading_mean=float(lead[m].mean()),projection_mean=float(P[m].mean()),
              projection_median=float(np.median(P[m])),cosine_mean=float(score[m].mean()),cosine_median=float(np.median(score[m])),
              angle_to_OC_mean_deg=float(np.degrees(np.arccos(np.clip(score[m],-1,1))).mean()),
              OC_cosine_below_09_n=int((score[m]<.9).sum()),
              mean_embedding_norm=float(np.linalg.norm(Z[m].mean(0))),
              total_duration_mean=float(F['total_s'][m].mean()))
        result.append(r)
    return result

T=np.zeros(len(rows),bool);T[test]=True
pos=lead[test]>0
summary={'plan':plan,'sample':{'n':len(rows),'speakers':len(set(speakers)),
 'codecs':dict(Counter(codecs)), 'cohorts':dict(Counter(m['cohort'] for m in rows)),
 'train_n':len(train),'train_speakers':len(set(speakers[train])),
 'test_n':len(test),'test_speakers':len(set(speakers[test])),
 'all_leading_quantiles_s':np.quantile(lead,[0,.1,.25,.5,.75,.9,1]).tolist(),
 'no_complete_lowenergy_frame_n':int((lead==0).sum()),
 'test_no_complete_lowenergy_frame_n':int((lead[test]==0).sum()),
 'all_pcm_hashes_verified':True,'source_manifest_sha256':hashlib.sha256((ROOT/'sample_manifest.json').read_bytes()).hexdigest()},
 'raw_axis':fit_metrics(raw),'adjusted_axis':fit_metrics(adjusted),'total_duration_adjusted_axis':fit_metrics(total_adjusted),
 'natural_lead_vs_OC':cluster_corr(lead[test],score[test],speakers[test]),
 'natural_lead_vs_OC_within_speaker':within_speaker_rank_corr(lead[test],score[test],speakers[test]),
 'axis_vs_lead_within_speaker':within_speaker_rank_corr(lead[test],P[test],speakers[test]),
 'axis_positive_duration_only':cluster_corr(lead[test][pos],P[test][pos],speakers[test][pos]),
 'positive_duration_vs_OC':cluster_corr(lead[test][pos],score[test][pos],speakers[test][pos]),
 'threshold_sensitivity_fixed_primary_axis':{str(db):cluster_corr(F[f'lead{db}_s'][test],P[test],speakers[test]) for db in [30,50]},
 'pca':{'train_explained_variance_ratio':pca.explained_variance_ratio_.tolist(),
   'heldout_PC1_lead_spearman':rho(lead[test],pc[test,0]),'heldout_PC2_lead_spearman':rho(lead[test],pc[test,1])},
 'bins_all':bins(np.ones(len(rows),bool)),'bins_heldout':bins(T),
 'codec_descriptive_heldout':[],
 'integrity':{'embedding_norm_max_abs_error':float(np.max(np.abs(np.linalg.norm(Z,axis=1)-1))),
   'recomputed_score_max_abs_error':float(np.max(np.abs(Z@center-score))),
   'raw_axis_vs_OC_angle_deg':float(np.degrees(np.arccos(np.clip(raw['u']@center,-1,1)))),
   'sphere_cosine_distance_identity_max_error':float(np.max(np.abs(np.sum((Z-center)**2,1)-(2-2*score))))}}
for c in sorted(set(codecs)):
    m=T&(codecs==c)
    summary['codec_descriptive_heldout'].append({'codec':c,'n':int(m.sum()),'speakers':len(set(speakers[m])),
      'lead_axis_spearman':rho(lead[m],P[m]),'lead_OC_spearman':rho(lead[m],score[m])})

# Private diagnostic tables and axes are never downloaded to the local client.
private=[dict(m,**f,split=('test' if T[i] else 'train'),axis_projection=float(P[i]),OC_cosine=float(score[i])) for i,(m,f) in enumerate(zip(rows,features))]
(OUT/'private/original_file_features.json').write_text(json.dumps(private,indent=2))
np.savez_compressed(OUT/'private/axis_arrays.npz',Z=Z,center=center,lead=lead,y=y,score=score,
 P=P,Q=Q,pc=pc,train=train,test=test,speaker=speakers,codec=codecs,u=raw['u'],adjusted_u=adjusted['u'],
 mean=raw['train_mean'],orth_axis=v,raw_pred=raw['test_pred'],adj_pred=adjusted['test_pred'],adj_target=adjusted['test_target'])
(OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':13,'axes.spines.top':False,
 'axes.spines.right':False,'axes.grid':True,'grid.alpha':.16,'figure.facecolor':'white','savefig.facecolor':'white'})
color='#34689b'; gold='#b68527'; norm=plt.Normalize(0,float(lead.max()))
fig,axes=plt.subplots(1,2,figsize=(14,5.5),constrained_layout=True)
for ax in axes:ax.set_aspect('equal',adjustable='datalim')
sc=axes[0].scatter(pc[test,0],pc[test,1],c=lead[test],cmap='cividis',norm=norm,s=22,alpha=.85,lw=.25,edgecolor='#333333')
axes[0].set(xlabel=f'PC1 ({100*pca.explained_variance_ratio_[0]:.1f}% train variance)',ylabel=f'PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)',title='A. PCA: duration not used to fit the axes')
axes[1].scatter(P[test],Q[test],c=lead[test],cmap='cividis',norm=norm,s=22,alpha=.85,lw=.25,edgecolor='#333333')
cx=float((center-raw['train_mean'])@raw['u']);cy=float((center-Z[train].mean(0))@v)
axes[1].scatter([cx],[cy],marker='*',s=190,c='#ce623b',edgecolor='black',lw=.6,label='Trained OC center',zorder=5)
axes[1].legend(loc='best',fontsize=10)
axes[1].set(xlabel='Duration-associated axis (longer direction →)',ylabel='Orthogonal PC1',title='B. Axis fitted on training speakers only')
fig.colorbar(sc,ax=axes,shrink=.8,label='Original leading low-energy duration (seconds)')
fig.suptitle(f'Original ASV bona fide: held-out speakers only (n={len(test)}, speakers={len(set(speakers[test]))})\nFull native audio; no waveform modification; color threshold −40 dB / 20 ms',fontsize=14)
fig.savefig(OUT/'natural_lowenergy_space.png',dpi=180);fig.savefig(OUT/'natural_lowenergy_space.pdf');plt.close(fig)

order=test[np.argsort(lead[test],kind='stable')]; rank=np.arange(1,len(order)+1)
fig,axes=plt.subplots(3,1,figsize=(12,9.2),sharex=True,constrained_layout=True)
for ax,vv,title,ylabel in zip(axes,[lead,P,score],
 ['A. Sorting key: original leading low-energy duration','B. Projection on independently fitted duration axis','C. Similarity to trained bonafide OC center'],
 ['Leading duration (s)','Axis projection','Cosine to OC center']):
    ax.scatter(rank,vv[order],c=lead[order],cmap='cividis',norm=norm,s=19,edgecolor='none',alpha=.85)
    ax.set(title=title,ylabel=ylabel)
axes[1].axhline(0,c='#666666',lw=.8)
axes[2].set_ylim(min(.0,float(score[test].min())-.05),1.01)
axes[2].axhline(.9,c='#888888',ls='--',lw=.8)
axes[2].text(len(order)*.99,.89,'0.9 descriptive tail cutoff; not a classification threshold',ha='right',va='top',fontsize=9)
axes[2].set_xlabel('File rank after sorting by leading low-energy duration (ties arbitrary; separate files)')
fig.suptitle('Original ASV bona fide: identical held-out files, ordered by low-energy duration\nPoints are different recordings, not a trajectory of transformed audio',fontsize=14)
fig.savefig(OUT/'natural_lowenergy_sorted.png',dpi=180);fig.savefig(OUT/'natural_lowenergy_sorted.pdf');plt.close(fig)

fig,axes=plt.subplots(1,3,figsize=(15,4.7),constrained_layout=True)
for ax,xx,vv,title,labx,laby in [
 (axes[0],lead[test],P[test],'A. Natural duration vs axis','Leading low-energy duration (s)','Axis projection'),
 (axes[1],adjusted['test_target'],adjusted['test_projection'],'B. Nuisance-adjusted association','Residual log(1 + leading seconds)','Residual axis projection'),
 (axes[2],lead[test],score[test],'C. Natural duration vs OC center','Leading low-energy duration (s)','Cosine to OC center')]:
    ax.scatter(xx,vv,s=18,c=color,alpha=.6,lw=0)
    ax.set(title=title,xlabel=labx,ylabel=laby)
    rr=rho(xx,vv);ax.text(.03,.97,f'held-out Spearman ρ = {rr:.3f}',transform=ax.transAxes,va='top',fontsize=10)
axes[2].set_ylim(min(.0,float(score[test].min())-.05),1.01)
fig.suptitle('Original ASV bona fide: held-out relationship checks\nControls: codec, post-leading duration, trailing duration and active-frame RMS',fontsize=14)
fig.savefig(OUT/'natural_lowenergy_relationships.png',dpi=180);fig.savefig(OUT/'natural_lowenergy_relationships.pdf');plt.close(fig)

def fm(x):return f'{x:.3f}'
text = ['# Original ASV bona fide: leading low-energy duration and embedding geometry','',
 'Scope: 1,050 sampled ASV-source bona fide files from ASVspoof2021 DF eval. No VCC. Full native waveform embeddings, no synthetic transformations.',
 f'Train: {len(train)} files / {len(set(speakers[train]))} speakers. Held out: {len(test)} files / {len(set(speakers[test]))} speakers.',
 'Leading low energy: consecutive initial 20 ms RMS frames below 1% of maximum frame RMS (−40 dB). Zero is resolution-dependent, not certified silence absence.',
 '', '| Held-out check | Estimate | 95% speaker-cluster CI |','|---|---:|---:|']
for name,m in [('Duration vs fitted axis',summary['raw_axis']['heldout_correlation']),
 ('Nuisance-adjusted residual duration vs residual axis',summary['adjusted_axis']['heldout_correlation']),
 ('Duration vs cosine to OC center',summary['natural_lead_vs_OC']),
 ('Positive-duration files only: duration vs axis',summary['axis_positive_duration_only'])]:
    text.append(f"|{name}|{m['spearman']:.4f}|[{m['ci95'][0]:.4f}, {m['ci95'][1]:.4f}]|")
text += ['',f"Held-out log-duration prediction R²: {summary['raw_axis']['heldout_target_r2']:.4f}. Nuisance-only R²: {summary['adjusted_axis']['nuisance_only_target_r2']:.4f}; combined R²: {summary['adjusted_axis']['heldout_total_target_r2']:.4f}.",'',
 '| Original leading low energy | Held-out n | Mean axis projection | Mean OC cosine | Mean angle to OC (deg) | OC < 0.9 n |',
 '|---|---:|---:|---:|---:|---:|']
for r in summary['bins_heldout']:
    if r['n']: text.append(f"|{r['bin']}|{r['n']}|{r['projection_mean']:.4f}|{r['cosine_mean']:.4f}|{r['angle_to_OC_mean_deg']:.2f}|{r['OC_cosine_below_09_n']}|")
text += ['','Important limits:',
 '- This is a natural cross-file association. It cannot establish that changing duration alone moves a recording, or that training causally exploited duration as a shortcut.',
 '- The axis is supervised on training speakers. Its held-out association is the relevant check; training-set visual alignment alone is circular.',
 '- Speaker clustering and speaker-disjoint fitting do not remove every recording, codec or utterance confound. Nuisance adjustment is model-dependent.',
 '- All samples are bona fide. This analysis alone cannot quantify discrimination benefit, EER changes or a B/S shortcut.',
 '- 21DF participated in earlier model selection. Held-out refers only to this axis fitting, not to a pristine model-development test.',
 '- Full native inputs differ from the fixed crop/repeat input used in some earlier figures. Those earlier tail plots are not directly reproduced here.',
 '- Bootstrap intervals condition on the fitted training axis. They do not cover retraining or axis selection uncertainty.',
 '', 'Artifacts: natural_lowenergy_space.png/pdf, natural_lowenergy_sorted.png/pdf, natural_lowenergy_relationships.png/pdf; summary.json. Per-file rows and axes stay in private/.']
(OUT/'report.md').write_text('\n'.join(text)+'\n')
completion={'status':'complete','elapsed_seconds':time.time()-start,'n':len(rows),'view':0,
 'artifacts':[p.name for p in OUT.iterdir() if p.is_file()],
 'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(OUT/'COMPLETED.json').write_text(json.dumps(completion,indent=2))
print(json.dumps({'complete':True,'sample':summary['sample'],'raw_axis':summary['raw_axis'],
 'adjusted_axis':summary['adjusted_axis'],'natural_lead_vs_OC':summary['natural_lead_vs_OC']},indent=2),flush=True)
