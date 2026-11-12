import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
os.environ['MPLCONFIGDIR']='/home/mentee/.cache/matplotlib'
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
assert Path('/.dockerenv').exists()
root=Path('/results/analysis/EBM80_native_quiet_strict');out=root/'analysis';out.mkdir(exist_ok=False)
idir=root/'inference';trials=json.loads((idir/'sample_manifest.json').read_text())
done=json.loads((idir/'COMPLETED.json').read_text());assert done['completed_trials']==4000
shards=[np.load(idir/f'shard_{rank}.npz') for rank in range(2)]
ix=np.concatenate([x['indices'] for x in shards]);order=np.argsort(ix)
assert sorted(ix.tolist())==list(range(4000))
data={k:np.concatenate([x[k] for x in shards])[order] for k in ['scores','embeddings','eligible','prefixes','lengths','checks','seams']}
assert np.array_equal(shards[0]['center'],shards[1]['center'])
center=shards[0]['center'].astype(float);s=data['scores'].astype(float);z=data['embeddings'].astype(float)
y=np.array([t['label'] for t in trials]);source=np.array([t['source'] for t in trials]);speaker=np.array([t['speaker'] for t in trials]);cohort=np.array([t['cohort'] for t in trials])
eligible=data['eligible'].astype(bool);native=s[:,0]
eps=max(1e-4,10*float(np.max(np.abs(s[:,1]-native))))
R=20000;rng=np.random.default_rng(904123)
family_indices={'zero':[(2,3),(6,7),(10,11)],'own40':[(4,5),(8,9),(12,13)]}
def angle(a):return np.degrees(np.arccos(np.clip(a,-1,1)))
def effect(before,after,mask):
    a=s[mask,before];b=s[mask,after];v=b-a;n=len(v)
    if n==0:return {'n':0,'available':False,'reason':'empty subgroup; no estimate'}
    zz0=z[mask,before];zz1=z[mask,after];disp=np.linalg.norm(zz1-zz0,axis=1)
    unit=center/np.linalg.norm(center);dz=zz1-zz0
    radial=dz@unit;transverse=np.linalg.norm(dz-radial[:,None]*unit,axis=1)
    paircos=np.sum(zz0*zz1,axis=1)/(np.linalg.norm(zz0,axis=1)*np.linalg.norm(zz1,axis=1))
    return {'n':n,'mean':float(v.mean()),'median':float(np.median(v)),
      'q10':float(np.quantile(v,.1)),'q90':float(np.quantile(v,.9)),
      'toward_n':int((v>eps).sum()),'away_n':int((v<-eps).sum()),'within_epsilon_n':int((np.abs(v)<=eps).sum()),
      'raw_positive_n':int((v>0).sum()),'toward_ge001_n':int((v>=.01).sum()),'away_ge001_n':int((v<=-.01).sum()),
      'toward_ge01_n':int((v>=.1).sum()),'away_ge01_n':int((v<=-.1).sum()),
      'mean_angle_to_center_change_deg':float((angle(b)-angle(a)).mean()),
      'mean_actual_embedding_displacement':float(disp.mean()),'median_actual_embedding_displacement':float(np.median(disp)),
      'mean_actual_embedding_rotation_deg':float(angle(paircos).mean()),
      'mean_transverse_displacement':float(transverse.mean()),
      'center_crossings_in_n':int(((a<.9)&(b>=.9)).sum()),'center_crossings_out_n':int(((a>=.9)&(b<.9)).sum())}
def boot_matrix(delta,mask):
    rows=np.flatnonzero(mask);dd=delta[mask]
    feat=np.concatenate([dd,(dd>eps).astype(float)],axis=1)
    boot=np.zeros((R,feat.shape[1]));cluster_counts={}
    for src in np.unique(source[mask]):
        loc=np.flatnonzero(source[rows]==src)
        groups=[loc[speaker[rows[loc]]==sp] for sp in np.unique(speaker[rows[loc]])]
        counts=np.array([len(g) for g in groups]);sums=np.stack([feat[g].sum(0) for g in groups])
        draws=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=R)
        boot+=(draws@sums)/(draws@counts)[:,None]*(len(loc)/len(rows));cluster_counts[str(src)]=len(groups)
    return {'q95':np.quantile(boot,[.025,.975],axis=0),'qsim':np.quantile(boot,[.003125,.996875],axis=0),'clusters':cluster_counts}
r={'provenance':{'plan_sha256':hashlib.sha256((root/'plan.json').read_bytes()).hexdigest(),
 'inference_provenance':json.loads((idir/'provenance.json').read_text()),'n':4000,'bootstrap_replicates':R,
 'primary_simultaneous_ci':'Per-endpoint percentile speaker-cluster interval 99.375%; Bonferroni across 8 prespecified new-file means gives approximate familywise 95%. No exact p-values claimed.',
 'direction_epsilon':eps,'identity_repeat_max_delta':float(np.max(np.abs(s[:,1]-native))),
 'score':'Cosine to trained bonafide OC center; positive differences mean toward center, not a probability.',
 'source_weighting':'Within each true-label population, retain observed source weights; resample speaker clusters within source.'},
 'cohorts':{},'primary':[],'source_effects':{},'matched_eligibility':{},'ramp_controls':{},'tail':{}}
for c in ['new_files','previous']:
    r['cohorts'][c]={}
    for fam,pairs in family_indices.items():
        pairdefs=[]
        for dose,(front,back) in zip([.25,.5,1.],pairs):
            pairdefs += [(dose,'front_original',0,front),(dose,'back_original',0,back),(dose,'front_back',back,front)]
        delta=np.stack([s[:,after]-s[:,before] for _,_,before,after in pairdefs],axis=1)
        r['cohorts'][c][fam]={}
        for label,word in [(1,'bonafide'),(0,'spoof')]:
            mask=(cohort==c)&(y==label)
            if fam=='own40':mask &= eligible
            assert mask.sum()>10
            bs=boot_matrix(delta,mask);records=[]
            for k,(dose,contrast,before,after) in enumerate(pairdefs):
                a=effect(before,after,mask)
                a.update(dose_s=dose,contrast=contrast,mean_ci95=bs['q95'][:,k].tolist(),
                   fraction_toward_ci95=bs['q95'][:,len(pairdefs)+k].tolist(),speaker_clusters=bs['clusters'])
                if c=='new_files' and dose==1 and contrast in ['front_original','front_back']:
                    a['mean_simultaneous_ci']=bs['qsim'][:,k].tolist()
                    a['primary_mean_direction']='toward' if bs['qsim'][0,k]>0 else ('away' if bs['qsim'][1,k]<0 else 'unresolved')
                    r['primary'].append(dict(a,family=fam,label=word))
                records.append(a)
            r['cohorts'][c][fam][word]={'eligible_n':int(mask.sum()),'total_label_n':int(((cohort==c)&(y==label)).sum()),
                'source_counts':{src:int((mask&(source==src)).sum()) for src in np.unique(source)},'effects':records}
            print(f'statistics {c}/{fam}/{word}, n={mask.sum()}',flush=True)
assert len(r['primary'])==8
# Secondary checks have no confirmatory hypothesis tests.
for fam,pairs in family_indices.items():
    f,b=pairs[-1];r['source_effects'][fam]={}
    for src in np.unique(source):
        r['source_effects'][fam][src]={}
        for lab,word in [(1,'bonafide'),(0,'spoof')]:
            mask=(cohort=='new_files')&(source==src)&(y==lab)
            if fam=='own40':mask &= eligible
            r['source_effects'][fam][src][word]={key:effect(before,after,mask) for key,before,after in [('front_original',0,f),('back_original',0,b),('front_back',b,f)]}
for lab,word in [(1,'bonafide'),(0,'spoof')]:
    mask=(cohort=='new_files')&(y==lab)&eligible
    r['matched_eligibility'][word]={fam:{key:effect(before,after,mask) for key,before,after in
       [('front_original',0,pairs[-1][0]),('back_original',0,pairs[-1][1]),('front_back',pairs[-1][1],pairs[-1][0])]} for fam,pairs in family_indices.items()}
    m=(cohort=='new_files')&(y==lab)
    r['ramp_controls'][word]={key:effect(before,after,m) for key,before,after in [('front_original',0,14),('back_original',0,15),('front_back',15,14),('ramp_front_minus_zero_front',10,14),('ramp_back_minus_zero_back',11,15)]}
    # Endpoint-matching ramp energy, relative to native energy, obtained from saved jump/RMS ratios.
    scale=float(np.sum(np.linspace(0,1,160)**2))
    rr=scale*data['seams'][m,2,0,:]**2/data['lengths'][m,None]
    r['ramp_controls'][word]['added_energy_over_native']={'front_median':float(np.median(rr[:,0])),'back_median':float(np.median(rr[:,1])),
        'front_max':float(rr[:,0].max()),'back_max':float(rr[:,1].max())}
    r['ramp_controls'][word]['zero_splice_jump_over_native_rms_median']=np.median(data['seams'][m,2,0],axis=0).tolist()
# Native-tail strata are outcome-defined exploratory summaries, never causal training subgroups.
for group,mask in [('native_tail',(cohort=='new_files')&(y==1)&(native<.9)),('native_core',(cohort=='new_files')&(y==1)&(native>=.9))]:
    r['tail'][group]={'n':int(mask.sum()),'zero_front_original':effect(0,10,mask),'zero_front_back':effect(11,10,mask)}
r['mechanical_checks']={'maximum_energy_error':float(data['checks'][:,0].max()),'preservation_flags_all_true':bool(np.all(data['checks'][:,1:]==1)),
 'native_lengths_match_manifest':bool(np.array_equal(data['lengths'],[t['samples'] for t in trials])),
 'primary_complete':len(r['primary'])==8}
assert all(r['mechanical_checks'][k] for k in ['preservation_flags_all_true','native_lengths_match_manifest','primary_complete'])
r['interpretation_limits']=['Same native samples preserved exactly in every primary intervention; no alteration of the existing speech or low-energy content.',
 'Front/back comparisons vary placement and the splice context at fixed added length/energy; front-original also increases total length.',
 'Own40 is a relative energy proxy; eligibility is not random and repetition can add periodic structure. Family comparisons use a shared eligible subset.',
 'Ramp controls alter only added filler but front and back filler waveforms/energies are not identical; they are a sensitivity check, not a pure join-mechanism proof.',
 'New-file results are prespecified confirmation within the same corpus, not new-speaker, independent-utterance, external-domain or pristine model-development validation.',
 'Percentile cluster bootstrap intervals are approximate and condition on one checkpoint, the observed source mixture, and selected intervention forms.',
 'Native evaluation differs from prior 64,600-sample crop/wrap; no claimed EER improvement against prior runs.',
 'A frozen-model input response does not prove the historical training origin, exclusive reliance on shortcuts, or a causal explanation for every tail point.']
(out/'summary.json').write_text(json.dumps(r,indent=2,allow_nan=False))
np.savez_compressed(out/'merged_private.npz',**data,labels=y,source=source,speaker=speaker,cohort=cohort,center=center)

fig,axes=plt.subplots(2,2,figsize=(12,8),sharey=True,layout='constrained')
for row,word in enumerate(['bonafide','spoof']):
    for col,fam in enumerate(['zero','own40']):
        ax=axes[row,col];entry=r['cohorts']['new_files'][fam][word]
        for contrast,title,color,marker in [('front_original','Add at front','#286FA2','o'),('back_original','Add at back','#C67B32','s')]:
            rr=[a for a in entry['effects'] if a['contrast']==contrast]
            xx=[a['dose_s'] for a in rr];mm=np.array([a['mean'] for a in rr]);ci=np.array([a['mean_ci95'] for a in rr])
            ax.plot(xx,mm,marker=marker,color=color,label=title);ax.fill_between(xx,ci[:,0],ci[:,1],color=color,alpha=.15)
        ax.axhline(0,c='#777',lw=1);ax.set(title=f'{word} | {fam} | n={entry["eligible_n"]}',xlabel='Added duration (seconds)',ylabel='Mean cosine change from native waveform');ax.legend();ax.grid(alpha=.15)
fig.suptitle('Complete native waveform preserved | new files only | positive = toward bonafide center\nBands: descriptive 95% speaker-cluster intervals; 8 primary endpoints use wider multiplicity-adjusted intervals',fontsize=12)
fig.savefig(out/'native_quiet_dose.png',dpi=170);plt.close(fig)
print(json.dumps({'primary':r['primary'],'checks':r['mechanical_checks'],'identity':r['provenance']['identity_repeat_max_delta']},indent=2),flush=True)
