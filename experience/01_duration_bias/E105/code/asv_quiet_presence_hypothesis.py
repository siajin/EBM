import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import json,hashlib,time
from collections import Counter
import numpy as np
import soundfile as sf
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
assert Path('/.dockerenv').exists()
root=Path('/results/analysis/EBM80_native_quiet_strict')
out=Path('/results/analysis/EBM80_ASV_quiet_presence_hypothesis_20261030')
out.mkdir(exist_ok=False);(out/'private').mkdir()
plan={'population':'Existing original/native samples only; ASVspoof2021 DF source=asvspoof, both true labels. No VCC inspection.',
 'checkpoint':'/home/mentee/results/CM/LA/EBM/model/BestCLS_80_21DF.pt',
 'input':'Entire native waveform, existing frozen-model view 0. No new inference or waveform intervention.',
 'definitions':'Leading consecutive nonoverlapping 20ms frame RMS below 10% of full-file maximum RMS (-20dB); -40dB sensitivity.',
 'fixed_cutoffs_seconds':[.02,.1,.5],
 'cutoff_meaning':'>=20ms means at least one measured leading quiet frame; >=100ms and >=500ms are descriptive sustained-prefix thresholds. They are not picked for accuracy.',
 'availability_vs_use':'Length-only balanced accuracy measures availability of a correlated cue, NOT whether the network uses it.',
 'interpretation':'Natural cross-file associations, not causal shortcut identification. Model selection involved 21DF. No VCC causal/generalization claim from this analysis.',
 'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
(out/'plan.json').write_text(json.dumps(plan,indent=2))
manifest=json.loads((root/'sample_manifest.json').read_text());a=np.load(root/'analysis/merged_private.npz')
ix=np.array([i for i,m in enumerate(manifest) if m['source']=='asvspoof']);rows=[manifest[i] for i in ix]
y=np.array([r['label'] for r in rows]);g=np.array([r['speaker'] for r in rows]);att=np.array([r['attack'] for r in rows]);codec=np.array([r['codec'] for r in rows]);score=a['scores'][ix,0].astype(float)
assert len(rows)==1298 and (y==1).sum()==1050 and (y==0).sum()==248
assert np.array_equal(y,a['labels'][ix]) and np.array_equal(g,a['speaker'][ix])
assert np.max(abs(a['embeddings'][ix,0].astype(float)@a['center'].astype(float)-score))<1e-5
prefix={20:[],40:[]};duration=[]
for j,r in enumerate(rows):
    audio,sr=sf.read('/data/ASVspoof2021_DF/ASVspoof2021_DF_eval/flac/'+r['id']+'.wav',dtype='float32')
    assert sr==16000 and audio.ndim==1 and len(audio)==r['samples']
    assert hashlib.sha256(audio.tobytes()).hexdigest()==r['pcm_sha256']
    rms=np.array([np.sqrt(np.mean(audio[k:k+320].astype(float)**2)) for k in range(0,len(audio),320)])
    assert rms.max()>0
    for db in prefix:prefix[db].append(int(np.flatnonzero(rms>=rms.max()*10**(-db/20))[0])*.02)
    duration.append(len(audio)/sr)
prefix={db:np.array(v) for db,v in prefix.items()}
lead=prefix[20]
def rho(x,z):
    return float(spearmanr(x,z).statistic) if len(x)>=3 and np.ptp(x)>0 and np.ptp(z)>0 else None
def describe(mask):
    n=int(mask.sum());r={'n':n,'speakers':len(set(g[mask]))}
    if n:r.update(lead_median_s=float(np.median(lead[mask])),cosine_mean=float(score[mask].mean()),
        cosine_median=float(np.median(score[mask])),cosine_q10=float(np.quantile(score[mask],.1)),cosine_q90=float(np.quantile(score[mask],.9)),
        OC_cosine_ge09_n=int((score[mask]>=.9).sum()))
    return r
def boot_rho(mask):
    z=score[mask];x=lead[mask];sg=g[mask];groups=[np.flatnonzero(sg==v) for v in np.unique(sg)]
    rng=np.random.default_rng(684092);vals=[]
    for _ in range(2000):
        j=np.concatenate([groups[k] for k in rng.integers(0,len(groups),len(groups))]);r=rho(x[j],z[j])
        if r is not None:vals.append(r)
    return {'n':len(x),'speakers':len(groups),'spearman':rho(x,z),'ci95_speaker_bootstrap':np.quantile(vals,[.025,.975]).tolist()}
summary={'plan':plan,'n':len(rows),'classes':{},'cutoff_rules':{},'score_strata':{},'natural_score_association':{},'spoof_attack_strata':[],
 'note_score':'Higher cosine means more bonafide-like. >=0.9 is a descriptive region, not calibrated classification accuracy.',
 'verification':{'all_original_pcm_hashes_verified':True,'cached_cosine_recomputed':True,'source_only_asvspoof':True}}
for lab,name in [(1,'bonafide'),(0,'spoof')]:
    m=y==lab;summary['classes'][name]=describe(m)
    summary['classes'][name]['duration_quantiles_s']={str(db):np.quantile(v[m],[0,.25,.5,.75,1]).tolist() for db,v in prefix.items()}
    summary['classes'][name]['source_cohort_counts']=dict(Counter(r['cohort'] for r in np.array(rows,dtype=object)[m]))
    summary['natural_score_association'][name]=boot_rho(m)
for db,lead_db in prefix.items():
    rules=[]
    for cutoff in [.02,.1,.5]:
        quiet=lead_db>=cutoff
        nb=int(((y==1)&quiet).sum());ns=int(((y==0)&quiet).sum())
        rules.append({'cutoff_s':cutoff,'bonafide_with_prefix_n':nb,'bonafide_total_n':int((y==1).sum()),
          'spoof_with_prefix_n':ns,'spoof_total_n':int((y==0).sum()),
          'bonafide_with_prefix_fraction':float(quiet[y==1].mean()),'spoof_with_prefix_fraction':float(quiet[y==0].mean()),
          'length_only_balanced_accuracy':float((quiet[y==1].mean()+(~quiet[y==0]).mean())/2)})
    summary['cutoff_rules'][str(db)]={'rules':rules,'length_only_AUC':float(roc_auc_score(y,lead_db))}
for cutoff in [.02,.1,.5]:
    summary['score_strata'][str(cutoff)]={}
    for lab,name in [(1,'bonafide'),(0,'spoof')]:
        summary['score_strata'][str(cutoff)][name]={'shorter':describe((y==lab)&(lead<cutoff)),
           'at_least_cutoff':describe((y==lab)&(lead>=cutoff))}
for attack in sorted(set(att[y==0])):
    m=(y==0)&(att==attack);r=describe(m);r.update(attack=attack,n_ge100ms=int((m&(lead>=.1)).sum()),lead_score_spearman=rho(lead[m],score[m]));summary['spoof_attack_strata'].append(r)
summary['frozen_model_native_ASV_AUC']=float(roc_auc_score(y,score))
(out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
np.savez_compressed(out/'private/original_presence_features.npz',manifest_index=ix,label=y,lead20=lead,lead40=prefix[40],score=score,speaker=g,codec=codec,attack=att)
report=['# ASV natural leading low-energy presence: cue availability, not causal use','',
 'Full original/native ASV-source sample only; B=1050, S=248; frozen epoch80. VCC not inspected.',
 'Primary low-energy criterion: 20ms RMS <10% of full-file max RMS (relative -20dB).','',
 '| Required leading prefix | B with prefix | S with prefix | Length-only balanced accuracy |','|---|---:|---:|---:|']
for r in summary['cutoff_rules']['20']['rules']:
    report.append(f"|>={r['cutoff_s']:.2f}s|{r['bonafide_with_prefix_n']}/1050 ({100*r['bonafide_with_prefix_fraction']:.1f}%)|{r['spoof_with_prefix_n']}/248 ({100*r['spoof_with_prefix_fraction']:.1f}%)|{100*r['length_only_balanced_accuracy']:.1f}%|")
report+=['','The threshold-only rule predicts bonafide if prefix duration is at least the cutoff. Balanced accuracy avoids reporting an inflated majority-class accuracy. This is not the model accuracy.',
 'Model score strata are observational: source, codec, attack and waveform properties can confound the association. A score region at cosine 0.9 is descriptive and is not a calibrated classification boundary.',
 'B-only natural length/score correlation does not test B-vs-S adoption of a presence cue, especially with near-complete B exposure.',
 'Testing whether this cue causes VCC degradation requires a controlled ASV-trained intervention/retraining comparison, with VCC held out from tuning. No such causal estimate is made here.']
(out/'report.md').write_text('\n'.join(report)+'\n')
(out/'COMPLETED.json').write_text(json.dumps({'status':'complete','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'n':len(rows)}))
print(json.dumps(summary,indent=2),flush=True)
