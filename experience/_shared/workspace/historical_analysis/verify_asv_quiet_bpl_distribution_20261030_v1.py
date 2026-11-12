import os,sys,json,random,gzip
from pathlib import Path
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1');sys.dont_write_bytecode=True
import numpy as np,soundfile as sf
import asv_quiet_bpl_distribution_20261030_v1 as m
assert Path('/.dockerenv').exists();R=m.ROOT;checks={}
# Explicit synthetic boundary cases, independent expected lengths.
for name,a,expected in [('edges',np.r_[np.zeros(6400),np.ones(16000),np.zeros(3200)],[.4,.2,0.,.6]),('internal',np.r_[np.ones(3200),np.zeros(6400),np.ones(3200)],[0.,0.,.4,.4]),('allzero',np.zeros(9600),[.6,0.,0.,.6]),('lastpartial',np.r_[np.ones(16000),0.],[0.,1/16000,0.,1/16000])]:
 f,p,_,_=m.describe(a);assert np.allclose(f[:,[1,2,3,4]],np.array(expected)[None],atol=1e-10);checks[name]=True
 f2,_,_,_=m.describe(a*.1);assert np.allclose(f[:,:12],f2[:,:12]);checks['relative_amplitude_invariance']=True
rows=json.loads((R/'native_manifest_private.json').read_text());d=np.load(R/'native_metrics_private.npz')['features'];rng=random.Random(439);chosen=[]
for dataset in sorted({t['dataset'] for t in rows}):
 for lab in [0,1]:chosen+=rng.sample([i for i,t in enumerate(rows) if t['dataset']==dataset and t['label']==lab],5)
err=0.
for i in chosen:
 a,sr=sf.read(rows[i]['path']);assert sr==16000;frames=[a[j:j+320] for j in range(0,len(a),320)];rms=np.array([np.sqrt(float(np.dot(v,v))/len(v)) for v in frames]);weights=np.array([len(v) for v in frames])
 for q,db in enumerate([20,40]):
  quiet=rms<rms.max()*10**(-db/20) if rms.max()>0 else np.ones(len(rms),bool);active=np.flatnonzero(~quiet)
  lead=(int(active[0])*320 if len(active) else len(a))/16000;tail=(len(a)-min(len(a),(int(active[-1])+1)*320) if len(active) else 0)/16000;total=float(weights[quiet].sum())/16000;inside=total-lead-tail
  ref=np.array([lead,tail,inside,total,total*16000/len(a)]);err=max(err,float(abs(ref-d[i,q,1:6]).max()))
assert err<2e-5;checks['native_independent_50_files_max_error']=err
sys.path.insert(0,str(m.SOURCE));import data_loaders as dl
from types import SimpleNamespace
pairs=json.loads((R/'pair_manifest_private.json').read_text());meta=[json.loads(line) for line in gzip.open(R/'pair_metadata_private.jsonl.gz','rt')];measured=np.load(R/'pair_metrics_private.npz')['features'];selected=set(rng.sample(range(len(pairs)),16))
for view in ['L','S']:
 for field in [view+'_codec',view+'_augtype']:
  for value in {a[field] for a in meta}:selected.add(next(i for i,a in enumerate(meta) if a[field]==value))
worst=0.
for i in sorted(selected):
 row=pairs[i];seed=910000+row['file_index']*17+row['repeat'];random.seed(seed);np.random.seed(seed);ds=dl.TrainSet({'num_train_frames':404,'num_train_frames_short':303},[SimpleNamespace(path=row['path'],label=row['label'])]);L,S,label=ds[0];assert label==row['label']
 for col,a in enumerate([L,S]):
  f,_,_,_=m.describe(a);worst=max(worst,float(abs(f-measured[i,2,col]).max()))
assert worst<2e-5;checks['original_loader_additional_records']=len(selected);checks['original_loader_metric_max_error']=worst
plan=json.loads((R/'plan.json').read_text());checks['source_unchanged']=m.digest(m.SOURCE/'data_loaders.py')==plan['source_sha256'];assert checks['source_unchanged']
receipt={'readiness':'Share with caveats','checks':checks,'scope':'Nativefull19LAtrain/dev;1000perclasseach19LAeval/21LAeval/21DF_ASVsource.Onlytrainpairsgenerated;all2580bonafide+2580randomspoofs,2drawseach.','qualifications':['LowenergyRMSproxyisnotannotatedsilence;exactzerosreportedseparately.','Internalquietcanincludelowamplitudephonemesandfricatives.','Repeatedfilesarecorrelated;2drawspersourcedofilearenotindependentfiles.','Evaluationcohortsarerandomsamples;21DFonlyasvspoofsource,VCCexcluded.','BPLonlyusesbonafide;spoofpairsdiagnosticonly.','Measuredinputdifferencesdonotprovehistoricalshortcutuse.','Rawsourcequietoverlapvalidonlyforno-codec/no-noisereference.','RawBoostmayraisenoisefloorwithoutremovingoriginalregion.']}
(R/'verification.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2),flush=True)
