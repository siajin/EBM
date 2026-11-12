import os,sys,json,time,random,hashlib,gzip,multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1');sys.dont_write_bytecode=True
import numpy as np
ROOT=Path('/results/analysis/ASV_quiet_BPL_distribution_20261030_v1');SOURCE=Path('/home/mentee/code/EBM')
FN=['duration_s','leading_s','trailing_s','internal_s','total_s','fraction','internal_runs','longest_internal_s','runs','exactzero_fraction','leading_exactzero_s','trailing_exactzero_s','max_rms_dbfs','quiet_energy_fraction','all_zero']
TH=[20,40];STAGES=['raw_crop_reference','codec_crop_before_RawBoost','final_relative_threshold','final_preRawBoost_fixed_threshold']
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def frameinfo(a):
 n=len(a);assert n>0 and np.isfinite(a).all();k=(n+319)//320;length=np.full(k,320.);length[-1]=n-320*(k-1)
 pad=np.pad(a,(0,k*320-n));energy=np.sum(pad.reshape(k,320)**2,axis=1);rms=np.sqrt(energy/length);bounds=np.r_[np.arange(k)*320,n]
 return rms,energy,length,bounds

def describe(a,reference_peak=None,with_intervals=False):
 a=np.asarray(a,dtype=np.float64);rms,energy,length,bounds=frameinfo(a);peak=float(rms.max());ref=peak if reference_peak is None else reference_peak
 nz=np.flatnonzero(a!=0);zlead=int(nz[0]) if len(nz) else len(a);ztail=len(a)-int(nz[-1])-1 if len(nz) else 0;zero=float(np.mean(a==0));features=[];profiles=[];intervals=[]
 for db in TH:
  quiet=(rms<ref*10**(-db/20)) if ref>0 else np.ones(len(rms),bool)
  switches=np.diff(np.r_[False,quiet,False].astype(np.int8));starts=np.flatnonzero(switches==1);ends=np.flatnonzero(switches==-1);spans=[(int(bounds[s]),int(bounds[e])) for s,e in zip(starts,ends)]
  leading=spans[0][1] if spans and spans[0][0]==0 else 0;trailing=len(a)-spans[-1][0] if spans and spans[-1][1]==len(a) and spans[-1][0]>0 else 0
  internal=[(s,e) for s,e in spans if s>0 and e<len(a)];total=float(length[quiet].sum());inside=sum(e-s for s,e in internal)
  assert abs(total-leading-trailing-inside)<1e-6
  feats=[len(a)/16000,leading/16000,trailing/16000,inside/16000,total/16000,total/len(a),len(internal),max([e-s for s,e in internal],default=0)/16000,len(spans),zero,zlead/16000,ztail/16000,20*np.log10(max(peak,1e-15)),float(energy[quiet].sum()/max(energy.sum(),1e-30)),float(peak==0)]
  cumulative=np.r_[0,np.cumsum(length*quiet)];edges=np.linspace(0,len(a),21);amount=np.interp(edges,bounds,cumulative);profile=np.diff(amount)/np.diff(edges)
  assert np.all((profile>=-1e-8)&(profile<=1+1e-8)) and abs(profile.mean()-total/len(a))<1e-7
  features.append(feats);profiles.append(profile);intervals.append([[s/16000,e/16000] for s,e in spans])
 return np.array(features),np.array(profiles),intervals,peak

def init_native():
 global NROWS,SF
 import soundfile as sf
 SF=sf;NROWS=json.loads((ROOT/'native_manifest_private.json').read_text())
def native_one(i):
 row=NROWS[i];a,sr=SF.read(row['path']);assert sr==16000 and a.ndim==1;f,p,r,peak=describe(a)
 return i,f,p,r if row['label']==1 else None

def crop(a,n,start=0):return np.pad(a,(0,n-len(a)),'wrap') if len(a)<n else a[start:start+n].copy()
def init_pair():
 global PROWS,SF,DL,Codec
 import soundfile as sf
 SF=sf;PROWS=json.loads((ROOT/'pair_manifest_private.json').read_text());sys.path.insert(0,str(SOURCE))
 import data_loaders as dl
 DL=dl;Codec=dl.CodecAugmentationLight

def apply(a,t):
 L=DL.LnL_convolutive_noise;I=DL.ISD_additive_noise;S=DL.SSI_additive_noise
 if t==1:return L(a)
 if t==2:return I(a)
 if t==3:return S(a)
 if t==4:return S(I(L(a)))
 if t==5:return I(L(a))
 if t==6:return S(L(a))
 if t==7:return S(I(a))
 if t==8:return DL.normWav(L(a)+I(a),0)
 return a

def quiet_intersection(raw,start):
 rms,energy,length,bounds=frameinfo(raw);ans=[];n=len(raw)
 for db in TH:
  q=rms<rms.max()*10**(-db/20) if rms.max()>0 else np.ones(len(rms),bool);cu=np.r_[0,np.cumsum(length*q)]
  def total(lo,hi):return float(np.interp(hi,bounds,cu)-np.interp(lo,bounds,cu)) if hi>lo else 0.
  l0,l1=0,min(n,64600);s0,s1=start,start+min(n,48440);il,ir=max(l0,s0),min(l1,s1);a=total(l0,l1);b=total(s0,s1);shared=total(il,ir);union=a+b-shared
  ans.append([a/16000,b/16000,shared/16000,shared/union if union>0 else -1.])
 return ans

def paired_one(i):
 from types import SimpleNamespace
 row=PROWS[i];raw,sr=SF.read(row['path']);assert sr==16000 and raw.ndim==1
 seed=910000+row['file_index']*17+row['repeat'];random.seed(seed);np.random.seed(seed)
 codec=Codec();meta={'file_index':row['file_index'],'repeat':row['repeat'],'label':row['label'],'speaker':row['speaker'],'id':row['id'],'raw_duration_s':len(raw)/16000};current='L';original_select=codec.selectCodec
 def select():
  c,q=original_select();meta[current+'_codec']=c;meta[current+'_quality']=q;return c,q
 codec.selectCodec=select
 stage_features=[];stage_profiles=[];actual=[];final_arrays=[]
 for current,n in [('L',64600),('S',48440)]:
  a=raw.copy();flag=random.random()<.8;meta[current+'_codec_flag']=flag;meta[current+'_codec']='none';meta[current+'_quality']=-1
  if flag:
   cap=int(n*1.2);meta[current+'_prefix_capped']=len(a)>cap
   if cap<len(a):a=a[:cap]
   a=codec.convert(a,sr)
  else:meta[current+'_prefix_capped']=False
  bf,bp,br,base_peak=describe(a);meta[current+'_before_crop_s']=len(a)/16000;meta[current+'_source_lead20_s']=float(bf[0,1]);meta[current+'_source_lead40_s']=float(bf[1,1]);meta[current+'_wrap_fraction']=max(0,n-len(a))/n
  start=0
  if len(a)<n:a=np.pad(a,(0,n-len(a)),'wrap')
  elif current=='L':a=a[:n]
  else:start=random.randint(0,len(a)-n);a=a[start:start+n]
  meta[current+'_crop_start_s']=start/16000;meta[current+'_max_start_s']=max(0,round(meta[current+'_before_crop_s']*16000)-n)/16000
  f,p,r,pre_peak=describe(a);t=random.randint(0,8);meta[current+'_augtype']=t;meta[current+'_has_SSI']=t in [3,4,6,7];a=apply(a,t);assert len(a)==n;final_arrays.append(a)
  ff,pp,_,_=describe(a);fx,px,_,_=describe(a,reference_peak=pre_peak);actual.append((f,p,ff,pp,fx,px))
 # Literal implementation equivalence against original loader for first eight file-repeat records.
 if i<8:
  random.seed(seed);np.random.seed(seed);ds=DL.TrainSet({'num_train_frames':404,'num_train_frames_short':303},[SimpleNamespace(path=row['path'],label=row['label'])]);aa,bb,_=ds[0]
  assert np.array_equal(aa,final_arrays[0]) and np.array_equal(bb,final_arrays[1]);meta['original_loader_bitwise_equal']=True
 rng=np.random.default_rng(seed+5000000);refstart=int(rng.integers(max(1,len(raw)-48440+1)));ref=[crop(raw,64600),crop(raw,48440,refstart)];meta['reference_short_start_s']=refstart/16000
 rf=[];rp=[]
 for a in ref:f,p,_,_=describe(a);rf.append(f);rp.append(p)
 stage_features=[rf,[v[0] for v in actual],[v[2] for v in actual],[v[4] for v in actual]];stage_profiles=[rp,[v[1] for v in actual],[v[3] for v in actual],[v[5] for v in actual]]
 return i,np.array(stage_features),np.array(stage_profiles),meta,quiet_intersection(raw,refstart)

def trials():
 rng=random.Random(20261030);allrows=[];protocols=[]
 base=Path('/data/ASVspoof2019/LA')
 for split,end in [('train','train.trn'),('dev','dev.trl'),('eval','eval.trl')]:
  path=base/'ASVspoof2019_LA_cm_protocols'/f'ASVspoof2019.LA.cm.{end}.txt';protocols.append(path);rr=[]
  for line in path.read_text().splitlines():
   f=line.split();rr.append(dict(id=f[1],speaker=f[0],label=int(f[4]=='bonafide'),attack=f[3],source='asvspoof',dataset='19LA_'+split,path=str(base/f'ASVspoof2019_LA_{split}/flac/{f[1]}.flac')))
  if split=='eval':rr=[t for lab in [1,0] for t in rng.sample([t for t in rr if t['label']==lab],1000)]
  allrows+=rr
 path=Path('/data/ASVspoof2021_LA_eval/keys/LA/CM/trial_metadata.txt');protocols.append(path);rr=[]
 for line in path.read_text().splitlines():
  f=line.split()
  if f[7]=='eval':rr.append(dict(id=f[1],speaker=f[0],label=int(f[4]=='bonafide'),attack='unknown',source='asvspoof',codec=f[2],dataset='21LA_eval',path=f'/data/ASVspoof2021_LA_eval/flac/{f[1]}.flac'))
 allrows += [t for lab in [1,0] for t in rng.sample([t for t in rr if t['label']==lab],1000)]
 path=Path('/data/ASVspoof2021_DF/keys/DF/CM/trial_metadata.txt');protocols.append(path);rr=[]
 for line in path.read_text().splitlines():
  f=line.split()
  if f[7]=='eval' and f[3]=='asvspoof':rr.append(dict(id=f[1],speaker=f[0],label=int(f[5]=='bonafide'),attack=f[4],source=f[3],codec=f[2],dataset='21DF_ASV_eval',path=f'/data/ASVspoof2021_DF/ASVspoof2021_DF_eval/flac/{f[1]}.wav'))
 allrows += [t for lab in [1,0] for t in rng.sample([t for t in rr if t['label']==lab],1000)]
 return allrows,protocols

def main():
 assert Path('/.dockerenv').exists();ROOT.mkdir(exist_ok=False);native,protocols=trials();train=[t for t in native if t['dataset']=='19LA_train'];rng=random.Random(77201);bona=[t for t in train if t['label']==1];spoof=rng.sample([t for t in train if t['label']==0],len(bona));files=bona+spoof;rng.shuffle(files);pairs=[dict(t,file_index=i,repeat=rep) for i,t in enumerate(files) for rep in range(2)]
 workers=8;quota=Path('/sys/fs/cgroup/cpu.max')
 if quota.exists():
  q,p=quota.read_text().split()
  if q!='max':workers=max(1,min(workers,int(q)//int(p)))
 workers=min(workers,len(os.sched_getaffinity(0)));print('CPU_WORKERS',workers,flush=True)
 (ROOT/'native_manifest_private.json').write_text(json.dumps(native,indent=2));(ROOT/'pair_manifest_private.json').write_text(json.dumps(pairs,indent=2))
 counts={ds:{str(lab):sum(t['dataset']==ds and t['label']==lab for t in native) for lab in [1,0]} for ds in sorted({t['dataset'] for t in native})};plan={'native_counts':counts,'pair_files_per_label':len(bona),'pair_repeats_per_file':2,'pair_records':len(pairs),'thresholds_db_below_own_max20msRMS':TH,'features':FN,'stages':STAGES,'profile_bins':20,'workers':workers,'source_sha256':digest(SOURCE/'data_loaders.py'),'protocol_sha256':{str(p):digest(p) for p in protocols},'scope':'ASVspoofonly.VCCexcluded.19LAtrain/devfullcensus;19LAeval/21LA/21DF_ASV1000perclassrandomsample.','pair_scope':'Only19LAtrain.Entire2580bonafide+equalrandomspoofsample;twoseededactualTrainSet-equivalentdrawsperfile.Nomodelinference/training.','definitions':'20msnonoverlappingRMS;partiallastframeusesactualsamplecount.Quiet<10^(-db/20)*viewmaximumRMS.Onlyall-zeroinputspecialcasequietall;noobservedspeechannotation.','stages_note':'Reference:noaugmentation,headlong/fullfilerandomshort,wrapifshort.Codec_crop:actualcodec/cropbeforerawboost.Final_relative:actualmodelwaveformwithownthreshold.Final_fixed:actualwaveformwithpreRawBoostthreshold.','limits':['Relativequietnotannotatedsilence.','Twodrawsperfiledonotreconstructallhistoricalepochs.','OnlynormalviewsusedinactualBPL;spoofviewsforcomparison.','Nativeandcropdifferencesarenotmodelcausalimportance.','21DFsourceasvspoofonly;noVCC.']};(ROOT/'plan.json').write_text(json.dumps(plan,indent=2));print('PLAN',json.dumps(plan),flush=True);started=time.time()
 nf=np.empty((len(native),2,len(FN)),np.float32);npf=np.empty((len(native),2,20),np.float32)
 with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=init_native) as pool,gzip.open(ROOT/'bonafide_intervals_private.jsonl.gz','wt') as detail:
  for done,(i,f,p,r) in enumerate(pool.map(native_one,range(len(native)),chunksize=32),1):
   nf[i]=f;npf[i]=p
   if r is not None:detail.write(json.dumps({'index':i,'intervals':r})+'\n')
   if done%4000==0:print('NATIVE',done,len(native),'seconds',round(time.time()-started),flush=True)
 np.savez_compressed(ROOT/'native_metrics_private.npz',features=nf,profiles=npf);print('NATIVE_DONE',round(time.time()-started),flush=True)
 pf=np.empty((len(pairs),4,2,2,len(FN)),np.float32);pp=np.empty((len(pairs),4,2,2,20),np.float32);overlap=np.empty((len(pairs),2,4),np.float32)
 with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=init_pair) as pool,gzip.open(ROOT/'pair_metadata_private.jsonl.gz','wt') as detail:
  for done,(i,f,p,m,ov) in enumerate(pool.map(paired_one,range(len(pairs)),chunksize=4),1):
   pf[i]=f;pp[i]=p;overlap[i]=ov;detail.write(json.dumps(dict(index=i,**m))+'\n')
   if done%500==0:print('PAIRS',done,len(pairs),'seconds',round(time.time()-started),flush=True)
 np.savez_compressed(ROOT/'pair_metrics_private.npz',features=pf,profiles=pp,raw_source_quiet_overlap=overlap);assert np.isfinite(nf).all() and np.isfinite(pf).all();assert digest(SOURCE/'data_loaders.py')==plan['source_sha256']
 (ROOT/'COMPLETED.json').write_text(json.dumps({'native':len(native),'pairs':len(pairs),'seconds':time.time()-started,'source_unchanged':True}));print('ALL_COMPLETE',round(time.time()-started),flush=True)
if __name__=='__main__':main()
