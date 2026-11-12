import os,json,gzip,importlib.util
from pathlib import Path
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1');import numpy as np
R=Path('/home/mentee/results/analysis/ASV_quiet_BPL_distribution_20261030_v1')
plan=json.loads((R/'plan.json').read_text());F=plan['features'];fi={s:i for i,s in enumerate(F)};TH=[20,40]
def stat(v):
 v=np.asarray(v,dtype=float);return {'n':len(v),'mean':float(v.mean()),'q10':float(np.quantile(v,.1)),'q25':float(np.quantile(v,.25)),'median':float(np.median(v)),'q75':float(np.quantile(v,.75)),'q90':float(np.quantile(v,.9)),'q99':float(np.quantile(v,.99)),'max':float(v.max()),'positive_pct':float(100*np.mean(v>0))}
def desc(a,p):
 out={name:stat(a[:,j]) for j,name in enumerate(F)};front=a[:,1]>0;tail=a[:,2]>0;inner=a[:,3]>0;cat=front.astype(int)+2*tail+4*inner
 out['position_categories_pct']={name:float(100*np.mean(cat==code)) for code,name in enumerate(['none','front_only','tail_only','front_tail','internal_only','front_internal','tail_internal','front_tail_internal'])};out['mean_position_profile']=p.mean(0).tolist();out['leading_bins_pct']={name:float(100*np.mean(mask)) for name,mask in [('none',a[:,1]==0),('(0,0.25]',(a[:,1]>0)&(a[:,1]<=.25)),('(0.25,0.5]',(a[:,1]>.25)&(a[:,1]<=.5)),('(0.5,1]',(a[:,1]>.5)&(a[:,1]<=1)),('>1',a[:,1]>1)]};out['front_pct_of_all_quiet_duration']=float(100*a[:,1].sum()/max(a[:,4].sum(),1e-20));return out
def auc(v,y):
 order=np.argsort(v,kind='stable');sv=v[order];ranks=np.empty(len(v),float);start=0
 while start<len(v):
  end=start+1
  while end<len(v) and sv[end]==sv[start]:end+=1
  ranks[order[start:end]]=(start+1+end)/2;start=end
 nb=int(y.sum());ns=len(y)-nb;return float((ranks[y==1].sum()-nb*(nb+1)/2)/(nb*ns))
def main():
 rows=json.loads((R/'native_manifest_private.json').read_text());d=np.load(R/'native_metrics_private.npz');a=d['features'].astype(float);p=d['profiles'].astype(float);assert np.isfinite(a).all() and np.allclose(a[:,:,4],a[:,:,1]+a[:,:,2]+a[:,:,3],atol=2e-6);out={'plan':plan,'native':{},'bonafide_speakers':{},'native_leading_auc':{}}
 for ds in sorted({r['dataset'] for r in rows}):
  out['native'][ds]={};out['bonafide_speakers'][ds]={};out['native_leading_auc'][ds]={}
  ds_mask=np.array([r['dataset']==ds for r in rows]);y=np.array([r['label'] for r in rows])[ds_mask]
  for q,db in enumerate(TH):
   key=str(db);out['native'][ds][key]={};out['bonafide_speakers'][ds][key]={};out['native_leading_auc'][ds][key]=auc(a[ds_mask,q,1],y)
   for lab,name in [(1,'bonafide'),(0,'spoof')]:
    mask=ds_mask & np.array([r['label']==lab for r in rows]);out['native'][ds][key][name]=desc(a[mask,q],p[mask,q])
   speakers={}
   for speaker in sorted({r['speaker'] for r in rows if r['dataset']==ds and r['label']==1}):
    mask=np.array([r['dataset']==ds and r['label']==1 and r['speaker']==speaker for r in rows]);v=a[mask,q];speakers[speaker]={'n':len(v),'leading':stat(v[:,1]),'quiet_fraction':stat(v[:,5]),'internal':stat(v[:,3])}
   out['bonafide_speakers'][ds][key]={'speaker_count':len(speakers),'speaker_median_lead_distribution':stat([v['leading']['median'] for v in speakers.values()]),'per_speaker_private':speakers}
 (R/'native_summary.json').write_text(json.dumps(out,indent=2));print('NATIVE_SUMMARY_READY',flush=True)
 if not (R/'COMPLETED.json').exists():return
 pairs=json.loads((R/'pair_manifest_private.json').read_text());d=np.load(R/'pair_metrics_private.npz');a=d['features'].astype(float);p=d['profiles'].astype(float);ov=d['raw_source_quiet_overlap'].astype(float);meta=[json.loads(line) for line in gzip.open(R/'pair_metadata_private.jsonl.gz','rt')];assert len(meta)==len(pairs) and all(m['index']==i for i,m in enumerate(meta));assert np.isfinite(a).all() and np.allclose(a[:,:,:,:,4],a[:,:,:,:,1]+a[:,:,:,:,2]+a[:,:,:,:,3],atol=2e-6)
 out={'plan':plan,'stages':{},'conditional':{},'raw_source_overlap':{},'checks':{'loader_bitwise_verified_records':sum(m.get('original_loader_bitwise_equal',False) for m in meta),'all_source_unchanged':json.loads((R/'COMPLETED.json').read_text())['source_unchanged']},'file_repeat_variation':{}}
 y=np.array([r['label'] for r in pairs]);fileidx=np.array([r['file_index'] for r in pairs]);assert all(np.sum(fileidx==v)==2 for v in np.unique(fileidx))
 for stage in range(4):
  sn=plan['stages'][stage];out['stages'][sn]={}
  for q,db in enumerate(TH):
   out['stages'][sn][str(db)]={'leading_auc_long':auc(a[:,stage,0,q,1],y),'leading_auc_short':auc(a[:,stage,1,q,1],y)}
   for lab,name in [(1,'bonafide'),(0,'spoof')]:
    mask=y==lab;v=a[mask,stage,:,q];profiles=p[mask,stage,:,q];leadL=v[:,0,1]>0;leadS=v[:,1,1]>0
    entry={'records':int(mask.sum()),'files':int(mask.sum()/2),'long':desc(v[:,0],profiles[:,0]),'short':desc(v[:,1],profiles[:,1]),'long_minus_short':{F[j]:stat(v[:,0,j]-v[:,1,j]) for j in [1,2,3,4,5]},'leading_presence_pct':{k:float(100*z.mean()) for k,z in [('both',leadL&leadS),('long_only',leadL&~leadS),('short_only',~leadL&leadS),('neither',~leadL&~leadS)]},'abs_leading_difference_ge025_pct':float(100*np.mean(abs(v[:,0,1]-v[:,1,1])>=.25))}
    out['stages'][sn][str(db)][name]=entry
 # Conditioning on codec flags and SSI shows whether changes come from crop or noise floor.
 for q,db in enumerate(TH):
  out['conditional'][str(db)]={}
  for lab,name in [(1,'bonafide'),(0,'spoof')]:
   out['conditional'][str(db)][name]={}
   flags={'L_codec':np.array([m['L_codec_flag'] for m in meta]),'S_codec':np.array([m['S_codec_flag'] for m in meta]),'L_SSI':np.array([m['L_has_SSI'] for m in meta]),'S_SSI':np.array([m['S_has_SSI'] for m in meta])}
   for condition,cm in [('all',np.ones(len(meta),bool)),('both_codec',flags['L_codec']&flags['S_codec']),('neither_codec',~flags['L_codec']&~flags['S_codec']),('both_SSI',flags['L_SSI']&flags['S_SSI']),('neither_SSI',~flags['L_SSI']&~flags['S_SSI']),('only_long_SSI',flags['L_SSI']&~flags['S_SSI']),('only_short_SSI',~flags['L_SSI']&flags['S_SSI'])]:
    m=(y==lab)&cm;v=a[m,2,:,q];rawlead=np.array([t['S_source_lead'+str(db)+'_s'] for t in meta])[m];start=np.array([t['S_crop_start_s'] for t in meta])[m];dropped=(rawlead>0)&(start>=rawlead)
    out['conditional'][str(db)][name][condition]={'records':int(m.sum()),'long_leading_present_pct':float(100*np.mean(v[:,0,1]>0)),'short_leading_present_pct':float(100*np.mean(v[:,1,1]>0)),'both_leading_present_pct':float(100*np.mean((v[:,0,1]>0)&(v[:,1,1]>0))),'short_original_prefix_removed_pct_among_prefix_present':float(100*dropped.sum()/max((rawlead>0).sum(),1)),'short_start_s':stat(start),'long_wrap_pct':float(100*np.mean([meta[i]['L_wrap_fraction']>0 for i in np.flatnonzero(m)])),'short_wrap_pct':float(100*np.mean([meta[i]['S_wrap_fraction']>0 for i in np.flatnonzero(m)]))}
   m=y==lab;v=ov[m,q];good=v[:,3]>=0;out['raw_source_overlap'].setdefault(str(db),{})[name]={'eligible_records':int(good.sum()),'shared_quiet_jaccard':stat(v[good,3]),'no_shared_source_quiet_pct':float(100*np.mean(v[good,2]==0)),'note':'No-codec/no-noise reference only. Uniqueoriginalsamplecoverage;not normalized-positionoverlap.'}
   # paired repeats: presence varies within the same sourcefile.
   inds=np.flatnonzero(y==lab);long=(a[inds,2,0,q,1]>0).reshape(-1,2);short=(a[inds,2,1,q,1]>0).reshape(-1,2)
   out['file_repeat_variation'].setdefault(str(db),{})[name]={view:{'both_draws_present_pct':float(100*np.mean(z.sum(1)==2)),'one_draw_present_pct':float(100*np.mean(z.sum(1)==1)),'neither_draw_present_pct':float(100*np.mean(z.sum(1)==0))} for view,z in [('long',long),('short',short)]}
 out['metadata_checks']={'codec_long_pct':float(100*np.mean([m['L_codec_flag'] for m in meta])),'codec_short_pct':float(100*np.mean([m['S_codec_flag'] for m in meta])),'SSI_long_pct':float(100*np.mean([m['L_has_SSI'] for m in meta])),'SSI_short_pct':float(100*np.mean([m['S_has_SSI'] for m in meta]))}
 (R/'pair_summary.json').write_text(json.dumps(out,indent=2));print('PAIR_SUMMARY_READY',flush=True)
if __name__=='__main__':main()
