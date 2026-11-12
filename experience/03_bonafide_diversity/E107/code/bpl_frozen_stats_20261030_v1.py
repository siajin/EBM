from pathlib import Path
import json,hashlib
import numpy as np
R=Path('/home/mentee/results/analysis/EBM80_BPL_frozen_audit_20261030_v1')

def merge(prefix,n,fields):
 ds=[np.load(R/f'{prefix}_{r}.npz') for r in range(2)];ix=np.concatenate([d['indices'] for d in ds]);assert np.array_equal(np.sort(ix),np.arange(n));order=np.argsort(ix)
 return {k:np.concatenate([d[k] for d in ds])[order].astype(float) for k in fields},ds

def stat(a):
 a=np.asarray(a);assert np.isfinite(a).all()
 return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),mean_abs=float(abs(a).mean()),p90_abs=float(np.quantile(abs(a),.9)),ge001=int((abs(a)>=.01).sum()),positive=int((a>.000121).sum()),negative=int((a<-.000121).sum()))

def ci(v,rows,mask):
 rng=np.random.default_rng(31087);means=np.zeros(2000);N=int(mask.sum())
 for src in sorted({rows[i]['source'] for i in np.flatnonzero(mask)}):
  ix=[i for i in np.flatnonzero(mask) if rows[i]['source']==src];speakers=sorted({rows[i]['speaker'] for i in ix});members=[[i for i in ix if rows[i]['speaker']==sp] for sp in speakers]
  sm=np.array([v[z].sum() for z in members]);nn=np.array([len(z) for z in members]);draw=rng.integers(len(members),size=(2000,len(members)));means+=(len(ix)/N)*sm[draw].sum(1)/nn[draw].sum(1)
 return np.quantile(means,[.025,.975]).tolist()

def effect(v,rows):
 out={}
 for lab,name in [(1,'bonafide'),(0,'spoof')]:
  m=np.array([r['label']==lab for r in rows]);out[name]=dict(stat(v[m]),speaker_cluster_mean_ci95=ci(v,rows,m),sources={s:stat(v[m & np.array([r['source']==s for r in rows])]) for s in sorted({r['source'] for r in rows}) if (m & np.array([r['source']==s for r in rows])).any()})
 return out

def excess(c):return np.logaddexp(0,30*(1-np.clip(c,-1,1)))-np.log(2)

def main():
 assert (R/'COMPLETED.json').exists();plan=json.loads((R/'plan.json').read_text());items=json.loads((R/'items_private.json').read_text());rec=json.loads((R/'donor_recipients_private.json').read_text());v,ds=merge('views',len(items),['z','scores','features']);d,dd=merge('donor',len(rec),['z','scores','checks']);z=v['z'];s=v['scores'];dz=d['z'];sc=d['scores'];center=ds[0]['center'].astype(float).reshape(-1)
 assert np.max(abs(np.linalg.norm(z,axis=2)-1))<1e-5 and np.max(abs(np.linalg.norm(dz,axis=2)-1))<1e-5
 assert np.max(abs(z@center-s[:,:6]))<1e-5 and np.max(abs(dz@center-sc))<1e-5
 assert max(float(q['bpl_toggle_max']) for q in ds)==0
 threshold=json.loads(Path('/home/mentee/results/analysis/EBM80_21DF_leading_quiet_paired/summary.json').read_text())['threshold']
 out={'plan':plan,'checks':dict(bpl_off_max_score_difference=0.,normalization_max_error=float(np.max(abs(np.linalg.norm(z,axis=2)-1))),score_reconstruction_max=float(np.max(abs(z@center-s[:,:6]))),repeat_views_max=float(np.max(abs(s[:,6]-s[:,0]))),repeat_donor_max=float(np.max(abs(sc[:,1]-sc[:,0]))),donor_mean_energy_max_error=float(d['checks'][:,0].max()),checkpoint_unchanged=json.loads((R/'COMPLETED.json').read_text())['checkpoint_unchanged']),'views':{},'donor':{},'threshold':threshold}
 for dataset in ['19LAdev','21DF']:
  out['views'][dataset]={}
  for lab,name in [(1,'bonafide'),(0,'spoof')]:
   m=np.array([r['dataset']==dataset and r['label']==lab for r in items]);zz=z[m];ss=s[m];f=v['features'][m];pair=np.sum(zz[:,4]*zz[:,5],axis=1);direct=np.sum(zz[:,0]*zz[:,1],axis=1);qs=np.sum(zz[:,2]*zz[:,3],axis=1)
   originalB=zz[:,4]@zz[:,5].T;unaugB=zz[:,0]@zz[:,1].T;shuffleB=zz[:,2]@zz[:,3].T
   ent=dict(n=int(m.sum()),actual_aug_pair_cos=stat(pair),actual_aug_pair_excess_loss=stat(excess(pair)),unaug_pair_cos=stat(direct),quiet_shuffled_pair_cos=stat(qs),quiet_long_score_delta=stat(ss[:,2]-ss[:,0]),quiet_short_score_delta=stat(ss[:,3]-ss[:,1]),quiet_long_embedding_cos=stat(np.sum(zz[:,2]*zz[:,0],axis=1)),actual_aug_pair_score_gap=stat(ss[:,4]-ss[:,5]),actual_aug_lowenergy_lead_mean_s=[float(f[:,7].mean()),float(f[:,8].mean())],actual_aug_both_leading_quiet=int(((f[:,7]>0)&(f[:,8]>0)).sum()),loss_note='Excess above min softplus(0)-0.3;spoof computed diagnostically, not optimized in original BPL.',allpairs_actual_aug_mean_cos=float(originalB.mean()),allpairs_actual_aug_excess=float(excess(originalB).mean()),allpairs_unaug_excess=float(excess(unaugB).mean()),allpairs_shuffle_excess=float(excess(shuffleB).mean()))
   # Fixed four-item groups reflect old per-rank bonafide batchsize; direct pairs vs allpairs.
   losses=[]
   for q in range(0,len(zz)-3,4):losses.append(float(excess(zz[q:q+4,4]@zz[q:q+4,5].T).mean()))
   ent['batch4_excess']=stat(losses)
   # High similarity does not imply negligible score sensitivity. Predefined cosine threshold.
   stable=pair>=.99;ent['actual_pair_cos_ge099']=dict(n=int(stable.sum()),score_gap_ge001=int((stable & (abs(ss[:,4]-ss[:,5])>=.01)).sum()))
   stablequiet=np.sum(zz[:,2]*zz[:,0],axis=1)>=.99;ent['quiet_cos_ge099']=dict(n=int(stablequiet.sum()),score_delta_ge001=int((stablequiet & (abs(ss[:,2]-ss[:,0])>=.01)).sum()))
   ent['both_unaug_length_pairs_cos_ge099_and_quiet_delta_ge001']=int(((direct>=.99)&(qs>=.99)&(abs(ss[:,2]-ss[:,0])>=.01)).sum())
   ent['actual_aug_pair_cos_ge099_and_unaug_quiet_delta_ge001']=int(((pair>=.99)&(abs(ss[:,2]-ss[:,0])>=.01)).sum())
   ent['actual_aug_pair_cos_below09']=int((pair<.9).sum())
   out['views'][dataset][name]=ent
 # Mean of two draws gives recipient-level label contrast.
 contrasts={'bona_donor_minus_original':(sc[:,2]+sc[:,4])/2-sc[:,0],'spoof_donor_minus_original':(sc[:,3]+sc[:,5])/2-sc[:,0],'bona_minus_spoof_donor':(sc[:,2]+sc[:,4]-sc[:,3]-sc[:,5])/2,'draw1_bona_minus_spoof':sc[:,2]-sc[:,3],'draw2_bona_minus_spoof':sc[:,4]-sc[:,5],'inner_shuffle_minus_original':sc[:,6]-sc[:,0]}
 for name,delta in contrasts.items():out['donor'][name]=effect(delta,rec)
 out['donor']['counts']={str(lab):sum(r['label']==lab for r in rec) for lab in [0,1]};out['donor']['strata']=len({(r['source'],r['speaker'],r['codec']) for r in rec})
 out['donor']['mean_pair_embedding_cos']={name:float(np.sum(dz[:,a]*dz[:,b],axis=1).mean()) for name,a,b in [('bona_vs_spoof_draw1',2,3),('bona_vs_original',2,0),('spoof_vs_original',3,0)]}
 out['donor']['fixed_threshold_errors']={}
 for col,name in enumerate(plan['donor_names']):
  out['donor']['fixed_threshold_errors'][name]={}
  for lab,labelname in [(1,'bonafide'),(0,'spoof')]:
   m=np.array([r['label']==lab for r in rec]);before=(sc[m,0]>=threshold)==bool(lab);after=(sc[m,col]>=threshold)==bool(lab)
   out['donor']['fixed_threshold_errors'][name][labelname]=dict(n=int(m.sum()),errors=int((~after).sum()),new_errors=int((before&~after).sum()),corrected=int((~before&after).sum()))
 # Independent geometric upper bound, uses same fixed unit center.
 delta=sc[:,2]-sc[:,3];bound=np.linalg.norm(dz[:,2]-dz[:,3],axis=1);assert (abs(delta)<=bound+1e-6).all();out['checks']['cosine_score_bound_passed']=True
 (R/'summary.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
