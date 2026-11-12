from pathlib import Path
import json,hashlib,time
import numpy as np
import soundfile as sf
assert Path('/.dockerenv').exists()
root=Path('/results/analysis/EBM80_quiet_donor_audit');root.mkdir(exist_ok=False)
tr=json.loads(Path('/results/analysis/EBM80_native_quiet_strict/sample_manifest.json').read_text());old=[t for t in tr if t['cohort']=='previous'];new=[t for t in tr if t['cohort']=='new_files']
audio=Path('/data/ASVspoof2021_DF/ASVspoof2021_DF_eval/flac');bank=[];meta=[];rec=[];start=time.time()
for cohort,ts in [('donor',old),('recipient',new)]:
 for i,t in enumerate(ts):
  raw,sr=sf.read(audio/(t['id']+'.wav'),dtype='float32');assert sr==16000 and hashlib.sha256(raw.tobytes()).hexdigest()==t['pcm_sha256']
  a=raw[:64600];a=np.pad(a,(0,64600-len(a)),mode='wrap');rr=np.array([np.sqrt(np.mean(a[q:q+320].astype(float)**2)+1e-20) for q in range(0,len(a),320)]);prefix=int(np.flatnonzero(rr>=rr.max()*.01)[0]);frames=a[:prefix*320].reshape(-1,320)
  means=frames[:,16:-16].astype(float).mean(1) if prefix else np.array([]);std=frames[:,16:-16].astype(float).std(1) if prefix else np.array([])
  good=np.flatnonzero(std>=1e-6)
  if cohort=='donor' and len(good):
   ids=np.arange(len(bank),len(bank)+len(good)).tolist();bank.extend(frames[good,16:-16]);meta.append(dict(t,frame_ids=ids,prefix_frames=prefix,eligible_frames=len(good)))
  elif cohort=='recipient':rec.append(dict(t,prefix_frames=prefix,eligible_frame_indices=good.tolist()))
  if i%500==0:print(cohort,i,'elapsed',round(time.time()-start),flush=True)
np.savez_compressed(root/'donor_frames_private.npz',frames=np.asarray(bank,dtype=np.float32));(root/'donor_manifest_private.json').write_text(json.dumps(meta,indent=2));(root/'recipient_manifest_private.json').write_text(json.dumps(rec,indent=2))
def support(keys):
 groups={}
 for i,t in enumerate(meta):
  key=tuple(t[k] for k in keys);groups.setdefault(key,{0:[],1:[]})[t['label']].append(i)
 shared=[(k,g) for k,g in groups.items() if g[0] and g[1]]
 return {src:{'shared_strata':sum(k[0]==src for k,g in shared),'matched_pairs':sum(min(len(g[0]),len(g[1])) for k,g in shared if k[0]==src),'two_file_disjoint_sets_strata':sum(k[0]==src and min(len(g[0]),len(g[1]))>=2 for k,g in shared),'donor_speakers':len({meta[j]['speaker'] for k,g in shared if k[0]==src for lab in [0,1] for j in g[lab]})} for src in ['asvspoof','vcc2018','vcc2020']}
summary={'bank_files':len(meta),'bank_frames':len(bank),'strict_source_speaker_codec':support(['source','speaker','codec']),'source_codec':support(['source','codec']),
 'recipient_eligible':{str(lab):sum(t['label']==lab and bool(t['eligible_frame_indices']) for t in rec) for lab in [0,1]},
 'minimum_inner_std':1e-6,'keep_original_samples_at_each_frame_edge':16,'recipient_outside_leading_quiet':'preserved bitwise','elapsed_s':time.time()-start}
(root/'preparation_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
