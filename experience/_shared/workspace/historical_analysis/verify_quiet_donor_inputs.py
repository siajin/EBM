from pathlib import Path
import json,hashlib,numpy as np
import soundfile as sf
assert Path('/.dockerenv').exists()
root=Path('/results/analysis/EBM80_quiet_donor_audit');tr=json.loads((root/'recipient_manifest_private.json').read_text());dm=json.loads((root/'donor_manifest_private.json').read_text());bank=np.load(root/'donor_frames_private.npz')['frames'];pairs=np.load(root/'assignment_private.npz')['pairs']
plan=json.loads((root/'plan.json').read_text());assert hashlib.sha256(Path('/home/mentee/analysis/quiet_donor_audit.py').read_bytes()).hexdigest()==plan['script_sha256']
sh=[np.load(root/f'inference/shard_{i}.npz') for i in range(2)];order=np.argsort(np.concatenate([a['indices'] for a in sh]));g=np.concatenate([a['gains'] for a in sh])[order];checks=np.concatenate([a['checks'] for a in sh])[order]
valid=np.flatnonzero(np.isfinite(g[:,5]));rng=np.random.default_rng(713901)
chosen=sorted(set(valid[np.argsort(np.max(g[valid,5:],axis=1))[-4:]].tolist()+valid[np.argsort(checks[valid,0])[-4:]].tolist()+rng.choice(valid,8,replace=False).tolist()))
audio=Path('/data/ASVspoof2021_DF/ASVspoof2021_DF_eval/flac');read_donors=set();varerr=0.;energyerr=0.;meanerr=0.;outsideok=True;transforms=0
def read(t):
 raw,sr=sf.read(audio/(t['id']+'.wav'),dtype='float32');assert sr==16000 and hashlib.sha256(raw.tobytes()).hexdigest()==t['pcm_sha256'];a=raw[:64600];return np.pad(a,(0,64600-len(a)),mode='wrap')
for idx in chosen:
 t=tr[idx];a=read(t);rms=np.array([np.sqrt(np.mean(a[q:q+320].astype(float)**2)+1e-20) for q in range(0,64600,320)]);prefix=np.flatnonzero(rms>=rms.max()*.01)[0]
 allframes=a[:prefix*320].reshape(-1,320);good=np.flatnonzero(allframes[:,16:304].astype(float).std(1)>=1e-6);assert np.array_equal(good,t['eligible_frame_indices'])
 own=allframes[good,16:304].astype(float);mu=own.mean(1);sigma=own.std(1);donorids=pairs[idx].reshape(-1);k=min(8,len(good),*(len(dm[int(j)]['frame_ids']) for j in donorids))
 rr=np.random.default_rng(915037+idx);pattern=np.tile(np.arange(k),int(np.ceil(len(good)/k)))[:len(good)];rr.shuffle(pattern)
 arrays=[own[rr.choice(len(good),k,replace=False)][pattern],np.stack([rr.permutation(v) for v in own]),rr.normal(size=(k,288))[pattern]]
 for donor in donorids:
  donor=int(donor);meta=dm[donor]
  if donor not in read_donors:
   raw=read(meta);f=raw[:meta['prefix_frames']*320].reshape(-1,320);q=f[f[:,16:304].astype(float).std(1)>=1e-6,16:304];assert np.array_equal(q,bank[meta['frame_ids']]);read_donors.add(donor)
  selected=rr.choice(meta['frame_ids'],k,replace=False);arrays.append(bank[selected].astype(float)[pattern])
 for j,v in enumerate(arrays,2):
  repl=((v-v.mean(1)[:,None])/v.std(1)[:,None]*sigma[:,None]+mu[:,None]).astype(np.float32)
  # Float64 arithmetic order differs slightly from inference; verifies the statistical constraints independently.
  varerr=max(varerr,float(np.max(abs(repl.astype(float).var(1)-own.var(1))/own.var(1))))
  meanerr=max(meanerr,float(np.max(abs(repl.astype(float).mean(1)-mu))))
  energyerr=max(energyerr,float(np.max(abs(np.mean(repl.astype(float)**2,axis=1)-np.mean(own**2,axis=1))/np.mean(own**2,axis=1))))
  b=a.copy();mask=np.zeros(64600,dtype=bool)
  for row,q in enumerate(good):b[q*320+16:q*320+304]=repl[row];mask[q*320+16:q*320+304]=True
  outsideok &= bool(np.array_equal(b[~mask],a[~mask]));assert len(b)==64600;transforms+=1
assert varerr<1e-4 and energyerr<1e-5 and meanerr<1e-8 and outsideok
summary={'recipient_sample_n':len(chosen),'sampling':'8 seeded random plus4largest donor amplification and4largest stored energy error; overlapsdeduplicated','donor_files_redecoded':len(read_donors),'transformations_checked':transforms,'maximum_inner_variance_relative_error':varerr,'maximum_inner_energy_relative_error':energyerr,'maximum_inner_mean_error':meanerr,'all_unmodified_samples_identical':outsideok,'inference_script_hash_verified':True,'limits':'Sampled independent input reconstruction; not new model inference. Full-run invariants were separately asserted during inference.'}
(root/'analysis/input_verification.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
