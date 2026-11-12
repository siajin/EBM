from pathlib import Path
import json,numpy as np
assert Path('/.dockerenv').exists()
root=Path('/results/analysis');out=root/'EBM80_ASV_only_recheck_20261030';out.mkdir(exist_ok=False)
native=np.load(root/'EBM80_native_quiet_strict/analysis/merged_private.npz');m0=(native['cohort']=='new_files')&(native['source']=='asvspoof');s=native['scores'].astype(float);y=native['labels'];elig=native['eligible'];eps=.000121
def stat(v,m):
 a=v[m];return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),mean_abs=float(abs(a).mean()),toward_n=int((a>eps).sum()),away_n=int((a<-eps).sum()),small_n=int((abs(a)<=eps).sum()),abs_ge001_n=int((abs(a)>=.01).sum()))
r={'scope':'Only ASVspoof-source recipients. Shape and duration estimates are intervention-specific, not universal importance shares. VCC excluded from new analyses.',
 'native_factorial':{},'asv_to_asv_donor':{},'fixed_input_existing_effects':{},'limits':['Native factorial preserves complete native body but added duration changes total input length.','Own vs zero changes waveform and added energy; duration is held, amplitude is not.','Donor comparison instead holds recipient frame mean/energy and uses matched source/speaker/codec; conditional finite donor pools.','Different contrast intensities and eligibility populations cannot be ranked as universal feature importance.']}
for word,lab in [('bonafide',1),('spoof',0)]:
 m=m0&(y==lab)&elig;rr={}
 for dose,z,o in [('025',2,4),('050',6,8),('100',10,12)]:
  rr[dose]={'zero_minus_original':stat(s[:,z]-s[:,0],m),'own_minus_original':stat(s[:,o]-s[:,0],m),'own_minus_zero':stat(s[:,o]-s[:,z],m)}
 rr['duration_025_to_100']={'zero':stat(s[:,10]-s[:,2],m),'own':stat(s[:,12]-s[:,4],m)};r['native_factorial'][word]=rr
# Matched-source transplantation: both recipient source and donor source are ASV.
dr=root/'EBM80_quiet_donor_audit';tr=json.loads((dr/'recipient_manifest_private.json').read_text());src=np.array([a['source'] for a in tr]);yy=np.array([a['label'] for a in tr])
sh=[np.load(dr/f'inference/shard_{i}.npz') for i in range(2)];order=np.argsort(np.concatenate([a['indices'] for a in sh]));sc=np.concatenate([a['scores'] for a in sh])[order].astype(float);g=np.concatenate([a['gains'] for a in sh])[order]
delta=np.stack([sc[:,5]-sc[:,6],sc[:,7]-sc[:,8]],axis=1);available=np.isfinite(delta).all(1);bounded=available&(np.max(g[:,5:9],axis=1)<=10)
for word,lab in [('bonafide',1),('spoof',0)]:
 m=(src=='asvspoof')&(yy==lab)&bounded
 r['asv_to_asv_donor'][word]={'average':stat(delta.mean(1),m),'bankA':stat(delta[:,0],m),'bankB':stat(delta[:,1],m)}
prior=json.loads((root/'EBM80_cue_factor_audit/analysis/summary.json').read_text())
for key in ['zero_lead','shuffle_lead','phase_both','added_zero_front','added_zero_position','added_texture_front']:
 r['fixed_input_existing_effects'][key]={word:prior['source_effects'][key][word]['asvspoof'] for word in ['bonafide','spoof']}
assert all(a['average']['n']<=525 for a in r['asv_to_asv_donor'].values())
(out/'summary.json').write_text(json.dumps(r,indent=2,allow_nan=False));print(json.dumps(r,indent=2))
