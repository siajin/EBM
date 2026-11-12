from pathlib import Path
import json,random,time,hashlib
import numpy as np
import soundfile as sf
assert Path('/.dockerenv').exists()
root=Path('/results/analysis/EBM80_native_quiet_strict');root.mkdir(exist_ok=False)
data=Path('/data/ASVspoof2021_DF');protocol=data/'keys/DF/CM/trial_metadata.txt'
meta={};groups={}
for line in protocol.read_text().splitlines():
    f=line.split()
    if len(f)<9 or f[7]!='eval':continue
    t={'id':f[1],'speaker':f[0],'source':f[3],'codec':f[2],'label':int(f[5]=='bonafide'),'attack':f[4]}
    meta[t['id']]=t;groups.setdefault((t['label'],t['source']),[]).append(t)
old=json.loads(Path('/results/analysis/EBM80_21DF_centroid/sample_manifest.json').read_text())
oldids={t['id'] for t in old};trials=[dict(meta[t['id']],cohort='previous') for t in old]
counts={k:sum((t['label'],t['source'])==k for t in trials) for k in groups}
rng=random.Random(20261030)
for key,n in sorted(counts.items()):
    pool=[t for t in groups[key] if t['id'] not in oldids]
    trials += [dict(t,cohort='new_files') for t in rng.sample(pool,n)]
assert len(trials)==4000 and len({t['id'] for t in trials})==4000
lengths=[];rates=set();hashes={};dups=[];started=time.time()
for i,t in enumerate(trials):
    path=data/'ASVspoof2021_DF_eval/flac'/(t['id']+'.wav')
    a,sr=sf.read(path,dtype='float32');assert sr==16000 and a.ndim==1 and len(a)>512 and np.isfinite(a).all()
    h=hashlib.sha256(a.tobytes()).hexdigest()
    if h in hashes:dups.append([hashes[h],i])
    else:hashes[h]=i
    t['samples']=len(a);t['pcm_sha256']=h;lengths.append(len(a))
    if i%1000==0:print(f'inventory {i}/4000 {time.time()-started:.0f}s',flush=True)
plan={'created_before_inference_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
 'question':'Does adding leading quiet, without changing any native audio samples, shift a frozen model toward its bonafide OC center, and is that effect specific to front placement?',
 'checkpoint':'/home/mentee/results/CM/LA/EBM/model/BestCLS_80_21DF.pt',
 'native_waveform':'Entire decoded mono 16kHz file; no crop, resample, normalization, wrap, or batch padding.',
 'cohorts':{'previous':'2000 repeatedly explored files; exploratory only','new_files':'2000 file-ID-disjoint files drawn before scores; source-label counts match previous sample. Same corpus and possibly same speakers/underlying utterances. Checkpoint was selected with 21DF evaluation: not a pristine model-development test.'},
 'source_label_counts_per_cohort':{str(k):v for k,v in counts.items()},
 'exact_pcm_duplicate_pairs':dups,
 'doses_seconds':[.25,.5,1.0],
 'families':{'zero':'all-zero filler','own40':'repeat/truncate the original leading prefix whose 20ms RMS is below 1% of maximum frame RMS. At least one complete frame required; ineligible rows are excluded, never replaced with zeros.'},
 'construction':'front=[q,x], back=[x,q], using the exact same q for each file/family/dose; preserve x bit-for-bit. Front/back have equal length, sample multiset and energy.',
 'controls':['Native identity repeated in the same inference batch','At dose 1s, zero fillers with 10ms linear ramps matching adjacent native endpoint; native samples still untouched. Secondary splice sensitivity control.'],
 'primary_population':'new_files, separately by true label; zeros on all eligible native files, own40 on prefix-eligible files. Compare families only on shared own40 eligibility.',
 'primary_endpoints':'At 1s: mean cosine(front)-cosine(original), and mean cosine(front)-cosine(back), for each of 2 families and 2 labels: 8 endpoints total.',
 'uncertainty':'20000 source-stratified speaker-cluster bootstrap replicates. Each source weight held fixed within label. Bonferroni intervals at quantiles .003125 and .996875 for 8 prespecified means (approximate familywise 95%). Direction fraction CIs at 95%, descriptive.',
 'direction_epsilon':'max(1e-4, 10 times maximum native identity-repeat delta); report raw positive count and epsilon-qualified count, mean, median, >=0.01 and >=0.1 movements.',
 'interpretation_rules':['A positive front-back alone proves neither positive front-original nor a universal leading-duration cue. Report both contrasts and back-original.',
 'A systematic toward-center claim additionally needs a positive median and a majority toward center; report source heterogeneity, never replace median with mean.',
 'Dose and tail/source strata are secondary descriptive analyses, not extra uncorrected confirmatory tests.',
 'Frozen-model intervention can establish an input effect for tested files. It cannot establish historical learning mechanism, exclusive shortcut use, or generalization improvement without controlled retraining.'],
 'raw_samples_quantiles':np.quantile(lengths,[0,.5,.9,.99,1]).tolist(),
 'protocol_sha256':hashlib.sha256(protocol.read_bytes()).hexdigest()}
(root/'plan.json').write_text(json.dumps(plan,indent=2));(root/'sample_manifest.json').write_text(json.dumps(trials,indent=2))
print(json.dumps(plan,indent=2))
