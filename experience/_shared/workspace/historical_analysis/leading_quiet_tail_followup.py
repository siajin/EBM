from pathlib import Path
import json
import numpy as np
assert Path('/.dockerenv').exists()
root=Path('/results/analysis/EBM80_21DF_leading_quiet_paired')
out=root/'tail_followup.json'
assert not out.exists(), 'Preserve existing follow-up'
d=np.load(root/'paired_scores.npz'); summary=json.loads((root/'summary.json').read_text())
base=Path('/results/analysis/EBM80_21DF_centroid')
assert json.loads((root/'sample_manifest.json').read_text())==json.loads((base/'sample_manifest.json').read_text())
old=np.load(base/'embeddings.npz')['cosine']; y=d['labels']; b=y==1
s=d['scores']; src=d['sources']; spk=d['speakers']; names=d['names'].tolist()
assert np.max(np.abs(old-s[:,0]))<1e-3
tail=b&(old<.9); core=b&~tail; assert tail.sum()==163 and core.sum()==837
previous=np.load('/results/analysis/EBM80_21DF_tail_interventions/intervention_scores.npz')
assert np.array_equal(previous['labels'],y)
lengths=previous['lengths']; eps=summary['meaningful_delta_epsilon']
def stats(v):
    return {'n':len(v),'mean':float(np.mean(v)),'median':float(np.median(v)),
            'q25':float(np.quantile(v,.25)),'q75':float(np.quantile(v,.75))} if len(v) else {'n':0}
rng=np.random.default_rng(991)
def ci(v,mask):
    boot=np.zeros(1000)
    for source in np.unique(src[mask]):
        m=mask&(src==source)
        groups=[np.flatnonzero(m&(spk==speaker)) for speaker in np.unique(spk[m])]
        counts=np.array([len(g) for g in groups]); sums=np.array([v[g].sum() for g in groups])
        draws=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=1000)
        boot+=(draws@sums)/(draws@counts)*m.sum()/mask.sum()
    return np.quantile(boot,[.025,.975]).tolist()
r={'definition':'Original bonafide cosine to trained OC center < 0.9; training margin, not classification-error threshold or a fresh PCA contour.',
   'original_tail_n':int(tail.sum()),'original_core_n':int(core.sum()),'original_groups':{},'conditions':{},'paired_long_short':{}}
for word,mask in [('tail',tail),('core',core)]:
    r['original_groups'][word]={'cosine':stats(old[mask]),'leading_s':stats(d['checks'][mask,0]/16000),
      'original_short_padded_n':int((mask&(lengths<64600)).sum()),
      'sources':{source:int((mask&(src==source)).sum()) for source in np.unique(src)}}
for k,name in enumerate(names):
    r['conditions'][name]={'total_bona_below09_n':int((b&(s[:,k]<.9)).sum()),
       'original_tail_recovered09_n':int((tail&(s[:,k]>=.9)).sum()),
       'original_core_new_below09_n':int((core&(s[:,k]<.9)).sum()),
       'original_tail_cosine':stats(s[tail,k]),'original_core_cosine':stats(s[core,k]),
       'original_tail_delta_vs_original':stats((s[:,k]-s[:,0])[tail])}
for family,(short,long) in {'zero20':(2,4),'ownquiet20':(5,7),'variable20':(8,4),'zero40':(10,12)}.items():
    delta=s[:,long]-s[:,short]; result={}
    for word,mask in [('tail',tail),('core',core)]:
        e={'cosine_delta_long_minus_short':stats(delta[mask]),'cluster_95ci_mean_delta':ci(delta,mask),
          'closer_to_center_n':int((mask&(delta>eps)).sum()),'farther_from_center_n':int((mask&(delta<-eps)).sum()),
          'below09_short_n':int((mask&(s[:,short]<.9)).sum()),'below09_long_n':int((mask&(s[:,long]<.9)).sum()),
          'crossed_up09_n':int((mask&(s[:,short]<.9)&(s[:,long]>=.9)).sum()),
          'crossed_down09_n':int((mask&(s[:,short]>=.9)&(s[:,long]<.9)).sum()),
          'by_source':{}}
        for source in np.unique(src[mask]):
            g=mask&(src==source); e['by_source'][source]={'delta':stats(delta[g]),
               'closer_to_center_n':int((g&(delta>eps)).sum()),'below09_long_n':int((g&(s[:,long]<.9)).sum())}
        result[word]=e
    r['paired_long_short'][family]=result
r['limitations']=['Tail defined from an original model score: subgroup response is exploratory; larger effects than core may reflect score ceiling and source mixture, not only an underlying causal difference.',
 'Fixed-length long/short views share the same selected body and filler sample multiset, but move quiet from end to start. Front duration is not independently isolated from end structure.',
 'Original-to-transformed recovery mixes common-body crop and filler changes with placement. Primary evidence is paired short-to-long movement.',
 'These scores quantify center proximity in the original 160-dimensional space. No new PCA coordinates or proof that the whole visual tail disappeared.',
 'Changing inputs and making them closer to center is not evidence of improved generalization or a reason to pad everything; spoof also moves toward the center.']
out.write_text(json.dumps(r,indent=2,allow_nan=False))
print(json.dumps(r,indent=2),flush=True)
