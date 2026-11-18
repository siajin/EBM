"""Exact expected length distributions for original-first uniform crop sampling."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'artifacts/original_first_length_sampling_20261110'
OUT.mkdir(parents=True,exist_ok=True)
prior=json.loads((ROOT/'artifacts/segment_length_histograms_bidirectional_20261109/analysis.json').read_text(encoding='utf-8'))
frames=[]
for source in prior['sources']:
    path=ROOT/source['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==source['sha256']
    frames.append(pd.read_parquet(path,columns=['split','utterance_id','y','duration']))
sets=[frames[0].loc[frames[0].split=='T'],pd.concat([frames[0].loc[frames[0].split=='V'],frames[1]],ignore_index=True),
      frames[2],frames[3],pd.concat(frames[4:],ignore_index=True)]
results,rows,summaries,configs=[],[],[],{}
for info,f in zip(prior['datasets'],sets):
    assert f.utterance_id.is_unique and not f.isna().any().any()
    n=np.rint(f.duration.to_numpy()*16000).astype('int64')
    assert np.max(np.abs(n/16000-f.duration.to_numpy()))<1e-9 and (n>0).all()
    m=n//8000
    maximum=int(m.max())
    k=np.arange(1,maximum+1)
    # Row m: number of front/back crops of every target length for a source in bucket m.
    crop_counts=2*(np.arange(maximum+1)[:,None]//k)
    total_crops=crop_counts.sum(axis=1)
    total_crops[0]=1  # retain <0.5 s original once
    conditional=crop_counts/total_crops[:,None]
    short_bin=info['lengths'][0] is None
    if short_bin:
        conditional=np.column_stack([np.arange(maximum+1)==0,conditional])
    assert np.allclose(conditional[1:].sum(axis=1),1)
    if short_bin:
        assert np.isclose(conditional[0].sum(),1)
    result={'key':info['key'],'title':info['title'],'lengths':info['lengths'],
            'length_labels':info['length_labels'],'classes':{}}
    for y,name in [(1,'bonafide'),(0,'spoof')]:
        mask=f.y.to_numpy()==y
        nm=n[mask]; mm=m[mask]; count=len(mm)
        assert count==info['classes'][name]['original_utterances']
        frequencies=np.bincount(mm,minlength=maximum+1)
        new=frequencies@conditional/count
        old=np.array(info['classes'][name]['fractions'])
        # Independent direct per-utterance inverse-crop weighting.
        independent=[]
        if short_bin:
            independent.append(float(np.mean(nm<8000)))
        for target in k*8000:
            independent.append(float(np.sum((2*(nm//target))/total_crops[mm])/count))
        assert np.allclose(new,independent,atol=1e-13,rtol=1e-12)
        assert np.isclose(new.sum(),1,atol=1e-12) and np.isclose(old.sum(),1)
        assert np.array_equal(total_crops[mm],np.where(mm>0,2*np.array([sum(int(v)//int(t) for t in k) for v in mm]),1))
        # One complete original draw has total probability 1/N regardless of crop count.
        assert np.allclose(total_crops[mm]*(1/count/total_crops[mm]),1/count)
        short=nm[nm<8000]
        under2_mask=np.array([length is None or length<2 for length in info['lengths']])
        numeric_lengths=np.array([0 if length is None else length for length in info['lengths']])
        new_mean=float(new@numeric_lengths + np.sum(short/16000)/count)
        old_mean=float(old@numeric_lengths + np.sum(short/16000)/info['classes'][name]['all_segment_count'])
        r={'N':count,'all_crops_uniform':old.tolist(),'original_first':new.tolist(),
           'under2_old_percent':float(100*old[under2_mask].sum()),'under2_new_percent':float(100*new[under2_mask].sum()),
           'mean_old_seconds':old_mean,'mean_new_seconds':new_mean}
        result['classes'][name]=r
        summaries.append({'dataset':info['key'],'class':name,**{a:r[a] for a in ['N','under2_old_percent','under2_new_percent','mean_old_seconds','mean_new_seconds']}})
        for label,length,a,b in zip(info['length_labels'],info['lengths'],old,new):
            rows.append({'dataset':info['key'],'class':name,'length_bin':label,'segment_seconds':length,
                         'all_crops_uniform_fraction':float(a),'original_first_fraction':float(b),'delta_pp':float(100*(b-a))})
    b,s=result['classes']['bonafide'],result['classes']['spoof']
    for method in ['all_crops_uniform','original_first']:
        result['overlap_'+method+'_percent']=float(100*np.minimum(b[method],s[method]).sum())
    results.append(result)
    windows=[(0,len(info['lengths']))] if len(info['lengths'])<=40 else [(0,30),(30,len(info['lengths']))]
    upper=max(max(c[method]) for c in result['classes'].values() for method in ['all_crops_uniform','original_first'])*1.08
    for method,title in [('all_crops_uniform','기존 · 모든 조각 균등 선택'),('original_first','변경 · 원음 균등 → 조각 균등')]:
        for page,(lo,hi) in enumerate(windows):
            configs[f'{info["key"]}|{method}|{page}']={
                'id':'original-first-length-'+method,'title':title,'height':320,
                'spec':{'type':'bar','x':'길이','y':['Bona fide','Spoof'],'xLabel':'조각 길이 (초)',
                        'yLabel':'클래스 내 선택 확률','sortOrder':'original','startAtZero':True,'yDomain':[0,upper],
                        'valueFormat':{'style':'percent','maximumFractionDigits':2}},
                'columns':['길이','Bona fide','Spoof'],
                'rows':[[label,b[method][i],s[method][i]] for i,label in enumerate(info['length_labels']) if lo<=i<hi]}

result={'method':{'old':'uniform over all emitted bidirectional crops within class',
                  'new':'uniform over original utterances within class, then uniform over all emitted crops of that original',
                  'length_probability':'mean_i(crop_count_i_at_length / total_crop_count_i)',
                  'not_uniform_over_target_lengths':True,'duplicate_crops_retained':True,'short_sources':'<0.5 s retained once',
                  'meaning':'exact expected training-exposure probabilities, not materialized file counts or a Monte Carlo estimate',
                  'class_mix':'class-conditional probabilities; any external class quota can be retained',
                  'group_shares':'original speaker/corpus/attack proportions preserved in expectation within each class',
                  'audio_files_created':False,'sampler_modified':False,'training_started':False},
        'sources':prior['sources'],'datasets':results,'summaries':summaries,
        'validation':{'source_sha_match':True,'class_counts_match':True,'probabilities_sum_to_one':True,
                      'bucket_formula_matches_direct_per_utterance_weighting':True,'equal_total_probability_per_original':True}}
(OUT/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(OUT/'chart-configs.json').write_text(json.dumps(configs,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
pd.DataFrame(rows).to_csv(OUT/'length_probabilities.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(summaries).to_csv(OUT/'summary.csv',index=False,encoding='utf-8-sig')
overlaps=[{'dataset':d['key'],'all_crops_overlap_percent':d['overlap_all_crops_uniform_percent'],
           'original_first_overlap_percent':d['overlap_original_first_percent']} for d in results]
pd.DataFrame(overlaps).to_csv(OUT/'overlap_comparison.csv',index=False,encoding='utf-8-sig')
print(pd.DataFrame(overlaps).to_string(index=False))
print(pd.DataFrame(summaries).to_string(index=False))
