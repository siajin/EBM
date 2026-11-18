"""Before/after group shares from cached metadata; no audio, fit, or server IO."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/segment_group_shifts_20261110'
OUT.mkdir(parents=True, exist_ok=True)
prior = json.loads((ROOT/'artifacts/segment_length_histograms_bidirectional_20261109/analysis.json').read_text(encoding='utf-8'))
frames = []
sources = []
for entry in prior['sources']:
    path = ROOT / entry['path']
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha == entry['sha256']
    available = pq.ParquetFile(path).schema_arrow.names
    cols = [c for c in ['split','utterance_id','speaker_group','speaker_id_known','y','duration',
                       'source_corpus','attack_id','codec','transmission'] if c in available]
    frames.append(pd.read_parquet(path,columns=cols))
    sources.append({'path':str(path.relative_to(ROOT)),'sha256':sha,'columns':cols})
sets = [frames[0].loc[frames[0].split=='T'],
        pd.concat([frames[0].loc[frames[0].split=='V'],frames[1]],ignore_index=True),
        frames[2],frames[3],pd.concat(frames[4:],ignore_index=True)]
dimensions = {'speaker_group':'화자','source_corpus':'원본 코퍼스','attack_id':'공격·합성 방식',
              'codec':'코덱','transmission':'전송 조건'}
classes = {'all':'전체','bonafide':'Bona fide','spoof':'Spoof'}
records, summaries, validations, dataset_meta = [], [], [], []

for info, original in zip(prior['datasets'],sets):
    f = original.copy()
    assert f.utterance_id.is_unique and not f[['speaker_group','y','duration']].isna().any().any()
    if 'speaker_id_known' in f:
        assert f.speaker_id_known.eq(True).all()
    # 2019 cache omits a redundant corpus column; its source is explicitly ASVspoof2019 LA.
    if 'source_corpus' not in f:
        f['source_corpus'] = 'asvspoof'
    n = np.rint(f.duration.to_numpy()*16000).astype('int64')
    assert np.max(np.abs(n/16000-f.duration.to_numpy()))<1e-9 and (n>0).all()
    # floor(n/(8000*k)) == floor(floor(n/8000)/k), independently checked below.
    m = n//8000
    divisor_sum = np.array([sum(x//k for k in range(1,x+1)) for x in range(int(m.max())+1)],dtype='int64')
    weights = np.where(m>0,2*divisor_sum[m],1)
    independent = np.zeros(len(f),dtype='int64')
    for target in range(8000,int(n.max())+1,8000):
        independent += 2*(n//target)
    independent[n<8000] = 1
    assert np.array_equal(weights,independent)
    f['segments'] = weights
    f['forward_segments'] = divisor_sum[m]
    for y,name in [(1,'bonafide'),(0,'spoof')]:
        g = f.loc[f.y==y]
        assert len(g)==info['classes'][name]['original_utterances']
        assert int(g.segments.sum())==info['classes'][name]['all_segment_count']
    usable = [dim for dim in dimensions if dim in f and f[dim].notna().any()]
    dataset_meta.append({'key':info['key'],'title':info['title'],'dimensions':usable})
    for class_key, y in [('all',None),('bonafide',1),('spoof',0)]:
        g = f if y is None else f.loc[f.y==y]
        before_total,after_total = len(g),int(g.segments.sum())
        for dim in usable:
            h = g.assign(group=g[dim].fillna('미상').astype(str)).groupby('group',sort=True).agg(
                original_utterances=('utterance_id','size'),segment_count=('segments','sum'),
                forward_count=('forward_segments','sum'),mean_original_seconds=('duration','mean'))
            names,codes = np.unique(g[dim].fillna('미상').astype(str).to_numpy(),return_inverse=True)
            assert list(names)==h.index.tolist()
            assert np.array_equal(np.bincount(codes),h.original_utterances.to_numpy())
            assert np.array_equal(np.bincount(codes,weights=g.segments.to_numpy()),h.segment_count.to_numpy())
            assert int(h.original_utterances.sum())==before_total and int(h.segment_count.sum())==after_total
            h['before_fraction'] = h.original_utterances/before_total
            h['after_fraction'] = h.segment_count/after_total
            h['delta_pp'] = 100*(h.after_fraction-h.before_fraction)
            h['relative_share_multiplier'] = h.after_fraction/h.before_fraction
            h['segments_per_original'] = h.segment_count/h.original_utterances
            assert np.isclose(h.before_fraction.sum(),1) and np.isclose(h.after_fraction.sum(),1)
            assert np.isclose(h.delta_pp.sum(),0,atol=1e-10)
            tv = float(np.abs(h.after_fraction-h.before_fraction).sum()/2)
            inc, dec = h.delta_pp.idxmax(),h.delta_pp.idxmin()
            summary = {'dataset':info['key'],'class':class_key,'dimension':dim,'groups':len(h),
                       'original_total':before_total,'segment_total':after_total,
                       'TV_percent':100*tv,'before_after_overlap_percent':100*(1-tv),
                       'effective_groups_before':float(1/np.square(h.before_fraction).sum()),
                       'effective_groups_after':float(1/np.square(h.after_fraction).sum()),
                       'largest_increase_group':inc,'largest_increase_pp':float(h.loc[inc,'delta_pp']),
                       'largest_decrease_group':dec,'largest_decrease_pp':float(h.loc[dec,'delta_pp'])}
            summaries.append(summary)
            for group,row in h.iterrows():
                records.append({'dataset':info['key'],'class':class_key,'dimension':dim,'group':group,
                                'original_total':before_total,'segment_total':after_total,
                                **{k:int(row[k]) for k in ['original_utterances','segment_count','forward_count']},
                                **{k:float(row[k]) for k in ['mean_original_seconds','before_fraction','after_fraction',
                                                           'delta_pp','relative_share_multiplier','segments_per_original']}})
            # Grouping preserves exactly the same whole-class denominator.
            validations.append({'dataset':info['key'],'class':class_key,'dimension':dim,'groups':len(h),
                                'counts_reconcile':True,'fractions_sum_to_one':True,'independent_bincount_match':True})

df = pd.DataFrame(records)
sf = pd.DataFrame(summaries)
df.to_csv(OUT/'group_shares_before_after.csv',index=False,encoding='utf-8-sig')
sf.to_csv(OUT/'group_shift_summary.csv',index=False,encoding='utf-8-sig')
result = {'method':{'before':'one count per original utterance','after':'one count per emitted front/back crop; duplicates retained; <0.5 s kept once',
                    'denominator':'separately per dataset and selected class; all is the natural overall mixture',
                    'training_interpretation':'uniform crop sampling within selected population; actual balanced sampler exposure is not measured',
                    'source_definition':'source_corpus is original corpus, not inferred microphone/room identity',
                    'speakers':'protocol speaker_group; corpus namespace retained; all speaker_id_known flags checked true',
                    'same_corpus_for_2019':'ASVspoof2019 LA source is known from dataset provenance; not an inferred recording condition',
                    'audio_files_created':False,'training_started':False},
          'datasets':dataset_meta,'dimensions':dimensions,'classes':classes,'sources':sources,
          'summaries':summaries,'rows':records,'validation':validations}
(OUT/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

configs = {}
for (dataset,cl,dim),g in df.groupby(['dataset','class','dimension'],sort=False):
    g = g.assign(abs_delta=g.delta_pp.abs()).sort_values(['abs_delta','group'],ascending=[False,True],kind='stable')
    limit=float(max(g.before_fraction.max(),g.after_fraction.max())*1.1)
    for page in range((len(g)+14)//15):
        chunk=g.iloc[page*15:(page+1)*15]
        cfg={'id':'group-share-change','title':'분할 전후 구성비','height':max(220,60+len(chunk)*30),
             'spec':{'type':'horizontalBar','y':'그룹','x':['분할 전','분할 후'],'xLabel':'선택한 데이터셋·클래스 내 비율',
                     'sortOrder':'original','startAtZero':True,'xDomain':[0,min(1.,limit)],
                     'valueFormat':{'style':'percent','maximumFractionDigits':2}},
             'columns':['그룹','분할 전','분할 후'],
             'rows':[[row.group,float(row.before_fraction),float(row.after_fraction)] for row in chunk.itertuples()]}
        configs[f'{dataset}|{cl}|{dim}|{page}']=cfg
(OUT/'chart-configs.json').write_text(json.dumps(configs,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('GROUPS',len(df),'SUMMARIES',len(sf),'CHART CONFIGS',len(configs))
print(sf[sf.dimension=='speaker_group'][['dataset','class','TV_percent','largest_increase_group','largest_increase_pp','largest_decrease_group','largest_decrease_pp']].to_string(index=False))
print(df[(df.dimension=='source_corpus')&(df.dataset=='21DF_eval')][['class','group','before_fraction','after_fraction','delta_pp']].to_string(index=False))
print(df[(df.dimension=='speaker_group')&(df.dataset=='19LA_train')].sort_values('delta_pp').iloc[[0,-1]][['class','group','before_fraction','after_fraction','delta_pp']].to_string(index=False))
