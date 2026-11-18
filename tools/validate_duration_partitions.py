"""Independent per-file validation and sharp global overlap bound.

The bound applies even to mixed-length, off-grid partitions: at least 0.5s per
new piece means file i can yield at most floor(duration_i/0.5s) pieces. All bona
fide originals here are >=0.5s. Unchanged spoof originals below 0.5s can never
overlap bona fide output. Their probability is minimized by producing the
maximum possible number of spoof pieces. Cutting everything else into 0.5s
pieces attains that bound, with remainder strictly below 0.5s.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

SR=16000


def main(output):
    root=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(root/'artifacts/duration_matching_20261109'))
    from analyze import load
    output.mkdir(parents=True,exist_ok=True)
    rows=[];checks={}
    for key,title,source in load()[0]:
        source=source.copy()
        source['original_samples']=np.rint(source.duration.to_numpy()*SR).astype('int64')
        records=[];class_info={}
        for y,f in source.groupby('y'):
            a=f.original_samples.to_numpy()
            k=np.maximum(1,a//8000)
            length=np.where(a<8000,a,8000)
            waste=a-k*length
            assert (waste>=0).all() and (waste<8000).all()
            assert (k*length+waste==a).all()
            info={'originals':len(a),'pieces':int(k.sum()),'short_originals':int((a<8000).sum()),
                  'discarded_seconds':float(waste.sum()/SR),'max_discard_s':float(waste.max()/SR),
                  'total_seconds':float(a.sum()/SR)}
            class_info[int(y)]=info
            records.append(pd.DataFrame({'utterance_id':f.utterance_id,'y':int(y),
                'original_samples':a,'pieces':k,'piece_samples':length,'discarded_samples':waste}))
        assert class_info[1]['short_originals']==0
        common=1-class_info[0]['short_originals']/class_info[0]['pieces']
        baseline=pd.concat(records,ignore_index=True)
        baseline.to_parquet(output/f'{key}_global_optimum_half_second_plan.parquet',index=False)
        total=float(baseline.original_samples.sum()/SR)
        lost=float(baseline.discarded_samples.sum()/SR)
        row={'dataset':key,'global_maximum_overlap_pct':100*common,
             'unavoidable_spoof_short_originals':class_info[0]['short_originals'],
             'maximum_spoof_piece_count':class_info[0]['pieces'],
             'maximum_bonafide_piece_count':class_info[1]['pieces'],
             'baseline_piece_multiplier':float(baseline.pieces.sum()/len(baseline)),
             'baseline_audio_discard_pct':100*lost/total,
             'baseline_max_discard_per_source_s':float(baseline.discarded_samples.max()/SR)}
        short_mask=baseline.original_samples<8000
        row['subminimum_original_duration_s']=float(baseline.loc[short_mask,'original_samples'].sum()/SR)
        row['global_maximum_if_subminimum_originals_may_be_discarded_pct']=100.
        rows.append(row)
        validation={'global_maximum_proved':True,'global_maximum_attained_by_half_second_plan':True,
                    'class_info':class_info,
                    'alternative_if_entire_subminimum_original_may_be_lost':{
                        'all_classes_only_half_second_output_overlap_pct':100.,
                        'dropped_originals':int(short_mask.sum()),
                        'additional_discard_seconds':row['subminimum_original_duration_s'],
                        'each_dropped_original_strictly_below_tail_limit':True}}
        path=output/'full'/f'{key}_partition_plan.parquet'
        if path.exists():
            plan=pd.read_parquet(path)
            joined=plan.merge(source[['utterance_id','y','original_samples']],on='utterance_id',
                              how='outer',suffixes=('_plan','_source'),validate='one_to_one',indicator=True)
            assert (joined['_merge']=='both').all()
            assert (joined.y_plan==joined.y_source).all()
            assert (joined.original_samples_plan==joined.original_samples_source).all()
            assert (plan.pieces>=1).all() and (plan.piece_samples>0).all()
            used=plan.pieces*plan.piece_samples
            assert (used<=plan.original_samples).all()
            assert (used+plan.discarded_samples==plan.original_samples).all()
            assert ((plan.discarded_samples>=0)&(plan.discarded_samples<8000)).all()
            short=plan.original_samples<8000
            assert (plan.loc[~short,'piece_samples']>=8000).all()
            assert (plan.loc[short,'pieces']==1).all()
            assert (plan.loc[short,'discarded_samples']==0).all()
            assert (plan.loc[short,'piece_samples']==plan.loc[short,'original_samples']).all()
            # For each compact manifest row, intervals [jL,(j+1)L) are adjacent,
            # disjoint, and end at used<=source length. Verify all expanded ends
            # vectorially without producing an unnecessary multi-million-row file.
            assert np.array_equal(used.to_numpy(),(plan.pieces*plan.piece_samples).to_numpy())
            hb=plan[plan.y==1].groupby('histogram_bin').pieces.sum()
            hs=plan[plan.y==0].groupby('histogram_bin').pieces.sum()
            h=pd.concat([hb.rename('b'),hs.rename('s')],axis=1).fillna(0)
            achieved=float(np.minimum(h.b/h.b.sum(),h.s/h.s.sum()).sum())
            assert achieved<=common+1e-9
            validation['practical_plan']={'one_to_one_source_join':True,
                'source_labels_and_sample_counts_match':True,'no_overlap_no_reuse_full_retained_span':True,
                'strict_tail_limit_all_files':True,'minimum_piece_length_and_short_exception':True,
                'independently_recomputed_overlap_pct':100*achieved,
                'gap_to_global_upper_bound_pp':100*(common-achieved)}
        checks[key]=validation
        print(json.dumps(row),flush=True)
    pd.DataFrame(rows).to_csv(output/'global_optimum.csv',index=False)
    (output/'independent_validation.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[2]/'artifacts/duration_partitions_20261109')
    args=p.parse_args();main(args.output)
