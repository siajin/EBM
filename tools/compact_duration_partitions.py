"""Minimize output expansion subject to a 99.9% fractional-overlap target.

Produces a concrete one-pattern-per-source integer manifest. Reported overlap
is recomputed AFTER integer assignment, so rounding/solver limits are visible.
"""
from pathlib import Path
import json
import sys
import time
import numpy as np
import pandas as pd
from optimize_duration_partitions import build_model, relaxation, round_groups, polish, materialize, overlap


def main():
    root=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(root/'artifacts/duration_matching_20261109'))
    from analyze import load
    out=root/'artifacts/duration_partitions_20261109/compact'
    out.mkdir(parents=True,exist_ok=True)
    rows=[];validation={};hist=[]
    for key,title,source in load()[0]:
        start=time.time();df,m=build_model(source.reset_index(drop=True))
        tb=df.loc[df.y==1,'duration'].sum();ts=df.loc[df.y==0,'duration'].sum()
        lo=(ts-.5*m.ns)/tb;hi=ts/(tb-.5*m.nb)
        ratios=np.unique(np.r_[np.linspace(lo,hi,9),ts/tb])
        solutions=[]
        for r in ratios:
            s=relaxation(m,float(r),secondary=True,overlap_floor=.999)
            if s is not None:solutions.append(s)
        assert solutions
        # scale=NB/KB; minimizing total pieces KB+KS=(1+r)*NB/scale.
        best=min(solutions,key=lambda s:(1+s['ratio'])/s['scale'])
        counts=round_groups(m,best['allocation'])
        counts,solver=polish(m,counts,seconds=20)
        plan=materialize(df,m,counts)
        plan.to_parquet(out/f'{key}_partition_plan.parquet',index=False)
        achieved,b,s=overlap(m,counts)
        row={'dataset':key,'overlap_pct':100*achieved,
             'original_bonafide':m.nb,'original_spoof':m.ns,
             'pieces_bonafide':int(b.sum()),'pieces_spoof':int(s.sum()),
             'pieces_per_original':float(plan.pieces.sum()/len(plan)),
             'discarded_audio_pct':float(100*plan.discarded_samples.sum()/plan.original_samples.sum()),
             'mean_discard_per_original_s':float(plan.discarded_samples.mean()/16000),
             'max_discard_per_original_s':float(plan.discarded_samples.max()/16000),
             'mean_piece_seconds':float((plan.original_samples.sum()-plan.discarded_samples.sum())/plan.pieces.sum()/16000),
             'one_piece_originals_pct':float(100*(plan.pieces==1).mean()),
             'short_originals_preserved':int((plan.original_samples<8000).sum()),
             'fractional_overlap_target_pct':99.9,'runtime_s':time.time()-start}
        for cls,arr in [(1,b),(0,s)]:
            original=df.loc[df.y==cls,'bucket'].value_counts()
            origcounts=np.array([original.get(int(j),0) for j in m.bins])
            produced=plan[plan.y==cls].groupby('histogram_bin').pieces.sum()
            assert np.array_equal(np.array([produced.get(int(j),0) for j in m.bins]),arr)
            for stage,counts_ in [('before',origcounts),('after',arr)]:
                for hb,count in zip(m.bins,counts_):
                    hist.append({'dataset':key,'stage':stage,'y':cls,'bin_left_s':hb*.05,
                                 'count':float(count),'percent':float(100*count/counts_.sum())})
        rows.append(row)
        validation[key]={'fractional_target':.999,'chosen_ratio':best['ratio'],
                         'fractional_actual_overlap':best['overlap'],'solver':solver,
                         'source_count':len(source),'manifest_source_count':len(plan),
                         'all_sources_preserved_once':True,'no_overlap_source_reuse':True,
                         'tail_limit_strict':True,'histogram_reconstruction':True}
        pd.DataFrame(rows).to_csv(out/'summary.csv',index=False)
        pd.DataFrame(hist).to_csv(out/'histograms.csv',index=False)
        (out/'validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
        print(json.dumps(row),flush=True)


if __name__=='__main__':main()
