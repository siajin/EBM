"""Compare wrap-extension vs bucketed zero-padding on identical T occurrences.

Waveform sample counts, before stochastic codec/RawBoost augmentation. Both
training views are counted; this is not measured runtime or a training run.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from audit_variable_length_metadata import read, schedule
from geometry_cache import DEFAULT_ROOT

OUT=DEFAULT_ROOT/'local_analysis/padding_budget_20261106_v1'
LONG,SHORT=64600,48440


def measure(data,steps):
    lengths=data.samples.to_numpy();batches=[q for step in steps for q in step]
    occurrences=np.concatenate(batches)
    assert len(occurrences)==25920 and len(set(occurrences))==25380
    old={};new={}
    for view,cap in [('long',LONG),('short',SHORT)]:
        n=lengths[occurrences]
        old[view]=dict(total_samples=int(len(n)*cap),real_samples=int(np.minimum(n,cap).sum()),
            extension_samples=int(np.maximum(cap-n,0).sum()),extension_occurrences=int((n<cap).sum()))
        total=real=pad=affected=0
        for q in batches:
            valid=lengths[q] if view=='long' else np.minimum(lengths[q],cap)
            total+=int(valid.max()*len(q));real+=int(valid.sum());pad+=int((valid.max()-valid).sum())
            affected+=int((valid<valid.max()).sum())
        new[view]=dict(total_samples=total,real_samples=real,extension_samples=pad,extension_occurrences=affected)
    for method in [old,new]:
        method['combined']={key:method['long'][key]+method['short'][key] for key in method['long']}
    rows={}
    for view in old:
        a,b=old[view],new[view]
        assert a['total_samples']==a['real_samples']+a['extension_samples']
        assert b['total_samples']==b['real_samples']+b['extension_samples']
        rows[view]=dict(old_wrap=a,new_zero=b,
            old_extension_percent=100*a['extension_samples']/a['total_samples'],
            new_extension_percent=100*b['extension_samples']/b['total_samples'],
            extension_fraction_drop_pp=100*(a['extension_samples']/a['total_samples']-b['extension_samples']/b['total_samples']),
            added_samples_reduction_percent=100*(1-b['extension_samples']/a['extension_samples']),
            total_input_samples_reduction_percent=100*(1-b['total_samples']/a['total_samples']),
            old_added_seconds_per_utterance=a['extension_samples']/16000/len(occurrences),
            new_added_seconds_per_utterance=b['extension_samples']/16000/len(occurrences),
            old_added_hours=a['extension_samples']/16000/3600,
            new_added_hours=b['extension_samples']/16000/3600,
            restored_original_hours=(b['real_samples']-a['real_samples'])/16000/3600)
    return rows


def independent_check(data,steps,rows):
    """Scalar, per-occurrence accounting independent of vector reductions."""
    lengths=data.samples.tolist();acc={v:{'old_total':0,'old_real':0,'old_pad':0,'new_total':0,'new_real':0,'new_pad':0} for v in ['long','short']}
    for step in steps:
        assert len(step)==2
        for ids in step:
            assert len(ids)==80 and data.iloc[ids].y.tolist()==[1]*8+[0]*72
            full=[lengths[int(i)] for i in ids];short=[min(n,SHORT) for n in full]
            for view,values,cap in [('long',full,LONG),('short',short,SHORT)]:
                maximum=max(values);r=acc[view]
                for original,valid in zip(full,values):
                    r['old_total']+=cap;r['old_real']+=min(original,cap);r['old_pad']+=max(cap-original,0)
                    r['new_total']+=maximum;r['new_real']+=valid;r['new_pad']+=maximum-valid
    for view,r in acc.items():
        for method,prefix in [('old_wrap','old'),('new_zero','new')]:
            for field,suffix in [('total_samples','total'),('real_samples','real'),('extension_samples','pad')]:
                assert rows[view][method][field]==r[prefix+'_'+suffix]
    # End-to-end conservation: replacing extension and restoring clipped content.
    for row in rows.values():
        a,b=row['old_wrap'],row['new_zero']
        assert b['total_samples']==a['total_samples']-a['extension_samples']+b['extension_samples']+b['real_samples']-a['real_samples']
    return dict(status='pass',scalar_vs_vector_exact=True,per_rank_class_ratio=True,total_sample_accounting_exact=True)


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    data,fingerprint=read('T');steps=schedule(data,'sortish32',20261106)
    primary=measure(data,steps);verification=independent_check(data,steps,primary)
    unique={}
    for view,cap in [('long',LONG),('short',SHORT)]:
        n=data.samples.to_numpy();p=int(np.maximum(cap-n,0).sum());total=int(len(n)*cap)
        unique[view]=dict(N=len(n),wrap_samples=p,total_samples=total,wrap_percent=100*p/total)
    sensitivity={}
    for mode in ['random','sorted']:
        sensitivity[mode]=measure(data,schedule(data,mode,20261106))
    result=dict(status='pass',source_fingerprint=fingerprint,unique_utterances=len(data),scheduled_occurrences=25920,
        sampling_duplicates=dict(bonafide=12,spoof=528),seed=20261106,global_batch=160,local_batch=80,world_size=2,
        schedule='random pools32 steps, class-wise length sort, match class length ranks, shuffle step order',
        comparison='Exactly the same utterance occurrences and batches for old and new conditions',
        unit='16kHz waveform samples, before augmentation; long full original, short min(original,48440)',
        denominator='Sum of allocated waveform samples for each method; combined is sum of both views, not mean of percentages',
        primary=primary,unique_each_once_old_reference=unique,batching_controls=sensitivity,verification=verification,
        qualifications=['Packing simulation, not measured runtime or completed retraining','Codec augmentation may change actual lengths','Old long clips to64600; new long restores the entire original','Same25920occurrences as previous audit; additional sampling is not waveform wrap'],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='pass',primary=primary,unique_each_once_old_reference=unique,verification=verification),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
