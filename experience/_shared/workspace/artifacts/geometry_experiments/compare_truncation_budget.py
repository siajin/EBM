"""Original-content cropping for the same occurrences as padding comparison.

Pre-augmentation sample accounting. Short-view remains capped at48440 samples;
long-view becomes full length. Combined denominators count both original views,
not distinct utterance content lost from every view.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from audit_variable_length_metadata import read,schedule
from compare_padding_budget import LONG,SHORT,OUT


def main():
    dest=OUT/'truncation_comparison.json'
    assert not dest.exists(),'Preserve existing result'
    data,fingerprint=read('T');steps=schedule(data,'sortish32',20261106)
    ids=np.concatenate([q for step in steps for q in step]);lengths=data.samples.to_numpy()[ids]
    original=int(lengths.sum());N=len(ids)
    rows={}
    for view,cap in [('long',LONG),('short',SHORT)]:
        old_cut=int(np.maximum(lengths-cap,0).sum())
        new_cut=0 if view=='long' else old_cut
        old_n=int((lengths>cap).sum());new_n=0 if view=='long' else old_n
        rows[view]=dict(original_samples=original,old_cut_samples=old_cut,new_cut_samples=new_cut,
            old_cut_percent=100*old_cut/original,new_cut_percent=100*new_cut/original,
            cut_amount_reduction_percent=100*(old_cut-new_cut)/old_cut,
            old_affected_occurrences=old_n,new_affected_occurrences=new_n,
            old_affected_percent=100*old_n/N,new_affected_percent=100*new_n/N,
            old_cut_seconds_per_utterance=old_cut/16000/N,new_cut_seconds_per_utterance=new_cut/16000/N,
            restored_hours=(old_cut-new_cut)/16000/3600)
        assert sum(max(int(n)-cap,0) for n in lengths)==old_cut
    old=rows['long']['old_cut_samples']+rows['short']['old_cut_samples']
    new=rows['long']['new_cut_samples']+rows['short']['new_cut_samples']
    rows['combined']=dict(original_samples=2*original,old_cut_samples=old,new_cut_samples=new,
        old_cut_percent=100*old/(2*original),new_cut_percent=100*new/(2*original),
        cut_amount_reduction_percent=100*(old-new)/old,
        old_cut_seconds_per_utterance=old/16000/N,new_cut_seconds_per_utterance=new/16000/N,
        restored_hours=(old-new)/16000/3600)
    # Independently reconcile used-original counts from prior padding comparison.
    prior=json.loads((OUT/'comparison.json').read_text())
    assert prior['source_fingerprint']==fingerprint and prior['scheduled_occurrences']==N
    for view,r in rows.items():
        p=prior['primary'][view]
        assert r['original_samples']-r['old_cut_samples']==p['old_wrap']['real_samples']
        assert r['original_samples']-r['new_cut_samples']==p['new_zero']['real_samples']
    result=dict(status='pass',source_fingerprint=fingerprint,N_unique=len(data),scheduled_occurrences=N,
        comparison='Same occurrences and short-view rule as previous padding calculation',
        denominator='Total original waveform samples per view; combined is two copies of original sample total',
        unit='16kHz waveform samples before codec/RawBoost augmentation',
        qualification='Short-view cropping remains intentional. Combined cut is view-wise omission, not unique content absent from both views; new long retains all original content.',
        rows=rows,verification=dict(scalar_vector_exact=True,prior_used_original_sample_counts_exact=True),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
