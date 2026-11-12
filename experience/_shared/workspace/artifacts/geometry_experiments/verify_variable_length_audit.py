"""Independent checks of exported feasibility diagnostics and metadata packing."""
import hashlib
import json
import math
import numpy as np
import pandas as pd
from audit_variable_length_metadata import OUT, read, schedule


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    audit=json.loads((OUT/'metadata_audit.json').read_text());complete=json.loads((OUT/'complete.json').read_text())
    checks=[]
    def check(name,ok):
        assert ok,name
        checks.append(name)
    for name,sha in complete['files'].items():check('probe_export_hash_'+name,digest(OUT/name)==sha)
    more=OUT/'augmentation_probe';sub=json.loads((more/'complete.json').read_text())
    for name,sha in sub['files'].items():check('augmentation_export_hash_'+name,digest(more/name)==sha)
    lock=json.loads((OUT/'probe_selection_lock.json').read_text())
    check('score_blind_selection_lock',digest(OUT/'probe_selection.parquet')==lock['sha256'])
    d,_=read('T');sample=d.samples.to_numpy()
    # Closed-form receptive field400/stride320, independent of looped conv helper.
    t=(sample-400)//320+1
    for mode,saved in audit['packing_simulation'].items():
        steps=schedule(d,mode);seen=[];useful=allocated=useful_att=allocated_att=0
        for step in steps:
            check(f'{mode}_same_rank_count_step{len(seen)//160}',len(step)==2 and len(step[0])==len(step[1])==80)
            for q in step:
                check(f'{mode}_class_order_step{len(seen)//80}',d.iloc[q].y.tolist()==[1]*8+[0]*72)
                a=t[q];useful+=sum(int(x) for x in a);allocated+=len(q)*int(max(a))
                useful_att+=sum(int(x)**2 for x in a);allocated_att+=len(q)*int(max(a))**2;seen.extend(q)
        check(mode+'_all_rows_covered',len(set(seen))==len(d))
        check(mode+'_full_epoch_count',len(seen)==25920 and len(steps)==162)
        check(mode+'_frame_fraction',abs(useful/allocated-saved['useful_frame_fraction'])<1e-14)
        check(mode+'_attention_fraction',abs(useful_att/allocated_att-saved['useful_attention_pair_fraction'])<1e-14)
    prep=json.loads((OUT/'prepare_complete.json').read_text())
    check('all_25380_actual_train_headers',prep['actual_train_headers_checked']==25380 and prep['all_train_header_samplecounts_match_metadata'])
    check('archive_source_checkpoint_unchanged',complete['all_archive_current_source_checkpoint_hashes_unchanged'])
    ranks=[json.loads((OUT/f'rank{rank}_results.json').read_text()) for rank in [0,1]]
    for rank,r in enumerate(ranks):
        check(f'rank{rank}_all_pass',r['status']=='pass' and r['classifier_state_before']==r['classifier_state_after'])
        check(f'rank{rank}_layer_norm_frontend',r['frontend_config']['feat_extract_norm']=='layer')
        check(f'rank{rank}_all_10_batch_comparisons',len(r['padded_batch_equivalence'])==10)
        check(f'rank{rank}_batch_score_equivalence',max(q['max_score_error'] for q in r['padded_batch_equivalence'])<1e-6)
        check(f'rank{rank}_arbitrary_pad_ignored',r['arbitrary_backend_padding_max_z_error']==0)
        check(f'rank{rank}_mask_negative_controls',r['omitted_all_masks_max_z_change']>1 and r['omitted_backend_masks_max_z_change']>.1)
        check(f'rank{rank}_two_stage_gradients',r['classifier_backward_stage1']['finite_gradient_tensors']==181 and r['centroid_backward_stage2']['centroid_gradient_norm']>0)
        actual=json.loads((more/f'rank{rank}_audio_backward.json').read_text())
        check(f'rank{rank}_actual_audio_long_short_backward',actual['status']=='pass' and actual['finite_gradient_tensors']==181)
        check(f'rank{rank}_ragged_pool_gradient_reference',actual['ragged_pool_gradient_error']<1e-12 and actual['padded_feature_gradient_exact_zero'])
        check(f'rank{rank}_no_audio_feature_saved',not r['raw_audio_or_SSL_features_saved'] and not actual['raw_audio_or_SSL_features_saved'])
    check('two_gpu_numeric_replication',ranks[0]['padded_batch_equivalence']==ranks[1]['padded_batch_equivalence'])
    codec=json.loads((more/'augmentation_results.json').read_text())
    check('codec_deltas_recomputed',all(q['output_samples']-q['input_samples']==q['delta_samples'] for q in codec['codec_cases']))
    check('codec_length_change_observed',len(codec['codec_cases'])==15 and max(q['delta_samples'] for q in codec['codec_cases'])==1985)
    result=dict(status='pass',checks=len(checks),check_names=checks,
        max_batch_z_error=max(q['max_z_error'] for r in ranks for q in r['padded_batch_equivalence']),
        max_batch_score_error=max(q['max_score_error'] for r in ranks for q in r['padded_batch_equivalence']),
        no_performance_improvement_claim=True,DDP_training_or_full_batch_capacity_not_verified=True)
    (OUT/'independent_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='check_names'},ensure_ascii=False))


if __name__=='__main__':main()
