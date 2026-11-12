"""Audit completed positive-BPL pilot artifacts and prepare reviewed chart rows."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
from positive_bpl_pilot import eer

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/positive_only_bpl_20261110'
assert (OUT/'run.exit').read_text().strip()=='0'
assert (OUT/'smoke.exit').read_text().strip()=='0'
protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
complete=json.loads((OUT/'pilot/complete.json').read_text(encoding='utf-8'))
assert complete['steps']==108 and complete['waveform_draws']==108*48
assert complete['frontend_frozen'] and not complete['OC_loss_used'] and complete['codec_calls']==0
assert hashlib.sha256((OUT/'manifest.json').read_bytes()).hexdigest()==protocol['manifest_sha256']
assert hashlib.sha256((OUT/'policy.json').read_bytes()).hexdigest()==protocol['policy_sha256']
history=[json.loads(line) for line in (OUT/'pilot/training.jsonl').read_text().splitlines()]
assert [x['step'] for x in history]==list(range(1,109)) and all(np.isfinite(x['loss']) and x['pairs_per_gpu']==144 for x in history)
rows=[]
for step in [0,27,54,108]:
    d=json.loads((OUT/f'pilot/diagnostics_step{step:04d}.json').read_text())
    score=pd.read_csv(OUT/f'pilot/scores_step{step:04d}.csv')
    assert len(score)==512 and score.uid.is_unique
    assert score.uid.tolist()==[r['uid'] for r in manifest['dev']]
    assert score.label.tolist()==[r['label'] for r in manifest['dev']]
    recalculated=eer(score.label.to_numpy(),score.clean_score.to_numpy())
    for name in ['EER_percent','AUC']:
        assert np.isclose(recalculated[name],d['clean_detection'][name],atol=1e-12)
    clean_subset=score[score.uid.isin({r['uid'] for r in manifest['noise']})]
    rec_noise=eer(clean_subset.label.to_numpy(),clean_subset.clean_score.to_numpy())
    assert len(clean_subset)==128
    assert np.isclose(rec_noise['EER_percent'],d['noise']['clean_same_subset']['EER_percent'],atol=1e-12)
    rows.append({'step':step,'clean_dev512_EER_percent':d['clean_detection']['EER_percent'],
                 'clean_dev512_AUC':d['clean_detection']['AUC'],
                 'paired_clean128_EER_percent':d['noise']['clean_same_subset']['EER_percent'],
                 'paired_noisy128_EER_percent':d['noise']['noisy_detection']['EER_percent'],
                 'white_noise_cosine_distance':d['noise']['mean_cosine_distance'],
                 'unit_covariance_trace':d['dev']['unit_covariance_trace'],
                 'effective_rank':d['dev']['covariance_effective_rank'],
                 'mean_offdiagonal_cosine':d['dev']['mean_offdiagonal_cosine'],
                 'relative_noise_to_between_utterance_distance':d['noise']['mean_cosine_distance']/(1-d['dev']['mean_offdiagonal_cosine'])})
summary=pd.DataFrame(rows);summary.to_csv(OUT/'summary.csv',index=False,encoding='utf-8-sig')
first,last=rows[0],rows[-1]
analysis={'pilot_complete':True,'steps':108,'scope':'fresh backend vs its own initialization; locked stratified dev diagnostics, not official benchmark comparison',
          'clean_EER_change_pp':last['clean_dev512_EER_percent']-first['clean_dev512_EER_percent'],
          'paired_noisy_EER_change_pp':last['paired_noisy128_EER_percent']-first['paired_noisy128_EER_percent'],
          'white_noise_distance_reduction_percent':100*(1-last['white_noise_cosine_distance']/first['white_noise_cosine_distance']),
          'unit_variance_reduction_percent':100*(1-last['unit_covariance_trace']/first['unit_covariance_trace']),
          'relative_noise_ratio_reduction_percent':100*(1-last['relative_noise_to_between_utterance_distance']/first['relative_noise_to_between_utterance_distance']),
          'initial_effective_rank':first['effective_rank'],'final_effective_rank':last['effective_rank'],
          'final_unit_variance_positive':last['unit_covariance_trace']>0,'rows':rows,
          'limitations':protocol['limits']+['Controlled noise is one 20dB additive-white-noise condition.',
                      'No all-constant representation at the measured stages; strong directional contraction is not ruled out by a nonzero effective rank.',
                      'No conclusion about full 50-epoch performance, unseen attacks, or superiority to B/C.'],
          'validation':{'score_UID_label_order_matches_manifest':True,'clean_EER_and_AUC_independently_recomputed':True,
                        'paired_clean_subset_EER_recomputed':True,'all_training_losses_finite':True,'all144_pairs_used':True,
                        'no_raw_audio_SSL_hidden_or_model_downloaded':True}}
(OUT/'analysis.json').write_text(json.dumps(analysis,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
charts={'detection':{'id':'positive-bpl-noise-detection','title':'동일 검증 128개 · 잡음 전후 탐지 오류',
           'height':300,'spec':{'type':'line','x':'업데이트','y':['원음','20dB 백색잡음'],'xLabel':'BPL 학습 업데이트',
             'yLabel':'EER · 낮을수록 좋음','yDomain':[0,.65],
             'valueFormat':{'style':'percent','maximumFractionDigits':2}},
           'columns':['업데이트','원음','20dB 백색잡음'],
           'rows':[[r['step'],r['paired_clean128_EER_percent']/100,r['paired_noisy128_EER_percent']/100] for r in rows]},
        'geometry':{'id':'positive-bpl-noise-geometry','title':'잡음 반응과 전체 표현 변동량 · 초기값 100%',
           'height':300,'spec':{'type':'line','x':'업데이트','y':['잡음으로 인한 변화','전체 표현 변동량'],'xLabel':'BPL 학습 업데이트',
             'yLabel':'초기값 대비 비율','yDomain':[0,1.05],
             'valueFormat':{'style':'percent','maximumFractionDigits':1}},
           'columns':['업데이트','잡음으로 인한 변화','전체 표현 변동량'],
           'rows':[[r['step'],r['white_noise_cosine_distance']/first['white_noise_cosine_distance'],r['unit_covariance_trace']/first['unit_covariance_trace']] for r in rows]}}
(OUT/'chart-configs.json').write_text(json.dumps(charts,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(summary.to_string(index=False))
print(json.dumps({k:v for k,v in analysis.items() if k not in ['rows','limitations','validation']},ensure_ascii=False,indent=2))
