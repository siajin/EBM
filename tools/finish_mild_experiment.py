"""Prepare fixed20 evaluation and independently compare existing B/C20 EERs."""
import csv
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(sys.argv[1]);sys.path.insert(0,str(ROOT/'code'))
from tools.compare_coverage_epoch20 import eer, validate_phase, save


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    config=json.loads((ROOT/'experiment.json').read_text())
    result=Path(config['host_result_path'])
    if '--configure' in sys.argv:
        # These /home paths are visible inside the existing container as well.
        audit=json.loads((ROOT/'final_training_audit.json').read_text())
        evaluation={'root':str(ROOT),'epoch':20,'code_root':str(ROOT/'code'),
            'source_sha256':config['source_sha256'],'checkpoint':str(result/'model/CLS_20.pt'),
            'checkpoint_sha256':audit['CLS20_sha256'],'train_state':str(result/'model/TrainState_20.pt'),
            'train_state_sha256':audit['TrainState20_sha256'],'frontend_revision':config['frontend_revision'],
            'evaluator_sha256':sha(ROOT/'code/tools/evaluate_coverage_checkpoint.py'),
            'phases':{'21LA':148176,'21DF':533928},'class_counts':{'21LA':[14816,133360],'21DF':[14869,519059]},
            'reference_19LA_scores':str(result/'text/19LA/scores_20.txt'),'evaluation_seed':1729}
        save(ROOT/'evaluation_config.json',evaluation)
    else:
        cfg=json.loads((ROOT/'evaluation_config.json').read_text())
        complete=json.loads((ROOT/'complete.json').read_text())
        assert complete['epoch']==20 and complete['status']=='complete'
        proofs=[validate_phase(ROOT,p,cfg) for p in cfg['phases']]
        current={r['phase']:r['independently_recomputed_EER_percent'] for r in proofs}
        with Path(cfg['reference_19LA_scores']).open() as source: rows=list(csv.DictReader(source))
        assert sorted(int(r['index']) for r in rows)==list(range(71237))
        assert sum(int(r['label']) for r in rows)==7355
        current['19LA']=eer([(float(r['score']),int(r['label'])) for r in rows])
        baseline=Path('/home/mentee/analysis/ebm_B_C_checkpoints_20261109/metrics.json')
        old=json.loads(baseline.read_text())
        legacy_path=Path('/home/mentee/analysis/ebm_B_final_eval_20261107_215451/metrics.json')
        legacy=json.loads(legacy_path.read_text())
        previous_path=Path(config['wait_for_evaluation_root'])/'comparison.json'
        previous=json.loads(previous_path.read_text())
        previous20={r['phase']:r['coverage_EER_percent'] for r in previous['same_epoch_comparison']}
        table=[]
        for phase in ['19LA','21LA','21DF']:
            b,c=[old[phase][f'{name}_e20']['EER_percent'] for name in ['B','C']]
            table.append({'phase':phase,'epoch':20,'mild_EER_percent':current[phase],
                'B_EER_percent':b,'C_EER_percent':c,'minus_B_percentage_points':current[phase]-b,
                'minus_C_percentage_points':current[phase]-c,
                'previous_short_activity_E20_EER_percent':previous20[phase],
                'legacy240_E50_reference_EER_percent':legacy[phase]['legacy240']['EER_percent']})
        save(ROOT/'comparison.json',{'status':'complete','fixed_epoch':20,'same_epoch_comparison':table,
            'validation':proofs,'checkpoint_sha256':cfg['checkpoint_sha256'],
            'baseline_metrics_sha256':sha(baseline),'legacy_metrics_sha256':sha(legacy_path),
            'previous_comparison_sha256':sha(previous_path),'protocol':'front64600, no repeat, mask, batch8, trained OC cosine',
            'limits':['Fixed20 selected before this run; 19LA every5 is monitoring only.',
                'Saved B/C scores use a different evaluation noise RNG seed.',
                'legacy240 fixed50 is a reference; epoch differs from this fixed20 run.',
                'No single-factor causal conclusion: input duration and pairing differ from B/C.',
                'One training seed; no significance claim.']})
        save(ROOT/'analysis_complete.json',{'status':'complete','comparison_sha256':sha(ROOT/'comparison.json')})


if __name__=='__main__': main()
