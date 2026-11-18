"""Fixed20 same-RNG original240 comparison; B/C remain secondary saved references."""
import csv
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(sys.argv[1]);sys.path.insert(0,str(ROOT/'code'))
from tools.compare_coverage_epoch20 import eer,validate_phase,save


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if '--configure' in sys.argv:
        from tools.finish_mild_experiment import main as configure
        return configure()
    cfg=json.loads((ROOT/'experiment.json').read_text())
    ev=json.loads((ROOT/'evaluation_config.json').read_text())
    assert json.loads((ROOT/'complete.json').read_text())['status']=='complete'
    proofs=[validate_phase(ROOT,p,ev) for p in ev['phases']]
    current={p['phase']:p['independently_recomputed_EER_percent'] for p in proofs}
    with Path(ev['reference_19LA_scores']).open() as stream:rows=list(csv.DictReader(stream))
    assert sorted(int(r['index']) for r in rows)==list(range(71237)) and sum(int(r['label']) for r in rows)==7355
    current['19LA']=eer([(float(r['score']),int(r['label'])) for r in rows])
    ref_root=ROOT/'original240_eval';ref_cfg=json.loads((ref_root/'evaluation_config.json').read_text())
    assert json.loads((ref_root/'complete.json').read_text())['status']=='complete'
    refs=[validate_phase(ref_root,p,ref_cfg) for p in ref_cfg['phases']]
    reference={p['phase']:p['independently_recomputed_EER_percent'] for p in refs}
    bc_path=Path('/home/mentee/analysis/ebm_B_C_checkpoints_20261109/metrics.json')
    bc=json.loads(bc_path.read_text());table=[]
    for phase in ['19LA','21LA','21DF']:
        table.append({'phase':phase,'epoch':20,'new_data_original240_training_EER_percent':current[phase],
            'preserved_original240_EER_percent':reference[phase],
            'difference_percentage_points':current[phase]-reference[phase],
            'B20_saved_EER_percent':bc[phase]['B_e20']['EER_percent'],
            'C20_saved_EER_percent':bc[phase]['C_e20']['EER_percent']})
    result={'status':'complete','fixed_epoch':20,'same_epoch_comparison':table,'score_validation':proofs+refs,
        'checkpoint_sha256':ev['checkpoint_sha256'],'original240_checkpoint_sha256':ref_cfg['checkpoint_sha256'],
        'training_core':'archived240 full graph, original class samplers, fixed repeat/crop, same-source long/short BPL, two-stage Adam',
        'evaluation_protocol':'both new and original240: front64600, no repeat, masks, batch8, own OC cosine, ASP1e-5, seed1729+stream525252+rank',
        'original240_and_new_evaluation_RNG_equal':True,'B_C_saved_evaluation_RNG_equal':False,
        'baseline_metrics_sha256':sha(bc_path),'limits':[
            'One training seed. No statistical significance claim.',
            'Original240 historical codec execution was not logged; current run explicitly keeps codec disabled.',
            'Raw original intervals retained by mild generation do not equal final network coverage after restored fixed views.',
            'Preserved original20 has no optimizer TrainState; do not claim an Adam-state verification for that historical reference.']}
    save(ROOT/'comparison.json',result)
    save(ROOT/'analysis_complete.json',{'status':'complete','comparison_sha256':sha(ROOT/'comparison.json')})


if __name__=='__main__':main()
