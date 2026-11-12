"""Bounded, CPU-only post-evaluation validation and existing-result comparison."""
import csv
import datetime
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import time


def save(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def eer(values):
    positive = sum(label for _, label in values)
    negative = len(values) - positive
    assert positive and negative and all(math.isfinite(value) for value, _ in values)
    values.sort(reverse=True)
    tp = fp = 0
    previous = (0., 1.)
    for _, group in itertools.groupby(values, key=lambda item: item[0]):
        labels = [label for _, label in group]
        tp += sum(labels)
        fp += len(labels)-sum(labels)
        current = (fp/negative, 1-tp/positive)
        if current[0] >= current[1]:
            before = previous[0]-previous[1]
            after = current[0]-current[1]
            alpha = -before/(after-before)
            return 100*(previous[0]+alpha*(current[0]-previous[0]))
        previous = current
    raise AssertionError('Missing EER crossing')


def validate_phase(root, phase, config):
    scores, indices, uids = [], set(), set()
    proofs = []
    for rank in [0, 1]:
        path = root/'evaluation'/f'scores_{phase}_rank{rank}.csv'
        done = json.loads((path.parent/f'done_{phase}_rank{rank}.json').read_text())
        assert sha(path) == done['sha256']
        assert done['optimizer_updates'] == 0 and done['parameters_unchanged']
        count = 0
        with path.open() as stream:
            for row in csv.DictReader(stream):
                index, uid, label = int(row['index']), row['uid'], int(row['label'])
                assert index % 2 == rank and index not in indices and uid not in uids
                assert int(row['used_samples']) == min(int(row['original_samples']), 64600)
                assert label in [0, 1]
                indices.add(index)
                uids.add(uid)
                scores.append((float(row['score']), label))
                count += 1
        assert count == done['rows']
        proofs.append({'rank': rank, 'rows': count, 'score_sha256': done['sha256']})
    assert len(scores) == config['phases'][phase]
    assert indices == set(range(len(scores)))
    positive = sum(label for _, label in scores)
    assert [positive, len(scores)-positive] == config['class_counts'][phase]
    actual = eer(scores)
    logged = json.loads((root/f'metrics_{phase}.json').read_text())['EER_percent']
    assert abs(actual-logged) < 1e-8, (phase, actual, logged)
    return {'phase': phase, 'rows': len(scores), 'bonafide': positive,
            'spoof': len(scores)-positive, 'independently_recomputed_EER_percent': actual,
            'matches_evaluation_metric': True, 'rank_score_proofs': proofs}


def compare(root):
    config = json.loads((root/'evaluation_config.json').read_text())
    assert int((root/'run.exit').read_text()) == 0
    complete = json.loads((root/'complete.json').read_text())
    assert complete['status'] == 'complete' and complete['epoch'] == 20
    assert sha(Path(config['checkpoint'])) == config['checkpoint_sha256']
    assert sha(Path(config['train_state'])) == config['train_state_sha256']
    proofs = [validate_phase(root, phase, config) for phase in config['phases']]
    current = {proof['phase']: proof['independently_recomputed_EER_percent'] for proof in proofs}
    reference = Path(config['reference_19LA_scores'])
    with reference.open() as stream:
        rows = list(csv.DictReader(stream))
    assert sorted(int(row['index']) for row in rows) == list(range(71237))
    current['19LA'] = eer([(float(row['score']), int(row['label'])) for row in rows])
    assert abs(current['19LA']-config['existing_19LA_EER_percent']) < 1e-8
    baseline_root = Path('/home/mentee/analysis/ebm_B_final_eval_20261107_215451')
    bc_root = Path('/home/mentee/analysis/ebm_B_C_checkpoints_20261109')
    baseline = json.loads((baseline_root/'metrics.json').read_text())
    bc = json.loads((bc_root/'metrics.json').read_text())
    historical = json.loads((root/'historical_legacy_final.json').read_text())
    main, fixed50, repeated, differences = [], [], [], []
    for phase in ['19LA', '21LA', '21DF']:
        main.append({'phase': phase, 'epoch': 20,
                     'B_EER_percent': bc[phase]['B_e20']['EER_percent'],
                     'C_EER_percent': bc[phase]['C_e20']['EER_percent'],
                     'coverage_EER_percent': current[phase],
                     'role': '19LA monitored before stopping' if phase == '19LA' else 'Post-stop target evaluation'})
        fixed50.append({'phase': phase, 'epoch': 50,
                        'legacy240_EER_percent': baseline[phase]['legacy240']['EER_percent'],
                        'B_EER_percent': bc[phase]['B_e50']['EER_percent'],
                        'C_EER_percent': bc[phase]['C_e50']['EER_percent'],
                        'comparability': 'Same input/score policy; epoch differs from coverage20'})
        repeated.append({'phase': phase, 'epoch': 50,
                         'legacy240_EER_percent': historical['final_EER_percent'][phase],
                         'comparability': 'Historical repeating-input pipeline; epoch and evaluation differ'})
        for model in ['B', 'C']:
            old = bc[phase][f'{model}_e20']['EER_percent']
            differences.append({'phase': phase, 'reference_model': model, 'epoch': 20,
                                'coverage_minus_reference_percentage_points': current[phase]-old,
                                'relative_change_percent': 100*(current[phase]/old-1)})
    result = {'status': 'complete', 'fixed_checkpoint_epoch': 20,
              'checkpoint_sha256': config['checkpoint_sha256'],
              'same_epoch_comparison': main, 'fixed50_reference': fixed50,
              'historical_repeating_reference': repeated, 'differences': differences,
              'score_validation': proofs, 'optimizer_updates': 0,
              'protocol': {'common': 'front64600 cap, no repeat, masks, batch8/rank, trained OC cosine, ASP noise1e-5',
                           'coverage_rng': 'evaluation_seed1729 plus stream525252 and rank',
                           'BC_rng': '20261107 plus rank and phase stream; different random execution',
                           'EER': 'full ROC tie-group linear crossing; percent; higher score means bonafide'},
              'interpretation_limits': [
                  'Same-epoch input and score policy match, but saved B/C evaluation RNG differs.',
                  'Baseline240 fixed50 and B/C fixed50 are references, not epoch-matched controls.',
                  'Historical repeating-input scores must not be mixed into the main comparison.',
                  'The user stopped training after observing19LA; epoch20 was explicitly chosen before21LA/DF results.',
                  'Several training input and pairing choices change together; no single-factor causal attribution.',
                  'Single training seed; no statistical significance claim.'],
              'sources': {str(baseline_root/'metrics.json'): sha(baseline_root/'metrics.json'),
                          str(bc_root/'metrics.json'): sha(bc_root/'metrics.json'),
                          str(reference): sha(reference)},
              'finished_KST': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()}
    save(root/'comparison.json', result)
    save(root/'analysis_complete.json', {'status': 'complete', 'finished_KST': result['finished_KST'],
                                       'comparison_sha256': sha(root/'comparison.json'), 'GPU_used': False})


if __name__ == '__main__':
    root = Path(sys.argv[1])
    if (root/'analysis_complete.json').exists():
        raise SystemExit('Analysis already complete; no overwrite')
    deadline = time.time()+8*3600
    try:
        while not (root/'run.exit').exists():
            if time.time() > deadline:
                raise TimeoutError('Bounded wait for evaluation completion expired')
            time.sleep(30)
        compare(root)
    except Exception as error:
        save(root/'analysis_failed.json', {'status': 'failed', 'error': str(error)})
        raise
