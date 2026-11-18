"""Frozen post-training evaluation: all checkpoints, native/wrap, no optimizer."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from paired_evaluation import (CAP, EPOCHS, PHASES, EXPECTED, META, sha, save, views,
                               common_scores, analyze_phase, metric)


def verify_sources(root, cfg):
    assert sha(root/'experiment.json') == (root/'experiment.sha256').read_text().strip(), 'Configuration changed'
    for name, expected in cfg['source_sha256'].items():
        assert sha(root/'code'/name) == expected, name
    for name, expected in cfg['payload_sha256'].items():
        assert sha(root/name) == expected, name
    for source in cfg['reference_checkpoints'].values():
        assert sha(source['path']) == source['sha256'], source['path']


def freeze_checkpoints(root):
    import torch
    cfg = json.loads((root/'experiment.json').read_text())
    verify_sources(root, cfg)
    manifest = dict(cfg['reference_checkpoints'])
    result = Path(cfg['current_result'])
    for epoch in EPOCHS:
        marker = json.loads((result/'short32_stages'/f'checkpoint_{epoch}.complete.json').read_text())
        assert marker['status'] == 'committed' and marker['epoch'] == epoch
        for filename, record in marker['files'].items():
            assert sha(result/'model'/filename) == record['sha256']
        filename = f'CLS_{epoch}.pt'
        manifest[f'short32_e{epoch}'] = dict(family='short32', epoch=epoch,
            path=str(result/'model'/filename), sha256=marker['files'][filename]['sha256'],
            TrainState_path=str(result/'model'/f'TrainState_{epoch}.pt'),
            TrainState_sha256=marker['files'][f'TrainState_{epoch}.pt']['sha256'])
    validation = {}
    for key, ck in manifest.items():
        assert sha(ck['path']) == ck['sha256']
        head = torch.load(ck['path'], map_location='cpu', weights_only=True)
        clean = {k.removeprefix('module.'): v for k, v in head.items()}
        assert all(torch.isfinite(v).all() for v in clean.values())
        details = dict(classifier_sha256=ck['sha256'], finite=True, epoch=ck['epoch'])
        if 'TrainState_path' in ck:
            assert sha(ck['TrainState_path']) == ck['TrainState_sha256']
            state = torch.load(ck['TrainState_path'], map_location='cpu', weights_only=True)
            assert state['epoch'] == ck['epoch']
            state_clean = {k.removeprefix('module.'): v for k, v in state['classifier'].items()}
            assert clean.keys() == state_clean.keys()
            assert all(torch.equal(v, state_clean[k]) for k, v in clean.items())
            steps = [int(v['step']) for v in state['optimizer']['state'].values()]
            assert min(steps) == max(steps) == ck['epoch']*108
            revision = state.get('frontend_revision') or state['args'].get('frontend_resolved_revision')
            if revision is not None: assert revision == cfg['frontend_revision']
            details.update(matches_TrainState=True, Adam_steps=min(steps))
            del state, state_clean
        else:
            details['TrainState_validation'] = 'not available in legacy archive'
        validation[key] = details
        del head, clean
    assert len(manifest) == 38
    assert not (root/'checkpoints.json').exists(), 'Refuse refreeze/overwrite'
    save(root/'checkpoints.json', manifest)
    save(root/'checkpoint_validation.json', validation)


def items_for_phase(args, phase):
    from tools.evaluate_B_C_checkpoints import make_items
    items = make_items(args, phase)
    if phase in ('21LA', '21DF'):
        base = args['path_21LA'] if phase == '21LA' else args['path_21DF']
        key = Path(base)/('keys/LA/CM/trial_metadata.txt' if phase == '21LA' else 'keys/DF/CM/trial_metadata.txt')
        mapping = {}
        for line in key.read_text().splitlines():
            t = line.split()
            if t[7] != 'eval': continue
            uid, speaker, codec = t[1], t[0], t[2]
            attack = t[4]
            label = int(t[5] == 'bonafide')
            assert uid not in mapping
            mapping[uid] = (speaker, codec, attack, label)
        for item in items:
            speaker, codec, attack, label = mapping[Path(item.path).stem]
            assert label == int(item.label)
            item.speaker = speaker; item.codec = codec
            item.type = 'bonafide' if label else attack
    else:
        for item in items: item.codec = 'not_applicable'
    return items


class PairedWaves:
    def __init__(self, items, phase): self.items, self.phase = items, phase
    def __len__(self): return len(self.items)
    def __getitem__(self, index):
        import soundfile as sf
        item = self.items[index]
        wave, sr = sf.read(item.path, dtype='float32')
        assert sr == 16000
        native, wrap = views(wave)
        row = dict(phase=self.phase, index=index, uid=Path(item.path).stem,
            speaker=str(item.speaker), label=int(item.label), original_samples=len(wave),
            attack=item.type, codec=item.codec)
        return native, wrap, row


def collate(batch): return batch


def load_models(root, rank, probe):
    import torch
    from transformers import Wav2Vec2Model
    import arguments, model
    cfg = json.loads((root/'experiment.json').read_text())
    args, _, _ = arguments.get_args()
    plm = Wav2Vec2Model.from_pretrained(args['PLM_name'], local_files_only=True).to(rank).eval().requires_grad_(False)
    assert getattr(plm.config, '_commit_hash', None) == cfg['frontend_revision']
    manifest = json.loads((root/'checkpoints.json').read_text())
    heads = {}
    for key, ck in manifest.items():
        if probe and ck['epoch'] != 50: continue
        assert sha(ck['path']) == ck['sha256']
        obj = model.B_Linear(args['num_layers'], args['hidden_size'], args['output_size'], args['agg_size'],
            loss=model.OCSoftmax(args['output_size'], r_real=args['r_real'], r_fake=args['r_fake'],
                alpha=args['alpha'], class_weight=[5/9, 5]), loss_bpl=model.BPL(args['alpha_cl']))
        state = torch.load(ck['path'], map_location='cpu', weights_only=True)
        obj.load_state_dict({k.removeprefix('module.'): v for k, v in state.items()}, strict=True)
        heads[key] = obj.to(rank).eval().requires_grad_(False)
    return args, plm, heads


def worker(rank, root_string, phase, smoke, probe):
    import torch
    import native_steps
    import soundfile as sf
    root = Path(root_string)
    torch.set_num_threads(2); torch.cuda.set_device(rank)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(1729+rank)
    args, plm, heads = load_models(root, rank, probe)
    initial_versions = [p._version for m in [plm, *heads.values()] for p in m.parameters()]
    tag = 'probe' if probe else 'full_'+phase
    if smoke: tag += '_smoke'
    dest = root/f'{tag}_rank{rank}.csv'; partial = dest.with_suffix('.partial.csv')
    assert not dest.exists() and not partial.exists(), 'Refuse score overwrite/retry'
    fields = META+['condition', 'used_samples', 'wave_sha256']+['score_'+k for k in heads]
    qc = {}; count = 0; source_count = 0; start = time.time()

    def infer(waves, seed, noise):
        x, mask = native_steps.features(plm, [torch.from_numpy(w) for w in waves], f'cuda:{rank}')
        values = common_scores(heads, x, mask, seed, noise)
        return values

    items = items_for_phase(args, phase)
    if probe:
        from tools.probe_repetition_shortcut import dose_views
        selection = pd.read_csv(root/'probe_selected.csv', dtype={'speaker': str})
        assert sha(root/'probe_selected.csv') == json.loads((root/'experiment.json').read_text())['probe_selection_sha256']
        selection = selection.iloc[rank::2]
        if smoke: selection = selection.groupby('label').head(1)
        def batches():
            for row in selection.to_dict('records'):
                item = items[int(row['index'])]
                assert Path(item.path).stem == row['uid'] and int(item.label) == row['label']
                assert str(item.speaker) == str(row['speaker'])
                wave, sr = sf.read(item.path, dtype='float32')
                assert sr == 16000 and len(wave) == row['original_samples']
                meta = dict(row, attack=item.type, codec=item.codec)
                yield [(w, meta, condition) for condition, w in dose_views(wave)]
        groups = batches(); expected_sources = len(selection)
    else:
        dataset = PairedWaves(items, phase)
        indices = list(range(rank, len(dataset), 2))
        if smoke:
            # Header-only stratification: real short/long and both labels on each GPU.
            selected = []
            for label in [0, 1]:
                counts = {True: 0, False: 0}
                for index in indices:
                    if int(items[index].label) != label: continue
                    short = sf.info(items[index].path).frames < CAP
                    if counts[short] < 2:
                        selected.append(index); counts[short] += 1
                    if min(counts.values()) == 2: break
                assert min(counts.values()) == 2, (phase, rank, label, counts)
            indices = sorted(selected)
        loader = torch.utils.data.DataLoader(dataset, batch_size=8, sampler=indices,
            num_workers=4, collate_fn=collate, generator=torch.Generator().manual_seed(100+rank))
        groups = loader; expected_sources = len(indices)
    with partial.open('x', newline='') as stream, torch.inference_mode():
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for batch_number, batch in enumerate(groups):
            seed = 1729+525252+rank+100000*PHASES.index(phase)+batch_number
            tasks = []
            if probe:
                for offset in range(0, len(batch), 8):
                    subset = batch[offset:offset+8]
                    values = infer([w for w, _, _ in subset], seed, 0.)
                    for i, (wave, meta, condition) in enumerate(subset):
                        tasks.append((wave, meta, condition, {k: v[i] for k, v in values.items()}))
                source_count += 1
            else:
                native = infer([n for n, _, _ in batch], seed, 1e-5)
                short = [i for i, (n, _, _) in enumerate(batch) if len(n) < CAP]
                wrapped = infer([batch[i][1] for i in short], seed, 1e-5) if short else {}
                position = {j: i for i, j in enumerate(short)}
                for j, (n, w, meta) in enumerate(batch):
                    sn = {k: v[j] for k, v in native.items()}
                    sw = {k: v[position[j]] for k, v in wrapped.items()} if j in position else sn
                    # Identical long waveforms are inferred once and explicitly reused.
                    tasks += [(n, meta, 'native', sn), (w, meta, 'wrap4', sw)]
                if smoke and not qc:
                    wave = batch[0][0]
                    one = infer([wave], seed, 0.)
                    again = infer([wave], seed, 0.)
                    together = infer([n for n, _, _ in batch], seed, 0.)
                    error = max(abs(float(one[k][0])-float(together[k][0])) for k in heads)
                    assert error < 3e-5, error
                    assert all(np.array_equal(one[k], again[k]) for k in heads)
                    qc = dict(batch_single_max_error=error, repeat_single_exact=True,
                              actual_DataLoader_workers=4, all_heads=len(heads))
                source_count += len(batch)
            for wave, meta, condition, scores in tasks:
                row = {k: meta[k] for k in META}
                row.update(condition=condition, used_samples=len(wave),
                    wave_sha256=hashlib.sha256(wave.tobytes()).hexdigest())
                row.update({'score_'+k: float(v) for k, v in scores.items()})
                writer.writerow(row); count += 1
            if batch_number % 25 == 0:
                stream.flush()
                save(root/f'{tag}_progress_rank{rank}.json', dict(phase=phase, sources=source_count,
                    expected_sources=expected_sources, rows=count, updated_unix=time.time()))
        stream.flush(); os.fsync(stream.fileno())
    assert source_count == expected_sources
    assert count == source_count*(31 if probe else 2)
    assert initial_versions == [p._version for m in [plm, *heads.values()] for p in m.parameters()]
    assert all(p.grad is None for m in [plm, *heads.values()] for p in m.parameters())
    partial.replace(dest)
    save(root/f'{tag}_done_rank{rank}.json', dict(status='complete', rows=count,
        sources=source_count, sha256=sha(dest), elapsed_seconds=time.time()-start,
        optimizer_updates=0, model_versions_unchanged=True, qc=qc))


def summarize_probe(root):
    manifest = json.loads((root/'checkpoints.json').read_text())
    frames = []
    for rank in [0, 1]:
        path = root/f'probe_rank{rank}.csv'
        receipt = json.loads((root/f'probe_done_rank{rank}.json').read_text())
        assert sha(path) == receipt['sha256']
        f = pd.read_csv(path, float_precision='round_trip', dtype={'speaker': str})
        assert len(f) == receipt['rows']; frames.append(f)
    f = pd.concat(frames, ignore_index=True)
    assert len(f) == 336*31 and not f.duplicated(['uid', 'condition']).any()
    selection = pd.read_csv(root/'probe_selected.csv')
    assert set(f.uid) == set(selection.uid)
    assert f.groupby('uid').condition.nunique().eq(31).all()
    assert not f.isna().any().any()
    rows = []
    for key, ck in manifest.items():
        if ck['epoch'] != 50: continue
        col = 'score_'+key
        assert np.isfinite(f[col]).all()
        identity = f.pivot(index='uid', columns='condition', values=col)
        for end in ['head', 'tail']:
            assert np.max(abs(identity[f'{end}_2s_repeat2']-identity[f'{end}_2s_native'])) < 3e-5
        for seconds in [1, 2]:
            for label in [0, 1]:
                g = f[f.label == label].pivot(index=['uid', 'speaker'], columns='condition', values=col)
                for intervention in ['repeat2', 'repeat3', 'repeat4', 'smooth_seams4', 'smooth_sham4', 'reflect4']:
                    delta = pd.concat([g[f'{end}_{seconds}s_{intervention}']-g[f'{end}_{seconds}s_native']
                        for end in ['head', 'tail']], axis=1).mean(axis=1)
                    speaker = delta.groupby(level='speaker').mean()
                    rng = np.random.default_rng(20261111)
                    boot = [float(speaker.iloc[rng.integers(0, len(speaker), len(speaker))].mean()) for _ in range(1000)]
                    rows.append(dict(family=ck['family'], epoch=50, seconds=seconds, label=label,
                        intervention=intervention, n=len(g), speakers=len(speaker), mean_delta=float(delta.mean()),
                        speaker_equal_mean=float(speaker.mean()), ci95_low=float(np.quantile(boot, .025)),
                        ci95_high=float(np.quantile(boot, .975)), noise_std=0.,
                        interval='speaker means resampled; exploratory, no multiple-test correction'))
                natural_delta = g[f'natural4_smooth_{seconds}s_positions']-g['natural4']
                rows.append(dict(family=ck['family'], epoch=50, seconds=seconds, label=label,
                    intervention='natural4_smoothing_control', n=len(g),
                    speakers=g.index.get_level_values('speaker').nunique(),
                    mean_delta=float(natural_delta.mean()), noise_std=0.))
    save(root/'probe_analysis.json', dict(status='complete', rows=len(f), effects=rows,
        limits='Long-source artificial crops differ from naturally short recordings; no training causality'))
    pd.DataFrame(rows).to_csv(root/'probe_effects.csv', index=False)


def make_report(root):
    phases = [json.loads((root/f'analysis_{phase}.json').read_text()) for phase in PHASES]
    minima = [m for p in phases for m in p['minima']]
    lines = ['# 반복·비반복 공통 조건 평가', '',
        '학습·optimizer 갱신 없이 저장 모델만 평가했다. 앞 64,600샘플, 반복은 wrap, 비반복은 유효 길이 mask를 썼다.',
        '모든 모델은 같은 동결 frontend/입력/난수, 학습된 각 OC 중심 cosine, ASP 잡음 1e-5를 사용했다. dev 추론은 없다.',
        '최소 EER는 집합별 사후 최솟값이다. 단일 배포 체크포인트의 비편향 성능으로 해석하지 않는다.', '',
        '|집합|모델|조건|전체 보존 최소 EER(%) / epoch|공통15~50 최소 EER(%) / epoch|',
        '|---|---|---|---|---|']
    for phase in PHASES:
        for family in ['legacy240', 'B', 'C', 'short32']:
            for condition in ['native', 'wrap4']:
                group = [m for m in minima if m['phase'] == phase and m['family'] == family and m['condition'] == condition]
                all_m = next(m for m in group if m['candidate_scope'] == 'all_available')
                common = next(m for m in group if m['candidate_scope'] == 'common_15_to_50')
                lines.append(f"|{phase}|{family}|{condition}|{all_m['eer_percent']:.6f} / {all_m['epoch']} ({all_m['candidate_count']}개)|{common['eer_percent']:.6f} / {common['epoch']}|")
    for n, title, evidence, interpretation in [
        (1, '전체 체크포인트 최소 EER', 'minima_*.csv / full_metrics_*.csv', '각 조건의 최적 epoch를 비교한다. 같은 epoch 고정 행에서 반복 효과와 선택 효과를 분리한다.'),
        (2, '길이별 분리력과 오류 전이', 'length_eer_*.csv / transitions_*.csv', '원음 길이로 나누고 정상/스푸프 분모를 따로 본다. 한 클래스만 있는 구간의 EER는 산출하지 않는다.'),
        (3, '임계값 이동 민감도', 'transitions_*.csv', 'native 고정 임계값과 조건별 임계값을 병기한다. target 유래의 설명용 임계값이며 운용 성능 주장이 아니다.'),
        (4, '공격·코덱·길이 구성', 'attack_codec_errors_21LA.csv / duration_weighting_21LA.csv', '같은 공격·코덱 비율을 보존한 길이 재가중 결과는 구성 민감도이며 학습 인과관계 증명이 아니다.'),
        (5, '같은 crop의 반복량·접합 통제', 'probe_effects.csv / probe_analysis.json', '50epoch 고정·잡음0에서 336개×31조건을 비교한다. head/tail을 원음별 평균하고 화자 단위 불확실성을 계산한다. 자연 단음과 인공 crop의 차이는 남는다.')]:
        lines += ['', f'## 실험 {n}. {title}', '', f'결과: `{evidence}`', '', '해석: '+interpretation]
    # Concrete paired effect at each fixed epoch; minimum rows never replace this table.
    lines += ['', '## 같은 50epoch에서 반복 효과', '', '|집합|모델|비반복 EER(%)|반복 EER(%)|반복-비반복(%p)|', '|---|---|---:|---:|---:|']
    for p in phases:
        for family in ['legacy240', 'B', 'C', 'short32']:
            group = [r for r in p['full_metrics'] if r['family'] == family and r['epoch'] == 50]
            a = next(r['eer_percent'] for r in group if r['condition'] == 'native')
            b = next(r['eer_percent'] for r in group if r['condition'] == 'wrap4')
            lines.append(f"|{p['phase']}|{family}|{a:.6f}|{b:.6f}|{b-a:+.6f}|")
    lines += ['', '## 길이·임계값 진단 수치 (21LA, 고정50)', '',
              '|모델|원음|조건|해당 길이 내 EER(%)|정상 수|스푸프 수|', '|---|---|---|---:|---:|---:|']
    la = next(p for p in phases if p['phase'] == '21LA')
    for r in la['length_eer']:
        if r['epoch'] == 50 and r['length_bin'] in ('le2', 'gt2'):
            value = f"{r['eer_percent']:.6f}" if r['eer_percent'] is not None else '한 클래스만 존재'
            lines.append(f"|{r['family']}|{r['length_bin']}|{r['condition']}|{value}|{r['bonafide']}|{r['spoof']}|")
    lines += ['', '|모델|임계값 정책|클래스(2초 이하)|비반복 오류/분모|반복 오류/분모|', '|---|---|---|---:|---:|']
    for r in la['transitions']:
        if r['epoch'] == 50 and r['length_bin'] == 'le2':
            label = '정상' if r['label'] else '스푸프'
            lines.append(f"|{r['family']}|{r['policy']}|{label}|{r['native_errors']}/{r['n']}|{r['wrap_errors']}/{r['n']}|")
    lines += ['', '해석: 길이 내 EER와 클래스별 오류 전이를 함께 본다. 조건별 임계값에서 오류가 회복되는 부분은 점수 기준 이동에 민감한 부분이다. 이 수치는 target 설명용이다.', '',
              '## 길이 구성 통제 (21LA, 고정50)', '', '|모델|조건|가중 정책|EER(%)|제외 수|', '|---|---|---|---:|---:|']
    for r in la['duration_weighting']:
        if r['epoch'] == 50:
            lines.append(f"|{r['family']}|{r['condition']}|{r['policy']}|{r['eer_percent']:.6f}|{r['excluded']}|")
    lines += ['', '해석: 공통 지지집합의 원래 EER와 공격×코덱 보존 길이 재가중 EER를 비교한다. EER 차이는 구성 민감도를 나타내며 모델이 길이를 학습 원인으로 사용했다는 증명은 아니다.', '',
              '## 반복량·접합부 통제 수치 (고정50, 정상, ASP 잡음0)', '',
              '|모델|원래 crop(초)|반복4초 Δscore|접합 smoothing의 추가 회복|자연4초 smoothing Δscore|', '|---|---:|---:|---:|---:|']
    effects = json.loads((root/'probe_analysis.json').read_text())['effects']
    for family in ['legacy240', 'B', 'C', 'short32']:
        for seconds in [1, 2]:
            e = {r['intervention']: r['mean_delta'] for r in effects
                 if r['family'] == family and r['seconds'] == seconds and r['label'] == 1}
            lines.append(f"|{family}|{seconds}|{e['repeat4']:+.6f}|{e['smooth_seams4']-e['repeat4']:+.6f}|{e['natural4_smoothing_control']:+.6f}|")
    lines += ['', '해석: Δscore<0은 반복 뒤 스푸프 방향 이동이다. smoothing 회복은 자연음 smoothing 대조 및 sham 대조와 함께 판단한다. 반복만의 주기성 원인, 자연 단음에서의 동일 효과, 학습 발생 원인은 이 개입만으로 확정하지 않는다.']
    lines += ['', '기존240은 보존8개(15~50), 나머지는10개(5~50)다. 과거 codec 사용 여부 등 학습 차이가 있어 순수 padding 학습 ablation은 아니다.',
              '현재 short32는 짧은 정상에 L43 비중 감소와 L32 추가가 함께 적용된다. 효과가 나타나더라도 L32만의 독립 효과를 분리하려면 별도 학습 ablation이 필요하다.']
    (root/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    verify_sources(root, json.loads((root/'experiment.json').read_text()))
    for ck in json.loads((root/'checkpoints.json').read_text()).values():
        assert sha(ck['path']) == ck['sha256']
        if 'TrainState_path' in ck: assert sha(ck['TrainState_path']) == ck['TrainState_sha256']
    save(root/'complete.json', dict(status='complete', minima=minima, optimizer_updates=0,
        report_sha256=sha(root/'report.md'), checkpoint_manifest_sha256=sha(root/'checkpoints.json'),
        completed_unix=time.time(), spending_ceiling_usd=0))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('root', type=Path)
    parser.add_argument('--mode', choices=['freeze', 'smoke', 'full', 'probe', 'report'], required=True)
    cli = parser.parse_args(); root = cli.root
    assert Path('/.dockerenv').exists() and os.environ.get('CUDA_VISIBLE_DEVICES') == '0,1'
    cfg = json.loads((root/'experiment.json').read_text()); verify_sources(root, cfg)
    if cli.mode == 'freeze': freeze_checkpoints(root); return
    if cli.mode == 'report':
        for phase in PHASES: analyze_phase(root, phase)
        summarize_probe(root); make_report(root); return
    import torch.multiprocessing as mp
    phases = PHASES if cli.mode in ('smoke', 'full') else ['21LA']
    for phase in phases:
        mp.spawn(worker, args=(str(root), phase, cli.mode == 'smoke', cli.mode == 'probe'), nprocs=2, join=True)
        if cli.mode == 'smoke':
            columns = ['score_'+k for k in json.loads((root/'checkpoints.json').read_text())]
            for rank in [0, 1]:
                path = root/f'full_{phase}_smoke_rank{rank}.csv'
                done = json.loads((root/f'full_{phase}_smoke_done_rank{rank}.json').read_text())
                assert sha(path) == done['sha256']
                f = pd.read_csv(path, float_precision='round_trip')
                assert len(f) == done['rows'] == 16 and f.uid.nunique() == 8
                assert len(columns) == 38 and np.isfinite(f[columns]).all().all()
                a = f[f.condition == 'native'].set_index('uid').sort_index()
                b = f[f.condition == 'wrap4'].set_index('uid').sort_index()
                assert a.index.equals(b.index) and set(a.label) == {0, 1}
                noop = a.original_samples >= CAP
                assert int(noop.sum()) == 4 and np.array_equal(a.loc[noop, columns], b.loc[noop, columns])
            save(root/f'smoke_{phase}_complete.json', dict(status='complete', rows=32, heads=38,
                both_labels_and_lengths=True, actual_workers=4, optimizer_updates=0))
    # Every checkpoint hash must still match after inference.
    for ck in json.loads((root/'checkpoints.json').read_text()).values(): assert sha(ck['path']) == ck['sha256']


if __name__ == '__main__': main()
