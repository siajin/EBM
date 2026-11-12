"""Visualize an EBM checkpoint using a reproducible balanced 21DF sample.

Run inside the mentee container. No training, network access, or source edits.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

sys.dont_write_bytecode = True
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['MPLCONFIGDIR'] = '/home/mentee/.cache/matplotlib'

ROOT = Path('/home/mentee/results/CM/LA/EBM')
CHECKPOINT = ROOT / 'model/BestCLS_80_21DF.pt'
SOURCE = ROOT / 'backup/scripts'
DATA = Path('/data/ASVspoof2021_DF')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def sample_trials(n, seed):
    groups = {0: [], 1: []}
    protocol = DATA / 'keys/DF/CM/trial_metadata.txt'
    for line in protocol.read_text().splitlines():
        fields = line.split()
        if len(fields) < 8 or fields[7] != 'eval':
            continue
        label = int(fields[5] == 'bonafide')
        groups[label].append({'id': fields[1], 'label': label, 'attack': fields[4]})
    rng = random.Random(seed)
    selected = []
    for label in (1, 0):
        if len(groups[label]) < n:
            raise ValueError('Requested sample exceeds population')
        selected.extend(rng.sample(groups[label], n))
    rng.shuffle(selected)
    for row in selected:
        path = DATA / 'ASVspoof2021_DF_eval/flac' / (row['id'] + '.wav')
        if not path.is_file():
            raise FileNotFoundError(path)
    return selected, {str(k): len(v) for k, v in groups.items()}, digest(protocol)


def worker(rank, cfg, trials):
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import Wav2Vec2Model
    sys.path.insert(0, str(SOURCE))
    from model import B_Linear, BPL, OCSoftmax

    torch.set_num_threads(2)
    torch.manual_seed(cfg['seed'] + rank)
    np.random.seed(cfg['seed'] + rank)
    device = torch.device('cpu' if cfg['cpu'] else f'cuda:{rank}')
    if not cfg['cpu']:
        torch.cuda.set_device(rank)
    state = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    state = {k.removeprefix('module.'): v for k, v in state.items()}
    net = B_Linear(num_layer=25, hidden_size=1024, output_size=160, agg_size=128,
                   loss=OCSoftmax(160, r_real=.9, r_fake=.2, alpha=20),
                   loss_bpl=BPL(alpha=30))
    net.load_state_dict(state, strict=True)
    net.to(device).eval()
    plm = Wav2Vec2Model.from_pretrained(cfg['plm_path'], local_files_only=True).to(device).eval()
    center = torch.nn.functional.normalize(net.loss.weight.detach(), dim=1)
    assert center.shape == (1, 160)
    indices = list(range(rank, len(trials), cfg['workers']))
    collected, used = [], []
    started = time.time()
    with torch.inference_mode():
        for offset in range(0, len(indices), cfg['batch_size']):
            batch_ids = indices[offset:offset + cfg['batch_size']]
            audio_batch = []
            for idx in batch_ids:
                path = DATA / 'ASVspoof2021_DF_eval/flac' / (trials[idx]['id'] + '.wav')
                audio, sr = sf.read(path, dtype='float32')
                if sr != 16000 or audio.ndim != 1 or len(audio) == 0:
                    raise ValueError(f'Unexpected audio format: {path}')
                crop = 404 * 160 - 40
                audio = np.pad(audio, (0, max(0, crop-len(audio))), mode='wrap')[:crop]
                audio_batch.append(audio)
            x = torch.from_numpy(np.stack(audio_batch)).to(device)
            hidden = torch.stack(plm(x, output_hidden_states=True).hidden_states, dim=1)
            z = net.layer_norm(net.linear(net.AHP_ASP(net.bridge_module(hidden))))
            z = torch.nn.functional.normalize(z, dim=1)
            if not torch.isfinite(z).all():
                raise ValueError('Non-finite embeddings')
            collected.append(z.cpu().numpy())
            used.extend(batch_ids)
            if offset == 0 or (offset // cfg['batch_size']) % 10 == 0:
                print(f'worker {rank}: {len(used)}/{len(indices)} samples, {time.time()-started:.0f}s', flush=True)
    np.savez_compressed(Path(cfg['out']) / f'shard_{rank}.npz',
                        indices=np.array(used), embeddings=np.concatenate(collected),
                        center=center.cpu().numpy()[0])


def render(out, trials, provenance):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    shards = [np.load(out / f'shard_{r}.npz') for r in range(provenance['workers'])]
    idx = np.concatenate([s['indices'] for s in shards])
    assert sorted(idx.tolist()) == list(range(len(trials)))
    z = np.concatenate([s['embeddings'] for s in shards])[np.argsort(idx)]
    c = shards[0]['center']
    for shard in shards:
        np.testing.assert_allclose(c, shard['center'], atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(z, axis=1), 1, atol=1e-5)
    labels = np.array([r['label'] for r in trials])
    means = np.stack([z[labels == 1].mean(0), z[labels == 0].mean(0)])
    centers = np.vstack([c, means])
    pca = PCA(n_components=3, svd_solver='full').fit(z)
    coords = pca.transform(z)
    centers_pca = pca.transform(centers)
    cosine = np.clip(z @ c, -1, 1)
    residual = z - cosine[:, None] * c
    _, _, vt = np.linalg.svd(residual, full_matrices=False)
    ortho = residual @ vt[0]
    center_cos = centers @ c
    center_y = (centers - center_cos[:, None] * c) @ vt[0]
    colors = {1: '#2563A6', 0: '#D87527'}
    names = {1: 'Bonafide', 0: 'Spoof'}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig = plt.figure(figsize=(16, 12), layout='constrained')
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222, projection='3d')
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    for label in (1, 0):
        mask = labels == label
        kw = dict(c=colors[label], s=10, alpha=.38,
                  marker='o' if label else '^', label=f'{names[label]} (n={mask.sum():,})')
        ax1.scatter(coords[mask, 0], coords[mask, 1], **kw)
        ax2.scatter(coords[mask, 0], coords[mask, 1], coords[mask, 2], **kw)
        ax3.scatter(cosine[mask], ortho[mask], **kw)
        ax4.hist(cosine[mask], bins=np.linspace(-1, 1, 61), density=True,
                 histtype='step', linewidth=2, color=colors[label],
                 linestyle='-' if label else '--', label=names[label])
    center_names = ['Learned OC center', 'Bonafide sample mean', 'Spoof sample mean']
    for j, (name, color, marker) in enumerate(zip(center_names,
                ['#171717', colors[1], colors[0]], ['*', 'D', 'D'])):
        kw = dict(c=color, marker=marker, s=220 if j == 0 else 95,
                  edgecolors='white', linewidths=1.1, label=name, zorder=8)
        ax1.scatter(*centers_pca[j, :2], **kw)
        ax2.scatter(*centers_pca[j], **kw)
        ax3.scatter(center_cos[j], center_y[j], **kw)
    ev = pca.explained_variance_ratio_ * 100
    ax1.set(title=f'PCA projection | retained variance {ev[:2].sum():.1f}%',
            xlabel=f'PC1 ({ev[0]:.1f}%)', ylabel=f'PC2 ({ev[1]:.1f}%)', aspect='equal')
    ax1.legend(fontsize=9, loc='best')
    ax2.set(title=f'3D PCA | retained variance {ev.sum():.1f}%',
            xlabel='PC1', ylabel='PC2', zlabel='PC3')
    ax3.set(title='Center-aligned view | true cosine on horizontal axis',
            xlabel='Cosine similarity to learned OC center',
            ylabel='Leading orthogonal direction (uncentered SVD)', xlim=(-1.05, 1.05))
    ax4.set(title='Original 160D space | similarity distribution',
            xlabel='Cosine similarity to learned OC center', ylabel='Density', xlim=(-1, 1))
    for boundary, name in ((.2, 'spoof margin 0.2'), (.9, 'bonafide margin 0.9')):
        ax4.axvline(boundary, color='#777777', linestyle=':', linewidth=1, label=name)
    ax4.legend(fontsize=9)
    for ax in (ax1, ax3, ax4):
        ax.grid(alpha=.14)
        ax.set_axisbelow(True)
    fig.suptitle('21DF embedding geometry | EBM checkpoint epoch 80\n'
                 'L2-normalized 160D embeddings; balanced random evaluation sample', fontsize=17)
    fig.supxlabel('PCA distances can be distorted. Class means are arithmetic means of normalized samples. '
                  'Loss margins are not calibrated decision thresholds. Sampling seed: ' + str(provenance['seed']),
                  fontsize=10)
    fig.savefig(out / 'centroid_geometry.png', dpi=180)
    fig.savefig(out / 'centroid_geometry.pdf')
    plt.close(fig)
    summaries = {}
    for label in (1, 0):
        values = cosine[labels == label]
        summaries[names[label]] = {'n': int(len(values)), 'cosine_mean': float(values.mean()),
            'cosine_std': float(values.std()),
            'cosine_q05_q50_q95': np.quantile(values, [.05, .5, .95]).tolist(),
            'angle_median_degrees': float(np.median(np.degrees(np.arccos(values))))}
    summary = {'provenance': provenance, 'classes': summaries,
               'pca_variance_ratio': pca.explained_variance_ratio_.tolist(),
               'limitations': ['Balanced sample does not reflect class prevalence.',
                   'No full-evaluation EER is inferred from this sample.',
                   'PCA can distort distances; cosine is measured in original 160D space.',
                   'Backbone was frozen in the saved training configuration.',
                   'Backbone uses a currently cached revision; original run did not pin its revision.',
                   'Original pooling injects small random noise even in eval mode; extraction seed is fixed.']}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    np.savez_compressed(out / 'embeddings.npz', embeddings=z, labels=labels, learned_center=c,
                        class_means=means, cosine=cosine, pca_coordinates=coords)
    print(json.dumps(summary, indent=2), flush=True)
    print('Figure saved:', out / 'centroid_geometry.png', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--per-class', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=4, help='Per GPU inference batch, independent of training')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--cpu', action='store_true', help='CPU-only inference inside container')
    parser.add_argument('--out', default='/results/analysis/EBM80_21DF_centroid')
    args = parser.parse_args()
    if not Path('/.dockerenv').exists():
        raise SystemExit('Run only inside the mentee container.')
    if args.per_class < 3 or args.batch_size < 1:
        raise SystemExit('per-class >= 3 and batch-size >= 1 are required')
    import torch
    from huggingface_hub import snapshot_download
    workers = 1 if args.cpu else 2
    if not args.cpu and (os.environ.get('CUDA_VISIBLE_DEVICES') != '0,1' or torch.cuda.device_count() != 2):
        raise SystemExit('Use CUDA_VISIBLE_DEVICES=0,1; exactly two GPUs are required.')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    trials, population, protocol_hash = sample_trials(args.per_class, args.seed)
    plm_path = snapshot_download('facebook/wav2vec2-xls-r-300m', local_files_only=True)
    cfg = dict(vars(args), workers=workers, plm_path=plm_path)
    provenance = dict(cfg, checkpoint=str(CHECKPOINT), checkpoint_sha256=digest(CHECKPOINT),
                      protocol_sha256=protocol_hash, population=population,
                      torch_version=torch.__version__, sampling='uniform random within label, without replacement',
                      crop_samples=64600, sample_rate=16000,
                      embedding='after layer_norm, L2 normalized',
                      source_sha256={f: digest(SOURCE / f) for f in ('model/b_linear.py', 'model/loss.py', 'train.py')})
    (out / 'sample_manifest.json').write_text(json.dumps(trials, indent=2))
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    if args.cpu:
        worker(0, cfg, trials)
    else:
        torch.multiprocessing.spawn(worker, args=(cfg, trials), nprocs=2, join=True)
    render(out, trials, provenance)


if __name__ == '__main__':
    main()
