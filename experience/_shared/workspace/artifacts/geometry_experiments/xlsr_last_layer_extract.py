"""Stream XLS-R final Transformer features for official 19LA train to laptop.

No EBM logger, optimizer, training augmentation, or original source mutation.
Feature bytes travel through kernel FIFOs and SSH; no feature file on server.
Only scripts, small audit JSON and path-free row metadata persist on server.
"""
import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import runpy
import select
import struct
import sys
import time
from types import SimpleNamespace

REFERENCE = Path('/home/mentee/results/CM/LA/EBM_e50_bs160_lr1e5')
TV = Path('/home/mentee/analysis/embedding_geometry/20261105_fast_p0_01')
COUNT, FRAMES, DIM, SAMPLES = 25380, 201, 1024, 64600


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(path)


def settings():
    archive = REFERENCE / 'backup/scripts'
    args = runpy.run_path(str(archive / 'arguments.py'))['get_args']()[0]
    assert args['PLM_name'] == 'facebook/wav2vec2-xls-r-300m'
    assert args['num_test_frames'] * 160 - 40 == SAMPLES
    return archive, args


def prepare(root):
    import pandas as pd
    archive, args = settings()
    frame = pd.read_parquet(TV / 'manifest.parquet')
    frame = frame[frame.split == 'T'].sort_values('utterance_id').reset_index(drop=True)
    protocol = Path(args['path_19LA']) / 'LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'
    official = {r[1]: r for line in protocol.read_text().splitlines() if (r := line.split())}
    assert len(frame) == len(official) == COUNT and frame.y.value_counts().to_dict() == {0: 22800, 1: 2580}
    assert not frame.utterance_id.duplicated().any() and set(frame.utterance_id) == set(official)
    for row in frame.itertuples():
        assert row.y == int(official[row.utterance_id][-1] == 'bonafide')
        assert Path(row.path).is_file()
    frame['global_row'] = range(COUNT)
    frame['rank'] = frame.global_row % 2
    frame['feature_row'] = frame.global_row // 2
    assert not (root / 'manifest.parquet').exists()
    frame.to_parquet(root / 'manifest.parquet', index=False)
    # This path-free index is the user-facing mapping to the two arrays.
    frame.drop(columns='path').to_parquet(root / 'index.parquet', index=False)
    needed = COUNT * FRAMES * DIM * 4
    audit = dict(status='prepared',dataset='19LA_train',sample_count=COUNT,
        shape=[COUNT,FRAMES,DIM],dtype='float32',array_bytes=needed,
        representation='XLS-R final Transformer layer, hidden_states[-1] / last_hidden_state; before EBM',
        frontend=args['PLM_name'],frontend_frozen=True,frontend_finetuning=False,
        preprocessing='Archived EnrollmentSet: first 64600 samples; wrap-pad short audio; no train augmentation',
        protocol_sha256=digest(protocol),manifest_sha256=digest(root/'manifest.parquet'),
        index_sha256=digest(root/'index.parquet'),script_sha256=digest(__file__),
        archived_source_sha256={str(p.relative_to(archive)):digest(p) for p in archive.rglob('*.py')},
        storage='laptop_only_FIFO_SSH_stream',server_feature_files=False,
        no_original_audio_code_or_checkpoint_export=True)
    save(root / 'input_audit.json', audit)
    print(json.dumps({k:v for k,v in audit.items() if k!='archived_source_sha256'}), flush=True)


class FeatureStream:
    """Nonblocking FIFO writer with a bounded reader-wait/write timeout."""
    def __init__(self, path, timeout=180):
        import stat
        assert stat.S_ISFIFO(path.stat().st_mode), 'Stream path must be a kernel FIFO'
        deadline = time.monotonic() + timeout
        while True:
            try:
                self.fd = os.open(str(path), os.O_WRONLY | os.O_NONBLOCK)
                break
            except OSError as error:
                if error.errno != errno.ENXIO or time.monotonic() >= deadline:
                    raise
                time.sleep(.2)

    def write(self, data):
        data = memoryview(data)
        deadline = time.monotonic() + 180
        while data:
            try:
                n = os.write(self.fd, data)
                assert n > 0
                data = data[n:]
                deadline = time.monotonic() + 180
            except BlockingIOError:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('Laptop receiver stalled; no disk spool is permitted')
                select.select([], [self.fd], [], min(remaining, 1))

    def message(self, header, payload=b''):
        encoded = json.dumps(header, separators=(',', ':')).encode('utf-8')
        assert len(encoded) < 65536
        self.write(struct.pack('<I', len(encoded)))
        self.write(encoded)
        if payload:
            self.write(payload)

    def close(self):
        os.close(self.fd)


def worker(rank, root_s, stage):
    import numpy as np
    import pandas as pd
    import soundfile as sf
    import torch
    from transformers import AutoConfig, Wav2Vec2Model
    root = Path(root_s)
    phase = root / stage
    torch.set_num_threads(2)
    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    archive, args = settings()
    sys.path.insert(0, str(archive))
    from data_loaders import EnrollmentSet
    config = AutoConfig.from_pretrained(args['PLM_name'], local_files_only=True)
    assert config.hidden_size == DIM and config.num_hidden_layers == 24
    model = Wav2Vec2Model.from_pretrained(args['PLM_name'], config=config, local_files_only=True).eval()
    state_hash = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        state_hash.update(name.encode())
        state_hash.update(str((str(tensor.dtype),tuple(tensor.shape))).encode())
        state_hash.update(tensor.detach().contiguous().numpy().tobytes())
    weight_hash = state_hash.hexdigest()
    model = model.to(device)
    model.requires_grad_(False)
    full = pd.read_parquet(root / 'manifest.parquet')
    if stage == 'qc':
        full = pd.concat([g.sample(n=32 if attack=='bonafide' else 16, random_state=20261105)
                          for attack,g in full.groupby('attack_id')]).sort_values('utterance_id').reset_index(drop=True)
        assert len(full) == 128
    frame = full.iloc[rank::2].reset_index(drop=True)

    def read(row):
        info = sf.info(row.path)
        assert info.samplerate == 16000 and info.channels == 1
        item = SimpleNamespace(path=row.path,type=row.attack_id,label=int(row.y))
        audio, _, label = EnrollmentSet(args,[item])[0]
        assert len(audio)==SAMPLES and int(label)==row.y
        return torch.as_tensor(audio,dtype=torch.float32).unsqueeze(0).to(device)

    with torch.inference_mode():
        gi = torch.zeros((1,SAMPLES),device=device)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(3): model(gi,output_hidden_states=False)
        torch.cuda.current_stream(device).wait_stream(stream)
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            gh = model(gi,output_hidden_states=False).last_hidden_state
        assert tuple(gh.shape)==(1,FRAMES,DIM)

        def infer(row):
            gi.copy_(read(row))
            graph.replay()
            result = gh[0].cpu().numpy().copy()
            assert result.dtype==np.float32 and np.isfinite(result).all()
            return result

        shared = full.head(8)
        values = [infer(row) for _,row in shared.iterrows()]
        repeated = [infer(row) for _,row in shared.iterrows()]
        repeat_error = max(float(np.max(np.abs(a-b))) for a,b in zip(values,repeated))
        assert repeat_error <= 1e-5
        shared_hash = hashlib.sha256(np.stack(values).astype('<f4',copy=False).tobytes()).hexdigest()
        reference_error = 0.
        start = time.monotonic()
        if stage == 'extract':
            output = FeatureStream(root / f'rank{rank}.fifo')
            output.message(dict(kind='begin',schema=1,rank=rank,count=len(frame),shape=[len(frame),FRAMES,DIM],dtype='<f4'))
            raw_hash = hashlib.sha256()
        for i,(_,row) in enumerate(frame.iterrows()):
            value = infer(row)
            if stage == 'qc':
                ref = model(read(row),output_hidden_states=True)
                assert len(ref.hidden_states)==25
                baseline = ref.hidden_states[-1][0].cpu().numpy()
                error = float(np.max(np.abs(value-baseline)))
                assert error <= 1e-5, ('eager last-layer reference differs',error)
                reference_error = max(reference_error,error)
            else:
                assert int(row['rank'])==rank and int(row.feature_row)==i
                payload = value.astype('<f4',copy=False).tobytes()
                raw_hash.update(payload)
                output.message(dict(kind='data',rank=rank,row=i,global_row=int(row.global_row),
                    utterance_id=row.utterance_id,bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest()),payload)
            if (i+1)%250==0 or i+1==len(frame):
                elapsed = time.monotonic()-start
                progress=dict(completed=i+1,total=len(frame),elapsed_seconds=elapsed,utterances_per_second=(i+1)/elapsed)
                save(phase / f'rank{rank}_progress.json',progress)
                print(json.dumps(dict(rank=rank,stage=stage,**progress)),flush=True)
        seconds=time.monotonic()-start
        report=dict(status='pass',rank=rank,count=len(frame),shape=[len(frame),FRAMES,DIM],
            dtype='float32',repeat_max_error=repeat_error,eager_hidden_states_last_max_error=reference_error,
            frontend_state_sha256=weight_hash,frontend_commit=getattr(config,'_commit_hash',None),
            shared_sha256=shared_hash,elapsed_seconds=seconds,gpu=torch.cuda.get_device_name(rank),
            server_feature_files=False)
        if stage=='extract':
            report.update(raw_bytes=len(frame)*FRAMES*DIM*4,raw_sha256=raw_hash.hexdigest(),stream_protocol='framed_FIFO_SSH_v1')
            output.message(dict(kind='end',rank=rank,count=len(frame),raw_sha256=raw_hash.hexdigest()))
            output.close()
        save(phase / f'rank{rank}_complete.json',report)


def finalize(root, stage):
    import numpy as np
    import pandas as pd
    phase=root/stage
    reports=[json.loads((phase/f'rank{r}_complete.json').read_text()) for r in [0,1]]
    assert all(r['status']=='pass' for r in reports)
    assert reports[0]['frontend_state_sha256']==reports[1]['frontend_state_sha256']
    assert reports[0]['shared_sha256']==reports[1]['shared_sha256'], 'GPU shared outputs differ bitwise'
    cross=0.
    if stage=='extract':
        qc=json.loads((root/'qc/summary.json').read_text())
        assert qc['frontend_state_sha256']==reports[0]['frontend_state_sha256']
        index=pd.read_parquet(root/'index.parquet')
        assert len(index)==COUNT and not index.utterance_id.duplicated().any()
        for rank,report in enumerate(reports):
            rows=index[index['rank']==rank]
            assert tuple(report['shape'])==(len(rows),FRAMES,DIM)
            assert rows.feature_row.tolist()==list(range(len(rows)))
        assert sum(r['count'] for r in reports)==COUNT
    audit=json.loads((root/'input_audit.json').read_text())
    archive,_=settings()
    assert audit['archived_source_sha256']=={str(p.relative_to(archive)):digest(p) for p in archive.rglob('*.py')}
    summary=dict(status='pass',stage=stage,sample_count=sum(r['count'] for r in reports),
        shape=[COUNT if stage=='extract' else 128,FRAMES,DIM],dtype='float32',
        cross_gpu_max_error=cross,reference_max_error=max(r['eager_hidden_states_last_max_error'] for r in reports),
        frontend_state_sha256=reports[0]['frontend_state_sha256'],rank_reports=reports,
        manifest_sha256=audit['manifest_sha256'],index_sha256=audit['index_sha256'],
        original_sources_unchanged=True,server_feature_files=False,
        storage='laptop_only_FIFO_SSH_stream',reference_QC_sha256=digest(root/'qc/summary.json') if stage=='extract' else None,
        finished_epoch=time.time())
    save(phase/'summary.json',summary)
    (phase/'complete').write_text('0\n')
    print(json.dumps(summary),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','qc','extract'])
    parser.add_argument('--root',required=True);options=parser.parse_args();root=Path(options.root)
    assert Path('/.dockerenv').is_file() and os.environ.get('CUDA_VISIBLE_DEVICES')=='0,1'
    if options.stage=='prepare':prepare(root);return
    import torch
    import torch.multiprocessing as mp
    assert torch.cuda.device_count()==2
    audit=json.loads((root/'input_audit.json').read_text())
    assert audit['script_sha256']==digest(__file__) and audit['manifest_sha256']==digest(root/'manifest.parquet')
    if options.stage=='extract':
        assert (root/'qc.exit').read_text().strip()=='0'
        assert json.loads((root/'qc/summary.json').read_text())['status']=='pass'
    (root/options.stage).mkdir(exist_ok=False)
    mp.spawn(worker,args=(str(root),options.stage),nprocs=2,join=True)
    finalize(root,options.stage)


if __name__=='__main__':main()
