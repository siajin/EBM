"""Conditional L32 hooks around the byte-preserved original240 training code.

The two training DataLoaders, their datasets and ModelTrainer.train are untouched.
No execution, model download, device initialization or job launch at import.
"""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import soundfile as sf
import torch
from torch.utils.data import DataLoader, DistributedSampler

SOURCE = Path(__file__).resolve().parent / 'baseline240_source'
TWO, SHORT = 32000, 48440


def verify_source():
    manifest = json.loads((SOURCE / 'manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['files'].items():
        if hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Original240 source changed: ' + name)
    return manifest


def load_source(name, namespace=None, member=None):
    """Load the frozen module, or compile one original class/function verbatim."""
    verify_source()
    path = SOURCE / name
    if member:
        tree = ast.parse(path.read_text(encoding='utf-8'))
        node = next(n for n in tree.body if getattr(n, 'name', None) == member)
        scope = dict(namespace or {})
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
        return scope[member]
    module_name = 'baseline240_source.' + name.removesuffix('.py').replace('/', '.')
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def validate_runtime():
    """Resolve training/evaluation/worker modules before any GPU/optimizer work."""
    import importlib
    verify_source()
    for name in ['short32_lifecycle', 'controlled_data_loaders', 'native_steps', 'train']:
        importlib.import_module(name)
    for name in ['data_loaders.py', 'model/b_linear.py', 'model/loss.py']:
        load_source(name)


def unwrap(module):
    return module.module if hasattr(module, 'module') else module


class Short32Hooks:
    """Read-only source schedule plus extra loss, inside a single DDP forward.

    DistributedSampler uses its own seeded Generator: enumerating its schedule
    does not advance the DataLoader/global RNG. No labels/rows are added to data.
    """
    def __init__(self, trainer):
        self.trainer = trainer
        self.model = unwrap(trainer.classifier)
        self.frontend = unwrap(trainer.plm)
        self.loaders = [trainer.train_loader_bona, trainer.train_loader_spoof]
        self.samplers = [trainer.sampler_bona, trainer.sampler_spoof]
        for loader, sampler in zip(self.loaders, self.samplers):
            if type(loader) is not DataLoader or type(sampler) is not DistributedSampler:
                raise ValueError('Require the original DataLoader and DistributedSampler')
            if loader.sampler is not sampler or loader.drop_last:
                raise ValueError('Original sampler/drop_last contract changed')
            if not len(loader) or not getattr(loader, 'in_order', True):
                raise ValueError('Original nonempty, ordered loading is required')
            if sampler.num_replicas != self.samplers[0].num_replicas or sampler.rank != self.samplers[0].rank:
                raise ValueError('Class samplers must share rank/world')
        if dict(trainer.classifier.named_parameters()).get('module.loss.weight') is not self.model.loss.weight:
            raise ValueError('Original two-stage train requires module.loss.weight')
        self.lengths = []
        for label, loader in zip([1,0], self.loaders):
            for item in loader.dataset.items:
                info = sf.info(item.path)
                if info.samplerate != 16000 or info.channels != 1 or info.frames < 720 or item.label != label:
                    raise ValueError('Expected original mono16k source/class: '+str(item.path))
                if label == 1:
                    self.lengths.append(info.frames)
        if trainer.da is not None or trainer.args['epoch'] > trainer.args['T_0']:
            raise ValueError('Short32 requires original codec-off/frozen-frontend regime')
        self.epoch = None
        self.step = 0
        self.raw_short = None
        self.embeddings = None
        self.handles = [self.frontend.register_forward_pre_hook(self.capture_wave),
                        self.model.register_forward_pre_hook(self.before, with_kwargs=True),
                        self.model.layer_norm.register_forward_hook(self.capture_embedding),
                        self.model.register_forward_hook(self.after, with_kwargs=True)]

    def capture_wave(self, module, inputs):
        if self.model.training and inputs and inputs[0].ndim == 2 and inputs[0].shape[1] == SHORT:
            self.raw_short = inputs[0].detach()

    def before(self, module, inputs, kwargs):
        if not module.training or kwargs.get('x_short') is None:
            return
        epoch = self.samplers[0].epoch
        if any(s.epoch != epoch for s in self.samplers):
            raise ValueError('Original class samplers must share epoch')
        if epoch != self.epoch:
            if self.epoch is not None and self.step != max(map(len, self.loaders)):
                raise ValueError('Previous original epoch was incomplete')
            self.epoch, self.step = epoch, 0
            self.batches = []
            for loader, sampler in zip(self.loaders, self.samplers):
                ids = list(sampler)
                self.batches.append([ids[i:i + loader.batch_size] for i in range(0, len(ids), loader.batch_size)])
        if self.step >= max(map(len, self.loaders)):
            raise ValueError('More forwards than original batches')
        rows = [b[self.step % len(b)] for b in self.batches]
        n = len(rows[0])
        labels = kwargs['label']
        if kwargs['bona_size'] != n or len(labels) != sum(map(len, rows)):
            raise ValueError('Original ragged batch schedule changed')
        if not torch.all(labels[:n] == 1) or not torch.all(labels[n:] == 0):
            raise ValueError('Original bona-first order changed')
        if self.raw_short is None or self.raw_short.shape != (len(labels), SHORT):
            raise ValueError('Missing exact original augmented short waveform')
        self.positions = [j for j, index in enumerate(rows[0]) if self.lengths[index] <= TWO]
        self.n = n
        self.embeddings = []

    def capture_embedding(self, module, inputs, output):
        if self.embeddings is not None:
            self.embeddings.append(output)

    def after(self, module, inputs, kwargs, output):
        if not module.training or kwargs.get('x_short') is None:
            return output
        if len(self.embeddings) != 2:
            raise ValueError('Expected original long and short embeddings')
        long, short = self.embeddings
        self.embeddings = None
        k, n = len(self.positions), self.n
        stats = dict(epoch=self.epoch, batch=self.step + 1, bona=n, eligible=k,
                     bpl43_pairs=n*n, bpl32_pairs=k*k, bpl32_weight=.5*k/n)
        self.step += 1
        raw, self.raw_short = self.raw_short, None
        if not k:
            self.last_stats = stats
            return output  # Exact original loss object and RNG, including K=0 ranks.
        if self.frontend.training:
            raise ValueError('Original frontend must remain in eval mode')
        positions = torch.tensor(self.positions, device=long.device)
        # Match the already augmented original3s view. No new dataset/RNG draws.
        # Call the unwrapped frozen frontend: K may differ across DDP ranks.
        # Extra dropout/ASP draws have a separate reproducible random stream.
        # Original center-pass and DataLoader worker seeds must not shift.
        devices = [long.device.index] if long.is_cuda else []
        seed = (int(self.trainer.args.get('rand_seed', 1)) + 1000003*self.epoch
                + 1009*self.step + 97*int(self.trainer.args.get('rank', 0)))
        generator = torch.Generator(device=long.device).manual_seed(seed)
        with torch.random.fork_rng(devices=devices):
            if long.is_cuda:
                torch.cuda.set_rng_state(generator.get_state(), long.device)
            else:
                torch.set_rng_state(generator.get_state())
            with torch.no_grad():
                hidden = self.frontend(raw[positions, :TWO], output_hidden_states=True).hidden_states
            two = module.bridge_module(torch.stack(hidden, dim=1))
            two = module.layer_norm(module.linear(module.AHP_ASP(two)))
        active43 = module.loss_bpl(long[positions], short[:n])
        loss32 = module.loss_bpl(short[positions], two)
        corrected = output[0] + (.5*k/n) * (loss32 - active43)
        stats.update(bpl32=float(loss32.detach()), bpl43_eligible=float(active43.detach()))
        self.last_stats = stats
        return corrected, output[1]

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.raw_short = self.embeddings = None


def run(process_id, args, experiment_args, environment):
    """Original main construction/train; existing no-dev monitoring stays outside."""
    from short32_bpl import validate_options
    args.setdefault('distributed_timeout_seconds', 600)
    validate_options(args)
    validate_runtime()
    import train as monitoring
    import controlled_data_loaders as evaluation_data
    original_data = load_source('data_loaders.py')
    original_backend = load_source('model/b_linear.py')
    original_losses = load_source('model/loss.py')
    original_trainer = load_source('train.py', dict(environment, tqdm=monitoring.tqdm), 'ModelTrainer')

    def loaders(a, items, val, evals):
        # Call the original function, retaining its two training loaders verbatim.
        result = original_data.get_loaders(a, items, val, evals)
        def evaluation(data):
            ds = evaluation_data.EnrollmentSet(dict(a, native_length=True, cap_long=True), data)
            return DataLoader(ds, batch_size=a.get('eval_batch_size', 8),
                              sampler=evaluation_data.ExactDistributedEvalSampler(ds, a),
                              num_workers=a['num_workers'], collate_fn=evaluation_data.collate_eval,
                              timeout=120 if a['num_workers'] else 0)
        return (*result[:4], evaluation(val), {name: evaluation(db.trials) for name, db in evals.items()})

    class Trainer(monitoring.ModelTrainer):
        train = original_trainer.train  # No wrapper/reimplementation/replay.

        def run(self):
            from short32_lifecycle import coordinated, run_training
            from positive_bpl import oc_evaluation_protocol
            revision = getattr(getattr(unwrap(self.plm), 'config', None), '_commit_hash', None)
            self.args['frontend_resolved_revision'] = revision
            self.evaluation_protocol = oc_evaluation_protocol(self.args, revision)
            self.evaluation_protocol['selection'] = '19LA monitoring every5 epochs; no dev; fixed training budget'
            self.short32 = coordinated('training_preflight', lambda: Short32Hooks(self))
            try:
                return run_training(self)
            finally:
                self.short32.close()

        def test(self, phase, epoch):
            from short32_lifecycle import evaluate
            return evaluate(self, phase, epoch, self.checkpoints)

    scope = dict(environment, data_loaders=SimpleNamespace(get_loaders=loaders),
                 train=SimpleNamespace(ModelTrainer=Trainer),
                 model=SimpleNamespace(B_Linear=original_backend.B_Linear,
                                       OCSoftmax=original_losses.OCSoftmax, BPL=original_losses.BPL))
    original_run = load_source('main.py', scope, 'run')
    return original_run(process_id, args, experiment_args)
