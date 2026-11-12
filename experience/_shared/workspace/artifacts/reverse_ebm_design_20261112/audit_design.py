"""Read-only source audit and synthetic math checks; never imports a runner."""
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / 'EBM'
OUT = Path(__file__).resolve().parent
REFERENCE = '10f3bcd057dc6a2ca687fbd5389ad6c9e15a3746'
sys.dont_write_bytecode = True


def git(*args):
    return subprocess.check_output(['git', *args], cwd=REPO)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def describe(data, filename):
    source = data.decode('utf-8-sig')
    result = {'bytes': len(data), 'lines': len(source.splitlines()), 'sha256': digest(data)}
    if filename.endswith('.py'):
        tree = ast.parse(source, filename=filename)
        result['definitions'] = [{'name': n.name, 'line': n.lineno, 'end_line': n.end_lineno}
                                 for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
        result['imports'] = sorted({n.module or '' for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} |
                                   {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names})
    return result


def source_audit():
    paths = git('ls-tree', '-r', '--name-only', REFERENCE).decode().splitlines()
    original = {p: describe(git('show', REFERENCE + ':' + p), p) for p in paths}
    manifest = json.loads((REPO / 'baseline240_source/manifest.json').read_text())
    frozen = {}
    for p, expected in manifest['files'].items():
        data = (REPO / 'baseline240_source' / p).read_bytes()
        frozen[p] = {'manifest_sha_match': digest(data) == expected,
                     'reference_text_match': data.replace(b'\r\n', b'\n') ==
                         git('show', REFERENCE + ':' + p).replace(b'\r\n', b'\n'),
                     **describe(data, p)}
        assert frozen[p]['manifest_sha_match'] and frozen[p]['reference_text_match'], p
    current = {}
    for path in sorted(REPO.rglob('*')):
        if not path.is_file() or '.git' in path.parts or '__pycache__' in path.parts:
            continue
        if path.suffix in ('.py', '.md', '.json', '.sh') or path.name in ('Dockerfile', '.gitignore', 'LICENSE'):
            relative = path.relative_to(REPO).as_posix()
            current[relative] = describe(path.read_bytes(), relative)
    receipt = dict(reference_commit=REFERENCE, head=git('rev-parse', 'HEAD').decode().strip(),
                   reference_file_count=len(paths), reference=original, frozen=frozen, current=current,
                   git_status=git('status', '--porcelain=v1').decode(),
                   scope='All original tree files read and Python AST inventoried; core and relevant extensions manually traced. Other extension utilities inventoried, not claimed to be line-by-line reviewed.')
    (OUT / 'source_audit.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    return receipt


def validate():
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)
    torch.manual_seed(1729)
    torch.set_default_dtype(torch.float64)
    spec = importlib.util.spec_from_file_location('audited_original_losses', REPO/'baseline240_source/model/loss.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checks = []

    def record(name, **details):
        checks.append(dict(name=name, status='passed', **details))

    def per_item(z, w, labels, compact):
        cosine = F.normalize(z, dim=1) @ F.normalize(w, dim=1).T
        compact_mask = labels[:, None] == compact
        penalty = torch.where(compact_mask, .9-cosine, cosine-.2)
        weight = torch.where(labels[:, None] == 1, 5., 5./9)
        return F.softplus(20*penalty) * weight

    z = torch.randn(120, 7, requires_grad=True)
    w = torch.randn(1, 7, requires_grad=True)
    labels = torch.tensor([1]*12+[0]*108)
    original = module.OCSoftmax(7, class_weight=[5/9,5]).double()
    with torch.no_grad(): original.weight.copy_(w)
    actual = original(z, labels)[0]
    expected = per_item(z, w, labels, 1).mean()
    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)
    ga = torch.autograd.grad(actual, (z, original.weight), retain_graph=True)
    ge = torch.autograd.grad(expected, (z,w), retain_graph=True)
    error = max(float((a-b).abs().max()) for a,b in zip(ga,ge))
    assert error < 1e-12
    record('original_OC_formula_and_gradient_identity', max_gradient_error=error)

    for compact in (0,1):
        scores = torch.tensor([[-.5],[.0],[.5],[.95]], requires_grad=True)
        for label in (0,1):
            loss = F.softplus(20*(.9-scores if label==compact else scores-.2)).mean()
            derivative = torch.autograd.grad(loss,scores)[0]
            assert bool((derivative < 0).all()) if label==compact else bool((derivative > 0).all())
    record('OC_score_derivatives_pull_compact_push_other')

    # Full embedding gradients, away from degenerate parallel/antiparallel points.
    steps = {}
    for label in (0,1):
        a = torch.tensor([[.3,.8,.5]], requires_grad=True)
        c = torch.tensor([[1.,0.,0.]])
        y = torch.tensor([label])
        loss = per_item(a,c,y,0).mean()
        g = torch.autograd.grad(loss,a)[0]
        before = float(F.cosine_similarity(a.detach(),c).item())
        after = float(F.cosine_similarity(a.detach()-.001*g,c).item())
        assert after > before if label==0 else after < before
        steps[str(label)] = dict(before=before, after=after)
    record('reverse_embedding_gradient_step_direction', labels=steps)

    for n in (6,12):
        a = torch.randn(n,7)
        y = torch.ones(n,dtype=torch.long)
        reference = original(a,y)[0]
        raw = F.softplus(20*(.9-F.normalize(a,dim=1)@F.normalize(original.weight,dim=1).T)).mean()
        assert torch.allclose(reference,5*raw)
    # This is an objective multiplier ratio, not an Adam update-size ratio.
    assert math.isclose((5/9)/5,1/9)
    record('center_scale_control', original_bona_multiplier=5., naive_spoof_multiplier=5/9,
           naive_objective_ratio=1/9, proposed_both_arm_multiplier=5.,
           caution='Adam moments, epsilon and weight decay prevent inferring a 9x parameter update ratio.')

    bpl = module.BPL(30.)
    a = torch.randn(120,7,requires_grad=True)
    b = torch.randn(120,7,requires_grad=True)
    mask = labels == 0
    loss = bpl(a[mask],b[mask])
    grads = torch.autograd.grad(loss,(a,b))
    assert all(torch.equal(g[~mask],torch.zeros_like(g[~mask])) and float(g[mask].norm())>0 for g in grads)
    assert torch.allclose(loss,bpl(a[mask].flip(0),b[mask]))
    record('spoof_BPL_routes_both_gradients_and_is_permutation_invariant', pairs=108**2)

    small_a=torch.randn(12,7)
    small_b=torch.randn(12,7)
    assert torch.allclose(bpl(small_a,small_b),bpl(small_a.repeat(9,1),small_b.repeat(9,1)))
    record('pair_mean_scale_is_not_multiplied_by_81')

    # Construct a simultaneously feasible two-class arrangement. A single
    # spoof center plus all-spoof pairs imposes no contradictory target.
    axis=torch.tensor([[1.,0.,0.]])
    spoof_z=torch.cat([-axis.repeat(12,1),axis.repeat(108,1)])
    bona_z=-spoof_z
    reverse=per_item(spoof_z,axis,labels,0).mean()+bpl(spoof_z[labels==0],spoof_z[labels==0])
    reference=per_item(bona_z,axis,labels,1).mean()+bpl(bona_z[labels==1],bona_z[labels==1])
    assert torch.allclose(reverse,reference,atol=1e-12,rtol=1e-12)
    record('constructive_two_class_feasibility', reverse_backend_loss=float(reverse),
           baseline_backend_loss=float(reference),
           note='Feasibility in embedding space does not prove a network learns this mapping on real data.')

    # Independent row/column subset averages exactly recover the full all-pairs mean.
    # A shared subset overweights same-UID diagonal terms.
    import itertools
    matrix = torch.arange(25,dtype=torch.float64).reshape(5,5).square()
    subsets = list(itertools.combinations(range(5),2))
    independent = torch.stack([matrix[list(i)][:,list(j)].mean() for i in subsets for j in subsets]).mean()
    shared = torch.stack([matrix[list(i)][:,list(i)].mean() for i in subsets]).mean()
    assert torch.allclose(independent,matrix.mean()) and not torch.allclose(shared,matrix.mean())
    record('independent_subset_pair_estimator', full_mean=float(matrix.mean()),
           independent_mean=float(independent), shared_subset_mean=float(shared))

    score = F.normalize(z.detach(),dim=1)@F.normalize(w.detach(),dim=1).T
    base = torch.where(labels[:,None]==1, F.softplus(20*(.9-score)), F.softplus(20*(score-.2)))
    inverted = -score
    equivalent = torch.where(labels[:,None]==0, F.softplus(20*(-.2-inverted)), F.softplus(20*(inverted+.9)))
    strict_reverse = torch.where(labels[:,None]==0,F.softplus(20*(.9-inverted)),F.softplus(20*(inverted-.2)))
    assert torch.allclose(base,equivalent) and not torch.allclose(base,strict_reverse)
    record('center_negation_is_not_role_reversal_with_same_margins')

    # Independent tied-block ROC implementation, with higher score=bonafide.
    def eer(ys,ss):
        ordered = sorted(zip(ss,ys), reverse=True)
        p,n = sum(ys),len(ys)-sum(ys)
        tp,fp,old_tp,old_fp = 0.,0.,0.,0.
        for s,group in itertools.groupby(ordered,key=lambda row:row[0]):
            rows=list(group); tp+=sum(y for _,y in rows)/p; fp+=sum(1-y for _,y in rows)/n
            if fp+tp>=1:
                mix=(1-old_fp-old_tp)/(fp+tp-old_fp-old_tp)
                return 100*(old_fp+mix*(fp-old_fp))
            old_tp,old_fp=tp,fp
        raise AssertionError('no crossing')
    ys = [0,0,0,1,1,1]
    raw = [.9,.8,.7,-.7,-.8,-.9]
    assert eer(ys,[-s for s in raw])==0 and eer(ys,raw)==100
    assert eer(ys,[0.]*6)==50
    record('canonical_score_sign_and_tied_ROC', correct_sign_eer=0., wrong_sign_eer=100., constant_eer=50.)

    # Execute the actual current EER method, without importing a training runner.
    import numpy as np
    from sklearn import metrics
    tree=ast.parse((REPO/'train.py').read_text(encoding='utf-8'))
    trainer=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ModelTrainer')
    method=next(n for n in trainer.body if isinstance(n,ast.FunctionDef) and n.name=='calculate_EER')
    namespace={'metrics':metrics,'np':np}
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(REPO/'train.py'),'exec'),namespace)
    cases=[(ys,[-s for s in raw]),(ys,raw),(ys,[0.]*6),
           ([0,1,0,1,0,1],[.1,.1,.4,.5,.4,.2])]
    differences=[abs(namespace['calculate_EER'](None,ss,y)-eer(y,ss)) for y,ss in cases]
    assert max(differences)<1e-12
    record('actual_current_EER_method_matches_independent_tied_ROC', max_error=max(differences))

    bona = [12]*107+[6]
    spoof = [108]*105+[60,108,108]
    assert len(bona)==len(spoof)==108 and sum(bona)==1290 and sum(spoof)==11616
    assert [b+s for b,s in zip(bona,spoof)][105:]==[72,120,114]
    record('original_per_rank_partial_batches', steps=108, bona_visits=sum(bona), spoof_visits=sum(spoof),
           step106=dict(bona=12,spoof=60,total=72), step108=dict(bona=6,spoof=108,total=114),
           bona_pairs_epoch=sum(n*n for n in bona),spoof_pairs_epoch=sum(n*n for n in spoof))

    # A constant embedding cannot make both OC classes small; exact angular
    # extrema can nevertheless be stationary because normalization projects gradients.
    aligned = torch.tensor([[1.,0.,0.]],requires_grad=True)
    c = torch.tensor([[1.,0.,0.]])
    loss = per_item(aligned,c,torch.tensor([1]),0).mean()
    g = torch.autograd.grad(loss,aligned)[0]
    assert float(loss.detach())>50 and torch.equal(g,torch.zeros_like(g))
    record('OC_does_not_guarantee_escape_from_exact_angular_stationary_collapse',
           wrong_class_loss=float(loss.detach()), gradient_norm=float(g.norm()))

    # No bona-bona term: many directions, including coincident directions, satisfy push margin.
    far = torch.tensor([[-1.,0.,0.],[-1.,0.,0.]])
    assert bool((F.cosine_similarity(far,c.expand_as(far))<=.2).all())
    assert torch.allclose(F.cosine_similarity(far[:1],far[1:]),torch.ones(1))
    record('bona_push_is_not_bona_mutual_repulsion')

    # Center-only differentiation leaves the encoder frozen, and theta-only
    # differentiation leaves center frozen. No optimizer is instantiated.
    layer=torch.nn.Linear(7,7,bias=False)
    center=torch.nn.Parameter(torch.randn(1,7))
    for p in layer.parameters(): p.requires_grad_(False)
    value=per_item(layer(torch.randn(108,7)),center,torch.zeros(108,dtype=torch.long),0).mean()
    value.backward()
    assert layer.weight.grad is None and float(center.grad.norm())>0
    center.grad=None; center.requires_grad_(False); layer.weight.requires_grad_(True)
    value=per_item(layer(torch.randn(120,7)),center,labels,0).mean()
    value.backward()
    assert center.grad is None and float(layer.weight.grad.norm())>0
    record('two_phase_gradient_isolation_without_optimizer_or_training')
    receipt = dict(status='passed', checks=len(checks), results=checks, torch_version=torch.__version__,
                   device='cpu', dtype='float64', synthetic_only=True, optimizers_instantiated=0,
                   remote_calls=0, model_inference_runs=0, spending_usd=0,
                   limitations='This validates source identities and candidate algebra, not an implemented reverse runner, DDP, audio, GPU memory, convergence or EER improvement.')
    (OUT/'validation.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    return receipt


def preservation(before):
    now={p:digest((REPO/p).read_bytes()) for p in before['current']}
    assert all(now[p]==entry['sha256'] for p,entry in before['current'].items())
    assert git('status','--porcelain=v1').decode()==before['git_status']
    receipt=dict(status='passed', files_checked=len(now), git_status_unchanged=True,
                 remote_contacted=False, new_training=0, new_inference=0)
    (OUT/'preservation.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    return receipt


if __name__=='__main__':
    before=source_audit()
    result=validate()
    preserved=preservation(before)
    print(json.dumps(dict(reference_files=before['reference_file_count'],
                          inventoried_current_files=len(before['current']),validation_checks=result['checks'],
                          status=result['status'],preservation=preserved),indent=2))
