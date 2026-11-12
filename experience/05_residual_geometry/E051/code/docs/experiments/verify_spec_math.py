"""Check the experiment specification on synthetic CPU arrays only.

This is not an EBM implementation, a data extractor, or a training runner.
Run: python -X utf8 docs/experiments/verify_spec_math.py
"""

import hashlib
import json
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "ebm_embedding_geometry_protocol.json"
SPEC = ROOT / "ebm_embedding_geometry_spec_20261105.md"


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def diagonal_metric(v):
    a = np.exp(2 * np.tanh(v))
    return np.diag(len(v) * a / a.sum())


def score(z, c, a):
    residual = z - c
    return 1 - 0.5 * np.einsum("ni,ij,nj->n", residual, a, residual)


def oc_loss(scores, labels, alpha=20, real_margin=0.9, fake_margin=0.2):
    logits = np.where(labels == 1, real_margin - scores, scores - fake_margin)
    return np.logaddexp(0, alpha * logits)


def main():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    spec = SPEC.read_text(encoding="utf-8")
    rng = np.random.default_rng(20261105)
    d = protocol["baseline"]["dimension"]
    z = unit(rng.normal(size=(256, d)))
    c = unit(rng.normal(size=(1, d)))[0]
    cosine = z @ c
    r = z - cosine[:, None] * c
    tangent_norm = np.linalg.norm(r, axis=1)
    u = r / tangent_norm[:, None]
    checks = []

    def check(name, condition, **details):
        if not bool(condition):
            raise AssertionError(f"{name}: {details}")
        checks.append({"name": name, "status": "pass", **details})

    err = float(np.max(np.abs(np.sum((z - c) ** 2, axis=1) - (2 - 2 * cosine))))
    check("normalized_euclidean_equals_cosine", err < 1e-10, max_error=err)
    err = float(np.max(np.abs(tangent_norm**2 - (1 - cosine**2))))
    check("tangent_norm_has_no_new_scalar_information", err < 1e-10, max_error=err)
    check("unit_direction_is_perpendicular", np.max(np.abs(u @ c)) < 1e-10)

    initial_a = diagonal_metric(np.zeros(d))
    check("metric_initialization_is_identity", np.allclose(initial_a, np.eye(d)))
    weighted_score = score(z, c, initial_a)
    check("identity_metric_score_equals_baseline", np.allclose(weighted_score, cosine, atol=1e-10))
    labels = np.tile([0, 1], len(z) // 2)
    sample_weights = np.where(labels == 1, 9.0, 1.0)
    baseline_loss = np.mean(sample_weights * oc_loss(cosine, labels))
    metric_loss = np.mean(sample_weights * oc_loss(weighted_score, labels))
    check("identity_metric_weighted_OC_loss_equals_baseline", abs(baseline_loss - metric_loss) < 1e-10)
    check("label_polarity_is_bonafide_positive", oc_loss(np.array([1.0]), np.array([1]))[0] < oc_loss(np.array([-1.0]), np.array([1]))[0])

    a = diagonal_metric(rng.normal(size=d))
    check("diagonal_metric_positive_definite_and_trace", np.linalg.eigvalsh(a).min() > 0 and abs(np.trace(a) - d) < 1e-10)
    low_rank = rng.normal(scale=0.001, size=(d, 8))
    a0 = a + low_rank @ low_rank.T
    full = d * a0 / np.trace(a0)
    check("low_rank_extension_positive_definite_and_trace", np.linalg.eigvalsh(full).min() > 0 and abs(np.trace(full) - d) < 1e-10)
    v = rng.normal(size=d)
    check("quadratic_residual_sign_symmetry", abs(v @ full @ v - (-v) @ full @ (-v)) < 1e-10)

    sigma = np.cov(z.T)
    gamma_one = np.trace(sigma) / d * np.eye(d)
    inverse = np.linalg.solve(gamma_one, np.eye(d))
    isotropic = d * inverse / np.trace(inverse)
    check("shrinkage_gamma_one_returns_identity", np.allclose(isotropic, np.eye(d)))

    # Same cosine and norm; different residual directions can yield different scores.
    za = np.array([[0.8, 0.6, 0.0], [0.8, 0.0, 0.6]])
    ca = np.array([1.0, 0.0, 0.0])
    sa = score(za, ca, np.diag([1.0, 0.5, 1.5]))
    check("equal_cosine_can_have_different_anisotropic_scores", np.isclose((za @ ca)[0], (za @ ca)[1]) and not np.isclose(sa[0], sa[1]), scores=sa.tolist())

    # Opposite modes cancel their mean despite distinct distributions.
    directions_b = np.array([[1.0, 0.0], [-1.0, 0.0]])
    directions_s = np.array([[0.0, 1.0], [0.0, -1.0]])
    check("zero_mean_direction_does_not_imply_equal_distribution", np.allclose(directions_b.mean(0), directions_s.mean(0)) and not np.allclose(directions_b.T @ directions_b, directions_s.T @ directions_s))

    hidden = protocol["attention"]["hidden"]
    parameter_count = d * hidden + hidden + hidden * d + d
    check("gate_and_residual_MLP_parameter_count", parameter_count == protocol["attention"]["dynamic_parameters"] == protocol["attention"]["matched_residual_mlp_parameters"], parameter_count=parameter_count)
    identity_gate = 2 / (1 + np.exp(-np.zeros(d)))
    check("gate_identity_initialization", np.array_equal(identity_gate, np.ones(d)))

    # A PSD low-rank factor initialized at zero is a stationary point in U.
    residual = z[0] - c
    factor_zero = np.zeros((d, 8))
    gradient_zero = 2 * np.outer(residual, residual) @ factor_zero
    gradient_nonzero = 2 * np.outer(residual, residual) @ low_rank
    check("zero_low_rank_factor_has_zero_quadratic_gradient", np.linalg.norm(gradient_zero) == 0 and np.linalg.norm(gradient_nonzero) > 0)

    variants = {row["id"] for row in protocol["variants"]}
    checks_count = len(protocol["statistics"]["primary_contrasts"])
    check("confirmatory_family_matches_adjusted_CI", checks_count == protocol["statistics"]["family_size"] == 5 and np.isclose(1 - 0.05 / checks_count, protocol["statistics"]["confirmatory_family_CI"]))
    screening = len(variants) * len(protocol["training"]["screening_seeds"])
    main_count = screening + (len(variants) - 1) * 2
    check("training_run_counts", screening == 18 and main_count == 28 and len(variants) * 5 == 30)
    covered = {h for e in protocol["experiments"] for h in e["hypotheses"]}
    check("all_seven_hypotheses_have_experiments", covered == {f"H{i}" for i in range(1, 8)})
    forbidden = set(protocol["splits"]["forbidden_fit_splits"])
    check("no_locked_split_in_fit_registry", all(not (set(e["fit_splits"]) & forbidden) for e in protocol["experiments"]))
    experiment_ids = {e["id"] for e in protocol["experiments"]}
    check("dependencies_resolve", all(set(e["depends_on"]) <= experiment_ids for e in protocol["experiments"]))
    dependencies = {e["id"]: e["depends_on"] for e in protocol["experiments"]}
    complete = set()
    def visit(experiment, active):
        if experiment in active:
            raise AssertionError(f"Dependency cycle: {active} -> {experiment}")
        if experiment in complete:
            return
        for dependency in dependencies[experiment]:
            visit(dependency, active | {experiment})
        complete.add(experiment)
    for experiment in dependencies:
        visit(experiment, set())
    check("experiment_dependencies_are_acyclic", complete == experiment_ids)
    check("no_accidental_training_execution_claim", not protocol["scope"]["start_training"] and not protocol["executable_runner"])
    check("all_spec_sections_exist", len(re.findall(r"^## \d+ ", spec, flags=re.M)) == 16)
    check("document_language_review", re.search(r"[\u3040-\u30ff]", spec) is None)

    report = {
        "status": "pass",
        "scope": "synthetic_numpy_CPU_math_and_protocol_checks_only",
        "does_not_validate": ["real_EBM_forward", "gradient_routing_in_PyTorch", "data_splits", "real_embeddings", "performance", "GPU_DDP_implementation"],
        "checks_passed": len(checks),
        "checks": checks,
        "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [SPEC, PROTOCOL, Path(__file__)]},
    }
    output = ROOT / "spec_validation.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "checks_passed": len(checks), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
