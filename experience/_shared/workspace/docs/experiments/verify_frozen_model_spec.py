"""CPU-only synthetic math checks for the frozen EBM experiment specification.

Does not import the laboratory model, read audio/embeddings, or fit experimental
heads. Tiny optimizer steps below check synthetic gradient paths only.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT = HERE / "ebm_frozen_model_spec_validation_20261105.json"
MD = HERE / "ebm_frozen_model_experiments_20261105.md"
PROTOCOL = HERE / "ebm_frozen_model_protocol_20261105.json"
AUDIT = HERE / "ebm_frozen_structure_audit_20261105.json"


def main():
    torch.set_num_threads(2)
    torch.manual_seed(1153)
    torch.set_default_dtype(torch.float64)
    p = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    d = p["architecture"]["embedding_size"]
    hidden = p["architecture"]["head_hidden_size"]
    results = []

    def check(name, ok, **details):
        if not bool(ok):
            raise AssertionError(f"{name}: {details}")
        results.append({"name": name, "status": "pass", **details})

    def close(name, a, b, tol=1e-10):
        err = float(torch.max(torch.abs(a - b)).detach())
        check(name, err <= tol, maximum_absolute_error=err, tolerance=tol)

    def per_sample_oc(s, y):
        cfg = p["original_training"]
        return torch.where(y, F.softplus(cfg["OC_alpha"] * (cfg["r_real"] - s)),
                           F.softplus(cfg["OC_alpha"] * (s - cfg["r_fake"])))

    def oc(s, y):
        cfg = p["original_training"]
        w = torch.where(y, cfg["bonafide_class_weight"], cfg["spoof_class_weight"])
        return (w * per_sample_oc(s, y)).mean()

    class StaticGate(nn.Module):
        def __init__(self):
            super().__init__()
            self.v = nn.Parameter(torch.zeros(d))

        def forward(self, x):
            return x * (1 + 0.5 * torch.tanh(self.v))

    class SmallHead(nn.Module):
        def __init__(self, residual=False):
            super().__init__()
            self.residual = residual
            self.first = nn.Linear(d, hidden, device="cpu")
            self.last = nn.Linear(hidden, d, device="cpu")
            nn.init.zeros_(self.first.bias)
            nn.init.zeros_(self.last.weight)
            nn.init.zeros_(self.last.bias)

        def forward(self, x):
            a = torch.tanh(self.last(F.gelu(self.first(x))))
            return x + 0.5 / math.sqrt(d) * a if self.residual else x * (1 + 0.5 * a)

    class DiagonalMetric(nn.Module):
        def __init__(self):
            super().__init__()
            self.v = nn.Parameter(torch.zeros(d))

        def weights(self):
            a = torch.exp(2 * torch.tanh(self.v))
            return d * a / a.sum()

        def forward(self, x, c):
            return 1 - 0.5 * ((x - c).square() * self.weights()).sum(-1)

    original = nn.Linear(11, d, device="cpu").requires_grad_(False)
    state_before = {k: v.detach().clone() for k, v in original.state_dict().items()}
    x = F.normalize(original(torch.randn(160, 11, device="cpu")), dim=-1).detach()
    c = F.normalize(torch.randn(d, device="cpu"), dim=0).detach()
    s = x @ c
    y = torch.arange(160, device="cpu") < 16
    loss = per_sample_oc(s, y)
    close("class_weight_reduction_16B_144S", oc(s, y),
          0.5 * (loss[y].mean() + loss[~y].mean()))

    heads = {"static": StaticGate(), "dynamic": SmallHead(), "MLP": SmallHead(True)}
    for name, head in heads.items():
        n = sum(v.numel() for v in head.parameters())
        check(f"{name}_parameter_count", n == p["posthoc_training"][f"{name}_parameters"], count=n)
        close(f"{name}_identity_vectors", head(x), x)
        close(f"{name}_identity_score", F.normalize(head(x), dim=-1) @ c, s)
        optimizer = torch.optim.SGD(head.parameters(), lr=0.01)
        ids = {id(q) for group in optimizer.param_groups for q in group["params"]}
        check(f"{name}_optimizer_isolation", ids == {id(q) for q in head.parameters()}
              and not ids.intersection({id(q) for q in original.parameters()} | {id(c)}))
        if name != "static":
            for step in range(2):
                optimizer.zero_grad(set_to_none=True)
                changed = head(x)
                objective = oc(F.normalize(changed, dim=-1) @ c, y)
                objective = objective + 0.01 * (changed - x).square().sum(-1).mean()
                objective.backward()
                first = float(head.first.weight.grad.norm())
                last = float(head.last.weight.grad.norm())
                if step == 0:
                    check(f"{name}_first_step_gradient", first < 1e-15 and last > 1e-12,
                          first_weight_gradient_norm=first, last_weight_gradient_norm=last)
                else:
                    check(f"{name}_second_step_gradient", first > 1e-12,
                          first_weight_gradient_norm=first)
                optimizer.step()
        with torch.no_grad():
            for q in head.parameters():
                q.normal_(0, 20)
            changed = head(x)
            delta = float((changed - x).norm(dim=-1).max())
            lower = float(changed.norm(dim=-1).min())
            check(f"{name}_bounded_transform", delta <= 0.5 + 1e-12 and lower >= 0.5 - 1e-12,
                  maximum_L2_change=delta, minimum_output_norm=lower)

    check("original_parameters_unchanged", all(torch.equal(v, state_before[k])
          for k, v in original.state_dict().items()) and
          all(q.grad is None for q in original.parameters()) and c.grad is None)
    close("normalized_Euclidean_equals_cosine", 1 - 0.5 * (x - c).square().sum(-1), s)
    metric = DiagonalMetric()
    close("metric_identity", metric(x, c), s)
    check("metric_parameter_count", sum(q.numel() for q in metric.parameters()) == d, count=d)
    oc(metric(x, c), y).backward()
    check("metric_nonzero_gradient", float(metric.v.grad.norm()) > 1e-12)
    with torch.no_grad():
        metric.v.normal_(0, 20)
    a = metric.weights().detach()
    check("metric_positive_trace_condition", bool((a > 0).all())
          and abs(float(a.sum()) - d) < 1e-10 and float(a.max() / a.min()) <= math.exp(4) + 1e-10,
          trace=float(a.sum()), condition_ratio=float(a.max() / a.min()))
    with torch.no_grad():
        metric.v.fill_(-10)
        metric.v[0] = 10
    extreme_c = F.one_hot(torch.tensor(0), d).to(torch.float64)
    extreme_s = float(metric(-extreme_c, extreme_c).detach())
    check("metric_can_be_less_than_minus_one", extreme_s < -1, synthetic_score=extreme_s)

    close("full_mask_native", F.normalize(x, dim=-1) @ F.normalize(c, dim=0), s)
    for k in p["F1"]["dimensions"]:
        mask = torch.zeros(d)
        mask[torch.randperm(d)[:k]] = 1
        masked_x = F.normalize(x * mask, dim=-1)
        masked_c = c * mask
        both = masked_x @ F.normalize(masked_c, dim=0)
        only_z = masked_x @ c
        close(f"mask_{k}_positive_scaling", only_z, both * masked_c.norm())
        check(f"mask_{k}_ranking", torch.equal(torch.argsort(only_z), torch.argsort(both)))

    # Synthetic embedding space only; no claim about attainable encoder outputs.
    u = torch.randn(160, d)
    u = F.normalize(u - (u @ c)[:, None] * c, dim=-1)
    v = torch.randn(160, d)
    v = v - (v @ c)[:, None] * c
    v = F.normalize(v - (v * u).sum(-1)[:, None] * u, dim=-1)
    fixed_s = torch.linspace(-0.9, 0.9, 160)
    tangent_norm = torch.sqrt(1 - fixed_s.square())[:, None]
    base = fixed_s[:, None] * c + tangent_norm * u
    for theta in (0.3, 1.0, 2.0):
        rotated = fixed_s[:, None] * c + tangent_norm * (math.cos(theta) * u + math.sin(theta) * v)
        close(f"rotation_{theta}_score", rotated @ c, fixed_s)
        close(f"rotation_{theta}_weighted_OC", oc(rotated @ c, y), oc(base @ c, y))
    z = base.detach().clone().requires_grad_(True)
    grad = torch.autograd.grad(oc(F.normalize(z, dim=-1) @ c, y), z)[0]
    close("azimuth_directional_derivative", (grad * v).sum(-1), torch.zeros(160))
    q, _ = torch.linalg.qr(torch.randn(d, d))
    close("joint_basis_rotation_native", (x @ q) @ (c @ q), s)

    h = torch.randn(12, d) * 0.02
    gamma, beta = torch.randn(d), torch.randn(d)
    eps = p["architecture"]["LayerNorm_eps"]
    ln = lambda value: F.layer_norm(value, (d,), gamma, beta, eps)
    close("LayerNorm_shift_invariance", ln(h + torch.randn(12, 1)), ln(h))
    tiny_h = torch.randn(12, d) * math.sqrt(eps)
    difference = float((ln(tiny_h) - ln(2 * tiny_h)).abs().max())
    check("LayerNorm_scaling_not_exact_with_epsilon", difference > 0.01,
          near_epsilon_scale_difference=difference)
    example = torch.tensor([[1., 3.], [2., -2.], [5., 1.], [9., -4.]])
    w = torch.softmax(torch.tensor([-2., -0.5, 0., 1.]), dim=0)[:, None]
    weighted = example * w
    actual_std = weighted.std(dim=0, correction=1)
    conventional_std = (w * (example - weighted.sum(0)).square()).sum(0).sqrt()
    difference = float((actual_std - conventional_std).abs().max())
    check("ASP_weighted_features_std_differs_from_weighted_variance", difference > 0.1,
          noise_disabled_to_isolate_formula=True, maximum_std_difference=difference)

    slope = p["posthoc_training"]["probability_calibration"]["slope"]
    intercept = p["posthoc_training"]["probability_calibration"]["intercept"]
    check("positive_calibration_rank_preservation", slope > 0 and
          torch.equal(torch.argsort(s), torch.argsort(slope * s + intercept)))
    check("reference_hash_consistency", p["reference_checkpoint_sha256"] ==
          "104873703a57cad98b686d770e0860e031a2cf9abef79b72668780c1660e0b24" and
          all(re.fullmatch(r"[a-f0-9]{64}", v["archive"])
              for v in audit["source_hashes"].values()))
    check("training_class_weights_and_length", p["original_training"]["class_weights_enabled"]
          and p["original_training"]["long_samples"] == 404 * 160 - 40
          and p["original_training"]["short_samples"] == 303 * 160 - 40
          and p["original_training"]["epochs"] <= p["original_training"]["T_0"])
    check("F1_condition_count", p["F1"]["main_score_conditions"] == 1 + 2 * 4 * (1 + 20)
          and p["F1"]["random_mask_seeds"] == list(range(200, 220)))
    core_audit_path = ROOT / "artifacts/geometry_cache/local_analysis/cached_core_v1/output_audit.json"
    core_audit = json.loads(core_audit_path.read_text(encoding="utf-8"))
    check("reused_diagnostic_audited", core_audit["status"] == "pass" and
          not p["reused_completed_diagnostic"]["same_as_new_spec_all_conditions"])
    plan = p['analysis_data_plan']
    check('equal_primary_evaluation_sizes',
          plan['default_evaluation'] == ['DF_pilot', '21LA_pilot'] and
          all(plan['source_data_counts'][name] == {'bonafide': 2000, 'spoof': 18000, 'total': 20000}
              for name in plan['default_evaluation']) and plan['unique_default_utterances'] == 80363)
    check('T_V_training_sizes_preserved',
          plan['source_data_counts']['T']['total'] == p['cache']['T_rows'] == 25380 and
          plan['source_data_counts']['V']['total'] == p['cache']['V_rows'] == 14983)
    lock_path = ROOT / plan['manifest_lock']
    sample_lock = json.loads(lock_path.read_text(encoding='utf-8'))
    check('sample_selection_metadata_only', sample_lock['uses_scores_or_embeddings'] is False and
          sample_lock['score_independent_selection'] is True and plan['resample_after_results'] is False)
    check('sample_source_fingerprints', all(
          item['source_cache_fingerprint'] == (p['cache']['DF_fingerprint'] if name == 'DF_small'
                                               else p['cache']['R21LA_fingerprint'])
          and sample_lock['views'][name]['checkpoint_sha256'] == p['reference_checkpoint_sha256']
          for name, item in plan['views'].items()))
    import pandas as pd
    view_frames = {}
    for name, item in plan['views'].items():
        path = ROOT / item['manifest']
        frame = pd.read_parquet(path)
        view_frames[name] = frame
        expected_counts = {1: 2000, 0: 18000} if name == '21LA_pilot' else {1: 200, 0: 1800}
        check(name + '_locked_hash', hashlib.sha256(path.read_bytes()).hexdigest() ==
              item['sha256'] == sample_lock['views'][name]['manifest_sha256'])
        check(name + '_counts_IDs_and_no_scores', frame.groupby('y').size().to_dict() == expected_counts and
              frame.utterance_id.nunique() == len(frame) == item['rows'] and
              not {'raw_score', 'cosine_s', 'raw_norm', 'tangent_norm'} & set(frame.columns))
    check('small_LA_nested_in_pilot', set(view_frames['21LA_small'].utterance_id) <=
          set(view_frames['21LA_pilot'].utterance_id))
    check('within_dataset_CV_separate', plan['within_dataset_CV']['pooled_40000_fit'] is False and
          plan['within_dataset_CV']['separate_fits'] is True)
    check('LA_default_vs_full_supplement', p['fit_routes']['transfer']['additional_evaluate'] == ['21LA_pilot']
          and p['fit_routes']['transfer']['final_supplementary_evaluate'] == ['21LA_eval'])
    broad = json.loads((HERE / 'ebm_embedding_geometry_protocol.json').read_text(encoding='utf-8'))
    override = broad['cached_analysis_data_size_policy']
    check('broad_spec_count_policy_matches', override['DF_pilot_rows'] == override['21LA_pilot_rows'] == 20000
          and override['small_panel_rows_each'] == 2000 and
          override['authoritative_config'] == PROTOCOL.name)
    text = MD.read_text(encoding="utf-8")
    links = []
    for target in re.findall(r"\]\(([^)]+)\)", text):
        if "://" in target or target.startswith("#"):
            continue
        path = (HERE / target.split("#", 1)[0]).resolve()
        check(f"document_link_{len(links)}", path.exists() or path == OUT,
              relative_target=target, generated_validation_output=(path == OUT))
        links.append(target)
    check("markdown_fences_balanced", sum(line.startswith("```") for line in text.splitlines()) % 2 == 0)
    check("document_key_contracts", all(s in text for s in (
          "use_class_weight", "param.require_grad", "cached_core_v1", "5,296", "source-family",
          "inconclusive_low_overlap", "W2=0", "1e-12", "후속 사용자 요청")))
    hashes = {str(path.relative_to(ROOT)).replace("\\", "/"):
              hashlib.sha256(path.read_bytes()).hexdigest()
              for path in (MD, PROTOCOL, AUDIT, Path(__file__), core_audit_path, lock_path)}
    report = {
        "status": "pass", "UTC": datetime.now(timezone.utc).isoformat(),
        "scope": "synthetic CPU mathematics, gradient paths, metadata sample locks and document/protocol consistency only",
        "torch_version": torch.__version__, "dtype": "float64", "device": "cpu", "seed": 1153,
        "real_embedding_training": False, "audio_inference": False, "original_model_loaded": False,
        "original_model_changed": False, "source_hashes": hashes,
        "check_count": len(results), "checks": results,
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "checks": len(results), "output": str(OUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
