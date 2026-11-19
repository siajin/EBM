# In-the-Wild shared-XLSR evaluation

This branch preserves the completed ITW evaluation of original240, spoof-sameUID
with normal all-cross BPL, and spoof-allcross with normal all-cross BPL.
No training weights, audio, utterance-level scores, dataset metadata or credentials
are included. Source/configuration and aggregate verification/results are included.

All runs use the same31779 verified16kHz mono recordings and wrap/front64600,
no mask, FP32, batch8 per GPU, pinned frozen XLSR300M revision
1a640f32ac3e39899438a2931f9924c02f080a54, and common matched_wrap:ITW seed1729
with ASP1e-5. Each batch has exactly one frontend call, shared by every head.
Legacy240 uses positive bona-center cosine, spoof-centered roles use negative
spoof-center cosine. No classifier is adapted or tuned on ITW.

Experiments contain two immutable execution configurations:18heads across two
families, followed by10heads of dual all-cross BPL. The latter reuses verified
audio and exact metadata order; the prior scores remain unchanged. No persistent
XLSR feature tensors were stored, so the additional family requires one fresh
frontend pass per batch, shared across its ten epochs.

Run unit tests before the51-record/two-GPU/full-and-partial/two-class smoke, then
evaluate all31779 records. The evaluator accepts ROOT and smoke/full as arguments.
Create manifest.json with source/source_root/models/dataset_manifest_sha256,
frontend_revision and fixed zero optimizer_updates/feature_files/new_spending_usd/
automatic_retries. Preserve the existing dataset_manifest.json and acceptance
format. Manifest paths and example pipeline paths are environment-specific and
must be configured for a NEW run directory before freezing control_manifest.json.
Do not edit or rerun any completed execution directory.

The six CPU tests cover original B_Linear equivalence, strict weights, role signs,
RNG/order independence, short wrapping, independent ROC, and immutability.
Actual GPU smoke reordered heads on identical hidden tensors with exact score
agreement. Every full evaluation checks UID/index/label, finite scores, independent
EER, call counters and source/model/control/data/score hashes. compare_prior.py
also verifies old results unchanged and exact cross-run UID/index/label ordering.

Observed ITW minima are retrospective target selections. The original240 has8
retained epochs15..50; the other families each have10epochs5..50. Full curves,
common-range minima and fixed50 comparisons accompany aggregate results.
