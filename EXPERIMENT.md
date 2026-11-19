# Normal all-cross BPL only, with spoof-centered OC

This branch disables ONLY spoof BPL. The primary reference is dual all-cross BPL
(role_v3): spoof weight1->0, normal all-cross weight1 unchanged. Pair target1,
OC pull0.9/push0.2, alpha30/20, spoof-only center multiplier5, original240 loader,
augmentation, fresh seed1, two Adam phases and frozen frontend remain unchanged.

Use tools/train_role_experiment.py, not main.py. Configuration-only example:
python tools/train_role_experiment.py --spoof-bpl-weight 0 --spoof-bpl-pairing all_cross --bona-bpl-weight 1 --bona-bpl-pairing all_cross --pair-target-cosine 1 --r-pull .9 --eval-input front64600_wrap --final-eval-policy deferred_shared

Default mode validates and does not train. This task completed CPU regression and
actual two-GPU108loader/three-update smoke only; full50epoch training has NOT
started and no trained-model EER exists. Smoke epoch0 is not a trained checkpoint
and must never initialize the next fresh training run.

For subsequent fresh training, preserve50epochs, save/evaluate19LA every5epochs,
and defer21LA/DF until all10heads can share one frozen XLSR output per batch.
Keep both full long/short forwards to preserve dropout/ASP RNG. Normal BPL is
exact original EBM all-local long x short mean; no class reweighting or merged
pair denominator. Spoof short direct gradient is exactly0, bona short positive on
nonaligned test inputs, spoof long OC and spoof center remain active.

role_v6 records the changed spoof weight. Default weight1 preserves exact old
v1-v5 metadata and behavior. New checkpoints require metadata-aware loading via
role_lifecycle.load_scoring_head. Parameters/Adam/RNG, partial batches, asymmetric
DDP, backward-compatible checkpoint reading and score sign were verified.

See docs/experiments/ebm_bona_only_allcross_bpl_20261120.md for formulas and
limitations and experiments/bona_only_bpl_20261120 for configuration/checks.
After future training use finalize_bona_only.py prepare_shared/compare, not the
archived v5 finalize.py used only by the unmodified validation reference tests.
The new v6 finalizer preserves previous spoof_uid results and adds bona_only as
a separate column against six completed baseline variants. Four local/server
CPU integration checks validate ten-head manifests for21LA/DF, reject wrong roles,
exercise real v6 shared scoring/RNG, and check synthetic full-result aggregation.
Removing spoof BPL does not guarantee aligned shared-parameter gradients,
prevent collapse from spoof OC, or improve generalization/EER.
No weights, audio, per-utterance scores, credentials or SSH administration helpers.
