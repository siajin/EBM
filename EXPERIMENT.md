# Spoof same-UID BPL with original normal all-cross BPL

Use tools/train_role_experiment.py, not main.py. The frozen source records role_v5:
spoof long/short views align by the same dataset item; normal BPL is exactly the
original EBM BPL(alpha30), averaged over every local normal long x short pair,
including same and different UIDs. Each class mean has coefficient1.

python tools/train_role_experiment.py --spoof-bpl-pairing same_uid --bona-bpl-weight 1 --bona-bpl-pairing all_cross --pair-target-cosine 1 --r-pull .9 --eval-input front64600_wrap --final-eval-policy deferred_shared

Spoof-centered OC pull0.9/push0.2, alpha20, center multiplier5, two Adam phases,
original240 batching/augmentation/worker RNG and fresh seed1 are unchanged.
Use50epochs; save/evaluate19LA every5epochs; after training share each frozen
XLSR batch across10heads for21LA/DF. No persistent feature files.

Before training, CPU math/gradient/partial-batch/DDP/checkpoint checks pass,
and previous normal-allcross vs normal-UID class errors were analyzed using
completed scores only. Actual2GPU108loader/3step smoke is mandatory before
fresh training, with spoof108/60/108 and normal144/144/36pairs and both short
gradients positive. No EER improvement or absence of gradient conflict is claimed.
The original v1-v4 metadata/checkpoints remain readable.

See experiments/spoof_uid_20261119 and docs/experiments for exact configuration,
pretraining analysis, validation and reproduction controls. No weights, audio,
per-utterance scores or environment/credential administration tools are committed.
