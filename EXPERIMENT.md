# Spoof center with normal same-UID BPL

The only objective change from the parent all-cross model is normal pairing:
normal long/short views are aligned by the same dataset item and averaged over n
pairs. Spoof BPL remains all-cross, with pair target1 and OC pull0.9/push0.2.
Normal coefficient1 and separate class means, two Adam phases, model architecture,
seed1, global batch240 and original loader/augmentation are unchanged.
No gradient projection, norm cap or Adam gate is used. Metadata v4 records pairing;
the v1/v2/v3 checkpoint readers and default metadata remain supported.

Use tools/train_role_experiment.py, not main.py. Configuration-only example:
python tools/train_role_experiment.py --bona-bpl-weight 1 --bona-bpl-pairing same_uid --pair-target-cosine 1 --r-pull .9 --eval-input front64600_wrap --final-eval-policy deferred_shared

Run a new two-GPU smoke before fresh seed1 training. During smoke and training,
evaluation is19LA only, after each5epoch committed checkpoint. After50epochs,
the shared evaluator runs21LA/DF by extracting one frozen XLSR batch and passing
that same tensor to all10 saved heads. No persistent frontend feature files.
The training_complete marker precedes complete shared evaluation; do not confuse them.
Pinned frontend revision:1a640f32ac3e39899438a2931f9924c02f080a54.

Reproduction controls, mathematical/batch review and matched historical metrics
are under experiments/bona_uid_20261118. No weights, audio, private credentials,
or environment administration helpers are included. EER improvement is unproven.
