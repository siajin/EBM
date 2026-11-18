# Spoof center with normal all-cross BPL

This branch preserves the exact frozen source of the completed November16 run.
Use tools/train_role_experiment.py, not main.py, for this opt-in experiment.

Backend objective: original OC + spoof all-cross BPL + normal all-cross BPL.
Both class BPL means have coefficient1. Pair target1, OC pull0.9/push0.2,
alpha_pair30/alpha_oc20, spoof-only center multiplier5. Seed1, global batch240,
50epochs, original fixed64600/48440 views and independent RawBoost, codec off.
The architecture, two Adam phases and frozen XLSR are unchanged.

Example configuration check (does not launch):
python tools/train_role_experiment.py --bona-bpl-weight 1 --pair-target-cosine 1 --r-pull .9 --eval-input front64600_wrap

Training requires a new isolated root, the pinned frontend revision, a matching
two-GPU smoke receipt and --mode train. Preserve existing run directories.
The historical pipeline is included as execution documentation, not an instruction
to restart its already completed run. Model weights and datasets are not committed.

The implementation is recorded as role metadata v3. Retrospective EER minima were
19LA0.540058(e45),21LA4.006449(e40),DF2.764140(e15) percent, at different checkpoints.
Fixed50 was0.679810/4.347630/3.766431 percent. Use matched wrap inputs for comparison.
