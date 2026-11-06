# Server baseline checkout

The server is on baseline/e50-bs240-lr1e5. main retains the B zero-padding experiment.
Source was restored from results/CM/LA/EBM_e50_bs240_lr1e5/backup/scripts.

Training/data/model code matches that archive except for the two online-codec
conditional blocks, which remain commented to honor the user's codec-off instruction.
The original fixed long/short repeat+crop path, class-specific loaders, original
two-stage update, local BPL, original evaluation and logger are restored.
No native microbatch helper, masks, new auxiliary losses, or length sampler is used.

Epoch50, global batch240, lr1e-5 and original experiment name are restored.
Source code restoration is not a new training run and does not restore/reload weights.
Existing result files and checkpoints have not been changed.

Important: the archived logger deletes an existing results directory of the same name.
Do not launch this restored configuration over the existing baseline results.
Use a new name for any explicitly requested rerun.

docs/CONTROL_B.md describes main, not this server branch. Do not pull main onto
this branch or restart an existing experiment automatically.
