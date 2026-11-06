# Private EBM workspace

This server branch restores the archived batch240 baseline. See SERVER_BASELINE.md.
The B control condition remains on main; CONTROL_B.md describes main.
The original project and its license are retained; upstream is kimho1wq/EBM.

- origin is the private siajin/EBM repository. Work on branches in this repository.
- The server training checkout is /home/mentee/code/EBM.
- Keep data, model weights, generated results, local backups, and credentials out of Git.
- Never stop or restart a running experiment while updating source without the user's request.
- GPU work uses the existing container and CUDA_VISIBLE_DEVICES=0,1.
- Check the active container, experiment, and GPU state before launching anything.
- Use a new experiment name; do not overwrite an existing results directory.
- Pull only with a clean working tree, and use git pull --ff-only.
- B has not completed full training. The existing tests are bounded correctness checks.

Code changes should be committed and pushed to origin. Then update the other checkout
with git pull --ff-only before starting a new, explicitly requested experiment.
