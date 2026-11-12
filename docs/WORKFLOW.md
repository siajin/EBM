# Private EBM workspace

This repository contains the current 4.0375초 상한 비반복 학습 control condition. See CONTROL_B.md.
The original project and its license are retained; upstream is kimho1wq/EBM.

- origin is the private siajin/EBM repository. Work on branches in this repository.
- The server training checkout is /home/mentee/code/EBM.
- Keep data, model weights, generated results, local backups, and credentials out of Git.
- Never stop or restart a running experiment while updating source without the user's request.
- GPU work uses the existing container and CUDA_VISIBLE_DEVICES=0,1.
- Check the active container, experiment, and GPU state before launching anything.
- Use a new experiment name; do not overwrite an existing results directory.
- Pull only with a clean working tree, and use git pull --ff-only.
- 4.0375초 상한 비반복 학습 has not completed full training. The existing tests are bounded correctness checks.

Code changes should be committed and pushed to origin. Then update the other checkout
with git pull --ff-only before starting a new, explicitly requested experiment.
