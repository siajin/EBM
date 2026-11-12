<a id="c-full-long-training-context-fixed-b-evaluation"></a>

# 원음 전체 비반복 학습: full long training context, fixed 4.0375초 상한 비반복 학습 evaluation

Parent: 4.0375초 상한 비반복 학습 commit `2c2c6ff65bbadf33b811983f7c648c97fffb81a1`.
Name: `EBM_e50_bs240_lr1e5_C_nativeLong_short303`.

The experimental factor is removal of the 64,600-sample training long-view cap.
Short remains a random continuous crop of at most 48,440 samples (3.0275 s).
Short utterances are never repeated. Right zero-padding is masked in frontend
and backend. Both auxiliary-loss coefficients stay zero; no wrapped view is
created. Original long OC + local bonafide long x short BPL, two Adam updates,
global batch240, rank120=12 bona+108 spoof, physical8, seed1, epoch50 and lr1e-5
remain unchanged. XLS-R is frozen. Codec calls and ACN stay disabled; the same
RawBoost policy is retained. Training starts from the same seeded initialization,
not from 4.0375초 상한 비반복 학습's trained checkpoint.

## Minimal runtime changes

- arguments.py: experiment name and training cap_long=False (plus its comment).
- data_loaders.py: evaluation loader explicitly retains cap_long=True.

The shared cap previously also controlled evaluation. Leaving that coupled would
change inference context and training together. Both in-training dev and future
target loaders now retain 4.0375초 상한 비반복 학습's front64,600 cap, no repetition, masks and batch8.
For 4.0375초 상한 비반복 학습's cap_long=True configuration, this loader change is behaviorally a no-op.
Direct EnrollmentSet users can still request uncapped input explicitly.
Model, losses, sampler, optimizer, gradient replay, logger and update code are
byte-identical to 4.0375초 상한 비반복 학습. No performance-dependent code tuning or checkpoint choice.

<a id="interpretation-and-analysis-fixed-before-c-training"></a>

## Interpretation and analysis fixed before 원음 전체 비반복 학습 training

Compare fixed epoch50 4.0375초 상한 비반복 학습 대 원음 전체 비반복 학습 with the same 4.0375초 상한 비반복 학습 cap/mask/batch8 evaluation, pooling
noise1e-5 and common RNG. Choose each model's operating threshold by the same
19LA-dev rule used in 4.0375초 상한 비반복 학습 (min |FPR-FNR|, lower FPR, higher threshold on ties).
Use the exact same dev/19LA/21LA/21DF trial lists, and original-length bins
[0,2), [2,3.0275), [3.0275,4.0375), [4.0375,infinity) seconds. Include counts,
normal rejection, spoof acceptance and paired speaker-cluster uncertainty.
Target scores never select an epoch, threshold or input policy.

This identifies the observed difference for full-long-context training under
the retained policy and seed. Longer input also changes content, augmentation
realizations, temporal pooling and stochastic RNG consumption. Same seed and
policy do not imply identical random tensors across different input shapes.
Do not call one trained pair a general causal proof or a pure duration effect.
Do not use the historical240 checkpoint as the controlled comparison for 원음 전체 비반복 학습.
Uncapped inference, if studied later, must be a separately labeled intervention.

## Before launch

Verify original-length metadata, 4.0375초 상한 비반복 학습과 원음 전체 비반복 학습 sampler UID schedules, unchanged <=64,600
input behavior, short-crop/augmentation selection, fixed evaluation input,
codec/wrap calls absent, finite objectives, full 12x12 BPL and two updates.
Measure real longest-input GPU memory on both existing assigned GPUs with the
same physical8. Never lower physical batch or alter the sampler in 원음 전체 비반복 학습 alone.
Use a new branch/name and retain 4.0375초 상한 비반복 학습 source, results and checkpoint hashes.
