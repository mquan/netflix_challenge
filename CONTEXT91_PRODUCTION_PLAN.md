# Verified context91 development best and qualifying production

The fixed tree91/contextNN91 mean scores **0.8599596977902322** on the same
1,128,164 known-ProbeX rows and462,858 users, improving the previous fixed
tree91/contextNN86 mean0.8599745622436025 by0.0000148644533703.
The lower overall comparable score is promoted without a minimum-gain or
all-fold veto. This is adaptive development evidence, not a qualifying score.

All five new neural folds passed saved-model replay. The original tree91
replay remains verified. A separate fixed-mean evaluator rehashed all ten
actual originating fold receipts and their model artifacts, matched row/fold
identities, checked the exact float32 mean, and independently scored it with
compensated summation. The two complete family scores are0.8606840430107859
for the tree and0.8600443942229535 for the context NN.

Evidence: `experiments/upstream-extension-context/{evaluation,nn-replay,independent-verification}.json`
and `experiments/upstream-tree91-context91/{plan,verification}.json`.

## Full qualifying production

Controller54968 uses the existing actual paired91 bank. Both families have
167 ridge columns and227 features plus their ridge prediction, with the same
full-known ridge verified byte-for-byte against the original full tree91.
The bank uses declared own retrains; some vectors differ from cached
development, as already quantified by paired-source verification.

The full-fit tree91 is reused only after exact bank, source, ridge, recipe,
version, seed, budget, artifact and saved-replay checks. Its actual median
tree budgets remain873/746/855. This run does not claim another tree refit.
Its source models are replayed again in a separate process before publishing
the new tree reuse replay receipt.

The context NN91 is newly fitted on all known ProbeX labels. The original
rank16/64-hidden architecture, three seeds, optimizer and scaling remain
unchanged. Budgets20/21/19 are per-seed medians of the five actual new neural
fold receipts, not inferred or fabricated receipts. Replay its saved full
models before freezing the equal float32 family mean.

The shared source fitting lock serializes this production with the next102
context fit. If a stronger verified102 development result exists before
production acquires the lock, the older production candidate is skipped.
All existing native source training jobs continue.

Qualifying outcomes do not select inputs, budgets or weights. Freeze the final
vector and replay both families before the recovered-group audit. Independently
verify canonical recovered labels, row counts and compensated group scores
afterward. The latest checked qualifying result remains0.8614776640541453 /
0.8624021195492101; both still miss0.85725. No new qualifying result is claimed
until its audit completes. These are adaptive recovered labels, not official
or untouched Netflix Test results.
