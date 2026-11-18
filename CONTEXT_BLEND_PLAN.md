# Context-weighted residual blend

The independently verified own-model development result is 0.8604038556527669
on adaptive ProbeX user folds. The first full qualifying recipe remains frozen
and is still waiting for paired model outputs. Neither qualifying group has
established the target below 0.85725.

This separate experiment keeps the parent's 85 fresh rating predictors, 130
fresh source voting features, original five user folds, ridge penalty, nested
budget selection, scaling, seeds, optimizer and stopping rules. It adds a
rank-16 interaction between predictor signals and query context to the existing
two-layer residual MLP. The signal branch receives the mean, 62 original model
columns, 23 added rating columns and the fold-specific ridge prediction. The
context branch receives five query contexts and 130 voting features. Both output
heads initialize to zero. All parameters are freshly trained on known ProbeX
labels with scoring users excluded from every supervised fitting stage.

Only the neural family is retrained. The parent tree's existing out-of-fold
vector is reused, with its manifest, saved-model replay, all five fold
receipts and artifact hashes checked. No new tree training receipt is invented.
The sole primary candidate is the fixed equal average of that tree and the new
NN. The unchanged ridge and tree are reported as controls. New model checkpoints
must reproduce every saved fold prediction bitwise, and the final score and fixed
average must pass the existing independent verifier. The parent result is the
comparison threshold. There is no qualifying prediction or qualifying-label
access in this development script. A production adapter is warranted only if
the measured development result improves.

Run `scripts/upstream_context_blend.py` from the reproduction worktree using its
venv. It owns separate `artifacts/upstream-context-blend` and
`experiments/upstream-context-blend` directories and an exclusive lock. Prepared
inputs and the reused tree prediction are copied into the new experiment and
SHA256-bound; source caches are not modified. Code is frozen at preparation.

The new child runs at nice 10 with two Torch threads. Admission waits for native
training RSS at most 18 GiB. Its own process group plus all compiled native
training groups must stay below 25 GiB, with at least 20 GiB free disk. A guard
failure stops only this new child and preserves partial work for inspection;
it does not restart training or stop the existing native fits. The other queued
heavy Python experiments wait for the first qualifying audit or missing paired
models, so they cannot currently coincide with this development experiment.

Validation before launch: targeted tests exercise zero initial correction,
learning through the new context head, immutable input arrays, bitwise saved
model replay, changed-recipe rejection and feature-layout rejection. These are
synthetic execution checks, not evidence of competition quality or goal success.
