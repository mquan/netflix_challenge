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

## Partial observation — 2026-11-18 22:07 UTC

Fold0 completed all three fresh neural seeds. Saved-model replay is bitwise
identical on 226,746 held-out rows. Independent compensated RMSE:
new NN 0.8593575065417106, parent NN 0.8595695005755923;
new fixed half 0.8592903561212312, parent fixed half
0.8594974065232871. This single fold does not establish the
complete candidate score or qualifying success. Fold1 is training; retain all
five folds and the fixed recipe before deciding on the candidate.

The partial verifier converts predictions to float64 before clipping, matching
the established scorer. A regression test covers float32 values immediately
outside both exact clipping limits; all three targeted context tests pass.

Fold1 also completed and independently replayed bitwise on 225,899 rows.
Its fixed-half RMSE is 0.8576817970921646, versus the parent
0.8578275582131777. Both completed folds improve; the unchanged
full five-fold comparison is still required. Fold2 is training.

Fold2 completed and independently replayed bitwise on 224,006 rows.
Fixed-half RMSE 0.8613082334525111, parent 0.8614397078251417.
All three completed folds improve; folds3/4 and final verification are pending.
