# RBM400 learned-context follow-up

Status: design only. No production declaration, queue, model extraction or
training has started for this follow-up. The existing RBM400 and own16 plans,
source files, numerical budgets and execution order stay unchanged.

The current full-trajectory neural study improved development CV by only
0.00005031. The frozen own15 recovered audit also shows only a small gain,
with its separate canonical check still pending at this plan's creation.
Adding more correlated prediction columns is unlikely to close the remaining
roughly0.0041 recovered-RMSE gap alone. This experiment tests whether a fresh
model's learned representation can supply useful conditional information to
the existing neural blend. It makes no target-achievement promise.

## Authoritative predictor contract

The existing factored-RBM worker saves two complete terminal predictors,
`known.pt` and `qualifying.pt`, after the full15 epochs of their respective
phases. The frozen terminal format preserves all seven tensors
`p,q,d,bv,bh,bu,but`; it has no resumable optimizer state. It does not retain
per-epoch predictive checkpoints. This follow-up must not assume checkpoint
trajectories exist, restart training or change the producer to create them.

The checked source contracts are
`scripts/upstream_factored_rbm_terminal.py`,
`scripts/upstream_factored_rbm_prediction.py` and
`scripts/upstream_factored_rbm_worker.py`. The model has400 hidden units and
50 factors. Its prediction routine returns the bipolar `hidden_expected`
posterior and uses `hidden_expected @ q.T` in its category logits. All
conditional-item addresses and empty-history behavior must remain exact.

Only after the original complete paired-model/source/data/epoch/query replay
proof and the original own16 completion may a separately declared successor
extract features. Both terminal payloads must pass the original terminal
validator, retain their original hashes and reproduce every original native
query bit under the admitted MPS predictor. No partial predictor is eligible.

## Fixed representation experiment

Extract50 user factors `U = hidden_expected @ q.T` using the original support,
conditional coordinates, user batches500 and query mapping. Retain the
posterior behavior for empty histories. Do not substitute zero user factors.
Cache only the complete query-user coverage needed by each original phase,
with an explicit coverage mask and canonical user IDs.

For every canonical movie, derive50 item factors from the fixed centered
rating contrast of `p[:,rating,:]`, with category weights
`[-2,-1,0,1,2]/sqrt(10)`. This contrast uses no query outcomes.
The150 new context coordinates are `U`, the item contrast `I`, and `U*I`.
Preserve all225 existing context positions and every existing operator bit.

The known-phase coordinates remain unchanged. Align qualifying-phase item
coordinates to the known basis by the fixed label-free ridge map
`A = solve(Iq.T@Iq + 1e-6*identity(50), Iq.T@Ik)` using all shared movie IDs.
Apply `Iq_aligned = Iq@A` and `Uq_aligned = Uq@inverse(A).T` to the qualifying
contexts only. Require finite matrices, full rank and condition number at
most1e6; reject and preserve the attempt if those checks fail. Do not tune
alignment against recovered scores. Verify contrast-dot invariance and
independently reproduce every feature bit under the declared arithmetic.
This is a proposed construction requiring CPU tests and full data validation,
not an already validated feature provider.

Append the one completed RBM400 prediction pair to the current486 raw streams,
subject to complete paired-alias checks before any removal. It is a terminal
pair, not a promised set of epoch predictions. Keep the150 new coordinates in
the nonlinear operator; the scalar ridge gains only the new raw prediction.
Maximum dimensions are487 raw predictors,563 ridge columns,863 nonlinear
columns and864 neural inputs. Exact scalar aliases may reduce those dimensions
mechanically; they must not reduce original contexts or declared seed slots.

## Known-only comparison and subsequent audit

Use two declared H128/rank32 arms: the scalar-only extension (up to714 inputs,
225 contexts) and the scalar-plus-latent extension (up to864 inputs,
375 contexts). Keep the full five held-user folds, three original seeds,
40-epoch maximum, patience4, batch8192, optimizer and original inner-selection/
fresh outer-refit semantics. Preserve every actual model and any explicitly
selected zero-iteration slot; never invent extra epochs or silently drop a slot.
Do not feed previous nonlinear OOF predictions into the new feature operator.

Select the lower complete separately canonical known fixed-half-with-tree154
CV score before any new qualifying outcome is read. Ties within1e-12 choose
the scalar-only arm. Retain both results and the tree's historical A/B caveat.
Complete the selected arm's full qualifying bag and all actual-model/operator
replays even if neither arm sets a development record.

Then add that provider to all16 original own16 families, preserving their
order. Predeclare a17-family known-only simplex primary, ridge10 with18
equations/unpenalized intercept, and the original new-neural/tree fixed-half
control. Freeze all six known/query vectors and independently check all
131071 simplex supports, actual KKT/objective conditions, complete ridge
equations and every arithmetic bit before auditing the three fixed candidates.
Apparent known calibration is not nested CV. No recovered outcome may select
an arm, coefficient, family, alignment or epoch.

## Admission and acceptance

This needs separate implementation, source declarations, CPU rejection and
independent numerical tests, review, and original-lane supervised queues before
execution. Keep all original local limits:3GiB data worker,16GiB neural/bag
worker,24GiB total in both physical/RSS domains, compressor below5GiB
(4.5GiB admission),128MiB swap growth from one original baseline per queue,
18.75GiB disk floor plus complete-output reservations. Preserve the same
process on pressure, inherited lane ownership, live watchdog and full budgets.

Two phase-specific user-factor caches require at most about200MB combined,
plus small item/alignment caches; use virtual query-row features rather than
another full matrix. Recheck actual free space before declaration. Retain all
current own models, checkpoints, predictions, data and working state.

Required evidence includes the complete original RBM400 paired proof; full
user/item/query coverage; raw prediction replay; latent/alignment invariants;
all inherited operator bits and canonical held-user folds; every actual model,
eligible scaler, normalized input and output; the selected full query bag;
the17-family label-free preaudit; and separate compensated recovered scores.
Only a predeclared primary strictly below0.85725 on BOTH full groups can satisfy
the goal. Community recovered audits remain adaptive, not official or untouched
Test. If own16 already achieves the goal, this additional study is optional.
