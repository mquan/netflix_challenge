# RBM400 learned-context follow-up

Status: array-only mathematical prototype implemented with eight CPU checks;
read-only extraction core implemented with twelve synthetic CPU checks.
Array-only operator and neural factories add seven and four CPU checks.
Authenticated read-only cache IO adds nine CPU checks. The complete-source
model adapter adds six, the owned extraction supervisor eight, and extraction/
alignment/replay orchestration seven, for61 combined CPU checks. The separately
reviewed supervisor can wait for complete original producers; actual model
extraction, actual cache proofs and neural studies have not yet run. Its
declaration must bind the reviewed protocol and every source hash before
allocating its namespace. Existing frozen producers and own16 plans, source
files, numerical budgets and execution order stay unchanged.

The current full-trajectory neural study improved development CV by only
0.00005031. The frozen own15 recovered audit also shows only a small gain,
with separate canonical verification now complete at0.8603412617/0.8613423892.
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
`scripts/upstream_factored_rbm_terminal_metal.py`,
`scripts/upstream_factored_rbm_prediction_metal.py` and
`scripts/upstream_factored_rbm_worker_metal.py`. Worker/training/terminal ASTs
match the originals under declared imports. The active producer is the reviewed
deterministic-Metal successor, with matching own16 consumer; all superseded
metadata and failed-preflight attempts remain explicit. Its ordered single-writer
reduction passes the actual full MPS device/reference/repeat checks. The model has400 hidden units and
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

The mathematical prototype is `scripts/upstream_rbm400_latent_context_math.py`.
Eight independent CPU checks cover all50factor contrasts, full alignment normal
equations and inverse maps, dot-product invariance after float32 cache casting,
rank/condition rejection, exact150coordinate row order and duplicates, explicit
coverage without zero substitution, invalid/nonfinite addresses or state and
finite inputs whose float32 cache casts overflow.
Item contrast accumulates float64 categories in fixed order then stores float32;
alignment solves in float64 and stores aligned caches in float32. The future
full-provider verifier must check the declared arithmetic on every real factor
and query, including measured casting error; these CPU checks do not admit an
actual terminal model or establish MPS inference/feature-bit agreement.

The read-only extraction core is `scripts/upstream_rbm400_latent_extraction.py`.
It accepts only complete original400/50/15 terminal records and an external
ownership/resource guard. It replays every native prediction bit before
returning query-user posterior factors, centered item factors and an explicit
coverage mask. Unqueried users remain NaN/uncovered; cold queried users retain
their original bias/conditional posterior. No model tensor or terminal RNG is
changed. Twelve CPU tests use an actually trained full-width synthetic terminal
model, including independent scalar posterior checks for all users, exact bit
mismatch rejection (literal uint32 bits, including a signed-zero regression),
source-row ordering/bounds/uniqueness, query-outcome access
rejection and unchanged model/RNG checks. CPU fixtures explicitly do not admit
an actual MPS provider. Production model loading, full source/data binding,
owned extraction and independent real-data replay now have separate source
implementations; actual real-data execution remains pending.

`scripts/upstream_rbm400_latent_cache.py` authenticates a separately frozen
manifest, all eight cache arrays and their source/model/proof references. It
maps the authenticated file descriptors read-only, requires exact shapes,
dtypes and payload sizes, rejects path escapes and symlinks, checks masks and
dual maps, and rehashes the closure even when the caller raises. Nine synthetic
IO tests cover those guarantees. These checks authenticate declared bytes;
the caller must still establish scientific provenance, actual model replay
and original-lane ownership. This module cannot admit a production provider.

`upstream_rbm400_latent_provider.py` requires the complete original paired
model/source/epoch/native-query proofs and finished own16 canonical lifecycle,
without using recovered numerical scores for extraction decisions. It rehashes
the complete frozen source closure and decodes the authenticated terminal file
descriptor read-only. Six CPU fixtures check lifecycle, environment and FD
identity contracts; the production adapter has no CPU mode or fit operation.

`upstream_rbm400_latent_execution.py` and its worker implement five owned
phases: known extraction, qualifying extraction/alignment, two fresh independent
full-phase cache/native/query-feature replays, then paired assembly. Eight
controller checks retain the original resource boundaries, same-frame pause,
two lane locks and watchdog behavior, and reject missing/changed review proofs.
Seven worker checks include a complete scaled synthetic paired cache workflow,
reopening both caches and independently reconstructing all query feature bits,
then rejecting tampered payloads. Device flags in that IO fixture are explicitly
mocked; it establishes no actual MPS or real-provider acceptance.

Only final known and aligned qualifying caches are stored; raw qualifying
posterior factors are temporary RAM values, so the two user caches stay about
200MB combined. Both actual terminal models are reopened by separate replay
processes. The real verifier must compare every cache/map bit, every native
prediction and every native query feature. It records a per-query bound from
the dual-map float64 residual, float32 cache casts and fixed float64 roundoff;
this measured invariance check supplements the mandatory exact cache-bit check.
No cache or completed extraction proof alone admits a trained neural provider.

The operator prototype `scripts/upstream_rbm400_latent_operators.py` keeps the
entire562-column design,712-column nonlinear parent and all original context
bits, retaining the final intercept. The scalar arm adds only its completely
paired-admitted prediction; the latent arm appends all150 new coordinates only
to the nonlinear design. Source admission, alias decisions and arm selection
are outside this pure component. Seven full-width tests check inherited uint
bits, raw scalar centering/clipping, intercept, both arms, scalar-alias cases,
complete coordinate partitions and rejection of missing/nonfinite inputs.

The factory prototype `scripts/upstream_rbm400_latent_neural.py` preserves all
original H128/rank32 optimizer/seeds/40epoch/patience4/batch8192 settings and
the zero-residual initialization. Four CPU construction checks cover every
alias/arm width, exact original parameter initialization when the scalar arm
aliases the parent, complete input/context buffers and rejection of altered
recipes. Only declared architecture identities and input/context widths differ.
This does not prepare real data, fit neural models or establish MPS model replay.

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
execution. Keep all original local limits:3GiB CPU data worker,8GiB actual-MPS
extraction worker (the original GPU cap),16GiB neural/bag
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
