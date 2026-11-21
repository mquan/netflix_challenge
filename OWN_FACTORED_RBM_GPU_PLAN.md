# Local factored bipolar RBM: separate GPU implementation

## Purpose and current boundary

Develop another freshly trained RBM family locally while the remote RBM
processes remain guard-stopped in their last pinned checkpoint. The original
local1200 training, audit continuation, CF intake, trajectory learning and all
remote-return requirements remain unchanged. No replacement native model has
been launched. No GPU kernel, production-data fit or accuracy claim is complete.

The first implementation work is a NumPy mathematical oracle and a tiled Torch
kernel. Eight CPU checks pass: literal per-user CD-3 gradients versus the batch
algebra; float64/float32 Torch states, gradients and updates; row tiling; bipolar
expectations; exact categorical boundaries; bias conservation; actual partial
batch scaling; momentum and decay on untouched shared parameters; and coordinate
ownership/conditional-deduplication checks in the oracle. These synthetic checks
read no production arrays or recovered outcomes. They do not establish actual
MPS support or throughput, native f32/RNG bit identity, or training accuracy.

## Source recipe and mathematical requirements

Reference: pinned original `src/rx.rs`, `src/bin/fit.rs`, `src/lib.rs` and the
`s3-rbm400pf` declaration in `pipeline-3.toml`. The proposed first production
study keeps400 bipolar hidden units, internal rank50,15 epochs for each of
train_3→probe_3 and fulltrain→qual, seed128, batch500 nonempty users, LR.02,
momentum.9, decay.001, LR_BU.001, LR_BUT.0005, bias decay.01, conditional
train/query item addresses, temperature1 and LR_BIF0. CD steps increase every
five epochs from1 through3 during the15-epoch run. No inferred reduction of the
remote recipe or its67-column return requirement is implied.

The new implementation is a separate numerical realization. Torch initialization,
random streams, matrix products and indexed reductions need not reproduce Rust
StdRng or its sequential f32 reductions. Give its artifacts a separate producer
name; never substitute them under a native producer's receipt or input path.
Only freshly initialized own parameters and raw known training ratings may enter.

Preserve the following source semantics:

- Hidden expectation is `2*sigmoid(a)-1`; sampling is ±1 with probability
  `sigmoid(a)`. It is not `tanh(a)`.
- For factored weights, rating-category factors are summed per user before
  projection by Q. Conditional D uses the deduplicated union of training and
  query item addresses, without query outcomes.
- Sample all five softmax categories at every CD step. Preserve the native
  `<=` categorical boundary and its last-category fallback in fixed-draw tests.
- User and training-day visible bias gradients are unaveraged. Their updates
  commute across distinct users in a batch. Shared P,Q,D,bv,bh remain fixed
  while the batch's gradients are accumulated.
- Shared learning rate is `.02 / actual_nonempty_batch_users`, including the
  last partial batch. Momentum and appropriate decay update every shared
  parameter, including untouched items. Bias slots use their own rates/decays.
- Every training-day slot of the selected users receives its momentum/decay
  update. A query on an unseen training day uses the user bias alone.
- Include all ratings and conditionals. Row tiles bound temporaries, not user
  batch size, rating histories, features, CD steps or epochs. Empty-history
  prediction must retain the source's hidden-bias/conditional computation.
- LR_BIF0 leaves zero frequency-bin biases zero. A later nonzero-LR_BIF arm
  would require its own implementation and tests because its per-example shared
  updates do not commute across users.

## Required implementation and admission before real fitting

1. Independently verify source/data/split/map bindings against the existing
   own split3 producer. Preserve each native row identity, raw rating and
   per-user rating order; exclude training `is_test` rows from positive states.
   Build compact user/day CSR metadata with full independent replay. Derive
   conditional unions from addresses, including masked/query addresses where
   the source does, while keeping their labels inaccessible. Verify the native
   train_3/known holdout disjointness and both known-row projections. Fulltrain
   is the ordinary final-fit input; qualifying target arrays must remain zero.
2. Implement full parameter initialization, the epoch/user shuffle and CD
   schedule, all-day bias storage, bounded transfers and query prediction. Bind
   initialization/RNG semantics explicitly as a new realization. Serialize the
   complete actual terminal model, biases, data bindings and prediction recipe
   for each phase. Intermediate checkpoints must be atomically retained or
   retired under an explicit restart protocol; no silent restart after failure.
3. Run the synthetic fixed-draw replay on actual MPS after acquiring the real
   GPU lane. Check all states/gradients/updates against the independent scalar
   reference, documenting f32 tolerances and ambiguous draw boundaries.
   Prove bounded temporaries on a synthetic heavy-user batch and all-day
   coverage. CPU checks alone do not admit this arm.
4. Before launch, compute real peak model/momentum/CSR/bias/output/checkpoint
   bytes and reserve disk for both phase models, prediction pairs and proofs
   above18.75GiB. Measure an actual full-data batch without shortening the
   declared15+15 experiment. Do not assume a GPU speedup or sufficient storage.
5. Production execution follows original local1200 and the existing local
   own15 completion. Acquire both existing primary and GPU lanes; respect
   earlier already-admitted work. Use original physical/RSS/compressor/swap/disk
   policy, a baseline established once, watchdog-before-dispatch and RAM
   preservation on guards. No new production queue is launched by this plan.
   No GPU work overlaps the current native owner holding that lane.

## Declared first candidate and independent verification

This new study's first candidate is the terminal prediction pair from the full
15+15 run. It is not a partial return of the remote epoch bank. Save all epoch
diagnostics as training provenance; qualifying-zero errors are not accuracy.
Retain both complete actual terminal models for independent replay of every
native query prediction, mapping and conditional support. Define acceptable
GPU/CPU numeric comparison tolerances before any recovered-label audit; store
the exact production f32 prediction vectors and independently check their
deterministic same-device replay. Investigate unsupported or nondeterministic
operations before claiming independent verification.

Only after those checks, add the distinctly named pair to a separately declared
known-label blend study with complete input/model provenance. Freeze its input
list, locally fitted coefficients and full vectors before reading recovered
outcomes. Recovered-group scores remain adaptive audits, never official or
untouched Test results. Neither the kernel checks nor this plan changes the
current best score or satisfies the below.85725/both-groups objective.

## Files

- `scripts/upstream_factored_rbm_reference.py`: small float64 mathematical oracle.
- `scripts/test_upstream_factored_rbm_reference.py`: separate scalar user loops
  and analytical invariants.
- `scripts/upstream_factored_rbm_torch.py`: lazy-imported tiled Torch gradients
  and updates, without a fitter or RNG.
- `scripts/test_upstream_factored_rbm_torch.py`: CPU-only f64/f32 comparisons.
