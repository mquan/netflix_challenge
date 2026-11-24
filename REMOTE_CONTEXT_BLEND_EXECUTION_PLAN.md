# Remote context-blend execution plan (frozen before launch)

Branch `research/remote-context-blend-execution`, frozen 2026-11-24 before any
full-scale producer run or blend comparison. Handoff:
[REMOTE_CONTEXT_BLEND_HANDOFF.md](REMOTE_CONTEXT_BLEND_HANDOFF.md). Arrival inventory:
`experiments/remote-context-blend/start.json`. Machine-readable version:
`experiments/remote-context-blend/plan.json`. If the owner changes anything, the change
is recorded here as a dated amendment **before** the affected step runs.

## 1. New temporal residual predictor (`knn3d`)

**What it adds.** TX300r models user/item drift inside a factor model. Its residual
is still structured over *time within a user*: items a user rated close in time to
the query are more informative about the query's residual (sessions, mood/rater
drift, shared raters on one account) than items rated years apart. The upstream
`knn3` residual regression (BellKor NNLS item-item weights) ignores dates. `knn3d`
keeps `knn3` exactly and adds one date kernel

  g(Δ) = a + (1 − a)·exp(−|Δ| / τ),  Δ in days,  **a = 0.5, τ = 30** (predeclared)

applied consistently in both places: in each item's regression the neighbour
signals of user u are `g(t_uj − t_um)·err_uj` (and the same for the bias-corrected
stabiliser), and at prediction the neighbour residual is `g(t_uj − t_query)·err_uj`.
All other knobs are the `knn3` defaults (threshold 0.25, k 10–60, shrinkage 20000,
reg 0.01, x 0.8, baseline regs 25/10). a and τ were fixed before any full-scale
score; the 2%-user smoke run was a crash/sanity check only and no setting was
changed after it.

**Matched control.** `s3-tx300r__knn3` (unchanged upstream stage, same base, same
binary) isolates the date kernel from the residual-neighbourhood effect itself.
TX300r itself is the reference and is **not** retrained.

**Source.** Tree `artifacts/rcb-r1/source` = APFS clone of the completed weekend
tree r10 (same data, split, row map `ffc16cb88b14…`, TX300r native outputs).
Additions only: `src/knn3d.rs` (new), `pub mod knn3d;` in `src/lib.rs`, a `knn3d`
branch in `chain()` of `src/bin/fit.rs` (before `knn3`); `knn3.rs` untouched.
Hashes are in `plan.json`.

**Commands** (both phases, train_3→probe_3 then fulltrain→qual, full data,
qualifying targets are the native zeros):

```sh
cd artifacts/rcb-r1/source
env RAYON_NUM_THREADS=6 ./target/release/fit chain s3-tx300r__knn3  -p pipeline-3.toml --final --preds preds_3
env RAYON_NUM_THREADS=6 KD_FLOOR=0.5 KD_TAU=30 ./target/release/fit chain s3-tx300r__knn3d -p pipeline-3.toml --final --preds preds_3
```

`knn3`/`knn3d` are deterministic (no RNG). Threads affect only parallel
scheduling, not the arithmetic of a per-item solve.

## 2. Admitted prediction bank (fixed before comparison)

Only producers trained without any probex outcome and with complete probex
(1,128,164) and qualifying (2,817,131) vectors on this machine:

* own fresh `preds_new` producers (split new: trainx→probex), terminal vectors of
  every job with both files (`upstream_repro/fresh/preds_new`);
* split-3 producers projected to probex by the verified row map: remote15h
  (attn32, bknbrx-100w, dnn24, mfrbmx-60h, rbm200e, tx50n, tx60o), weekend
  (tx300r, rbm800p, rbm400pf-lr015 terminal), and the two new chains;
* CFNADE48 terminal (remote9h).

Excluded: epoch snapshots, the partial rbm400pf LR 0.02 run, any downloaded or
reference vector, and anything fitted on probex outcomes. The exact list and
hashes go into `experiments/remote-context-blend/bank.json` before the first fit.

## 3. Global vs context-dependent blend

Rows: probex, original five user folds (`upstream_exchange.user_folds`, seed
20261116). Inputs clipped to [0, 6]; output clipped to [1.05, 4.95].

* **Global**: ridge, λ = 1000, unpenalized intercept.
* **Context**: same columns plus context×column interactions. Contexts, each in
  4 quantile bins with cutoffs fitted on the training-fold rows only:
  user support (log count of the user's ratings in the training split: trainx for
  known, fulltrain for qual), movie popularity (log item count in the same
  split), recency (days from the user's last training-split rating to the rated
  date). Additive (not crossed) blocks; interaction coefficients are shrunk
  toward 0, i.e. toward the global weights, by λ_ctx, while the global part keeps
  λ = 1000. λ_ctx is chosen from {1e4, 3e4, 1e5, 3e5, 1e6} by an inner 4-fold user
  split inside each outer training fold; tie → larger λ_ctx.
* **Rule**: context gets the primary slot only if pooled known RMSE improves by
  ≥ 0.001 **and** at least 4 of 5 outer folds improve; otherwise the global blend
  is primary. Both are frozen and returned either way.
* Also reported under the global blend: bank without `knn3`/`knn3d`, with `knn3`
  only, with both — the incremental effect of the new predictor on identical rows.

The probex predictors exclude the entire known population, so every outer
comparison uses predictor features produced without that fold's outcomes.

## 4. Complementary work

* Repackage the completed RBM400 LR 0.015 native files (r11) with full arrays,
  trace, log and lineage; byte counts + SHA-256 per member and per part.
* Fresh full latent CatBoost arm only after measuring its input inventory and
  throughput here (no CatBoost in the upstream venv yet).

## 5. Return

Frozen primary + matched global control: config, coefficients, all 2,817,131
qualifying predictions, known vectors and fold receipts. Large files go to a
draft GitHub release (transfer only); git gets code, manifests and receipts.
No recovered labels are read on this machine.

## Amendment 2026-11-24 — context v2 (declared after v1, before running v2)

v1 is final and unchanged (global primary). v1's λ_ctx was 1e4, the grid floor,
in all 5 folds. v2 changes **only** the grid to {1e2, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5}
(tie → larger). Same bank, folds, contexts, bins, cutoffs, inner seeds and
primary rule (≥0.001 pooled and ≥4/5 folds vs the same global blend). Because v2
is chosen after seeing v1, a v2 win is reported as adaptive known-row evidence
and the coordinator decides whether it replaces the v1 primary.

## Owner decision 2026-11-24 (recorded before any further comparison)

1. **Context v2 is admitted as the primary** (owner: "yes"), overriding the 0.001
   bar for this bank: −0.000985 pooled, 5/5 folds. Global ridge stays frozen as the
   matched control. Qualifying vectors: `rcb-blend-v2.tar.gz` (`qual-context.f32`
   primary, `qual-global.f32` control).
2. Rerun both blends on the coordinator's own17 known/qualifying bank once it is
   available here.
3. Run more knn3d temporal chains on other temporal bases and the fresh full
   latent CatBoost arm.

## Amendment 2026-11-24 — knn3d wave (declared before launch)

Same `knn3d` (a = 0.5, τ = 30) and `knn3` control, unchanged binary, on every other
base with training predictions on this machine: split-3 tx50n, tx60o, rbm800p,
dnn24, attn32, rbm200e, bknbrx-100w, mfrbmx-60h, rbm400pf (LR 0.015 run); split-new
tsvdx4-60, tsvdx5-150o (`data/trainx` cloned from the fresh tree, byte-checked).
22 jobs, both phases, 3 concurrent × 4 threads (`knn3d/wave.sh`). All 22 columns
enter a bank v2; the v2 blends (global control, context v2 primary) are rerun
unchanged on bank v2 and reported as an adaptive follow-up.
