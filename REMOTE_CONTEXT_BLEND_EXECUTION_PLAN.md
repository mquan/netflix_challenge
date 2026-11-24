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

## Amendment 2026-11-24 — local CatBoost tree-residual arm (declared before any full fit)

The full-context latent CatBoost arm is blocked here on its ~110 GiB data closure, which
has been requested from the coordinator. In the meantime, the same recipe runs on this
machine's bank v2: CPU CatBoost 1.2.10, RMSE, depth 8, lr 0.03, l2 10, border 64,
Bernoulli 0.8, rsm 0.9, SymmetricTree, thread_count 2, max 2000 with patience 100.
It keeps all 5 outer user folds and 3 seed slots (2026/7/13 + fold; +99 for the final
all-known fit). The inner split is user-index parity inside the outer-training users.
Features: clipped bank columns, the 3 contexts, and the global-ridge baseline as the
last feature. Target: the residual from that baseline. The inner baseline is refit on
inner-train rows. A slot that doesn't beat the zero correction keeps 0 iterations;
otherwise a fresh refit on all outer-training rows uses best_iteration+1.
Prediction = clip(baseline + mean of the 3 slot corrections). Code: `scripts/rcb_tree.py`
and `scripts/rcb_tree_combine.py`.
**Rule:** the tree arm replaces context v2 as primary only if it improves pooled known
RMSE by ≥ 0.001 over context v2 **and** improves ≥ 4/5 folds, on identical rows. Context
v2 and global stay frozen as comparisons.

## Amendment 2026-11-24 — own17 rerun (declared before any own17 fit)

Input: coordinator assets `own17-bank-known.npy` (sha `6c267209…`), `own17-bank-qual.npy` (`9fdf753b…`),
`own17-bank-columns.json` (`a6b172c8…`). All 17 per-column hashes were verified, the probex label hash
`5a554ca0…` matches ours, and per-column RMSE (≈0.859) confirms canonical row order.
**Caveat:** 16 of the 17 columns are user-fold OOF outputs of probex-fitted models (one is native-holdout).
Their non-held rows were fitted with outer-fold labels, so known-row CV here is **non-nested** and
optimistic for every arm. It is reported only as relative evidence; the recovered audit decides.
Runs: `rcb_blend.py` unchanged (v2 grid {1e2…1e5}, same folds, contexts, cutoffs, seeds, rule):
(a) bank `own17` (17 columns); (b) bank `own17x` = own17 + bank v2 (82 columns, incl. tx300r
knn3/knn3d and the wave). Contexts are the bank-v2 context files (same rows). No tree arm.

## Amendment 2026-11-24 — Lane 1: CF-NADE residual neighbourhoods (declared before any chain fit)

Coordinator note `coordinator-next-work-v2-20261124.md` (release asset). **Base: own CFNADE48** (remote9h, 160 hidden /
rank 48 / 95 epochs, not in the knn3d wave). Own CF96 vectors are **not on this machine**. Only the upstream reference
`cfnade-96*` vectors are here, and they are excluded. See the missing-input list in `lane1/README.md`.
Training-row predictions came from the epoch-95 checkpoint with `cfnade48-export` (`cfnade48-stage` with only
`save_train=true`, preds dir `preds_export48`, 0 training epochs). **Replay gate passed:** the re-predicted probex and qual
vectors are byte-identical to the saved terminal vectors (sha `353ecdbf…`, `9043511f…`). The exports are
`cfnade-48.trainx.npy` (99,352,343 rows, sha `a05150f4…`) and `cfnade-48.fulltrain.npy` (100,480,507 rows, `bb02c1a9…`).
All data/{trainx,probex,fulltrain,qual} arrays are byte-identical to the rcb-r1 tree.
Chains (split new trainx→probex and fulltrain→qual, both phases, same `fit` binary as the wave):
* `cfnade-48__knn3`: regression residual chain (reference), defaults;
* `cfnade-48__knn3d`: date kernel a=0.5, τ=30 (as in the wave);
* `cfnade-48__knn-25c`: cosine, `KN_K=25 KN_SCALE=2.5 KN_SIM=cosine KN_STAT0=rtg_supp KN_STAT2=cfnade-48_prod KN_TAU=0.05`;
* `cfnade-48__knn-25m`: MSE, `KN_K=25 KN_SCALE=7.5 KN_SIM=mse KN_STAT0=rtg_supp KN_STAT2=cfnade-48_diff2 KN_TAU=0.05`.
The 25c/25m knobs are the published split-1 `s1-als8-8__knn-25c/25m` recipes, unchanged; split-new and split-3 have no
knn-25 recipe. Stats: `fit nbstats {rtg_supp, cfnade-48_prod, cfnade-48_diff2}`.
**Comparison** on the native-held-out bank v2 (65 cols) with the same 5 user folds and global ridge λ=1000. Arms: bank v2
without CF48 (CF48 is already a column, so it is dropped), with the base only, base + knn3, base + knn3d, base + 25c,
base + 25m, and base + all. Every column is a native trainx→probex or train_3→probe_3 output, so no probex outcome enters
any base or residual fit. Standalone probex RMSE is also reported.

## Amendment 2026-11-24 — Lane 2: fresh TX300 on a second split (declared before launch)

**Split.** `split_3 20261124 2`, the same generator and K as split-3; only the generator seed differs. This writes
`data/{train,probe}_3s20261124` in tree `artifacts/rcb-l2/source`, an APFS clone of rcb-r1. The generator is verified:
at its default seed it reproduces `train_3` and `probe_3` byte for byte. The new probe has 1,760,494 rows; 307,627 of
them differ from probe_3. **Probex ⊂ probe_3s and probex ∩ train_3s = ∅**, with dates matching. Row map:
`artifacts/rcb-l2/probex-row-map-3s.npy`. Probex is therefore held out from every TX arm below, and from every other
bank column.
**Recipe.** `s3-tx300r` exactly (EPOCHS=24 F=300 LR_I=0.006 LR_U=0.0045 LR_W=0.001 LR_WD=0.0005 LR_Y=0.00075 NBR=100
REG_I=0.015 REG_U=0.06 REG_WD=0.002 RESET=12, EPSAVE=1) except **SEED=20261124** in both new arms. The `fit` binary is
the rcb-r1 build (sha 13cd19df…).
* **A**: the existing s3-tx300r (split-3, seed 42), not retrained.
* **B, the matched single-split control**: split-3, SEED=20261124, both phases (`--final`), preds `preds_3b`.
* **C, the second-split arm**: split-3s, SEED=20261124, known phase only (train_3s→probe_3s), preds `preds_3s`.
  Its fulltrain→qual phase would be B's phase-2 recipe on identical data, so **C's qual vector is B's**. This is
  declared now, and is the only way the known-side split effect can be paired with a qual vector.
**Comparison** on probex rows with the same 5 user folds, global ridge λ=1000, inner-fold fitting only:
(i) standalone B vs C; (ii) bank v2 + B vs bank v2 + C (single-split vs second-split, each with one new seed);
(iii) bank v2 + A + B vs bank v2 + A + C. B vs C differ only in the split.
Threads: 5 per arm (RAYON/OPENBLAS). ETA ≈ 8–10 h per phase, so C ≈ 9 h and B ≈ 18 h.

## Amendment 2026-11-24 — Own17x full-bank refit with CF-NADE residual columns (declared before any fit)

Source note: `coordinator-own17x-full-bank-followup-20261124.md`. The CF96 terminal pair (`own-cf96-terminal-pair-20261124.json`)
is verified: 4/4 sha256 match, float32, finite, native row counts; probex RMSE 0.89327, trainx 0.77561. It is
not an exact duplicate of any Own17x column (max corr 0.9936, with cfnade-48). Its probex vector is byte-identical to
the upstream reference `cfnade-96_ep95`; this is consistent with the coordinator's statement that our own CF96 run
replays the published recipe bitwise. The upstream file itself is not used.
**CF96 chains** use the lane-1 recipe unchanged with base `cfnade-96`: `knn3`, `knn3d` (a=0.5, τ=30), `knn-25c`, `knn-25m`,
plus stats `cfnade-96_prod` and `cfnade-96_diff2`; `rtg_supp` is reused (it depends only on the ratings).
**Arms**, all global ridge λ=1000, clip [0,6]→[1.05,4.95], 5 user folds seed 20261116, final fit on all known rows:

| arm | columns |
|---|---|
| X0 | Own17x control (82), recreated; must reproduce pooled 0.858340 and qual sha `717f9afa…` |
| X1 | X0 + 4 CF48 chains |
| X2 | X0 + CF96 terminal |
| X3 | X0 + CF96 + 4 CF96 chains |
| X4 | X0 + CF96 + 4 CF96 chains + 4 CF48 chains |

The native-held-out companion runs the same arms N0–N4 on bank v2 (65) in place of Own17x. Exact duplicate columns,
byte-identical in both the known and qual arrays, are dropped before fitting and reported; every original control
column is kept.
**Known-only selection rule:** the candidate is the lowest-pooled Own17x arm among X1–X4 that improves on X0 in ≥4/5
folds **and** whose native companion Nk improves on N0 in ≥4/5 folds. If none qualifies, X0 stays. Every arm's
qual vector is frozen once and returned. Own17x CV is non-nested (as before); the native companion is the
held-out evidence.
