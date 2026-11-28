# Four-day autonomous plan, 2026-11-26 → 2026-11-30 (declared before any result below exists)

The owner is away for four days. `driver.sh` (tmux `cf-lanes:driver4d`) runs the queue below. Each step verifies,
releases to `remote-context-blend-20261124`, evaluates, commits and pushes. It has no decisions
left for a human except the pause switch: `touch experiments/remote-context-blend/driver4d/STOP` stops new launches.
Running jobs are never killed by the driver. Each launcher's memory guard (<8% free) stops only its own stage.

## Queue and expected timing (UTC)
| # | job | source | start | expected end | notes |
|---|---|---|---|---|---|
| 0 | `rtg_supp` nbstats (4800 tree) | pinned, data only | 00:45 Nov 26 | minutes–1 h | input for `knn-d`; started now on spare cores |
| 1 | `tsvdx4-4800lm` both phases | `run4800.sh` (running) | 03:47 Nov 25 | ≈21:30 Nov 27 | 71 min/epoch, single core, 34 GB |
| 1a | release base + all epoch snapshots, then parents (train-row preds, ifeat) | driver | on base exit | +1 h | published as soon as both phases verify (handoff) |
| 2 | `__knnf` + `__knns` (parallel) | `run4800.sh` | after 1 | +1 h | |
| 2a | `tsvdx4-4800lm_diff1` nbstats → `__knn-d` | driver | after 2 | +1–3 h | deferred job from the 4800 handoff, now that the primary package is measured |
| 3 | `tsvdx5-300` → `__nlpp_paropt` → `__nlpp` → `__nlpp__knn3` | `run_nlpp300.sh` | after 2 | ≈ Nov 28–29 | ETA quoted after first measured epochs |
| 4 | `tsvdx5-400` → paropt → nlpp → knn3 | `run_nlpp.sh tsvdx5-400` | after 3 | likely after Nov 30 | next item on the coordinator's screen list (0.001600 historical, 64.96 GB floor); runs alone |

Item 4 is the only addition beyond the two handoffs. It is the next-ranked missing package in
`MISSING_PREDICTOR_PROGRESS.md` that this machine has not produced. It uses the same pinned tree and binary, and its
outputs follow the same return rules. It is launched unconditionally after item 3, unless STOP exists.

## Measurements (same for every package)
`scripts/rcb_pkg_eval.py`: fixed native bank v2 (65 columns, known probex rows) as the control. Groups are added
cumulatively (4800lm: base → snapshots 3/13/21/24 → other epochs; then knnf+knns; then knn-d. NLPP: base → nlpp → knn3).
Global ridge λ=1000, 5 user folds (seed 20261116), outer-training rows only. Pooled and per-fold deltas. Every column is
checked for shape (1,128,164 / 2,817,131), float32 and finite values, and its sha256 and mtime are recorded. NLPP rows after
paropt are development diagnostics (paropt sees all probex labels). No qualifying or recovered labels are read; qual
targets in these trees are zeros. The coordinator does the recovered-group audit of the frozen vectors.

## Release assets (names fixed now)
`rcb-4800lm-base`, `rcb-4800lm-parents`, `rcb-4800lm-resid`, `rcb-4800lm-knnd`, `rcb-tsvdx5-300`,
`rcb-tsvdx5-300__nlpp_paropt`, `rcb-tsvdx5-300__nlpp`, `rcb-tsvdx5-300__nlpp__knn3` (and the same for 400). Each is a `.tar.gz`,
split into `.part-aa…` above 1.9 GB. Member hashes are in `driver4d/<name>.members.sha256`; asset sizes and hashes are in `<name>.assets.txt`.
A failed stage still has its partial outputs and logs released.

Owner approved item 4 (tsvdx5-400 chain) on 2026-11-26T04:52:42Z.

## Amendment 2026-11-28T04:45Z — extension queue (declared before these jobs run)
The NLPP300 chain finished about 1.5 days earlier than planned. Base, paropt, nlpp and knn3 all exited 0. Against bank v2:
+base −0.000025, +nlpp −0.000097, +knn3 −0.000126 (5/5; probex diagnostics after paropt). The 400 chain started
04:21Z Nov 28. To keep the machine busy through Nov 30, `ext.sh` (tmux `cf-lanes:ext`) adds the next resolved pinned
packages from the coordinator's screen ranking (`experiments/upstream-missing-screen`), in order, skipping the
unresolved `rbmx2-500__knn-w`:
* **E1 `rbmx2-500bp` → `__knnf`** (screen #6, −0.001359). This is an RBM family, so it is decorrelated from the Tx
  parents. Its memory is small, so it runs beside the 400 chain with 6 threads. Tree `artifacts/rcb-rbm500bp` (pin, data identical),
  `rbmx2-new` sha256 `9c34678f…1a03`.
* **E2 `tsvdx6-300`** (screen #10 bundle, −0.001280; columns: base + epochs 3/12/20). The source-derived floor is 50.17 GB, so it
  runs alone after the 400 chain ends. `tsvdx6-300-new` sha256 `d7a41ecc…baa4`.
The same return and measurement rules apply: release asset per package, members and hashes, and a bank v2 control with cumulative groups.
STOP is honoured before each launch.

## Amendment 2026-11-28T18:38Z — extension queue 2 (declared before these jobs run)
The 400 chain finished at 15:19Z (+knn3 −0.000161 vs bank v2, 5/5). rbmx2-500bp→knnf finished at 17:55Z (−0.000548, 5/5). The RBM family is
paying more than extra Tx capacity. tsvdx6-300 (E2) started at 17:57Z (≈12 min/epoch, 49 GB, ETA ≈03:30Z Nov 29). `ext2.sh` (tmux
`cf-lanes:ext2`) takes the next resolved, feasible items from the screen ranking. Both are small-memory RBMs, so they run beside tsvdx6-300:
* **E3 `rbmx2-500` → `__knnf` → `__knn3`** (bundle #10 −0.001317; knnf and knn3 are marked verified upstream). Eval groups base, knnf, knn3.
* **E4 `rbmx2-600`** (#24 rbmx2-600_ep27 −0.001100; 27 epochs, so ep27 is the terminal vector). Eval group base.
Skipped: `tsvdx5-1000` NLPP chain (#1). Extrapolating 300→53.5 GB and 400→67.2 GB puts it at about 150 GB, more than the 96 GB machine.
`rbmx2-800bp` (#16): the pinned source sets `n_epochs: 3`, but the pipeline keeps epochs 3/24/27, so the screened bundle is not reproducible
from the pin. `rbmx2-500__knn-w`/`__knn-m` are unresolved. Same rules as before: same tree and binary (`rbmx2-new` 9c34678f…1a03), STOP honoured,
one release asset per package, and a bank v2 control.
