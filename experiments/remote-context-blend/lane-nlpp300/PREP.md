# tsvdx5-300 → NLPP → KNN3: preparation (reply to REMOTE_NLPP300_HANDOFF.md), 2026-11-25

## Source, build, data
* Tree `artifacts/rcb-nlpp300/source`: APFS clone of the 4800lm tree at pin **49c9b3676e6bab90fdf77e9a2da984edbfe4fe5c**.
  `src/`, `Cargo.*`, `pipeline-new.toml` are clean against the pin. It does not copy `preds_new`, `sim` or `data/qual_ratings`.
* Fresh build (cargo 1.93.0, `--release --locked --features blas`, Accelerate link flags):
  `tsvdx5-new` sha256 `27d1725381bd15474b8279114f63eacebd2f6ad7c8d8d179a0ef3d8a1c2ed4cc` (`binaries.sha256`).
* Data: 36 `.npy` files (trainx, probex, fulltrain, qual), byte-identical to the 4800lm tree (`data.sha256`).
  Qual ratings: 2,817,131 rows, 0 non-zero.
* Presets are unchanged from the pin: `tsvdx5-300 = run_base(300, 23 epochs, 100 nbrs, lr_w/c 0.0015, save_ifeat=false,
  reset_u_epoch=10, low_memory=false, subscores=false, save_probe_each_epoch=true)`, seed 42; paropt with the
  pinned 300 simplex (Nelder–Mead, 150 iters, 0.5); nlpp with 2 ALS iters, shrinkage 10/25, regs from paropt;
  knn3 with `Knn3Config::default()` on target `1.0*tsvdx5-300__nlpp`.

## Dependencies (from pinned `tsvdx5-new.rs`, `pipeline-new.toml`, `nlpp.rs`)
| order | job | reads | writes |
|---|---|---|---|
| 1 | `tsvdx5-300` | data | `.probex/.qual`, `_ep01…23` snapshots, `.trainx/.fulltrain` train-row preds, `.out` |
| 2 | `tsvdx5-300__nlpp_paropt` | base `.trainx/.probex` + trainx/probex labels (**sees all probex labels**) | `tsvdx5-300__nlpp_regs.npy`, `__nlpp_paropt.out` (trainx/probex only) |
| 3 | `tsvdx5-300__nlpp` | base train-row + query preds (both phases), regs | `.probex/.qual`, `.trainx/.fulltrain` |
| 4 | `tsvdx5-300__nlpp__knn3` | nlpp train-row + query preds | `.probex/.qual`, `.out` |
No `sim/` statistics are needed.

## Storage
Train-row preds ≈ 0.8 GB per producer (base and nlpp); 23 × (4.5 + 11.3) MB snapshots ≈ 0.36 GB; query vectors and
knn3 ≈ 0.1 GB. **≈ 2.1 GB in total.** 184 GB free.

## Memory and schedule
96 GB RAM on this machine. The source-derived lower bound is 49.74 GB (non-low-memory Tx: two dense user/day arrays).
It does not fit alongside 4800lm (34 GB) with margin, so the chain starts only after the 4800lm package ends.
`run_nlpp300.sh` (tmux `cf-lanes:nlpp300`) waits for `rcb-4800/logs/DONE` and no `tsvdx4-new` process, then runs
the four stages strictly in sequence (8 threads), stopping at the first failure. Each stage writes start/end JSON
(binary hash, free memory, disk), a full log, `/usr/bin/time -l` totals, and a 30-s sampler of the **fit process
itself** (footprint, compressed, CPU, free %, disk). The guard terminates only the running stage if free memory drops below 8%.
Expected start ≈ 21:00Z Nov 27 plus the knnf/knns time. Peak memory and epoch cadence are reported from the first
epochs before an ETA is quoted.

## Returns
Each stage's outputs are released as soon as they finish, with shapes, hashes, row-order evidence (native probex/qual
order, identical to the data files above) and timestamps. The comparison is: fixed bank control, then +base, +NLPP, +KNN3,
on the same probex rows and 5 user folds. ProbeX comparisons downstream of paropt are development diagnostics.
