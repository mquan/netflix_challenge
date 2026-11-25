# tsvdx4-4800lm — preparation (reply to REMOTE_4800LM_HANDOFF.md)

## Source and build
* Tree: `artifacts/rcb-4800/source`, an APFS clone of the fresh upstream-repro tree at pinned commit
  **49c9b3676e6bab90fdf77e9a2da984edbfe4fe5c**. `src/`, `Cargo.*` and `pipeline-new.toml` are clean against the pin.
  The preset is unchanged: `run_lm(4800, 27, 2400, 32, save_ifeat=true, low_memory=true, save_train=true,
  subscores=false, save_probe_each_epoch=true)`.
* Data: `trainx`, `probex`, `fulltrain` and `qual` only. All four are byte-identical to the rcb-r1 tree, and qual
  ratings are all zero. `data/qual_ratings` is **not** copied.
* Fresh build: cargo 1.93.0, `RUSTFLAGS=-C link-arg=-framework -C link-arg=Accelerate`,
  `--release --locked --features blas`. Binary hashes: `tsvdx4-new 8d0ab459…2e0d`, `nbstats-new 7f993bd8…3dca`
  (`binaries.sha256`). The fresh tree's older binary had been rebuilt since its receipt, so it was not reused.

## Native dependencies (resolved from pinned `tsvdx4-new.rs` and `pipeline-new.toml`)
| job | needs | writes (both phases) |
|---|---|---|
| `tsvdx4-4800lm` | data only | `.probex/.qual`, `_ep01…27` snapshots, `.trainx/.fulltrain` train-row preds, `.ifeat.{trainx,fulltrain}` item factors, `.out` |
| `tsvdx4-4800lm__knnf` | parent train-row preds + `.ifeat` | `.probex/.qual` |
| `tsvdx4-4800lm__knns` | parent train-row preds | `.probex/.qual` |
| `tsvdx4-4800lm__knn-d` (deferred) | parent train-row preds, `sim/rtg_supp`, `sim/tsvdx4-4800lm_diff1` (`nbstats-new`) | `.probex/.qual` |

## Storage estimate (both phases)
These are source-derived sizes; actual sizes will be recorded.
* Train-row preds: 397 MB + 402 MB.
* Terminal and per-epoch snapshots: 27 × 4.5 MB probex plus 27 × 11.3 MB qual, about 0.43 GB (published
  epochs 3, 13, 21 and 24 are a subset).
* Item factors: about 17,770 × 4,800 × 4 B = 0.34 GB per phase per factor array, so ≤ 2 GB total.
* knnf and knns: about 32 MB.
* knn-d stats: 2 × 1.26 GB (diff1) + 2 × 1.26 GB (rtg_supp).
* **Total ≈ 4 GB without knn-d, ≈ 9 GB with it.** Free disk is 175 GB.

## Memory and schedule
The source-derived lower bound is 35.35 GB; the real peak is unmeasured. For comparison, the two TX300 arms each grew
from 17 GB to about 49 GB footprint (mostly compressed). Swap is 22.5 / 23.5 GB in use, and macOS reports 37% free.
Starting 4800lm next to both TX300 arms would risk them, so `run4800.sh` (tmux `cf-lanes:4800`):
* waits for TX300 arm C to exit (≈ 04:30Z Nov 25). B's qual phase then runs alone until ≈ 13:30Z, which frees
  about 49 GB;
* runs the base with 8 threads, then knnf and knns in parallel with 4 threads each;
* samples footprint, compressed memory, swap, free % and disk every 30 s into `logs/*.resources`;
* has a **guard**: if system memory free drops below 8%, it kills only the running 4800lm-package process, never
  TX300. Phase-1 epoch time and peak memory will be reported in the first hour to update the two-phase ETA.

## Coordinator retrieval: existing returns
| asset (release) | bytes | sha256 |
|---|---|---|
| `s3-rbm800p-index.json` (remote-weekend-20261120) | 23,388 | — (index lists the parts) |
| `s3-rbm800p-part01.tar.gz` | 890,918,695 | `c65de0703d3ced9749bc80d7b7722ae682bfc08bfc7108faed81c3acf87ece65` |
| `s3-rbm800p-part02.tar.gz` | 944,100,583 | `5def76d9101aece8a8581d9edb2e4173ff1dd097b94fb3bee29ad69d7eaa5a8d` |
| `s3-rbm800p-part03.tar.gz` | 136,411,517 | `7a566b1d52901d80e77d27934ceb2f8e3b26cc3e5a08e40a367721f0c75c5ec1` |
| `rcb-knn3d-wave.tar.gz` (remote-context-blend-20261124) | 818,104,142 | `4784440bbc17fd48bd2c36909ed81e50986c21e58bf37e8e533760874ad0d5b8` |

Wave package members are listed in `knn3d/wave/package-members.sha256`.

## Measured (2026-11-25T17:46Z)
Started 03:47Z. Phase 1 epoch 11/27 at 17:26Z (probex 0.88326). Epoch time ≈71 min (epochs 4–11 at 70–71 min;
the first ones ran at 80 min next to TX300 B). Footprint 34 GB RSS (4.5 GB compressed) since B exited; it peaked near
34 GB + 31 GB compressed while B was running. Process CPU ≈100%: this preset's training loop runs on one core,
so the 8-thread setting does not speed it up. Disk 48 MB so far.
ETA: phase 1 ends ≈12:30Z Nov 26; phase 2 (27 epochs, ≈72 min) ≈21:00Z Nov 27; then knnf+knns in parallel.
