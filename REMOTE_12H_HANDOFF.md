# Remote 12-hour fresh-model assignment: handoff

Branch `research/fresh-models-12h`. The plan is
[REMOTE_12H_PLAN.md](REMOTE_12H_PLAN.md) and the full timeline is in
[REMOTE_12H_PROGRESS.md](REMOTE_12H_PROGRESS.md). Host: Apple M3 Max, 14 cores,
96 GiB RAM, macOS 27.2.1. Checkout: `~/workspace/netflix_upstream_repro`, which was
switched from `research/upstream-reproduction` @ db394c1 (already on origin).

Clock (UTC): start 2026-11-18T03:01:05Z, no new launches after 12:31:05Z, hard
training stop 13:01:05Z, final deadline 15:01:05Z.

## Summary

- **16 native jobs completed**, each with a fresh receipt. Every receipt
  output hash was re-verified before export.
- **25 ProbeX NPZ files, 24 unique vectors**, are in `experiments/upstream-remote-12h/outputs/`.
  The index with all hashes is `experiments/upstream-remote-12h/transfer-index.json`.
- **cfnade-96 was not launched.** The measured cost is ≥50 h single-threaded.
- **No ensemble evaluation was run here.** The coordinator's control bank
  (the M1 receipts and outputs) is absent from this machine.
- **No owned processes remain.** All `r12-*` tmux sessions are closed, and
  `ps` shows no supervisor, wrapper or native model process.
- Scope extension: after the assigned primary finished early (and CF-NADE was
  ruled out), the user approved training further candidates from
  `experiments/upstream-missing-screen/recipe-costs.json` inside the same window.
  The user asked to continue toward the 10% goal, but that can't be measured
  here, because the control bank is absent.

## Jobs

| Job | Start–finish (UTC) | Native min | Peak group RSS | Kept epochs exported | Diagnostic ProbeX RMSE | Fresh vs released RMS diff |
| --- | --- | ---: | ---: | --- | ---: | --- |
| `bknbrx-100` (assigned primary) | 03:03–03:12 | 9.1 | 10.0 GB | 01, 05 (≡ terminal) | 0.905514 | ep05 0.0101 (Hogwild) |
| `tsvdx5-150o` | 03:50–09:36 | 346.3 | 28.7 GB | 06, 16, 24 | 0.875583 | 6.8e-7 |
| `bknbrx-50` | 09:37–10:14 | 36.8 | 10.2 GB | 05, 09, 17, 21 | 0.900374 | ep05 8.0e-3, ep17 7.3e-3 (Hogwild) |
| `mf-60` → `mf-60__knn3` | 10:14–10:22 | 5.3 + 2.5 | 3.5 GB | — | 0.926041 / 0.907782 | 0 / 0 |
| `als8-8` → `als8-8__knn3` | 10:22–10:29 | 4.2 + 2.7 | 7.9 GB | — | 0.956807 / 0.946445 | 2e-7 / 2e-7 |
| `dnn-24__dnn-16p50__asym-16p75` | 10:29–10:31 | 1.4 | | — | 0.897370 | no fixture |
| `…__asym-16p75__knn3b` | 10:31–10:36 | 4.8 | 5.4 GB | — | 0.886205 | 0 |
| `…__asym-16p75__knn3c` | 10:36–10:39 | 3.6 | 5.1 GB | — | 0.887052 | 0 |
| `dnn-24__asym-16p75` → `__knn3` | 10:39–10:43 | 1.3 + 2.6 | 4.3 GB | — | 0.901566 / 0.890907 | no fixture / 0 |
| `dnn-24__dnn-16p50__asym-16p50` | 10:43–10:45 | 1.3 | 3.5 GB | — | 0.901492 | 0 |
| `dnn-24__dnn-24p50` | 10:45–12:13 | 88.7 | 4.8 GB | — | 0.910467 | 0 |
| `dnn-24__dnn-16` → `__knn3` | 12:13–12:59 | 42.7 + 2.8 | 5.4 GB | — | 0.917338 / 0.908362 | no fixture / 0 |

Notes:

- **Swap growth was 0 throughout.** Free memory stayed at or above 85%.
- The dnn-24 chains reused the receipt-verified local `dnn-24` and
  `dnn-24__dnn-16p50` parents, which the wrapper verified and skipped.
  Intermediate parents with no released fixture (`…asym-16p75`, `dnn-24__dnn-16`)
  are exported too, but they aren't columns of the released bank. Include them
  only as a separately declared arm.
- "Fresh vs released" is a post-freeze diagnostic. Released vectors never
  entered training. A zero diff means the fresh vector is bit-identical to the
  released one.
- Standalone RMSEs are diagnostics, not ensemble gains.
- **Skipped at the launch cutoff:** `mf-61__knn3` and `mf-61__asym-16`. Neither
  was started, and neither has partial outputs.

## Per-job transfer contents

Each job has a directory `experiments/upstream-remote-12h/<job>/` containing:

- `transfer-manifest.json`: per output, the source NPY SHA256, NPZ SHA256,
  prediction-content SHA256 and owning native receipt SHA256. It also holds the
  alignment SHA256s (ratings, user_idxs, item_idxs, dates), the embedded
  `fresh-data-receipt.json`, the source pin, host details, and the command and
  environment.
- `native-receipt.json`, `native.log`, `native.out`.
- `fixture-comparison.json`.

The supervisor files (`status.json`, `resources.jsonl` sampled every 60 s,
`supervisor.log` and `wrapper.out`) sit in the directory of the job that was
launched. For a chain, that is the endpoint's directory. Exports are raw, finite
float32 arrays of shape (1128164,), in native order and unclipped. Weights and
fulltrain/qual outputs stay in `artifacts/upstream-repro/fresh/preds_new/` on
this machine.

The alignment hashes are identical for every job:

| Array | SHA256 |
| --- | --- |
| ratings | `5a554ca04854b25a0422771b40a02415e488212513a475f3eda10c01238fa2f7` |
| user_idxs | `d961f5966fc6af74a3a00754068929e689a4d86b0d749f211e70582183bd7b06` |
| item_idxs | `b6e42d9de67ede3b121e49df81968d1a30bd9913b18a3dc599fedf2e2bf2a856` |
| dates | `c9555bb419b28094811a16ef8f05821f38eed307f998b8da89708b9f1f9d4f47` |

## cfnade-96: resource-limit finding

The profile is in `experiments/upstream-remote-12h/cfnade-96-profile/profile-summary.json`.
It is **profiling only**: `--test`, 4 epochs, swa 3, no `--save`, with
`preds_new` unchanged before and after.

- Epoch 1 took **942.8 s** using one core; `src/cfnade.rs` has no rayon
  parallelism. RSS was 4.1 GB.
- At 95 epochs × 2 phases, that is ≥49.8 h, or about 75 h with the plan's margin.
- The full recipe needs a dedicated host for 2–3 days.
- The first profile attempt failed with exit 127 (binary not yet built). That
  record is kept in `cfnade-96-profile-attempt0-missing-binary/`.

## Deviation to note

`tsvdx5-150o`'s measured projection was about 6 h. The plan's strict
1.5× + 30 min rule gives about 9.5 h, which would have exceeded the 13:01 stop.
I kept it running anyway, because the timing was measured and the supervisor
still enforced the hard stop. It finished in 5.77 h.

## Coordinator evaluation (receiving side)

1. Verify the NPZ, content and receipt hashes against each `transfer-manifest.json`,
   and the alignment hashes against your bank.
2. Reproduce the controls 0.8677119362283443, 0.8663200739003126 and
   0.866286461848646, each within 1e-9.
3. Run the predeclared arms, deduplicating by content hash and not selecting
   epochs from fold scores:
   - control + bknbrx-100 {ep01, ep05}
   - control + tsvdx5-150o {terminal, ep06, ep16, ep24}
   - control + bknbrx-50 {terminal, ep05, ep09, ep17, ep21}
   - control + the released-bank dnn-24 bundle columns (`…asym-16p75__knn3b`,
     `…__knn3c`, `dnn-24__asym-16p75__knn3`, `…asym-16p50`, `dnn-24__dnn-24p50`,
     `dnn-24__dnn-16__knn3`)
   - control + `mf-60`, `mf-60__knn3`
   - control + `als8-8`, `als8-8__knn3`
   - control + all of the above

   Use five user folds (seed 20261116), lambda 1000, an unpenalized intercept,
   native input clipping, output clip [1.05, 4.95], and fourteen-only context
   interactions. Report every fold delta, including negative ones.
4. Released-vector diagnostics (against the earlier 0.86632 control, so they are
   selection evidence only) were: tsvdx5-150o −0.00134, bknbrx-100 −0.00078,
   bknbrx-50 −0.00075, dnn-24 bundle −0.00052, mf-60 −0.00017, als8-8 −0.00016.
   These overlap and won't add up. Even all together they are very unlikely to
   reach the 10% target (≈0.8563 ProbeX-equivalent) from 0.86629. The large
   remaining gap is in big bundles that don't fit 96 GB or 12 h on this host:
   tsvdx5-1000/400/300 and tsvdx4-4800lm exceed physical RAM, and the
   rbmx2-400–800 models are about 15+ h.

## Restart requirements

- Every completed job resumes as verified-and-skipped through
  `upstream_repro.py --run-job`.
- Next cheap jobs: `mf-61__knn3` and `mf-61__asym-16`. Run them with
  `scripts/remote12h_queue.sh CUTOFF DEADLINE mf-61__knn3 mf-61__asym-16`.
- cfnade-96 needs a multi-day supervisor deadline. Its binary is already built.
  Native training can't resume, so an interrupted run starts over.
- Tools:
  - `scripts/remote12h_supervisor.py`: process-group supervisor with deadline,
    memory and disk limits. Tests are in `tests/test_remote12h_supervisor.py`.
  - `scripts/remote12h_launch.sh` and `scripts/remote12h_queue.sh`.
  - `scripts/remote12h_export.py --job NAME [--kept EE ...]`.
  - `experiments/upstream-remote-12h/fixture_compare.py JOB`.
