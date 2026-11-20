# Fifteen-hour remote assignment: nine fresh original prediction pairs

Branch: `research/remote15h-handoff`. Coordinator remains on
`research/full-upstream-reproduction`. Execute this assignment now; no further
planning round is needed. Preserve existing remote outputs and running jobs.

Our independently trained qualifying best remains **0.8605716962 / 0.8615832225**.
The target is strictly below **0.85725 on both recovered groups**. These are
adaptive recovered audits, not an official untouched Test. This assignment
adds model diversity; it cannot promise the remaining improvement. Public
numerical predictions, weights, and qualifying outcomes are prohibited inputs.

## Assignment and budget

Run these producers **serially**, in this order, at the original complete
capacity, data size, seeds, and epoch budgets. Every producer must finish both
`train_3 → probe_3` and `fulltrain → qual` before any of its snapshots qualifies
for admission. The machine-readable commands, environment, source, and data
hashes are frozen in `experiments/upstream-remote15h/plan.json`.

| Producer | Original epochs per phase | Published pairs to return | Rough training time |
|---|---:|---|---:|
| `s3-dnn24` | 8 + 8 | terminal, ep03 | 1–2h |
| `s3-attn32` | 12 + 12 | terminal, ep06 | about 1h |
| `s3-rbm200e` | 25 + 25 | terminal, ep01/05/10/17 | 6–9h, uncertain |

Expected total training is about 8–12h, plus bootstrap, verification, and
transfer. These estimates are based on related completed producers, not a
measured runtime for every exact recipe. Start the 15-hour clock on arrival;
set the **training deadline to arrival +14h**, leaving one hour for return.
The helper requires at least 2h remaining before either short job and 10h
before the RBM. Export and upload each completed producer immediately. If
bootstrap takes too long to admit the RBM, return the four short-job pairs and
the timing evidence. Do not shorten or change a model to fit the deadline.

DNN: D24 / H1=96 / H2=64 / NMF256 / decay .87 / seed43.
Attention: original default D32 / pool32 / NMF192 / default seed.
RBM: H200 / exponential hidden type / EA0 / EB1 / LR.02.
Original model threads: Rayon4 and OpenBLAS4. No inherited model knobs.

Do not duplicate local CFNADE96, Bernoulli RBM200, attention64, tx100r,
item-autoencoder, attention2 blend, or the queued KNN3 chains. Do not migrate
the CF96 trajectory or restart old completed producers.

## 1. Isolate checkout and locate existing data

Use the existing remote research checkout and Python environment. Adjust paths
only to match the laptop; do not overwrite a dirty checkout or live tree.

```sh
cd ~/workspace/netflix_upstream_repro
git fetch origin research/remote15h-handoff
git worktree add ~/workspace/netflix_remote15h -b research/remote15h-execution origin/research/remote15h-handoff
cd ~/workspace/netflix_remote15h
```

Use `~/workspace/netflix_upstream_repro/.venv/bin/python` for all commands below.
Check actual hardware, free disk, memory, and current processes first. This plan
assumes the previously available M3 Max /96 GiB machine. Run one native producer
at a time, on power, with sleep prevented. If materially different hardware or
competing jobs make full-budget completion implausible, return the observation
and prioritize the two short producers; never kill unrelated jobs.

The existing native source should be
`~/workspace/netflix_upstream_repro/artifacts/upstream-repro/fresh` at source
pin `49c9b3676e6bab90fdf77e9a2da984edbfe4fe5c`. It must contain unchanged original
code. The data directory must have our canonical **train, probe, fulltrain,
qual, probex** arrays. The helper verifies all44 array hashes and all115 code
and configuration file hashes before creating the isolated native tree.
Qualifying targets must be all zero. No raw ratings are uploaded.

**Never run the original ingestion binary:** it imports recovered qualifying
labels. Never download numerical prediction banks. If canonical arrays are
missing or hashes differ, preserve the evidence, push a blocker report promptly,
and identify any existing matching canonical copy on this laptop. Do not invent
a substitute split. The coordinator can arrange a separate private transfer if
no matching data exists.

## 2. Prepare the exact original split

Record UTC arrival and the fixed UTC deadline (arrival +14h) in
`experiments/upstream-remote15h/execution/start.json`, including hardware,
paths, disk, and live-job observations. Keep the same deadline throughout.

```sh
~/workspace/netflix_upstream_repro/.venv/bin/python scripts/upstream_remote15h.py prepare \
  --source-tree ~/workspace/netflix_upstream_repro/artifacts/upstream-repro/fresh \
  --data-root ~/workspace/netflix_upstream_repro/artifacts/upstream-repro/fresh/data \
  --tree ~/workspace/netflix_remote15h/artifacts/upstream-remote15h/source
```

This copies only hash-verified original code/configuration, builds original
`fit` and `split_3` binaries with the existing toolchain and BLAS, links the
verified own datasets, and runs the unchanged splitter with default seed42,
K2, add fraction .25. It compares every generated split-array hash to the
coordinator's split, checks every known-label match, proves known rows absent
from training, and recomputes the exact1,128,164-row known projection from
1,760,494 native holdout rows. It records actual build and binary identity.
It requires20 GiB free at preparation and each subsequent native-job admission.

Never rerun `prepare` over an existing partial directory. Inspect and preserve
any partial setup; use a fresh owned root if repair is necessary, documenting
why. `verify-inputs --tree ...` rechecks a completed preparation without fitting.

## 3. Run, verify, and return after each producer

For each job in the table, run the following with its actual name and the
**same explicit arrival+14h UTC deadline**, e.g. ISO `2026-11-20T17:30:00Z` is
only a syntax example; compute the actual deadline on arrival.

```sh
~/workspace/netflix_upstream_repro/.venv/bin/python scripts/upstream_remote15h.py run-job \
  --tree ~/workspace/netflix_remote15h/artifacts/upstream-remote15h/source \
  --job s3-dnn24 --deadline-utc ACTUAL_UTC_DEADLINE

~/workspace/netflix_upstream_repro/.venv/bin/python scripts/upstream_remote15h.py export \
  --tree ~/workspace/netflix_remote15h/artifacts/upstream-remote15h/source \
  --job s3-dnn24 \
  --output ~/workspace/netflix_remote15h/artifacts/upstream-remote15h/exports/s3-dnn24
```

Repeat for `s3-attn32`, then `s3-rbm200e`, subject to time and disk admission.
Use tmux and `caffeinate -i` around the work session. The runner owns only its
launched direct native process; at the training deadline it terminates that
process and records the producer as **incomplete and ineligible**. Preserve
all native evidence. It never automatically restarts an interrupted trajectory.
A process lock prevents overlapping helper operations in the same native root.

The export is a separate process: it rehashes source/data/binary and all native
outputs, checks both complete epoch traces, validates finite float32 native
queries of exactly1,760,494 /2,817,131 rows, checks terminal against the final
epoch, recomputes the row projection, and verifies each compressed published
pair bit for bit. Every pair contains known `probex` and full `qual`, with no
qualifying labels. No blend weights or qualifying score are fitted remotely.

After **each** export:

1. Copy its JSON receipts and logs, but **not prediction arrays**, to
   `experiments/upstream-remote15h/execution/JOB/`. Include the exported
   verification, plan, preparation, launch, native receipt, build, and split logs.
2. Archive the complete export directory into `JOB-export.tar.gz`. Calculate
   its SHA256 and byte count; record the release tag, asset name, archive hash,
   and native/payload hashes in the tracked return report.
3. Create a **draft transfer release**, e.g. `remote15h-own-20261120`, with
   `gh release create TAG --draft --title 'Own remote15h training transfer' --notes 'Transfer of our own original-source training outputs; no raw ratings.'`.
   Reuse that release for subsequent jobs. Upload each completed archive with
   `gh release upload TAG PATH_TO_ARCHIVE --repo mquan/netflix_challenge`.
   Never overwrite an existing asset with different bytes; use a fresh name.
4. Commit named execution reports and push with
   `git push origin HEAD:research/remote15h-handoff`. Do this immediately for
   each completed producer, not only at the end of the fifteen-hour window.

If release upload is unavailable, preserve the archives and push the small
report promptly, explicitly marking transfer incomplete. Do not claim the
coordinator has received numerical arrays before the asset is available.

Keep **all** generated native files, including training/fulltrain vectors,
all epoch snapshots, RBM factors, native logs, and actual binaries. Once compact
published pairs are safely uploaded, use remaining transfer time to archive and
upload complete native outputs as a separate lossless asset. Record its exact
membership/hashes. Exclude linked raw datasets and recovered labels. Do not
delete native artifacts when the window ends; report retained paths and any
untransferred files clearly.

## 4. Progress and completion

Push a status at start, after bootstrap, after every producer, and at least
hourly while training: observed UTC, PID, actual phase/epoch from logs, elapsed,
time left, memory/disk, completed exports and asset identities, next action.
Incomplete epochs never count as a completed original producer. With less
than one hour left, focus entirely on verification, transfer, and reporting.

At completion, push a final checklist with: 3 complete producers /9 verified
published pairs if achieved; otherwise the exact completed subset and reasons.
No qualifying labels or outcome-based snapshot selection. Coordinator will
independently validate the return and evaluate the fixed new features under
our existing nested user-fold protocol before any qualifying promotion.

## Verification of this handoff

The coordinator checks that recipes exactly match the pinned registry; tests
reject inherited model knobs, missing/reordered/shortened two-phase traces,
wrong vector shapes/dtypes/nonfinite values, and changed file hashes. These
checks validate the handoff helper; full native fits execute on the remote.
