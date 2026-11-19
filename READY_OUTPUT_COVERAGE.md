# Completed-output iteration and public-bank coverage

Current snapshot (2026-11-19 23:08 UTC): verified own development best
**0.8592309819864167**, recovered qualifying **0.8607516601651759 /
0.8617378856802073**, still missing the strict0.85725 target. The split3
attention16 pilot and subsequent DNN16t/DNN32nm/NADE12s full original models
are complete and independently verified. Exact original attention64, RBM200
and temporal100r are now prepared/queued, with a separate completion watcher
and five fixed published output pairs. See [SPLIT3_NEXT3_PLAN.md](SPLIT3_NEXT3_PLAN.md).
These are own-training completion and queue states, not complete public-bank
reproduction or additional qualifying gains. Earlier observations below are
retained as history.

Observed November 19, 2026, approximately 10:55 UTC. Goal remains active:
both recovered qualifying groups must score strictly below 0.85725 using
our own trained outputs and locally fitted weights. Best audited worse group
remains **0.8624021195492101**; no new qualifying score is claimed here.

## Eight available additions

The fixed rule was to add every unused published new-split raw feature with
a complete original native receipt and both prediction splits, removing
exact paired-value duplicates before scoring. Public input was JSON feature
names only. All numerical vectors came from our own training.

Ten ready columns were found. `bknbrx-100_ep05` equals our existing terminal
`bknbrx-100` on both splits, and `cfnade-12_ep40` equals our existing CF12
terminal. The remaining eight are:

- `dnn-24__asym-16p75`
- `dnn-24__dnn-16`
- `dnn-24__dnn-16p50__asym-16p75`
- `mf-61__asym-16`
- `mf-61__asym-16_ep02`
- `mf-61__asym-16_ep07`
- `rbmx2-0`
- `tsvdx4-l32nb_ep03`

On the same 1,128,164 known ProbeX rows and five user folds, the own93 ridge
control is **0.8631793373172530**, and the all-eight arm is
**0.8631599062596309**. One fold worsens slightly. Separate five-fold sufficient
statistics, linear solves, prediction replay and compensated scoring confirm
both results; maximum replay error is 5.30e-12 with tolerance 1e-10.
This is a ridge diagnostic, not a new nonlinear best or qualifying promotion.
The fixed main arm retains all eight; no post-score feature subset was chosen.

Evidence: `experiments/upstream-ready-public-schema/{plan,ridge-screen,ridge-verification}.json`.
No active training recipe or frozen bank was changed.

## Source coverage changes the next training direction

The public bank contains **536 raw predictors + 83 meta predictions**, plus
**220 support features**. The 536 raw predictors comprise:

| Source split | Raw columns | Exact declared source outputs |
| --- | ---: | ---: |
| old | 122 | 20 |
| 3 | 263 | 263 |
| new | 151 | 134 |

These are recipe counts, not completed training counts. The original current
201-job queue covers `pipeline-new.toml`; it does not reproduce the entire
public bank. Own93 includes 39 of the public 151 newest-split raw column names,
with many additional own snapshots outside that published selection.

The counts above come from upstream's actual `Pipeline::load` and
`resolve_pipeline`, at the pinned source commit. The regenerated new-split
job registry must equal the original registry. Some additional new-split
components are emitted auxiliaries rather than declared outputs, so 134 is
a conservative declared-output count, not proof that all other recipes are absent.
Old-split `pipeline-1.toml` offers further candidate reconstructions, but
similar names alone do not establish exact reproduction.

Evidence: `experiments/upstream-all-pipeline-coverage/{registry,coverage}.json`.
The coverage report binds all tracked source files, registry helper/binary,
public metadata, and its own implementation.

## Next executable lane

Build an isolated split3 reproduction lane, beginning with the original
`s3-attn16` pilot and then dependency-ordered source jobs for the 263 raw columns.
Preserve original capacities, epochs and native commands. Use verified own
ingested arrays; skip the source ingest job because its inputs mention recovered
qualifying labels. Qualifying targets remain zero throughout training.

Before admitting any split3 feature, verify unique `(user,item,date)` row
identities, project `probe_3` predictions onto known ProbeX, and prove those
evaluation rows are absent from the split3 training data. Native split3 starts
from original train and adds training rows to its holdout; known ProbeX is a
subset of original Probe, but actual row-disjointness must still be checked.
Keep the fulltrain/qual phase separate, and bind complete two-phase receipts.

This new lane is **not launched yet**. Current uncheckpointed temporal natives,
the coordinated CF96 resume, and the queued context experiments retain ownership
of their jobs. Admit additional work from measured memory availability rather
than interrupting them. After the pilot completes, use known-label-only fold
selection and locally trained meta learners; a source recipe is not a trained
feature, and a development gain is not target success.
