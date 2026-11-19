# Fixed own102 iteration using our published CF48 epoch25 snapshot

The goal remains RMSE below0.85725 on both recovered qualifying groups. Best
audited worse-group RMSE remains0.8624021195492101. This iteration has a new
known-ProbeX ridge diagnostic, not a new nonlinear or qualifying score.

## Completed diagnostic

The public new-split raw schema names `cfnade-48_ep25` and
`cfnade-48_ep95`. Our independently trained original160-hidden/rank48/95-epoch
trajectory already generated the epoch25 prediction pair before its checkpoint
transfer. The local vectors match the exact SHA256s in that captured own package:

- ProbeX: `e12b59bee7cac79157a80198c56c5e630f7ff48ede6f02f4ac4dfd7e8b1938da`
- Qual: `67093d060930f34a8a9122263e63027c51bb9b21a8df5120ff0e07abd1737465`

The frozen source, input arrays, binaries, configuration and package manifest
were reverified before using these values. They are our predictions, not public
prediction values. Snapshot25 is a complete generated output for both query
splits; **the whole95-epoch job is still running remotely**. Its capacity,
optimizer, schedule and total epoch count are unchanged. No complete95-epoch
job receipt or terminal prediction is fabricated here.

The fixed all-nine arm adds the previously verified ready-eight bundle and
this published snapshot to own93. All arms were declared before scoring:

| Known-ProbeX ridge arm | RMSE |
| --- | ---: |
| Own93 control | 0.8631793373172530 |
| Ready-eight | 0.8631599062596309 |
| CF48 epoch25 only | 0.8627234642672663 |
| Fixed all-nine | **0.8626981139126103** |

All1,128,164 rows and the same five user folds are retained. A separate solver
recomputed five-fold sufficient statistics, weights, predictions and
compensated scores for control and all-nine. Maximum prediction replay error
was5.77e-12 at tolerance1e-10. Neither qualifying labels nor public numerical
features entered the experiment. The all-nine arm remains fixed; no post-score
feature subset was selected.

Evidence: `experiments/upstream-cf48-prefix-screen/{plan,ridge-screen,ridge-verification}.json`.

## Prepared cache and queued context fit

Own102 has178 ridge columns and238 neural features; the context model receives
239 inputs including its fitted ridge prediction. The cache retains the
cached own93 development realization, adding all nine rating features with
their fixed centering. Every retained and new column was checked against the
parent and actual own vectors, row/fold identities matched, and the baseline
was bound to the independently replayed all-nine prediction. This cache check
does not claim another independent ridge solve. Paired full-fit production
assembly remains separate work.

The next model preserves the original rank16,64-hidden context architecture,
five nested user folds, seeds, optimizer and stopping policy. It reuses the
actual verified tree91 predictions and model receipts. A train/save/reload
test confirms the239-input model factory and exact saved-state prediction
replay. No active model source or previous artifact was changed.

Live controller47665 waits for all91 context training/replay and memory
admission. The idle pending93 controller29784 was verified to have no child
or prepared model cache, then stopped to give this larger measured iteration
priority. Its code and prior reports remain preserved. The queue handoff
binds24 implementation files and the actual own102 data manifest. All91 worker
45335 and the three native source training jobs continue unchanged.

Evidence: `experiments/upstream-cf48-102/{manifest,cache-column-verification}.json`
and `experiments/upstream-cf48-102-context/queue-handoff.json`.

After fitting, replay all saved neural folds and independently score the fixed
mean before any development promotion. A production fit must use declared
paired own inputs with their exact family layouts; its vector must be frozen
and replayed before the recovered-group audit. No recovered outcomes select
features, stopping budgets or weights. These audits are adaptive recovered
labels, not official or untouched Test results.
