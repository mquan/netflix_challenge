# Verified context blend development result

Observed 2026-11-18 22:34 UTC. The complete five-fold own-predictor fixed average improves
from 0.8604038556527668 to **0.8602161957664950**.
Every outer user fold improves. All five NN fold receipts were rehashed,
all saved networks replayed bitwise, and the independent compensated scorer and
fixed-average identity passed. The original parent tree is reused for development;
no tree training receipts were invented. This is adaptive ProbeX development,
not a qualifying or official Netflix Test result.

Paired MSE delta: -0.0003228913631665345. Conditional user-cluster 95%
interval: [-0.00038365744044361276, -0.00026212528588945614]. This interval has no
selection correction. Evidence is in `experiments/upstream-context-blend/`.

The final qualifying adapter is `scripts/upstream_context_production.py`.
It preserves the original85-column paired-bank transform, refits its actual
full-known-ProbeX ridge, and freshly refits both model families. Tree budgets
come from the five verified original tree fold receipts; context NN budgets
come from the five new NN receipts. The fixed median budgets are tree
[698,737,860], NN[21,19,19], with the unchanged three seeds. No qualifying
outcomes choose architecture, predictors, budgets or weights.

The queue waits for the first frozen85-column qualifying audit. It skips if
that audit establishes both groups below0.85725. It serializes against the
extension's development lock and final-production status, checks any already
verified91-column development result, and skips if that result is stronger.
A completed91-column target pass also makes this refit unnecessary. These
pass/fail checks only avoid work after success; qualifying residuals and group
scores do not select weights or recipes.

Each production child repeats adapter bindings in a fresh process. The
unchanged production core performs full fits, bitwise model replay, frozen
prediction hashing, recovered-label audit and independent group scoring. The
current fitter PID is relayed to the existing native memory-accounting status
only while this queue owns final production. Separate models/reports are in
`artifacts/upstream-context-production` and
`experiments/upstream-context-production`. Its queue status is separate from
the original first qualifying attempt. The active goal remains incomplete
until the actual required qualifying results and provenance pass review.

Validation: three context neural/scoring checks and two production checks
passed. Production checks use real isolated Torch fitting/inference, test the
zero-budget seed, reject semantic prediction substitution despite an updated
receipt byte hash, and reject a budget receipt bound to the wrong source
manifest. Synthetic checks do not establish competition quality.
