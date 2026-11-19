# Exact sparse Adam candidate: measured, not promoted

The active goal still requires both recovered qualifying groups below0.85725.
The best verified development result remains0.8602161957664950. Native fits
continue unchanged; this candidate has not been used for full-data training.

A short sample of the running canonical CFNADE phase showed substantial time
in per-user training and sparse factor-row Adam updates. The candidate changes
only sparse3D Adam indexing into explicit contiguous row slices. The scalar
normalization, moment updates, bias corrections, square root, parameter step
and subsequent weight-decay step retain their original order. Shape/index
checks precede each row. Source model capacity, configuration, RNG and checkpoint
layout remain unchanged. The isolated source is in
`artifacts/upstream-cfnade-sparse-rows`; no live source tree was modified.

The inherited24-case parity suite passes initialization plus7 epochs, including
all84 parameter/Adam/SWA arrays and RNG state, and direct/grouped predictions.
The checkpoint proof executable also builds, but no checkpoint conversion,
full-data continuation or production handover has been performed.

A separate controlled benchmark compares the current checkpoint-capable trainer
with the candidate, freshly initialized on identical synthetic data. It uses
canonical320 hidden units, rank96, batch8,512 users with200 ratings each,2048
items, and nine timed epochs per implementation in alternating execution order.
Every epoch checks all state/RNG and query prediction bits for exact equality.
Median training time is3.617453750s baseline versus3.522200500s candidate,
a1.0270x ratio on this concurrently busy host. This is roughly2.7%, not a
full-data throughput result. It does not justify interrupting the existing
checkpointed fits. Original scheduler35987 and phases41719/41720 are retained.

Reproduce with `scripts/upstream_cfnade_sparse_rows.py`, followed by
`scripts/upstream_cfnade_sparse_benchmark.py`. Reports, proof hashes and timing
samples are in `experiments/upstream-cfnade-sparse-rows`. Candidate scripts and
source/proof bindings are frozen; further changes need a new experiment.
No qualifying outcomes or downloaded prediction/model weights were used.

Subsequent use: the exact backend is now prepared for an additional original
CFNADE48 preset, after capacity-specific parity checks. Existing CFNADE96
workers stay on their current backend; no checkpoint handover is performed.
See CFNADE48_TRAINING_PLAN.md for the new model's independent resource guard.
