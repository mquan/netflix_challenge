# Apple-GPU larger neural recipe: admitted after execution benchmark

Use a separately declared float32 MPS recipe for the queued own110/all90
hidden128/rank32 network. The original native model recipes and active tree
study continue unchanged. Retire only the verified idle CPU whole110 queue,
with no child or prepared model cache. All three native20/20 completion proofs,
the independent fixed all-three ridge solve and full column checks remain
mandatory before any full-data neural fit. The prior107/all90 controller is
retired only if still idle after the new cache is checked; active fits finish.

A real small32768-row/337-input,6-epoch fit benchmark measured CPU1.5819s
versus MPS0.4709s, a3.36x throughput ratio after warmup. Repeated65536-row
inference and actual saved-model reload/inference matched bitwise. This is
synthetic execution evidence, not model accuracy or a full-data speed promise.
The device is available locally under Torch2.14.0; no external compute is used.

Preserve128 hidden units, rank32, optimizer, learning rate, batch8192, dropout,
weight decay, five nested user folds and stopping. Device dropout has its own
seeded random stream, so GPU training is a new recipe, not CPU weight parity.
Inputs/weights/outputs remain float32. MPS training and inference are explicit;
CPU substitution is refused. Actual saved-model inference replay must pass
again on every full held-out fold before any score is promoted. Full qualifying
inference must use the same declared backend and pass saved-model replay.

Per-process MPS allocation is capped at25% of recommended device working
memory. Existing conservative native/meta admission remains in force: physical
footprints below18GiB and compressor below5GiB, with stable swap and the shared
fit lock. The macro-score remains unknown until actual fitting and scoring.
Known labels choose candidates and budgets; qualifying labels remain excluded.
Best current qualifying remains0.8609266298 /0.8619141823 (9.5103%); target unmet.
