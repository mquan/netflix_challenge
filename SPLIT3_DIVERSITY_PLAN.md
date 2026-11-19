# Next native diversity batch: three original split3 models

Reproduce the exact published s3-dnn16t, s3-dnn32nm and s3-nade12s recipes,
with original dimensions, seeds/defaults and both complete native phases.
These add different families rather than another subset of the same temporal
snapshots: transposed DNN, DNN without matrix factors, and small conditional
NADE. Preserve their published terminal columns and NADE epoch1, fixed in
advance from the public schema. No public numerical prediction files are read.

The private source tree copies only the pinned original source/configuration
and verified original fit executable. It reuses our verified split3 data:
98,720,013 training rows,1,760,494 source holdout rows, and an exact map to all
1,128,164 known-ProbeX rows. The prior full absence-of-known-rows training check
is bound through unchanged data hashes. Original fulltrain and zero-target
qualifying data are reused for phase2. Known projection is checked again before
admitting any output to a future blend.

The controller waits for the current original RBM50/Tx20lat lane to exit before
admission. Native models run serially after two stable memory samples: existing
native/meta physical footprints below20GiB, compressor below5GiB, swap growth
below32MiB per30seconds, and free disk above20GiB. This reserves4GiB for the
small original model. No active native fitting is stopped. Observation timeouts
do not cancel training. Partial native outputs are preserved on failure.

Each receipt requires every source-declared output, both full epoch traces,
finite float32 query predictions, complete hashes and the exact known query
projection. A separate completion checker must recheck those bindings and
projection before the fixed all-ready batch enters nested blending. This queue
is training preparation, not completed features or a new score.

Current best remains NN103 development0.8594692564 and recovered qualifying
0.8609916544 /0.8619938614. Both groups must fall below0.85725; no success is
claimed. The107-predictor plus50-RBM-context study is running independently.
