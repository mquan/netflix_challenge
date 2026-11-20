# Close missing high-capacity original model coverage

The remaining qualifying gap is0.0043162023 on the worse recovered group.
Recent raw-prediction additions show modest ridge gains; they do not justify
assuming small blend changes will cross the target. Test the larger-data
matched study first, then add structurally missing original model capacity.

The local original registry declares two exact low-memory presets absent
from both fresh execution receipts and declared fresh output paths:

1. `tsvdx4-1600lm`: original1,600 factors and2 epochs in each phase.
2. `tsvdx5-1200lm`: original1,200 factors and23 epochs in each phase,
   including its original100-neighbor component.

Two epochs for the first model is the published recipe, not a shortened
substitute. Likewise `low_memory=true` belongs to both original presets;
it must not be retrofitted onto other models. Do not import public epoch
blend coefficients: any selected fresh snapshots require known-only local
weight fitting within the existing user-held-out evaluation.

The coverage audit binds the original source/registry/data receipt and
records source-derived allocations. Three user-factor arrays plus item
and frequency-factor arrays need10.49GiB for1600 or7.87GiB for1200.
The1200 preset additionally allocates2.35GiB dense neighbor arrays.
These are lower bounds, excluding datasets, residual histories, optimizer
scratch, prediction storage and other allocations. They are not admission
estimates. The full-upstream runner deliberately deferred these large jobs;
do not bypass that deferral by invoking its unguarded job runner.

Before launch, construct an isolated, hash-bound original execution with
full source/data/input closure, complete two-phase output accounting,
actual memory admission and continuous compressed-inclusive monitoring.
Preserve all original factors, rows, threads and epochs. Require canonical
training/query disjointness and zero qualifying fitting targets. Avoid any
public numerical dependency. Preserve partial output/state on guard stops;
do not silently restart or replace the recipe.

Run only after existing temporal/CF/native and queued neural work can
coexist within measured resources, or after their memory is released.
Prepare the smaller-footprint feasible job first after accounting for its
neighbor histories; selection is based on source coverage and resource
feasibility, not qualifying outcomes. No high-capacity training job has
been launched by this plan or its coverage audit.

Independently verify full phase traces, original source/binary/data and
every native query vector before adding a predictor. Compare the augmented
bank on the existing canonical held-user folds. Only a development-eligible,
fully replayed model may receive a frozen qualifying audit. Success remains
below0.85725 on both complete recovered groups, with freshly trained
predictions and locally fitted weights.

[Coverage and allocation evidence](experiments/upstream-high-capacity-original/coverage-audit.json).
