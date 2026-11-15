AS1 implements a bounded, NumPy-only training-only screen for the hypothesis in
`AUDIENCE_SHOCK_PLAN.md`. Run from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/experiment.py \
  --name AS1-new-run --strategy audience-shock -- \
  .venv/bin/python scripts/audience_shocks.py \
  --data data/processed/netflix.bin --users 100 \
  --output artifacts/audience-shocks/AS1-new-run
.venv/bin/python -m unittest discover -s tests -p test_audience_shocks.py
```

Use a new name and output directory for every run. The experiment runner saves
source snapshots. The pilot records the source dataset hash, source revision,
NumPy version, selected users, fixed configuration and output hashes. Its generic
runner metric label says Probe; the pilot report actually scores a fresh
training-only outer partition. No Probe values are read by the pilot.

This first implementation deliberately uses a weaker bias baseline and fixed
random item geometry because clean held-out checkpoint inputs are unavailable.
A profile is the regularized sum of centered other-movie ratings times these
item vectors. It subtracts every candidate-movie contribution. No upstream
rating-dependent feature is fitted on residual, inner or outer rows. Out-of-sample
residuals come from a separate 20% partition, rather than in-sample predictions.
Inner validation selects ridge strength; upstream parameters and event boundaries
stay fixed during downstream refitting. Smooth and full arms use the same profile
dimensions. A hierarchical penalty shrinks event parameters toward a penalized
shared response. Event windows are half open and adjacent detections merge;
overlap normalization applies to smooth and placebo responses as well.

The report includes every arm, five user-fold scores, per-user paired MSE sums,
a user-cluster bootstrap interval against the base, event RMSE and coverage.
`state.npz` retains partition IDs, source rows and coefficients; all upstream
state is reproducible from the immutable input and manifest. `outer.f32` is a
complete prediction column in outer source-row order. Raw correction, confidence and distinct-user support exports share that order. The export function also accepts Probe
IDs/dates in their original order without requiring labels. It does not clip
predictions, so corrections are exactly zero outside events. Full Probe export
and blender integration remain gated, and must not be inferred from this pilot.

The null is a coarse timing/association null, not a causal model. Nineteen
permutations provide only a coarse maximum-score cutoff; a full pilot needs a
larger frozen calibration budget. Shifting placebo dates can reduce support;
report that limitation rather than treating placebo performance as definitive.
A 100-user sample may fail support simply because histories are sparse. A failed
screen rejects that bounded configuration, not every audience-shock method.
Fixed random geometry is a feasibility approximation; stronger conclusions need
clean learned geometry, richer stratified coverage reports, matched placebo
support, paired comparisons against each control, and the frozen R9 bank with
its original nested blending protocol. No expensive follow-up launches
automatically. The private-test target remains unverified.
