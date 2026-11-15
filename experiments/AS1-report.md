# AS1: bounded pilot rejected

The registered viewer-conditioned event correction failed the outer pilot gate.
No component was promoted and no full Probe or ensemble comparison was launched.
This is a reproducible rejection of the bounded configuration, not a rejection of
all possible audience-shock models.

The pilot sampled 500 complete user histories by seed 711, totaling 111,114
training interactions. It fitted a fresh weak bias baseline on 66,557 upstream
rows, correction models on 22,369 residual rows, selected ridge on 11,029 inner
rows, and scored once on 11,159 outer rows. All rating-dependent upstream inputs
exclude residual, inner and outer partitions. The fixed random four-dimensional
item geometry is independent of ratings; profiles exclude the candidate movie.
It does not reuse the potentially contaminated K1 checkpoint or R9 predictions.

| Arm | Outer RMSE |
| --- | ---: |
| Frozen weak bias base | 0.9479354382 |
| Smooth viewer-conditioned movie-time | 0.9479538511 |
| Abrupt event bias | 0.9479576511 |
| Full viewer-conditioned abrupt correction | 0.9479576704 |
| Eight-week-shift placebo | 0.9479340106 |

Full correction worsened RMSE by 0.00002223 versus the base, rather than achieving
the required >=0.0001 improvement over the strongest control. Its paired
user-cluster bootstrap 95% MSE-difference interval versus base was
[-0.00005077, 0.00017224], which does not favor correction reliably. All arms
selected ridge 10 using inner labels. The JSON report retains all five user-fold
scores, per-user differences, and failed arms.

The detector scanned 113 supported movie/windows; 19 stratified null permutations
gave a maximum-score cutoff of 2.15299. Three merged events survived, covering
only 8 outer rows (0.07169%). Full correction worsened event-row RMSE from
0.735594 to 0.774528. At this coverage, closing the historical planning gap alone
would require about 5.436 event-row MSE improvement, exceeding the base event-row
MSE of about 0.5411. This is a feasibility calculation for this sample, not a
private-test estimate or a general coverage bound.

The optimized run took 38.27 seconds (2,903 source rows/second), used one compute
thread and roughly 155 MiB RSS in a process snapshot. The initial 100-user timing
run was terminated after finding an unnecessary sparse-movie scan; its manifest
and source archive remain. The equivalent support precheck reduced the rerun to
1.70 seconds. Other training processes and neighboring workspaces were untouched.

Five tests pass: target-movie exclusion, nonzero-model outer-label isolation,
half-open boundaries and overlap normalization, ordered export with zero outside
support, activity-event recovery/no-shift null, and stable-volume composition
nomination (the final test covers both activity and null). No new dependencies.

Limitations: weak baseline and random geometry; very small user sample and event
coverage; coarse null budget; date-shift placebo is not support matched; no
stratified popularity/activity coverage analysis, strong clean geometry, or
matched R9 nested blender evaluation. The pilot gate fails before those expensive
follow-ups are justified. Event weeks are dataset-day bins, not displayed calendar
dates; movie indices are zero based. This retrospective training-only screen uses
no Probe labels and establishes neither incremental R9 value nor the private-test
10% target.

See `AS1-report.json` for inputs, hashes, events, confidence, coverage, all scores
and run records; `AS1-plan.json` for the bounded configuration. Full pre-scoring
manifests, logs and source snapshots are retained as `AS1-pilot-20261115.*`.
The generic experiment runner's Probe metric label does not apply to this fresh
training-only outer score. Reproduction commands and implementation details are
in `scripts/AUDIENCE_SHOCKS.md`.
