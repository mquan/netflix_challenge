# Movie audience shocks: experiment plan

Status: closed — bounded pilot failed; no further exploration planned.
Written 2026-11-15. Result and stop decision recorded 2026-11-15.

## Result and decision

The AS1 implementation and bounded 500-user training-only pilot are complete.
The pilot failed its registered gate, and the user decided not to pursue this
method further. Do not launch larger samples, stronger geometry/base retraining,
Probe export, or ensemble integration for this plan unless explicitly reopened.
The remaining stages below are retained as the original experimental proposal,
not an active work queue.

The pilot used 111,114 training interactions, with disjoint upstream, residual-fit,
inner-selection and outer-scoring partitions. Its fresh regularized user/item
bias baseline and fixed random item geometry were explicitly weaker controls
than R9-LAT-A2h; clean reusable pilot checkpoints were unavailable.

| Arm | Outer RMSE |
| --- | ---: |
| Frozen weak bias base | 0.9479354382 |
| Smooth viewer-conditioned movie-time | 0.9479538511 |
| Abrupt event bias | 0.9479576511 |
| Full viewer-conditioned abrupt correction | 0.9479576704 |
| Shifted-date placebo | 0.9479340106 |

The full correction worsened RMSE by 0.00002223 versus the base and failed to
beat the temporal controls or placebo. Its paired user-cluster 95% interval for
MSE difference versus base was [-0.00005077, 0.00017224], providing no reliable
evidence of improvement. Three detected events covered only 8 of 11,159 outer
rows (0.07169%); event-row RMSE worsened from 0.735594 to 0.774528. Closing the
historical planning gap at that coverage would require about 5.436 event-row
MSE improvement, exceeding the observed base event-row MSE of about 0.5411.

Exploratory subgroup inspection found improvement for Million Dollar Baby
(2 rows, RMSE 0.5798 to 0.5378), but deterioration for Spider-Man 2 (4 rows,
0.7908 to 0.8154) and Harry Potter and the Prisoner of Azkaban (2 rows,
0.7600 to 0.8835). These samples are too small to identify a reliable winning
movie or user subgroup. Activity surprise dominated composition scores, and
viewer-conditioned corrections were nearly constant within each event, offering
little evidence for the proposed audience-taste mechanism.

This rejects the bounded configuration and records a practical decision to stop;
it does not establish that every possible audience-shock model would fail.
No incremental R9 ensemble gain or private-test target achievement was measured.
The optimized pilot took 38.27 seconds; five implementation tests passed.

Evidence: [readable report](experiments/AS1-report.md),
[full results and input manifest](experiments/AS1-report.json),
[bounded configuration](experiments/AS1-plan.json), and
[implementation and reproduction notes](scripts/AUDIENCE_SHOCKS.md).

---

## Objective and hypothesis

Test whether abrupt changes in a movie's audience create predictable rating
errors beyond the existing temporal models, and whether correcting those errors
improves the strongest available frozen ensemble.

A movie may suddenly attract viewers with different tastes after a release,
broadcast, or publicity event. The dataset does not identify these causes, and
this experiment will not claim to infer them. It detects changes in activity and
audience composition directly from dated interactions.

The hypothesis is that an event can affect different viewer types differently.
A movie-time bias captures the average effect; a viewer-conditioned event
correction may capture additional structure. An audience change alone does not
establish predictive value: existing user factors may already explain it.

This is a proposed additional prediction column, not a replacement ensemble or
a separate blender fitted independently for each movie.

## Relationship to existing work

The inspected repository records cover temporal factors and biases, user-day
context, neighborhoods, local GP models, RBMs, neural history models, and
nonlinear blending. The neighboring `netflix_personas` and `netflix_content`
workspaces contain household-persona and content-feature work, respectively;
the latter also records selection-objective and ordinal experiments.

No explicit experiment for detected, abrupt audience changes with
viewer-conditioned movie-event corrections was found in the records inspected.
This is not a claim of worldwide novelty or an exhaustive audit of every active
process. Before implementation, recheck those workspaces for overlapping work.
Do not modify their files, artifacts, queues, or running processes.

## Success target and comparison baseline

The private-test target recorded in `CONVERGENT_SOLUTION.md` is RMSE <= 0.85725,
10% below the 0.9525 private-test Cinematch reference. Do not substitute quiz
references or interpret Probe scores as private-test scores.

That document records R9-LAT-A2h at 0.8668706425 Probe development CV and a
provisional development target around 0.86462 based on an earlier transfer
rehearsal. The resulting roughly 0.00225 RMSE gap is planning context, not a
validated conversion for this method. Verify the latest promoted result before
freezing the comparison. Without private-test labels, do not claim the 10%
target is achieved.

## Stage 0: input audit and isolation

1. Locate the processed training data, exact Probe row ordering, dated
   interactions, and reusable user/item geometry.
2. Verify which data trained every proposed input checkpoint. A checkpoint
   trained on a proposed pilot holdout is not a valid pilot input.
3. Locate the strongest reproducible ensemble, its completed prediction bank,
   fold definitions, and blend recipe. If unavailable, name a weaker control
   explicitly; do not present its incremental gains as gains over R9.
4. Record hashes, source revision, split definitions, and dependencies in a new
   experiment manifest. Never read a prediction artifact still being written.
5. Inspect available memory and concurrent workload before launching compute.
   Use a unique experiment prefix and separate output directory. Do not pause
   or restart other strategies.

Deliverable: input manifest and a decision on whether a clean pilot can reuse
existing artifacts or requires a new training-only base fit.

## Stage 1: identify candidate movies and events

Scan every movie; do not choose titles based on observed validation errors.

### Activity signal

Aggregate counts of distinct rating users by movie and calendar week. Compare
the movie's share of all rating activity in a window with its surrounding
baseline, rather than comparing raw counts alone. This reduces confounding
from platform growth. Account for sparse counts and changing global activity;
report counts alongside any standardized surprise score.

Start with weekly bins and adjacent four-week comparison windows. Treat these
as initial pilot settings, to be frozen or changed only on inner validation.
Account for truncated windows at the beginning and end of the dataset.

### Audience-composition signal

For each movie-window, summarize the taste vectors of users who rated it.
Compare that distribution with the movie's surrounding audience, using a
regularized, variance-normalized distance between mean vectors as the first
implementation. Center against contemporaneous platform-wide audience changes
so general changes in the user population do not automatically become movie
events. More complicated distribution distances are outside the first pilot.

User taste vectors must exclude ratings of the movie being examined. Simply
reusing a user factor trained on that movie does not satisfy this requirement.
One practical construction is a regularized profile formed from frozen item
vectors and a user's other training ratings, with the candidate movie's
contribution removed. Item geometry must itself obey the pilot training split.

### Candidate selection

- Allow activity surprise or audience-composition change to nominate an event;
  requiring both would miss audience substitutions with stable volume.
- Require adequate distinct users on both sides and shrink sparse estimates.
  Select the support threshold on inner training validation, not Probe.
- Calibrate detection against null permutations that preserve movie totals and
  broad calendar activity. For example, permute movie assignments among dated
  user events within broad time strata. Document exactly what each null
  preserves and destroys; a naive uniform date shuffle is not sufficient.
- Control false discoveries across the many movies and windows using an
  empirical null cutoff, chosen before the outer evaluation.
- Merge adjacent flagged windows into events and define deterministic handling
  of overlaps. Fix boundaries before examining outer held-out rating values.

The detector uses interaction identities, dates, and training-derived profiles,
not the candidate event's observed rating errors. Movie titles may be displayed
after selection for interpretation but do not influence selection.

Deliverables: event table with movie ID, start/end dates, support, activity and
composition scores, confidence, and the fraction of evaluation rows covered.

## Stage 2: feasibility screen

Use inner training validation to inspect whether the frozen base model has
predictable errors within detected events, especially errors varying with
viewer taste. Report coverage by movie popularity, date, and user activity.

Quantify the required subgroup gain before committing to larger training. If
events cover fraction f of rows, an overall MSE improvement D requires average
event-row improvement D/f when other rows are unchanged. For the historical
provisional gap, D is about 0.00390. At 10% coverage, this would require about
0.039 event-row MSE improvement to close the entire gap alone. This calculation
is a feasibility benchmark, not an expected result or an oracle estimate.

Stop if reliable events cover too little data for plausible impact, the clean
input pipeline cannot be built within the pilot budget, or inner validation
shows no stable residual signal beyond existing temporal controls. Coverage
alone does not justify training.

## Stage 3: fit the correction

For user u, movie i, and date t, start with:

```text
prediction(u,i,t) = base(u,i,t) + correction(u,i,t)
correction(u,i,t) = sum over applicable events e:
    confidence(e) * [event_bias(e) + dot(event_direction(e), profile(u,-i))]
```

The profile excludes movie i. Set correction to zero outside supported events.
Use a low-dimensional fixed projection of profiles for the first pilot, strong
ridge regularization, and hierarchical shrinkage toward a shared event
response. Sparse event-specific parameters shrink toward zero or the shared
response. Confidence is derived from permitted support statistics, not outer
evaluation errors. Specify overlap normalization so adjacent events cannot
accidentally multiply corrections.

Fit on training residuals. Prefer out-of-fold residual targets: in-sample
residuals from an overfit base can produce misleading corrections. If clean
cross-fitted base predictions are unavailable, account for their generation in
the budget rather than silently using contaminated targets.

The model is deliberately small. Do not add a neural event detector, external
release metadata, or a broad hyperparameter sweep in the first pilot.

## Stage 4: controls and leakage protection

Evaluate these predeclared arms with the same splits and comparable capacity:

1. Frozen base, with no new correction.
2. Additional smooth movie-time correction, with the same profile inputs where
   applicable, to test whether any gain is simply extra temporal capacity.
3. Abrupt event bias, shared by all viewer types for a given event.
4. Full abrupt event correction conditioned on viewer taste.
5. Matched placebo events with shifted/permuted dates, preserving support as
   closely as possible, to test whether detected timing matters.

Use a training-only holdout with inner selection and a separate outer scoring
partition. Freeze detection settings, regularization, projection dimension,
and fit budget before scoring the outer partition. Prefer a fresh outer
partition if existing P2 labels have already been repeatedly inspected; if
reusing P2, label the result as development evidence.

All rating-dependent inputs, including geometry, profiles, base predictions,
residual targets, and blend weights, must respect these splits. Exclude target
ratings from profile features and training-target context. For residual-model
cross-fitting, regenerate rating-dependent upstream features within each fold.
Test that changing outer held-out rating values leaves all fitted parameters
and predictions unchanged.

The initial pilot should detect events from training interactions only. A
separate, explicitly declared transductive variant may later use unlabeled
Probe/qualifying IDs and dates under the repository protocol. Never use their
rating values. Using later dates is acceptable for retrospective competition
completion but does not establish prospective forecasting performance; report
which setting was evaluated.

## Stage 5: evaluate incremental ensemble value

Only after the pilot gate passes:

1. Refit the frozen recipe on the permitted full training split.
2. Emit a complete Probe-order float32 prediction column. Outside events, its
   value equals the base prediction; it is not a sparse file or missing value.
   Also emit the raw correction and support/confidence features for auditing.
3. Add the predeclared corrected column and confidence feature to the frozen
   ensemble. Use the existing folds and nested blender selection protocol.
4. Compare against the identical frozen bank and blend recipe without the new
   component. Keep seeds and training budgets fixed across arms.
5. Report overall RMSE, each user-fold RMSE, paired per-user error differences,
   user-cluster bootstrap intervals, event-row RMSE, and coverage. Inspect
   concentration of gains in a few movies and sensitivity to calendar period.

Proposed gates, to register before scoring:

- Pilot: full correction beats the frozen base and both genuine temporal
  controls on outer held-out ratings, with an overall gain of at least 0.0001
  RMSE over the strongest genuine control. Placebo results must not provide an
  equally convincing explanation of the gain.
- Promotion: at least 0.0001 overall RMSE improvement over the frozen full
  ensemble, improvement in all five user folds, and a paired user-cluster 95%
  interval favoring the new model. These are proposed practical thresholds,
  not expected gains or a correction for repeated research selection.
- A passing component may still explain only a small fraction of the target
  gap. Do not relax a gate after seeing results or promote on event rows alone.
- If the component materially advances the target, repeat the transfer
  rehearsal with the current bank before making private-test estimates.

## Implementation artifacts and verification

Suggested new files, with names finalized before implementation:

- `scripts/audience_shocks.py`: split-aware summaries, event detection, fitting,
  prediction export, and saved configuration.
- `tests/test_audience_shocks.py`: meaningful tests for exclusion of target
  ratings, split isolation, event boundaries, zero correction outside support,
  deterministic outputs, and Probe row-order preservation.
- `experiments/AS1-plan.json`: frozen inputs, hashes, arms, gates, and budget.
- `experiments/AS1-report.json` plus a readable report: results including failed
  arms, runtime, coverage, and limitations.

Use the repository experiment runner where compatible, retaining source
snapshots and input/output hashes. First verify on synthetic events with known
audience shifts and on a no-shift null fixture. Then run a small real-data
timing sample before estimating full runtime. Avoid new dependencies unless
existing NumPy/scientific tooling cannot support the implementation.

## Timing, resource budget, and checkpoints

Planning estimate: 4–8 hours for a credible first pilot if clean reusable
geometry, residual predictions, and split infrastructure are available:

| Work | Estimated elapsed effort |
| --- | --- |
| Input audit and weekly summaries | 1–2 hours |
| Event detector and candidate inspection | 1–2 hours |
| Correction, isolation checks, controlled pilot | 2–4 hours |

A full ensemble evaluation may add several hours to a day of compute. These
are unbenchmarked estimates. New cross-fitted base training, missing artifacts,
memory pressure, or competing jobs can extend them substantially.

At approximately two hours, report input readiness, measured throughput,
candidate coverage if available, and a revised compute estimate. If a clean
pilot needs major upstream retraining, stop the bounded pilot and document that
dependency. Do not launch an open-ended training queue under the initial
estimate. Do not interfere with concurrent methods to meet the estimate.

## Completion criteria

The experiment is complete when it produces either a reproducible rejection
with the failed gate and evidence, or a reproducible prediction artifact and
matched ensemble comparison meeting the registered promotion criteria.

Neither detected events nor an improved standalone predictor constitute
success. The desired result is additional overall predictive accuracy beyond
the existing ensemble, with an honest account of its contribution toward the
10% target.
