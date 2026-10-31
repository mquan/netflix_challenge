# What the headroom map establishes

Independent interpretation audit, 2026-10-31. Read alongside
[HEADROOM.md](HEADROOM.md) and [STRATEGY.md](STRATEGY.md). This does not change
H1–H3's reported scores or the other session's implementation.

## Measured finding

`scripts/headroom_map.py` constructs leave-one-out pooled residual sums and
counts for each user/movie/time group, then fits a ridge correction. Its
results show that these **additive group-shift features** help much more on
the user side than on the movie side. This supports prioritizing observable
proxies for a user's current shift, which H4 is already investigating.

The measurements are diagnostic oracle results. They read other held-out
labels and are not predictions available at deployment. They are not a
promotion of a model or a change to the best dev-CV score.

## Limits on the conclusion

A small movie-group gain does **not** upper-bound the gain from all information
about movies or their other raters. Averaging a group's residuals discards
interactions with the target user. For example, let observable user type A and
movie type B be independent, equally likely +1/−1, and let an omitted rating
component be 0.4·A·B. Its population mean within every movie and within every
user is zero, yet a predictor that learns the interaction removes the entire
component. Adding independent noise does not change those zero group means.
Finite leave-one-out samples introduce noise and self-exclusion effects, so
this example is about the population information claim, not an assertion of
exactly zero finite-sample oracle gain.

E2's peer codes and ratings could describe a movie differently for different
target-user codes. Its attention head can therefore test interactions that
movie-group mean residuals do not measure. The existing results lower the
priority of a simple movie-wide correction; they do not establish an E2
ceiling of 0.0001. E2 itself remains unmeasured, and this distinction is not
evidence that E2 will help.

Likewise, the H2 b/a ratio is an observed ratio between two diagnostic oracle
gains. Multiplying H3's gain by that ratio is a planning heuristic, not a
validated Stage C correction or an upper bound on achievable improvement.
The user, user-day and movie features overlap; their gains cannot be summed
as independent opportunities. Combined-feature rows are the appropriate
measurements of the tested combinations.

A negative H4 result would rule out its particular feature/learner/protocol,
not every model of the unlabelled event stream. Calling any attainable gain
an upper bound requires a separate argument that the tested predictor
contains all relevant information and can exploit it adequately; pooled
means and a finite feature set do not provide that argument.

## Operational consequence

Keep the existing GPU queue and E5 pilots running. Await H4 before selecting
the next learning run. Keep E2 prepared and lower priority rather than
closing it on the movie-group oracle alone. Do not select a new recipe or
claim the 10% goal impossible solely from these diagnostic measurements.
The actual target and promotion gates remain those in GOAL.md.

## H4 result and next validation

The first `artifacts/headroom/H4-p2-proxies.json` reports fold-0 P2 RMSE
0.8477476174 → 0.8469337276 (gain 0.0008138899). This is a useful candidate
signal under its recorded protocol, not an original-probe or full-bank
promotion. Its `all_folds_improve` is false because the report evaluates only
fold 0; the other fold entries are null. Do not subtract this P2 gain from the
promoted 0.8668706425 dev-CV score.

The initial `scripts/headroom_proxies.py` constructs `base_cf` with
`headroom_map.crossfit(bank, y, folds)` over all five folds, before fitting the
residual learner on folds 2–4. Each training row's base prediction excludes its
own user fold, but the ridge fit producing that prediction includes fold-0
labels. Those base values also affect training residual targets and pooled
prediction features. Thus the complete residual-learning pipeline is not
isolated from outer fold 0, even though its feature builder does not directly
read query rating fields. The practical effect has not been measured.

Before treating this as an independently held-out gate, isolate the outer
fold throughout the pipeline: construct all training/selection residuals and
base-dependent features without outer-fold labels, and obtain outer-fold base
predictions from fits on the permitted training partitions. Freeze that
protocol before comparing scores. The other session owns H4; this note neither
modifies nor reruns it. A subsequent five-fold/full-bank evaluation and the
existing promotion requirements remain necessary.
