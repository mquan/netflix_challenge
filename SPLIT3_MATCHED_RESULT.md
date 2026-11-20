# More original known training labels help, but this bank is not competitive

Both matched arms completed all five user folds and three seeds. The seven
original completed pipelines were retained; recovery fitted only the three
missing expanded folds. Independent verification reopened all30 actual saved
models and reproduced both original eligible scalers, all ridge statistics
and every stored held prediction across all ten pipelines.

| Same canonical1,128,164-row evaluation | NN RMSE | Fixed half with tree154 |
| --- | ---: | ---: |
| Canonical-only meta-training | 0.8727302874706305 | 0.8624405419017913 |
| Complete original holdout meta-training | 0.8722568222466668 | 0.8622259939637812 |

The additional632,330 legitimate known ratings improve the NN by0.0004734652
and the fixed tree half by0.0002145479. Every held user's additional labels
were excluded from fitting and preprocessing. Both arms used the same73
own native predictors,130 original voting features and full H128/rank32
recipe; the evaluation population and user folds are identical.

The expanded arm remains behind the established0.8591267885072646 global
development score. No qualifying refit or global promotion follows. The
finding supports additional known labels as useful training evidence;
it does not show that this weaker source bank closes the qualifying gap.

[Complete independent replay](experiments/upstream-split3-matched-resume/independent-verification.json) ·
[Recovery completion](experiments/upstream-split3-matched-resume-queue/completion.json).
