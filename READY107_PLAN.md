# Next fixed raw-predictor batch: own107

Retain own105 exactly and append both ready public-schema columns together:
our published CF96 epoch10 snapshot pair and the completed original temporal60
pair. The temporal model completed both23-epoch phases, with every native output
hash, vector shape/dtype and finite value checked. CF96 retains its original
320-hidden/96-rank/95-epoch trajectory; its epoch10 pair is complete, while the
whole95-epoch model continues training.

| Known-ProbeX ridge diagnostic | RMSE |
| --- | ---: |
| Own105 control | 0.8625974317905872 |
| Fixed all-two addition | 0.8625801963952328 |

Independent five-user-fold statistics, solves and vector replay passed, with
maximum discrepancy5.56e-12. Four folds improve and one worsens; the lower
overall score is retained without an all-fold or minimum-gain veto. This is
preparation for a nonlinear candidate, not a new global development or qualifying
record.

`scripts/upstream_ready107.py` prepared183 ridge columns and243 features on the
same1,128,164 rows. Every retained/new cache column and unchanged label/user/fold
identity was checked. The baseline vector binds to the independently replayed
all-two comparison. The current own105 neural run remains unchanged.

At the next candidate boundary, use the completed own105 comparison as well as
the current verified NN103 best. Prefer batching any newly ready **complete
original latent support families** before another neural refit. Those non-rating
supports must enter as context features, without rating clipping or centering.
They do not add raw predictor/ridge columns. Keep the same user folds and exclude
qualifying outcomes from every fit and candidate choice.

The best verified qualifying result remains0.8609916544 /0.8619938614. Both
groups still miss0.85725. All numerical prediction features in this batch were
trained by us; no downloaded pretrained prediction features are used.
