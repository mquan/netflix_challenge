# First split3 outputs add a small independent ridge gain

The fixed addition of our original split3 attention terminal and published ep08
outputs improves the known-ProbeX five-user-fold ridge diagnostic:

| Fixed bank | RMSE |
| --- | ---: |
| Own103 control | 0.8626072996291720 |
| Own103 plus both split3 attention columns | 0.8625974317905872 |

All five folds improve in this diagnostic. Selection did not require all-fold
agreement: both columns were fixed from the source schema before this comparison.
An independent implementation recomputed fold statistics and ridge solves and
replayed both saved vectors; maximum prediction discrepancy was 5.56e-12.

The source model completed both original 12-epoch phases. Its holdout contains
every known-ProbeX row, with an exact unique row map and matching known labels;
the completion check binds the earlier full scan proving those rows absent
from split3 training. Native qual outputs are retained without projection.
No downloaded numerical features or qualifying labels enter this experiment.

This is a ridge diagnostic, not a new global blend score or qualifying result.
The current own103 neural iteration continues uninterrupted. Once its production
decision has completed, these two verified predictors can be added together at
the next fixed candidate boundary, with any further completed native outputs
batched before freezing that candidate. Keep the existing nested user folds.

Evidence is in `experiments/upstream-split3-105-screen/`, produced by
`scripts/upstream_split3_105_screen.py` and independently replayed by
`scripts/verify_upstream_split3_105_screen.py`.
