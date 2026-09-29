# Strategy reset: reproduce strong models before adding small variants

## Reference state

The best historical development score is R5-meta at **0.8748094089** on the
original probe. Its [manifest](experiments/round4/R5-meta.json) needs 61
prediction/support files. An audit on 2026-09-28 found only **one** matching
file anywhere under the available `artifacts/` directory, under a different
name. The former `/private/tmp/netflix-moonshot` bank and `artifacts/r4/` bank
are absent, and no backup is available. The R5 score must therefore remain a
historical result until its bank is regenerated. Do not compare a new model
only with an older ensemble and claim it beats R5.
The [R5 input audit](experiments/reference/R5-input-audit.json) can be rerun
with `python3 scripts/audit_artifact_bank.py experiments/round4/R5-meta.json`.

The strongest presently reproducible ensemble is B7. All eleven input hashes
(dataset plus ten prediction columns) match
[its manifest](experiments/B7-add-neighborhood128.json). Re-running the exact
five-user-fold quadratic contextual blend produced **0.8858460517** RMSE,
identical to the historical B7 result. This is the *provisional* control, not a
replacement claim for R5.
The [B7 input audit](experiments/reference/B7-input-audit.json) records the
hash check.

All scores on the original probe are development estimates. Base-model epochs
and research directions were repeatedly selected using probe labels; the five
user folds protect only blend fitting. The private 2009 score is on a different
split and is not a directly comparable local target.

## Published-method work

BellKor's [2009 report](https://www.asc.ohio-state.edu/statistics/dmsl/GrandPrize2009_BPC_BellKor.pdf)
describes two useful gaps in the current code:

* Its temporal neighborhood model jointly fits time-dependent biases, item
  pair weights, and a per-user time-decay coefficient, with normalization by
  the complete user history. The prior local N3 model used a fixed decay and a
  frozen baseline.
* Its PQ2 timeSVD++ variant uses 200 factors, frequency-specific movie factors,
  and 40 iterations. The existing local frequency-factor test was a six-epoch
  warm-start pilot, not a reproduction of PQ2.

The new `temporal-neighborhood` command implements the former learning
mechanism with the available capped residual item graph. It is still an
approximation: graph discovery is sampled, only positive-similarity top-k
edges are retained, and the graph pilot used 64 neighbors. The command
supports `--fixed-epoch` to avoid probe-based checkpoint selection in future
training-only validation runs. Numeric tests verify probe-label isolation,
nonzero decay learning, saved-model reload, and fingerprint rejection.

The two-epoch full-data pilot and matched zero-decay control show:

| Predictor | Standalone probe RMSE | Added to B7 CV RMSE |
| --- | ---: | ---: |
| B7 alone | — | 0.8858460517 |
| Prior fixed-decay N3 | 0.9108568308 | 0.8857123999 |
| Joint neighborhood, zero learned decay (TN0) | 0.9153227163 | 0.8853229372 |
| Same with learned user decay (TN1) | 0.9153348766 | 0.8853232230 |

TN0 and TN1 improve the provisional B7 bank by about 0.00052, but learned
decay contributes no measurable gain at this stage. These are two-epoch pilots;
they do not establish the converged value of the published model. The gain is
below our **0.001 RMSE per substantial training effort** continuation gate, so
we are stopping this 64-neighbor branch. TN0's source snapshot exposed a
default-argument rounding bug that disabled decay; the run is explicitly
classified as the zero-decay control. The default was fixed before TN1.
See [the ensemble comparison](experiments/TN-ensemble-comparison.json) and
individual TN0/TN1 experiment manifests.

## Factor-model decision

The 200-factor frequency-specific timeSVD++ run used the published factor
rate, regularization, decay, and a 40-epoch cap. It reached **0.8981178718**
standalone probe RMSE at epoch 6. Epochs 7 and 8 worsened to 0.8992588531
and 0.9006915938 while training error fell, so the run was stopped during
epoch 9 instead of spending more time on overfitting. The saved epoch-6
checkpoint reloads to an identical prediction column.

| Ensemble | Development CV RMSE |
| --- | ---: |
| B7 alone | 0.8858460517 |
| B7 + 200-factor frequency model | 0.8851865146 |
| B7 + HN3 date-local history | 0.8830615 |
| B7 + HN3 + 200-factor frequency model | 0.8826206638 |

The factor model adds only 0.00044 after HN3, below the continuation gate.
The [run manifest](experiments/PQ2-factor200-frequency.json) and
[ensemble comparison](experiments/PQ2-ensemble-comparison.json) preserve this
decision. BellKor's published RMSE values use the quiz set and train on probe
ratings, so the paper's 0.8777 score is not directly comparable to this local
0.8981 probe score.

## Reproducible compact reference

The residual-tree blender can now run from local `--extra` columns without
the absent union-bank manifest. Its report hashes every input. The matched
five-fold comparison used the same tree settings and context features:

| Reproducible bank | Development CV RMSE |
| --- | ---: |
| B7 ten columns + HN1/HN3 (C0) | 0.8802683275 |
| Same bank + TN0/PQ2 (C1) | **0.8792984371** |

C1 improves every held-out user fold over C0. Its 0.000970 gain is close to,
but below, the 0.001 continuation gate. We retain the already trained columns
in this compact reference, then stop tuning those families. The saved float32
prediction files independently reproduce both scores to rounding precision.
See the [C0 report](artifacts/compact/C0.json) and
[C1 report](artifacts/compact/C1.json), plus their experiment manifests under
`experiments/`. The [C1 input audit](experiments/reference/C1-input-audit.json)
verifies all fourteen column hashes. The local artifacts are ignored by Git;
the manifests are small enough to track in Git.

C1 is the strongest **currently reproducible** local reference, still
0.00449 RMSE behind historical R5. New predictor bets must beat C1 under the
same folds, then be checked against R5 once its bank exists. The next major
project choice is a deliberate R5 rebuild or a faithful later neural model
using C1 as the interim control. Further two-epoch neighborhood variants and
minor tree-feature changes do not meet the measured payoff threshold.

The original R5 bank should be rebuilt only through a dependency-aware batch
that validates every input hash and preserves the resulting files. It involves
many trained models and thousands of tree rounds, so do not silently substitute
a partial bank and label its score R5.
Seven identifiable source runs for missing R5 inputs alone total 6.7 hours of
recorded wall time, before upstream dependencies, dozens of derived columns,
or the R5 blender itself. Recorded wall time can include idle machine time.
