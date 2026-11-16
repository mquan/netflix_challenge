# Training-only validation pilot

The original probe has been used repeatedly to choose model checkpoints and
research directions. This pilot builds an additional validation set from labeled
training rows. It does not make the original probe a fresh test set, but it
allows a correction model to learn from ratings that its source model did not
see during training.

## Split

`scripts/train_holdout.py` writes another `NFPDAT2` dataset. It preserves all
implicit user/movie/date pairs, including the original probe and the selected
training rows. Only the selected training **ratings** move to the new probe;
original probe ratings are omitted. Every user retains at least one training
rating. The sampler uses each original probe user's dates to select nearby
training rows. It does not inspect either set of rating values.

| Statistic | Original probe | Random 1.4% training sample | Matched training sample |
|---|---:|---:|---:|
| Rows | 1,408,395 | 1,386,219 | 1,407,566 |
| Mean day | 2422 | 2103 | 2377 |
| Median remaining training ratings per user | 98 | 458 | 94 |
| Rows with fewer than 50 training ratings for their user | 32.0% | 4.2% | 33.9% |
| Five-star share | 27.3% | 23.0% | 27.2% |

The matched split is the one used below. It is still imperfect: its median item
support is 43,307 versus 37,576 on the original probe. Both generated binary
files are under ignored `artifacts/validation/`; their SHA-256 manifests are
archived in `experiments/validation/`.

## Source and correction

`--fixed-epoch` makes `joint` save the final specified epoch even if an earlier
one scores better on the new holdout. The pilot uses J3's existing architecture
and its previously chosen nine epochs. Changing the new holdout labels cannot
change its parameters or predictions; the unit test checks this contract.

`scripts/train_only_residual.py` uses J3's predictions for the matched holdout,
then fits a small LightGBM model to the residual. It selects the tree count on a
user-disjoint portion of the matched holdout, refits on all matched rows, and
applies the correction to the original probe J3 column. Changing original
probe labels leaves the resulting predictions unchanged. Its real-probe score
is a development estimate because J3's architecture and epoch came from
earlier probe research.

```sh
PY=/path/to/python-with-numpy-and-lightgbm
"$PY" scripts/train_holdout.py --data data/processed/netflix.bin \
  --output artifacts/validation/train-matched-20260927.bin
build/netflix joint --data artifacts/validation/train-matched-20260927.bin \
  --model artifacts/validation/J3-matched-fixed9.bin --factors 128 \
  --iterations 9 --fixed-epoch --regularization .005 --decay .95 \
  --predictions artifacts/validation/J3-matched-fixed9.f32
"$PY" scripts/train_only_residual.py \
  --holdout-data artifacts/validation/train-matched-20260927.bin \
  --holdout-pred artifacts/validation/J3-matched-fixed9.f32 \
  --target-data data/processed/netflix.bin \
  --target-pred artifacts/joint128-reg005.f32 \
  --output artifacts/validation/J3-corrected.f32
```

This is a one-split pilot of out-of-sample residual correction. If the corrected
column contributes to the existing ensemble, the next version should use
multiple training folds so more than 1.4 million training rows can fit the
correction.

Before scoring the corrected original-probe column, the fixed ensemble check is
90% R5-meta plus 10% corrected J3. Its control is 90% R5-meta plus 10% the
uncorrected J3 column. Neither blend weight is fitted to probe labels.

## Result

The fixed nine-epoch J3 scored 0.8762377769 on the matched holdout. This
absolute score is much lower than J3's 0.8978688121 on the original probe,
despite the similar marginal distributions above; the split does not reproduce
all of the original probe's difficulty. We therefore judge the correction on
its original-probe predictions.

The residual tree selected 1,323 rounds using user-disjoint inner holdout
ratings. Its inner validation RMSE moved from 0.8756357 to 0.8703532. Applied
to the original probe, it moved J3 from **0.8978688 to 0.8934085**. This is
evidence that a correction trained without the target ratings transfers, but
J3 is still a weaker standalone model than the current ensemble.

| Original-probe prediction | Development RMSE |
|---|---:|
| R5-meta | **0.8748094** |
| 90% R5 + 10% original J3 | 0.8750240 |
| 90% R5 + 10% corrected J3 | 0.8749309 |

The correction improves its matched fixed-blend control by 0.0000932, but the
blend remains 0.0001214 worse than R5 alone. It is **not promoted**. The
full configurations and input hashes are in
`experiments/validation/J3-corrected.f32.json` and
`experiments/H1-blend-check.json`. Expanding this residual correction to many
training folds is not justified by this one-split ensemble result.

## Verification

`make test` passes, including the fixed-epoch checkpoint test. The new
`tests/test_train_holdout.py` checks rating-independent split assignment,
preserved implicit interactions, and retained training support.
`tests/test_train_only_residual.py` verifies that changing original probe
labels leaves correction predictions unchanged. Both new test files pass.
