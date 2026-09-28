# Combining the two research branches

The other agent's `score-iteration-2` branch reports B6 at 0.8804020935:
LightGBM over 15 raw predictors, two kNN support features and training-derived
metadata. The moonshot branch's NN2 reaches 0.8803262843 using a different raw
bank and a neural correction to contextual ridge. Similar aggregate errors
do not establish that their predictions are interchangeable.

## Immediate result: fixed equal blend

Use a predeclared 50/50 average of the two saved cross-fitted prediction
columns, then clip to [1, 5]. There is no weight optimization or second-level
model fitting. Both source models exclude the same outer user fold when fitting
their blenders. The constant combination therefore preserves that exclusion.

| Model | Probe development CV RMSE |
| --- | ---: |
| Other branch B6 | 0.8804020935 |
| Moonshot NN2 | 0.8803262843 |
| Fixed equal average | **0.8786685218** |

The average improves every fold relative to both components. Relative to B6,
the gain is 0.0017335718 RMSE. This demonstrates useful complementary errors;
it does not isolate whether the source is the predictors or blending methods.
The paired user-cluster 95% interval for change in MSE versus B6 is
[-0.00323710, -0.00286186], conditional on these selected models. The recorded
`outside_pilot` subset is inherited from earlier GP analysis and is not an
independent test set.

Base checkpoints and research decisions still use probe labels. All scores are
development estimates, not a hidden-test audit. Fitting another blender directly
to these cross-fitted columns would require additional nesting; this experiment
does not do that.

## Controlled raw-bank comparison

To separate predictor and blender contributions, run the same nested neural
protocol from [NEURAL_BLEND.md](NEURAL_BLEND.md) on:

1. The other branch's exact 15 raw columns.
2. Those same columns plus six moonshot columns: three local GP variants, one
   temporal GP, and two residual-factor models.

Both runs also output the matched contextual ridge control. These comparisons
use four context variables; they do not introduce B6's richer LightGBM metadata.

On the original 15-column bank, contextual ridge reproduces B5 at
0.8816901997 and neural correction reaches 0.8808815323. B6's LightGBM result
is better on that bank, so the evidence supports combining the approaches rather
than replacing its blender with the neural implementation.

The completed comparison shows that the raw moonshot predictors transfer:

| Raw predictor bank | Contextual ridge | Ridge + neural correction |
| --- | ---: | ---: |
| B6's 15 raw columns | 0.8816901997 | 0.8808815323 |
| Same 15 + six moonshot columns | 0.8796294478 | **0.8788478728** |

Holding the blender fixed, adding the six columns improves RMSE by 0.0020607518
for ridge and 0.0020336595 for neural correction. Every fold improves for both
methods. This ablation establishes usefulness of the six-column group, without
attributing the gain to any one member.

Applying the same predeclared 50/50 rule to B6 and the combined-bank neural
ensemble reaches **0.8782199562**, better than both components in all five
folds. That is 0.0021821373 below B6 and 0.0004485656 below the first equal blend.
The final output is `artifacts/UNION-combined-equal.f32`. No blend weights were
searched. These are single-seed results on the development probe.

**Transfer recommendation:** retain B6, add the six raw moonshot columns, fit the
nested neural correction, and average those two predictions equally. Future
work can test the GP columns directly in LightGBM or ablate individual members;
neither result is claimed here. Model weights for inference on new pairs are
not exported by this experimental neural script.

## Provenance and reproduction

The completed B6 manifest is copied to `experiments/UNION-b6-source.json`.
Every one of its 15 raw columns and its final prediction column was checked
against the manifest SHA-256 before copying into `artifacts/other-branch/`.
This includes fixed AutoRec snapshots from the completed blend, without reading
the live A3 trainer's changing output. All original files and processes are
untouched. Predictor training on that branch remains the other agent's work.

```sh
export MOONSHOT_PYTHON=/path/to/python-with-numpy-and-pytorch
export VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
DATA=/path/to/netflix.bin
"$MOONSHOT_PYTHON" scripts/moonshot_union_bet.py --data "$DATA" --mode average
nice -n 15 "$MOONSHOT_PYTHON" scripts/moonshot_union_bet.py --data "$DATA" --mode other
nice -n 15 "$MOONSHOT_PYTHON" scripts/moonshot_union_bet.py --data "$DATA" --mode union
"$MOONSHOT_PYTHON" scripts/moonshot_union_bet.py --data "$DATA" --mode union-average
```

The fixed-average output is `artifacts/UNION-equal.f32`; its report is
`experiments/UNION-equal.json`. The run requires the frozen input files; large
prediction artifacts are local and not committed.
Complete raw-bank reports, source/output hashes and software versions are
recorded in `experiments/UNION-round-manifest.json`. Final paired statistics
against B6 are in `experiments/UNION-combined-paired.json`.
