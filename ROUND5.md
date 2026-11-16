# Round 5: user and movie context

The R4-C residual-tree ensemble scored 0.8754534170 on the original probe in
five user folds. R5-meta adds seven features computed without probe ratings to the same 55 raw
prediction columns: movie age at rating, user's smoothed one-star and five-star
fractions, unrated interaction count, count of active rating days, same-day
activity relative to that user's average day, and movie rating standard
deviation. The additional features are implemented in `extra_meta()` in
`scripts/round4_boost.py`. All other training settings match R4-C.

| Run | Full probe development CV RMSE | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---:|---:|---:|---:|---:|---:|
| R4-C | 0.8754534170 | .8765894 | .8752087 | .8716600 | .8769770 | .8768432 |
| **R5-meta** | **0.8748094089** | .8759153 | .8744987 | .8711588 | .8763427 | .8761425 |

The saved float32 prediction file scores 0.8748094090 when independently read,
converted to float64, and clipped to [1, 5]. Every fold improves. The paired user-cluster descriptive 95% interval
for the MSE difference is [-0.0012695, -0.0009849]. It does not adjust for
multiple rounds of probe-based model selection.

R5's most used new features by mean normalized tree gain are the user's
five-star fraction (0.0357), movie rating standard deviation (0.0342), movie
age (0.0327), user's one-star fraction (0.0317), and same-day activity ratio
(0.0304). Gain ranks describe tree usage, not causal attribution. The selected
round counts were [1606, 1300, 1255, 942, 1379].

The gain holds across user-history sizes. It is largest for users with fewer
than 20 training ratings (−0.00097 RMSE), then 20–49 (−0.00077), and remains
positive for users with at least 500 (−0.00030). These are descriptive slices
of the same development probe.

The output is `artifacts/r4/R5-meta.refit.f32`, SHA-256
`9f5466f4f7cb8e8715053ebb376eaa141b61315f3f889c0d4f848c2f452d1f6a`.
The complete input hashes, configuration, fold metrics, and LightGBM version
are archived in `experiments/round4/R5-meta.json`; the log is `experiments/R5-meta.log`.
The paired comparison is `experiments/R5-meta-paired.json`.

This is still a development estimate. Base model checkpoint choices and many
research decisions used the original probe; the five user folds protect only
the final blender's fit and early stopping.

A copy of the report is committed as `experiments/round4/R5-meta.json`.

## In progress

- **R5-meta-s3**: R5-meta averaged over three LightGBM seeds (`--seeds 2026 7 13`).
  Each seed selects its own round count on the inner split and is refit on all
  outer-training users; the fold corrections are averaged.
- **IA1 item-based AutoRec** (`scripts/autorec_item.py`, experiment
  `IA1-autorec-item512`): each movie is encoded from its raters' centered
  training ratings plus an indicator of every known rater, with a sparse first
  layer and reconstruction evaluated only at the needed users. 512 sigmoid units,
  input drop .25, code dropout .2, AdamW (lr 3e-3, weight decay .01),
  250k-rating batches. Probe RMSE 0.98287, 0.96286, 0.95816 after epochs 1–3;
  it is a new model axis for the blend, not a standalone contender.
  `torch.bincount` is very slow on MPS (0.67 s per call), so counts use `index_add`.

## Reproduce

```sh
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1
$PY scripts/round4_boost.py --data data/processed/netflix.bin --output artifacts/r4/R5-meta \
  --learning-rate .02 --rounds 8000 --extra $(cat artifacts/r4/extra-C.txt) \
  --extra-support $(cat artifacts/r4/support-B.txt) artifacts/r4/knn256td-T6R.support.f32 \
  --extra-meta
```
