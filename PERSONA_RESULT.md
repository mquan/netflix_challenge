# Day-gated recurring personas: isolated implementation and result

Branch: `codex/household-personas`, based on `origin/codex/local-context` at `ccbd2a2`. Worktree: `/Users/quan/workspace/netflix_personas`. User-specified success target: **private-test RMSE ≤ 0.85725**. No hidden private-test labels are available locally. This report concerns the original Netflix Probe development set.

## Method and data boundary

`src/persona_base.cpp` exports training residuals and Probe predictions from a frozen 256-factor joint checkpoint. The Probe prediction file matches the previously saved `joint256-reg01.f32` byte for byte. `scripts/persona_fit.py` projects frozen item factors onto eight principal directions. Within each account with at least 100 training ratings, it fits a one-person ridge control and a two-person mixture of ridge residual models. The two persona offsets recur across rating days; the gate uses query-movie affinity, target-excluded same-day training responsibilities, and other same-day implicit movie IDs. Fits use only training ratings; the Probe labels are read after predictions are saved for scoring. The new columns are additive candidates for the existing R9 blend.

The geometry checkpoint is `artifacts/joint256-reg01.bin` from the original checkout, linked read-only into this worktree. It is older and weaker than the K1 model in the original proposal. K1's split checkpoint is tracked in the fetched branch, but it was not assembled or used for this pilot. This limits the test's ability to falsify a K1-based version. The full R9 input bank (79 prediction columns, 46 support features) is tracked in the fetched branch and stays unchanged.

## Feasibility and direct result

`experiments/PERSONA-feasibility.json` finds 543,554 of 1,408,395 Probe rows (38.6%) with ≥100 training ratings for the account and another same-day implicit event. Of those, 236,932 (16.8% of all Probe rows) have a same-day *training rating*. For the 38.6% group, closing the provisional development shortfall solely within that group would require about 0.0101 MSE reduction per eligible row. That is a demanding threshold, not a predicted gain.

The full fit covers 697,045 Probe rows from 226,268 eligible accounts and takes 124 seconds after frozen residual export. Overall Probe RMSE:

| Prediction | RMSE |
|---|---:|
| Frozen joint base | 0.89516641 |
| One-person residual correction | 0.89512799 |
| Two recurring personas | 0.89503934 |

The two-persona gain versus its one-person control is **0.00008865 RMSE**. A deterministic synthetic fixture with two recurring raters confirms that the implementation can recover a large two-person signal when it exists (`tests/test_persona.py`). That fixture does not forecast a gain on Netflix.

The preplanned mechanism ablations use the same frozen geometry, eight dimensions, ridge penalty and four EM iterations:

| Two-persona gate | Direct Probe RMSE | One-weight R9 change |
|---|---:|---:|
| Query movie only; no day evidence | 0.89505188 | −0.00000095 |
| Query movie + other same-day movie IDs | 0.89505022 | −0.00000123 |
| Query movie + IDs + same-day training ratings | **0.89503934** | **−0.00000145** |

The day information has the expected direction, but its incremental value is tiny. These comparisons use the Probe for diagnosis and do not justify trying many more day-weight settings on the same labels.

## R9 ensemble comparison

`scripts/persona_screen.py` fits one scalar correction weight on four held-out-user folds and scores the fifth against the frozen R9-LAT-A2h out-of-fold column. The two-persona correction changes R9 from 0.8668706424 to **0.8668691916**: a gain of **0.00000145 RMSE**, with fold 4 worse by 0.00000302. The one-person correction is worse overall. This is a diagnostic screen, not a full blender retrain or independent test. It fails the existing all-five-fold promotion rule.

The fixed full-bank LightGBM stage finished. Adding the persona prediction and confidence feature changes the tree CV from **0.86711222 to 0.86711754 RMSE** (worse by 0.00000532). Its five fold changes are **+0.00001980, +0.00010104, +0.00005005, −0.00004843, −0.00009589** (`experiments/PERSONA-P1-R9-tree-comparison.json`). The paired user-cluster 95% interval for its MSE change is [−0.00004062, +0.00005908], which includes zero.

The registered three-seed NN2 and fixed 50/50 average completed. The average scores **0.86684476 RMSE**, versus R9's **0.86687064**: a **0.00002588** improvement. Fold changes are **+0.00000197, +0.00004932, −0.00003061, −0.00005027, −0.00009985**. The paired user-cluster 95% interval for the MSE change is [−0.00008547, −0.00000426] (`experiments/PERSONA-P1-R9-final-comparison.json`). This interval is descriptive: it does not account for architecture selection, the earlier Probe-guided bank, or correlated blend refits.

**Decision: close this candidate; do not promote it.** Folds 0 and 1 worsen, violating the predeclared all-fold rule. Its 0.00002588 average gain is about 1% of the approximately 0.00225 *provisional development* shortfall associated with the **0.85725 private-test target**. The older geometry and in-sample base residuals limit what can be concluded about a different implementation, but they do not justify a costly Stage C retrain of this failed candidate. The retained promoted best remains R9-LAT-A2h at 0.86687064 development CV.

## Integrity and limits

- The 890 MB processed dataset is read through an ignored symlink; the original checkout and its models are not edited.
- The frozen joint exporter reproduces the saved Probe prediction file byte for byte (`cmp` exit code 0).
- The synthetic recovery test passes; the exporter compiles with the existing warning flags and separate binary name.
- The current method fits residuals of a base model that saw all training ratings. Its training residuals are therefore in-sample, although Probe labels never enter the fit. A clean inner P2 training run would be needed before treating any small Probe gain as robust.
- This experiment's Probe result cannot establish the **0.85725 private-test target**. The Stage C transfer offset still needs remeasurement on the current bank.

## Reproduce the isolated experiment

The `data/processed/netflix.bin` and `artifacts/persona/geometry-joint256.bin` paths in this worktree are ignored read-only symlinks to the original dataset and frozen `joint256-reg01.bin`. The full R9 bank is tracked in this branch.

```sh
c++ -O3 -DNDEBUG -std=c++17 -Wall -Wextra -Wpedantic \
  src/persona_base.cpp -framework Accelerate -o build/persona_base
build/persona_base --data data/processed/netflix.bin \
  --model artifacts/persona/geometry-joint256.bin --output artifacts/persona/J256
OPENBLAS_NUM_THREADS=1 python scripts/persona_fit.py \
  --data data/processed/netflix.bin --model artifacts/persona/geometry-joint256.bin \
  --train-residual artifacts/persona/J256.train-residual.f32 \
  --base-probe artifacts/persona/J256.probe.f32 --output artifacts/persona/P1-full
OPENBLAS_NUM_THREADS=1 python scripts/persona_screen.py \
  --data data/processed/netflix.bin \
  --reference artifacts/r9/R9-LAT-A2h-20261010-NN2.avg.f32 \
  --base artifacts/persona/J256.probe.f32 \
  --one artifacts/persona/P1-full.one.f32 \
  --two artifacts/persona/P1-full.two.f32 \
  --output experiments/PERSONA-P1-screen.json
python -m unittest tests/test_persona.py
```

The full-blend commands and fold records are in `experiments/PERSONA-P1-R9-{s3,NN2}.log` and `artifacts/persona/P1-R9-{s3,NN2}.json`. They use the recorded R9 settings with the new two-persona prediction as one extra column and its confidence as one extra tree feature.
