# Bets beyond 0.87822

The fixed reference is `UNION-combined-equal.f32`, 0.8782199562 development CV.
The concurrent branch is training integrated neighborhoods, further factor
models and RBMs. This round uses a separate worktree and low-priority CPU jobs.
Its starting raw bank is the frozen 21-column union, not changing live outputs.

## A. Nonlinear user-space residual functions

Earlier GP models borrow residuals from movies in a user's history. Here each
movie learns a function over *users*, trained only on its training raters.
Normalize the frozen J3 effective user vectors (explicit plus implicit factors),
then apply 64 random Fourier features of an RBF kernel with gamma 4. Fit a
regularized least-squares residual model per movie, with penalties 1, 10, 100.
The residual target is the rating minus the frozen joint model prediction.

The matched control replaces the cosine map with a linear random projection
using the same Gaussian directions and width. Both include a penalized constant
feature. Variants 0–2 are linear; 3–5 are nonlinear. No probe labels fit the
kernel functions or select their parameters. The predictor bank is assessed by
adding each three-column family to the same contextual ridge ensemble.

[Rahimi and Recht's random-feature method](https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf)
approximates shift-invariant kernels with explicit low-dimensional features.
This application is our experiment, not a reported Netflix result from that
paper. Sixty-four features are a deliberately small first bet; they do not
reproduce an exact full user-user Gaussian process. Historical
[BellKor user-neighborhood models](https://www2.seas.gwu.edu/~simhaweb/champalg/cf/papers/BellkorNetflix2008.pdf)
also motivate seeking signal from the user axis.

### User-kernel result: park this family

The full run took 97.1 seconds, including feature construction and both families.
All 1,408,395 probe rows were covered. The best nonlinear standalone column is
0.8981731804, slightly worse than frozen J3 (0.8978688248).

| Contextual ridge bank | Development CV RMSE |
| --- | ---: |
| Frozen 21-column union | 0.8796294478 |
| Add three linear user-kernel variants | 0.8795813025 |
| Add three nonlinear user-kernel variants | 0.8796111957 |
| Add all six | 0.8795720343 |

The combined gain is only 0.0000574135; nonlinear features add 0.0000092682
beyond the linear family. This does not justify expanding the random-feature
rank in this round. The negative result is specific to this frozen geometry,
64-feature approximation and residual-fitting protocol, not a rejection of
all user-neighborhood models. These columns are not promoted to the final bank.

## B. Tree blending on the union bank, with full outer-training refits

B6 used 15 predictions and selected a boosted model on an inner training split,
leaving the inner validation users out of the final fit. Test its recipe on
the 21 raw union columns, the same 11 metadata features, and copies of the two
kNN support columns. The support files were stable across repeated reads and
are hashed in `experiments/BOOST-support-inputs.json`; the original B6 manifest
did not record their hashes.

For every outer user fold, choose tree count using the same user-disjoint inner
split. Save that selected model's predictions as one control, and train a new
model with the selected tree count on all outer training users as the treatment.
The second control/treatment pair learns a correction to contextual ridge,
instead of predicting the rating directly. Each inner/outer ridge fit uses only
the corresponding training users. Hyperparameters match B6: learning rate .05,
63 leaves, minimum leaf size 200, L2 10, feature/bagging fractions .8, patience
100 and a 3,000-tree maximum. Execution uses one thread.

The raw predictions are inputs; previous CV ensemble outputs are never training
features. Fixed equal averages with the existing union neural model will assess
complementarity without fitting a second-level blender. A different loss and
residual parameterization can produce useful errors even when the standalone
score does not improve. This is a hypothesis tested here, not an assumed gain.

Before the tree runs complete, the fixed combinations are declared as: each
of four tree outputs (direct/residual, selected/refitted) averaged equally with
the existing union neural output, plus a three-way equal average of the neural
output and the two refitted trees. No weights are optimized. Selecting among
these candidates remains a development decision.

### Completed tree results

| Tree target | Inner-selected model | Refit on all outer-training users |
| --- | ---: | ---: |
| Rating directly | 0.8786048801 | 0.8783968225 |
| Residual from contextual ridge | 0.8775907197 | **0.8774504869** |

The refitted residual tree improves the previous 0.8782199562 reference by
0.0007694693, in all five folds. The residual parameterization improves over
direct prediction with the same features and tree settings. Refitting adds a
further 0.0001402328. The fixed equal average with the old neural model scores
0.8776890856, so that two-way average is not retained.

The completed prediction column is `artifacts/BOOST-residual.refit.f32`.
Its recipe, inputs, tree counts, fold scores and hashes are recorded in
`experiments/ROUND3-progress.json`; large artifacts remain local.

After the predeclared combinations finished, the equal three-way average of
the old union neural model, direct refitted tree and residual refitted tree
reached **0.8773860815**. This improves all five folds versus 0.87822. Its margin
over the residual tree alone is only 0.0000644054, so the main gain comes from
the residual-tree bet. The best observed output is `artifacts/BETS-three-way.f32`;
all candidates, including weaker averages, are in `experiments/BETS-combinations.json`.

## C. A full rating-distribution head

Compare two networks with 32 tanh hidden units and the same inputs: all 21
predictions expressed as their mean and deviations, the four ridge contexts,
and the 13 cached metadata/support features. The control learns a scalar MSE
residual. The treatment outputs five logits correcting a Gaussian-shaped prior
centered on the ridge prediction, trains with cross-entropy, and predicts the
expected star rating. Their output heads necessarily have different widths.

Both use training-only normalization, AdamW (.001 learning rate, .01 weight
decay), 8192-row batches, a maximum of 60 epochs, and patience six. The inner
validation criterion is RMSE for both. A zero-epoch outcome falls back to ridge.
Refits use all outer training users. The stronger synthetic fixture verifies
exact outer-label isolation while requiring a nonzero selected epoch; the first
fixture allowed the fallback and therefore did not exercise the intended path.

[CF-NADE](https://arxiv.org/abs/1605.09477) motivates modeling rating distributions.
This is a small distributional blender experiment, not CF-NADE's autoregressive
rating-history architecture or its ordinal training loss. Its reported dataset
scores are not treated as comparable to this probe split.

The distributional head must be compared to its matched mean head before credit
is assigned to the loss. Each is also eligible for a fixed equal average with
the refitted residual tree; no learned weights on CV outputs are used.

**Status at the round-three commit:** the initial distributional head
selected zero epochs in all five folds, falling back to ridge at 0.8796294478.
Its truncated five-star Gaussian prior initially changes the ridge mean. A
calibrated-prior option now solves for logits preserving that mean; analytic
mean-preservation and outer-label isolation tests pass. Its full experiment is
a future bet. The matched mean head completed at 0.8782273971, and its equal
average with the residual tree reaches 0.8773995093. The initial distributional
head is weaker than its matched mean control. No distributional gain is claimed.

## Verification and limitations

`tests/test_moonshot_bets.py` checks the ridge solve analytically and verifies
that changed outer/probe labels leave training, selection and predictions
unchanged. It checks both direct and residual trees, selected and refitted.
The user-kernel test also requires a nonzero correction and full query coverage.

The initial tree launch failed before training because importing both PyTorch
and LightGBM initialized conflicting OpenMP libraries. PyTorch imports are now
local to the neural functions; the tree process does not import it. The failed
run's log is retained, and the corrected run has a new name.

These remain development CV estimates: base checkpoint selection and repeated
research decisions used the probe. No qualifying/test labels are consulted.
No new-pair inference checkpoint is exported for these experimental predictors.

## Commands

```sh
export MOONSHOT_PYTHON=/path/to/python-with-numpy-pytorch-lightgbm
export VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
DATA=/path/to/netflix.bin
make moonshot-user-kernel
"$MOONSHOT_PYTHON" tests/test_moonshot_bets.py
"$MOONSHOT_PYTHON" tests/test_moonshot_distribution.py
nice -n 15 build/moonshot_user_kernel --data "$DATA" \
  --model artifacts/snapshot/joint128-reg005.bin --output artifacts/USER-kernel64 \
  --features 64 --gamma 4
nice -n 15 python3 scripts/moonshot_user_compare.py --data "$DATA" \
  --kernel-prefix artifacts/USER-kernel64 --output USER-bank
nice -n 15 "$MOONSHOT_PYTHON" scripts/moonshot_boost.py --data "$DATA" \
  --output artifacts/BOOST-direct-v2
nice -n 15 "$MOONSHOT_PYTHON" scripts/moonshot_boost.py --data "$DATA" \
  --output artifacts/BOOST-residual --residual
nice -n 15 "$MOONSHOT_PYTHON" scripts/moonshot_distribution.py --data "$DATA" \
  --output artifacts/DIST-mean --head mean
nice -n 15 "$MOONSHOT_PYTHON" scripts/moonshot_distribution.py --data "$DATA" \
  --output artifacts/DIST-probability --head distribution
"$MOONSHOT_PYTHON" scripts/moonshot_bet_summary.py --data "$DATA"
```

Run PyTorch and LightGBM tests as separate processes in this environment because
their installed wheels bundle different OpenMP runtimes. Generated metadata are
cached locally under `artifacts/boost-union-features.npz`; artifacts are ignored
by git. Use new output names for new experiments.
