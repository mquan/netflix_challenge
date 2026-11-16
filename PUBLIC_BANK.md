# Public model bank continuation — 2026-11-15

The target **RMSE ≤ 0.85725** was numerically verified on 2026-11-16:
the 619-input contextual ridge scores **0.8571903481** on recovered source1
and **0.8561542082** on source0. Both pass, with **10.00626267%** improvement
over 0.9525 on source1. This is a measured, independently rescored community
audit, not a development score minus an estimated retraining offset.
It is not official Netflix scoring or an untouched private test.
See [RESULTS.md](RESULTS.md) for the frozen artifact and exact verification.
The rest of this file retains the earlier raw-bank development history.

The original 79-model R9 ensemble scores 0.8668706425 on five user folds of the
original Netflix Probe. All Probe values are excluded from base-model fitting;
Probe has driven model selection, so the metric remains development evidence.

## Measured result

Adding 122 public original-split base-model columns to the 79 local columns
produces **0.8660701636 contextual ridge CV RMSE**. This already improves the
previous tree/neural ensemble by **0.0008004789**, in every user fold:

| Fold | RMSE | Change from previous ensemble |
| --- | ---: | ---: |
| 0 | 0.86704379 | −0.00081531 |
| 1 | 0.86617384 | −0.00051500 |
| 2 | 0.86215898 | −0.00102913 |
| 3 | 0.86752809 | −0.00068032 |
| 4 | 0.86745800 | −0.00096277 |

The matched local-only contextual ridge is 0.8685312898. Its recorded original
score is reproduced within 6e-14 RMSE and a maximum prediction difference of
4.77e-7 after float32 export. This verifies the memory-bounded sufficient-statistic
implementation against the existing learner. The wider bank gains 0.00246113
against that matched ridge. It is 0.00882016 above the direct target.

Result: `experiments/PUB1-union-ridge-screen.json`. Prediction artifact:
`artifacts/public-bank/union-ridge-screen.f32`.

A nested movie-specific calibration lowers this to **0.8659848000** (saved
float32 column independently rescored at 0.86598479995). It fits seven fixed
residual features per movie using outer-training users only. A user-disjoint
inner split chooses from no correction and penalties 10/100/1000. Every fold
selects 1000 and improves, by 0.000069–0.000103 RMSE. This is a separate
calibrated-ridge result, not a measured improvement to the unfinished tree/NN
ensemble. Report: `artifacts/public-bank/item-calibration.json`.

The EASE adaptation finishes at 0.96470997 standalone. Adding it to the
201-column contextual ridge improves by only 0.00002597, to 0.86604419;
all folds improve, but the gain is too small to prioritize a larger sweep.
Report: `artifacts/ease/E1-union-ridge.json`. No EASE gain against the tree/NN
ensemble is claimed.

## Imported evidence and provenance

Source: [gtakacs/netflix_prize](https://github.com/gtakacs/netflix_prize), audited
at commit `49c9b3676e6bab90fdf77e9a2da984edbfe4fe5c`. The public bucket index
has SHA-256 `8714114b33345a63f5b254348015a5ff04427c9c901ffead120e6d4bc46e15e8`.
All 122 original-split `.probe.npy` base-model files match its MD5 and size.
The mapped float32 files also have SHA-256 receipts in the import manifest.

The row mapping was established by the neighboring content experiment: replay
original training-file user IDs; match each Probe pair; check all 1,408,395
ratings against the original training files; require a bijection. Source Probe
rows group movies numerically, whereas raw `probe.txt` groups them lexicographically.
The mapping SHA-256 is
`4355330f53ca5c51f7bcfec273779ebe9c03e3445063ac2f5d33aa1945d0f9a8`.
The importer additionally checks permutation validity and every column's shape
and finiteness. Imported standalone scores agree with the neighboring experiment.

The import excludes alternative splits (whose fitting uses some original Probe
labels), qualifying predictions/labels, voting features, and trained blender
outputs. Public models still have Probe-based checkpoint/hyperparameter selection.
No untouched-test claim follows from importing them. The 201-column bank uses
the existing nested user-fold fitting protocol, with a user-disjoint inner
partition for tree/NN stopping. It never fits a second blender to globally
cross-fitted ensemble columns.

The published integrated predictor family and larger RBMs supply useful
diversity. The strongest imported standalone result is 0.87941751, from
`tsvdx5-300__nlpp__knn3`. The experiment imports all eligible columns rather
than selecting them by their observed standalone scores.

## Experiments running

- `PUB1-union-tree-20261115`: 201 raw columns, existing 46 support columns,
  original metadata, nested residual tree, seed 2026. Prefix:
  `artifacts/public-bank/union-tree-s1`.
- `PUB1-union-NN2-20261115`: same bank, original NN2 architecture/settings,
  seeds 2026/7/13. A 50/50 average with the tree is fixed in advance. Prefix:
  `artifacts/public-bank/union-NN2`.
- `EASE1-residual-20261115`: an independent explicit-rating adaptation of
  [EASE](https://arxiv.org/abs/1905.03375). Training-only user/item bias
  residuals; user-normalized Gram; zero self-diagonal; 8,192 training-popular
  movies (covering 96,654,544 training entries); fixed ridge 100. Prefix:
  `artifacts/ease/E1-top8192-r100`. The paper's ranking results
  are not evidence of an RMSE gain for this adaptation.

## Strict-training-subset public bank

Source `split_3` removes 352,099 additional original-training ratings and
holds out **every original Probe rating**. Its 1,760,494 query rows include
all 1,408,395 original Probe queries. This differs from source `newsplit`,
which puts some original Probe labels into training and is excluded here.

`src/public_split3_map.cpp` reproduces the split from user IDs and training
dates, without reading rating values. It reproduces rand 0.9.2's ChaCha12
stream and PCG32 seed expansion, including both published StdRng reference
vectors. Tests also change every training and Probe rating in a fixture and
require identical removal and mapping files.

The reconstruction was independently checked against source metadata on every
original Probe row. Training-history counts, query dates, and same-day training
rating means relative to the user mean all match **exactly (maximum error 0)**.
This checks both mapping and the random removal set, without using Probe
rating values. `experiments/PUB3-mapping-verification.json` binds dataset,
mapping, removal and source-metadata hashes. The original-split mapper was
also rebuilt independently and reproduces the earlier all-pair mapping exactly.

Imported and MD5-verified: **265 base-model columns and 220 training-derived
metadata/latent-coordinate columns**, projected to original Probe order.
Source blender outputs remain excluded. All source models still carry their
own development selection history.

Running comparisons:

- `PUB3-union-ridge-20261115`: local 79 + original public 122 + subset public
  265 = **466 raw predictors**, contextual ridge only.
- `PUB3-union-full-tree-20261115`: those 466 predictors plus the original
  supports and 220 new source supports, for **750 tree features**. The original
  nested tree recipe and seed 2026 are fixed before scoring.

Feature preparation now writes contiguous columns. A profile of the initial
466-column preparation showed 99% of samples in strided float32 assignment;
the column-major representation avoids that cost without changing features or
folds. The redundant original feature-preparation job was stopped after the
equivalent complete statistics were available; its receipt is retained.

Code: `scripts/public_probe_bank.py`, `scripts/streaming_blend.py`,
`scripts/streaming_nn_blend.py`, `scripts/ease_residual.py`.
Tests verify safe source selection and remapping, dense-versus-grouped ridge
equivalence and outer-label isolation, and EASE's constrained regression solution
and exclusion of self-copying. Blockwise neural normalization also matches the
existing implementation exactly on a numerical fixture.

The existing household-persona job in `/Users/quan/workspace/netflix_personas`
was left unchanged. All new outputs are isolated in this checkout.

The later 466-base NN2 comparison reached **0.864533752805203** on the entire
original Probe. Newer split imports and actual qualifying evaluation are
documented separately in [QUALIFYING_PROGRESS.md](QUALIFYING_PROGRESS.md),
including the alternative holdout distinction and recovered-label provenance.
