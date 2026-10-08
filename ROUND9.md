# Round 9: latent blend features, retrieved history, Bayesian frequency effects

Started 2026-10-07 in `/Users/quan/workspace/netflix_round9`, branch
`round9-breakthrough`, based on `round4-combined` at e5dc7dc.
Existing datasets and artifacts are symlinked as inputs. New outputs live in
this worktree's `artifacts/r9`; experiment records live in its `experiments`.

## Fixed comparison

Use the promoted R8-B4-A2e bank and its three-seed LightGBM + three-seed NN2
50/50 average, **0.8674131 development CV**, as the control. The A2f result
0.8673301 is best by mean but was not promoted. Each new arm is evaluated
against the same frozen promoted bank, independently. Promotion requires
improvement in all five user folds. These are development scores, not test scores.
No new results are available at launch.

## LAT: user/movie coordinates and bias terms

`scripts/round9_latent.py` exports 22 features from frozen K1:
eight effective user coordinates (including implicit history and temporal drift),
eight movie coordinates in the same item-PCA basis, and six bias components.
PCA uses frozen movie factors only. The feature computation never reads probe
ratings. K1 itself is an older probe-selected checkpoint, retaining the existing
development-CV limitation.

A full-rank reconstruction on 2,048 evenly spaced probe rows matches K1's saved
raw predictions with maximum absolute error **6.34e-7**, checking session mapping,
frequency/time bins, implicit normalization and factor layout. The manifest
records dataset/model/prediction hashes and every output feature hash.

`scripts/round9_latent_run.py` runs R9-LAT-s3, then R9-LAT-NN2, then writes
`artifacts/r9/latent-comparison.json` with overall/fold deltas and promotion status.
Only additional features change; the prediction bank, ridge, folds, seeds and
blender hyperparameters match the control. Three CPU fold workers are used.
Launch PID: 85218. Log: `artifacts/r9/latent-pipeline.log`.

## A3: retrieve relevant history at a fixed 768-token budget

Import the sequence implementation from `bet-a2-sequence` (bf3c719). Keep A2e's
architecture, seed, masking and batch settings. Change the context sampler:
50% is chosen by movie similarity to the current targets; the remaining slots
are split between most recent and random older ratings. All masked targets
remain in the sequence. With multiple targets in one user's sample, retrieval
uses the maximum similarity to any target, giving a shared context union.

Similarity is cosine in 32 PCA dimensions of frozen movie factors. Selection
uses `artifacts/rehearsal/A-K1.bin`, trained on train minus P2. Refit uses K1
trained on the full training data. Thus P2 labels do not enter the retrieval
geometry used for holdout selection. This experiment measures the entire
retrieval procedure; it does not isolate the benefit of its pretrained geometry.

`scripts/round9_retrieval_run.py` waits until no other MPS sequence trainer is
active, selects up to 14 epochs on P2 (patience 3), refits for the selected epoch
count, then evaluates the new column through both frozen blender recipes.
Selection and refit run sequentially, one MPS trainer at a time. On launch another
worktree was running A2h, so this arm is **queued**, not yet training.
Launch PID: 99502. Log: `artifacts/r9/retrieval-pipeline.log`.
Final comparison: `artifacts/r9/R9-A3-comparison.json`.

## B5 pilot: isolate movie × rating-frequency bias

Import the sampler from `bet-b-bayes-timesvdpp` (0c1f29d). Add an item-frequency
row group using the existing implicit session counts and frequency bins. All
row-group loops now derive their length from the group array. The new checkpoint
format uses NFPBAYS2 and deliberately rejects old sampler checkpoints.

`scripts/round9_bayes_pilot.py` compares group off versus linear bias, on the
**full P2 dataset**, rank 16, 40 sweeps, burn-in 16, seed 29, two CPU threads.
Runs are sequential and otherwise identical. It records posterior-mean overall
and fold RMSE deltas. This is a diagnostic pilot, not evidence of ensemble gain;
rank/session scaling has not been launched.
Launch PID: 98200. Log: `artifacts/r9/bayes-pipeline.log`.
Final comparison: `artifacts/r9/bayes-pilot-comparison.json`.

## Verification

- C++ build succeeds with the existing warning flags.
- Five Bayesian contracts pass, exercising the new group: exact incremental
  residuals, learning, training/probe feature agreement, probe-label isolation,
  and checkpoint resume/snapshot behavior.
- Thirteen sequence/round-9 test executions pass: original learning/reload and
  isolation contracts, retrieved-context hidden-rating isolation, old relevant
  movie selection, token budget, and latent-feature rating-value isolation.
- Python compilation and `git diff --check` pass.

Commands (Python: `/Users/quan/.cache/netflix-venv/bin/python`):

```sh
make
python -m unittest discover -s tests -p test_bayes.py
python -m unittest discover -s tests -p test_round9.py
```

The three pipeline processes are detached and continue after the interactive
session. Check their logs and `ps` before launching any duplicate. Each experiment
uses a new output name and stops on subprocess failure.
