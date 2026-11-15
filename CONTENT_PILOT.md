# Isolated personalized content pilot

Worktree: `codex/content-features`. The household-persona run uses another
worktree and its own output paths. This pilot reads the original dataset and
frozen R9 predictions but writes only under `artifacts/content/`.

## Frozen first arm

The first arm uses only Netflix's supplied title and release year. It hashes
IDF-weighted title tokens into 64 fixed coordinates and adds eight smooth year
coordinates. This is a lexical control, not an LLM or a claim that titles
capture movie semantics. It establishes the batch and personalization pipeline
without external metadata joins or API calls.

For each user, training ratings are centered by shrunk user and movie means.
The residual-weighted average of rated-movie vectors yields one 72-coordinate
profile. The profile is shrunk by 30 pseudo-ratings. Probe features are its
dot product with the target movie and log training support. Probe labels never
enter the profile. Movie vectors and feature outputs are frozen and hashed.

## Decision rule

First run a one-weight, five-user-fold diagnostic against the frozen R9
out-of-fold column. It is a screen only: the R9 out-of-fold fit for one fold can
depend on labels in other folds. If the diagnostic gains at least 0.0001 RMSE
overall with four of five folds improving, run a matched 79-column R9
LightGBM/NN2 comparison with these two support columns and otherwise identical
settings. Promote only if that full comparison improves all five folds.
If the screen fails, close this lexical arm; richer semantic vectors can be
supplied to the same feature generator later as a separate preregistered arm.

The target for any final claim remains hidden-test RMSE, which cannot be
measured here. Probe results are development evidence only.

## Result (2026-11-15)

The 72-dimensional lexical control generated 1,408,395 probe features in
about 2.3 seconds using a low-memory C++ pass. Repeating the pass produced
byte-identical affinity and support files. The one-weight R9 diagnostic scored
0.8668698374 versus the frozen 0.8668706424 baseline, a gain of only
0.0000008050 RMSE. Folds 1 and 4 worsened. The fixed gate failed, so this arm
is closed without the costly full R9 blend. The result says little about richer
plot or genre embeddings; those require verified content and their own arm.

Inputs and per-fold scores: `experiments/CONTENT-title-year-screen.json`.

## Exploratory slice audit

Among users with fewer than 50 training ratings (450,585 Probe rows), the
fixed correction gained 0.00000487 RMSE. Movies released in 2000 or later
(911,904 rows) gained 0.00000419. Both are tiny compared with the 0.0001
aggregate gate; no broad user or movie-support slice performed strongly.
Movies with fewer than 100 training ratings worsened 0.00006569 RMSE.

Of 261 individual movies with at least 1,000 Probe rows, High Fidelity gained
0.00016158 on 1,363 rows and Being John Malkovich gained 0.00013368 on 2,348
rows; both were positive in all five user folds. These are selected after
looking at 261 movies, so their apparent consistency is not independent
confirmation. Other movies worsened, and the full-Probe result stayed flat.
No slice is promoted or used to retune the closed arm. Detailed slices are in
`experiments/CONTENT-title-year-slices.json` and
`experiments/CONTENT-title-year-movie-slices.json`.
