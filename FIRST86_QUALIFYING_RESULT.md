# Matching86 own-training qualifying audit

The frozen matching86 tree/context-NN blend scores **0.8615073689623157** on
source0 (1,408,789 rows) and **0.8624355592570154** on source1 (1,408,342 rows).
Both improve over our first85 audit, but both still miss the strict0.85725
target. The worse group improves9.45558433% over Cinematch0.9525; its remaining
RMSE gap is0.0051855593. These are adaptive community-recovered source groups,
not official Netflix or untouched Test scores.

Both families were fitted on all known ProbeX rows using our own paired
86-predictor bank, with separate saved-model replay before the mean was frozen.
Tree budgets are809/793/875 rounds and context-NN budgets19/20/20 epochs for
seeds2026/7/13, derived from their actual nested development receipts. Both
families consume162 ridge columns and222 features plus the ridge baseline.
No downloaded pretrained prediction features or qualifying outcomes enter
fitting or model selection.

The frozen float32 mean SHA256 is
`34c013c0e6934b5218d78848ad839fec2bef310c0317b9190552c64b7c305378`.
A separate checker verified the full production source/code bindings, actual
saved-model hashes and semantic-replay receipts, the exact float32 mean, the
canonical recovered-label Git blob, exact group counts and compensated scores.
That check verifies existing full replay evidence; it does not rerun inference.
See `experiments/upstream-cf12-matching-production/frozen-mean-score-verification.json`.

The paired bank intentionally includes some independently retrained own source
vectors rather than the exact cached development realization. This was already
disclosed by its original parent manifest. A full-row independent check verifies
all declared paired input transforms and quantifies nine changed predictor
columns from four source producers. Known labels and user identities match.
The development score is evidence for the recipe, not a measured score of the
rebuilt production bank. See
`experiments/upstream-extension-bank/paired-source-differences.json`.

The newer tree91/contextNN86 candidate was fixed on known-ProbeX evidence before
this audit. Its verified development mean is0.8599745622436025. Its production
controller waits for memory, will fit tree91 with its own ridge and budgets,
and may reuse this exact own full-fit NN86 only after all origin checks pass.
No new predictor, budget or weight is selected using the qualifying audit.
