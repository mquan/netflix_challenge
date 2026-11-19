# Fixed tree follow-up on own107 plus50 RBM context

Fit the unchanged three-seed LightGBM residual recipe on the verified
107-predictor/50-latent-context cache:183 ridge columns,293 nonlinear features
plus each eligible ridge prediction. Same five outer user folds, disjoint inner
stopping users, ridge penalty, tree capacity and stopping rules. All50 learned
RBM context values remain unchanged. This is one fixed family comparison,
with no feature-subset or snapshot sweep.

Wait for the current complete five-fold neural comparison and all saved-model
replays before launch. Reuse the lower independently verified known-CV NN from
the completed NN103 and own107+50 studies, binding actual original model,
receipt, input and replay hashes. Preserve its own source schema; do not invent
new NN fitting receipts. Train and replay all new trees, then independently
score the tree alone, reused NN and fixed half mean. Lower aggregate RMSE can
be promoted without an all-fold or minimum-gain veto. Qualifying outcomes do
not enter any selection or fit.

The controller shares the existing meta-fit lock and waits for two stable
memory samples: existing native/meta physical footprints below21GiB,
compressor below6GiB, swap growth below32MiB per30seconds, reserving5GiB.
Partial outputs remain preserved on failure. Native training is not interrupted.
The paired107+50 production bank is already separately verified, but a full
production refit requires completed lower development scores and saved-tree
replay. If blending, the reused NN must use its original exact bank/ridge.
No new score is claimed by this plan; both qualifying groups still miss0.85725.
