# Fixed own autoencoder additions to the nonlinear ensemble

Completed result: NN **0.8594001511750141**; the predeclared fixed half mean
with the expanded tree is **0.859274524524327**, higher than the existing
0.8592720612630157 record by0.0000024632613113. All five folds and15 actual
saved-model replays passed. The separate canonical known-label/stored-mean
checker rehashed135 source/model files and confirmed the exact score. No
global promotion or qualifying refit. Evidence:
experiments/upstream-rbm50-tree-autorec109-mps/{verification,
independent-mean-check}.json.

Our verified development record remains **0.8592720612630157**. Its frozen
qualifying audit remains **0.8608774626094220 / 0.8618605526405846**; the target
below0.85725 on both recovered groups is still unmet.

The new own masked-denoising autoencoder has complete epoch5 and epoch10
pairs, including actual saved-model inference replay on every query row.
Separate source/model/query checks give known-only component scores
0.9332148094511349 and0.9321417324180398. These are raw component scores.
The original50/50 training trajectory continues at durable epoch boundaries.

The fixed control/all-two ridge diagnostic improves from0.8625801963952328
to0.8625774187158269. A separate solver recomputed all five user-fold
statistics and every held-out prediction, with maximum prediction discrepancy
5.01e-12. This small improvement is not an ensemble record or qualifying gain.

The next experiment adds both fixed snapshots to own107, retaining all90
original RBM/Tx context components. Its cache has185 ridge columns and335
nonlinear features; the NN adds the held-user ridge prediction as input336.
The111 signal and225 context indices are disjoint and cover every input.
All retained and new cache columns must match their bound source values.

Fit the same H128/rank32 MPS residual/gating architecture, optimizer, seeds,
five nested user folds and stopping rule. Require actual replay of all15
saved NN models, independent scoring and original source/model hashes.
Before fitting, declare a single fixed half mean with the verified own107/50
tree model. Compare it against the current0.8592720612630157 record. There
is no weight search or automatic qualifying production; a better verified
development candidate needs a separately frozen full-fit recipe and audit.

Controller: scripts/upstream_autorec109_queue.py.
Declaration: experiments/upstream-autorec109-queue/plan.json.
Data/model: scripts/upstream_autorec109{,_mps_context}.py.
Mean: scripts/upstream_rbm50_tree_autorec109_mps.py.
Diagnostic: experiments/upstream-autorec-pair2-screen/{ridge-screen,
ridge-verification}.json.

GPU access uses the existing shared fitting lock. The raw trainer yields at
an actual completed epoch boundary and resumes after the nonlinear study.
CF96 and split3 native training continue independently. The completed negative
whole110 paired feature cache was unused by fitted models; its four derived
matrices were hash-checked and retired to recover11,173,075,952 bytes. Original
predictions, models, metadata and receipts remain preserved; see its retirement
receipt. The consumed development whole110 and best paired90 caches remain.
