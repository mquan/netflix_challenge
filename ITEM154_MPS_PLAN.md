# Fixed own item context extension

Completed result: NN **0.8593837348929274**; the predeclared expanded-tree half
mean is **0.8592309819864167**, improving the preceding consensus0.8592432880.
All15 actual saved-model replays and separate stored-mean/canonical known-label
scoring pass. The first four fold receipts remain unchanged after explicit
recovery of the disk-guard interruption; the incomplete fifth fold remains
archived. No model recipe, folds, seeds or stopping rule changed.
Evidence: experiments/upstream-rbm50-tree-item154-mps/{verification,
independent-mean-check}.json and experiments/upstream-item154-recovery/completion.json.
This is the latest development result; its qualifying audit is still pending.

The following is the original plan, retained as history:

The current independently verified development best is0.8592720612630157;
qualifying remains0.8608774626094220 /0.8618605526405846. Both qualifying
groups must fall below0.85725. This experiment has no accuracy result yet.

Use the already freshly trained native trainx item factors from rbmx2-150
(17,770x750) and tsvdx4-400lm (17,770x400). Original execution receipts,
the pinned source commit and exact original factor hashes are checked.
This introduces no downloaded numerical features or recovered outcomes.

For each table, center all17,770 items, take the top32 covariance eigenvectors,
choose each sign by its maximum absolute loading, and scale projected items
to unit population variance. The same trainx table supplies both canonical
queries. These64 components explain67.164% /72.125% of the original two
factor tables' variance. This is an unsupervised representation statistic,
not predictive accuracy or exact public-schema parity.

Preparation is complete. A separate checker recomputed covariance and the top
eigenvalue spectrum, checked eigenpair residuals/orthogonality/signs, replayed
every projected item bitwise, and checked all1,128,164 known and2,817,131
qualifying query-row gathers bitwise. No qualifying labels were read.

Append the64 components to the verified own107/all90 development features.
Retain the existing183-column ridge problem and exact row/user/fold identities.
The resulting397 nonlinear columns plus the nested ridge prediction yield398
NN inputs:109 signal and289 context indices, disjoint and complete.
Keep H128/rank32 and the same optimizer, seeds and five nested user folds.
Require all15 saved NN model replays and independent scoring.

Before fitting, declare one fixed half mean with the current own107/50 tree.
Compare against the verified current development best. No weight search,
automatic production or qualifying-based selection is authorized by this
experiment. A better development result requires a separately frozen full-fit
recipe and qualifying audit. All studies remain adaptive known-label research.

Controller: scripts/upstream_item154_queue.py.
Declaration: experiments/upstream-item154-queue/plan.json.
Support proof: experiments/upstream-item64-support/{plan,receipt,verification}.json.
Data/model: scripts/upstream_item154_features.py and upstream_item154_mps_context.py.
Mean: scripts/upstream_rbm50_tree_item154_mps.py.

The shared fitting lock serializes this study with the autoencoder and the
already running own109 nonlinear study. Native CF96 and split3 NADE continue.
