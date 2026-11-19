# Match the tree to the verified86-predictor neural component

The current best, 0.8600637147575518 on adaptive known ProbeX, combines the
original85-predictor tree with the new86-predictor context NN. With the neural
fits complete, test the corresponding tree on those same86-predictor inputs.

`scripts/upstream_cf12_tree.py` reuses the verified CFNADE-12 experiment's actual
labels, users, groups, design, features, grouped ridge statistics and baseline.
It verifies the preceding evaluation/independent-score binding, complete neural
replay, all five neural fold receipts/checkpoints and the reused OOF vector.
Read-only inputs and the NN vector are linked into a separate experiment; the
old tree vector is excluded. No neural fitting is repeated or claimed.

Train the original tree recipe without hyperparameter changes: the same five
user folds, inner budget selection, eligible-group ridge, three seeds, up to
2000 rounds, patience75, 31 leaves, minimum1000 rows per leaf, learning rate0.03,
feature fraction0.9, bagging fraction0.8, and two threads. The new tree consumes
the 162-column ridge design and222 features plus the training-only ridge,
matching the existing neural component's predictors. Reconstruct all saved trees
bit-for-bit, then independently score the predeclared equal tree/NN mean against
the preceding best. One completed fold or a successful process is not a result.

A focused test runs actual learned trees on the223-input layout, changes all
outer scoring labels to100, and verifies identical selected budgets, ridge,
prediction and saved-model bytes. Only that small synthetic test changes rounds
or leaf limits; the production-size development experiment retains the original
recipe above.

The worker runs at lower priority with initial other-RSS admission below17 GiB.
Sustained total RSS above24 GiB or free disk below22 GiB stops only this owned
experiment and preserves partial outputs. No existing native job, completed
candidate, qualifying queue or public-source recipe is changed.

This is an additional adaptive known-ProbeX comparison. Qualifying outcomes are
not read, and no qualifying fit starts automatically from this experiment.
A full verified development improvement would require a matching full-data
production plan before a frozen qualifying prediction and recovered-group audit.
The goal remains strictly below0.85725 on both recovered groups.

First-fold evidence: `verify_upstream_cf12_tree_partial.py --fold 0` replays
all three saved trees bit-for-bit and independently scores the fixed mean on
226,746 held rows. The mean is0.8592639386651046 versus0.8593164235053121 for
the preceding best on those same rows, a change of-0.0000524848402075.
This is partial evidence only; fold1 is running and no promotion is allowed
before all five folds and final replay/scoring complete.

The second completed fold (fold1) also passes all three saved-tree replays
bit-for-bit. Its fixed mean is0.8574717065320298 versus0.8575002796649305 on
the same225,899 held rows, a change of-0.0000285731329007. Both first folds
improve; three folds remain. The two partial reports are independent evidence
for their own rows, not a complete development score or qualifying result.
