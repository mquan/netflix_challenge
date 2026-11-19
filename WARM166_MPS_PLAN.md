# Own166 raw predictions with all90 original supports

After the committed warm59 control/all-additions ridge diagnostic and its
separate five-fold replay complete, prepare the fixed all59 nonlinear candidate.
This candidate is declared before either ridge score is observed; the ridge
result does not select a subset or replace nonlinear validation.

Retain the existing own107/all90 context design, inserting the59 captured own
raw prediction snapshots before the learned support components. The result is
166 raw predictors,242 ridge columns and392 nonlinear features. The fitted ridge
prediction makes393 model inputs:168 signal and225 context indices, covering
every input exactly once. The90 original learned supports retain their values
and order. Each raw addition uses the existing clipping/centering transforms.

Use the tested float32 MPS context architecture, hidden128/rank32 and unchanged
optimizer, dropout, batch size, five nested user folds and stopping procedure.
The small393-input fit/save/load/inference check passed bitwise. This is an
integration check, not full-data accuracy evidence. The full study must pass
actual saved-model replay on all five folds and independent compensated scoring.

Also predeclare the fixed half mean with the independently verified expanded
own107/50 tree. Its actual models, fold receipts and replays must be rehashed;
the393-input NN and294-input tree preserve distinct feature schemas. No
qualifying outcome enters either the weights or the candidate choice. The
declaration-time reference is0.8592720613; publishing must also compare against
any newer completed global result.

The data queue waits for the existing shared fitting lock, stable memory and
more than28 GiB free disk. Known-only cache growth is about4 GiB; no large paired
qualifying cache is precomputed. The existing qualifying refit keeps priority.
A future winning production implementation must preserve both family schemas
and account for disk headroom before assembling paired inputs.

These features are our completed earlier snapshot pairs. CF96 full95-epoch
completion remains unclaimed; the additional snapshots extend the published raw
selection. No public numerical predictions, new qualifying score or target
achievement is claimed by the plan or cache.

Code: scripts/upstream_warm166.py, scripts/upstream_warm166_mps_context.py,
scripts/upstream_warm166_queue.py and
scripts/upstream_rbm50_tree_warm166_mps.py.
Evidence: experiments/upstream-warm166-queue/integration-check.json and the
committed experiments/upstream-warm59-screen/plan.json.
