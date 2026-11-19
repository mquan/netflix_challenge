# Own epoch20 user context: fixed nonlinear study

The verified development best is0.8592309819864167. The completed item154
qualifying audit independently scores0.8607516601651759 /0.8617378856802073:
9.5288309% improvement on the worse group, still missing the strict0.85725
target by0.0044878857 RMSE. These are adaptive recovered groups, not official
or untouched Test. No qualifying outcomes select this new experiment.

The own raw autoencoder has now completed both epoch20 snapshots with actual
saved-model query replay. Keep its declared50/50 trajectory unchanged. Before
the epoch20 pair and its scores were available, declare a fixed64-component
unsupervised user representation from the trainx encoder: all480,189 users,
complete available trainx histories, batch256, no dropout, same learned
trainx table for both canonical queries. No query ratings fit the projection,
and there is no epoch/rank search or full50-epoch completion claim.

Preparation and separate verification are complete. Two full saved-encoder
passes reproduce every512-dimensional hidden vector bitwise. Centered top64
covariance components, canonical loading signs and population-variance scaling
produce a480,189x64 float32 table. These components explain46.74294% of hidden
variance; that statistic is not predictive accuracy. A third saved-encoder
pass independently reconstructs every CSR history batch and every projected
user row. Stable covariance merging, top spectrum, orthogonality, signs and
complete virtual query gathers pass; maximum eigenpair residual5.45e-15.
Only the123MB user table and small projection/moment files are stored; no
984MB hidden matrix or duplicated qualifying feature array is required.

Append all64 components to own107/item154, preserving its397 nonlinear columns,
183-column ridge problem, labels/users/groups and original five nested user
folds. The resulting461 feature columns plus nested ridge prediction give462
inputs:109 signals and353 context values, disjoint and complete. Keep
H128/rank32, optimizer, seeds and stopping unchanged. Require every new/retained
column check, all15 actual saved NN model replays and independent scoring.

Declare one fixed half mean with the verified own107/50 tree before fitting.
Compare against0.8592309819864167. No weight search, automatic qualifying
production or qualifying-based selection. A better complete development result
needs a separately frozen full-known recipe and post-freeze recovered audit.

The shared GPU lock serializes extraction, fitting and the current raw run.
The meta worker keeps the original24GiB memory guard and a documented8GiB
disk reserve for its small remaining outputs, avoiding the prior generic22GiB
guard interruption. Source/model histories, snapshot weights and prior fitting
artifacts remain preserved.

Code: scripts/upstream_autorec_user64.py,
scripts/verify_upstream_autorec_user64.py,
scripts/upstream_user218_features.py,
scripts/upstream_user218_mps_context.py and
scripts/upstream_user218_queue.py.
Evidence: experiments/upstream-autorec-user64/{plan,receipt,verification}.json.
Fixed NN/mean declaration: experiments/upstream-user218-queue/plan.json.
