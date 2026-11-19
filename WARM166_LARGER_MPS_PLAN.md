# Larger own166 MPS residual blender

Evaluate one fixed larger architecture on the same own166 raw/all90 support
bank: hidden512 and context rank64, retaining393 inputs,168 signal indices and
225 context indices. Optimizer, dropout, batch size, seeds, five nested user
folds, stopping rule and original maximum epoch cap remain unchanged. This is
a new declared architecture, not a change to any active fitted model.

The32-row actual MPS fit/save/load/inference check passed bitwise. It confirms
the architecture and loader integrate; it supplies no full-data accuracy or
timing evidence. All15 real fold models must later reload and reproduce held-user
predictions bitwise, followed by independent compensated scoring on all known
rows. A new global record or full-known refit requires beating the actual current
global development best. Improvement against an individual control is recorded
separately; it does not establish a new record.

The controller waits for the complete own166 hidden128/rank32 control and its
expanded-tree comparison, then uses the existing shared fitting lock and measured
memory admission. Read-only cache files are reused by hard link. No additional
large feature matrix, base-model retraining, qualifying-cache expansion or
qualifying-outcome access is required.

Predeclare the fixed half mean with the expanded own107/50 tree as a separate
known-only comparison. Actual original tree receipts and saved-model replay
evidence must be rehashed. The tree keeps294 inputs; the larger NN keeps393.
Weights are not fitted to qualifying outcomes.

No qualifying success is claimed. The best recovered worse-group RMSE is
0.8618605526, still above0.85725. The previous95-epoch CF96 run continues unchanged;
its full completion remains unclaimed. All numerical inputs here are trained
by us, including the explicitly additional captured earlier snapshot pairs.

Code: scripts/upstream_warm166_h512_mps_context.py.
Integration evidence:
experiments/upstream-warm166-h512-mps-context/integration-check.json.

Completed: all five folds and15 actual saved-model replays passed. The larger
neural model scores0.8596061246; its expanded-tree half mean0.8593417716 is
higher than0.8592720613. It establishes no new record or qualifying refit.
Evidence: experiments/upstream-warm166-h512-mps-context/{evaluation,
independent-verification,nn-replay}.json and
experiments/upstream-rbm50-tree-warm166-h512-mps/verification.json.
