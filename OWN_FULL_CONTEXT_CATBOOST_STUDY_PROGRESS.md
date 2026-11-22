# Durable full-arm tree study wrapper

The study wrapper now prepares paired source/contract/weight manifests and
retains all model, population, fold and held-prediction records for a complete
five-fold study. It keeps the original depth8/2000-round maximum/100-patience
configuration, all three seed slots per fold, exact original source bindings
and the byte-identical original scientific baseline. A partial or complete
namespace cannot be overwritten or automatically restarted.

The dedicated producer may import Torch during source preparation. The fitting
path remains Torch-free. It first receives the full bounded stream, then calls
the future owned exit gate to await the actual producer's zero-exit receipt.
This ordering avoids waiting for producer exit while its output pipe is full.
The original consumer independently authenticates the completed stream, exact
receipt, source files and known populations before any fit.

Forty-nine combined component checks pass. The new wrapper fixture uses
synthetic source/ownership proofs and mocked iteration selection/fold
scheduling, but actually fits and saves ten pinned small CPU CatBoost models
across all five folds. Fifteen seed slots, including zero slots, remain in
the inventory. Separate checks reject changed coefficients, altered manifests
or implementation bindings, a replaced baseline and partial-run reuse.
No production dataset or numerical score is admitted by these fixtures.

Still required: the owned study controller and role/exit/resource gates,
independent full paired-arm operator/ridge/model replay and canonical scoring,
known-only arm selection, full qualifying bag, and own18 calibration/fixed
audits. The existing scalar/latent neural and own17 chain retains priority;
this follow-up does not start ahead of its predeclared condition. No current
new score or target achievement is claimed.

[Study wrapper checks](experiments/upstream-full-context-catboost-study-protocol/verification.json).
