# MPS mathematics preflight, pending actual device execution

This extends the [separate study plan](OWN_FACTORED_RBM_GPU_PLAN.md). The device
preflight is implemented and its control/math paths have CPU fixture coverage.
Actual MPS and the full production-width stress case have not run.

`scripts/upstream_factored_rbm_preflight.py` requires a fresh MPS worker with
fallback and fast math explicitly disabled before Torch import. An existing
Torch runtime cannot qualify by changing environment strings afterward. Actual
MPS availability is required; CPU substitution is rejected. Strict deterministic
algorithms are enabled during checks, with prior runtime flags restored afterward.
The future controller must first supply its original ownership/resource guard
and acquire the actual required lanes; this library grants no permission itself.

The small fixed-draw case compares every CD-3 gradient/update with the separate
NumPy float64 oracle and checks cold/unseen-day query predictions and three exact
same-device repeated query vectors. Fixed tolerances are atol/rtol2e-5.

The required full stress case has500 users,17,770 items,400 hidden units,rank50,
117,570 training rows,118,069 conditional rows and1,000 queries. One complete
17,770-row user history spans multiple16,384-row tiles. No history or user is
dropped. Large-case fixed draws are valid zeros, avoiding ambiguous f32/f64
categorical-boundary divergence while exercising full indexed reductions.
All gradients/updates compare with the full NumPy float64 oracle at declared
atol5e-4/rtol5e-5, and prediction vectors must repeat exactly on the same device.
Process peak RSS and current/driver MPS allocations are reported separately;
actual continuous guard enforcement and full-fit peaks remain requirements.

Only a real MPS run with the exact full stress case can set the device-qualified
flag. CPU-only and reduced fixtures cannot. The production worker must preserve
the required environment/runtime policy for each subsequent fit/replay process.
Device qualification alone does not admit actual data, resource headroom or a
production model.

Five new CPU checks cover small fixed-draw math, the reduced heavy-user generator,
unchanged full stress specification, fallback/pre-import rejection and runtime
restoration after a synthetic guard error. They join the earlier50 CPU checks.
No MPS operation, full stress allocation, production data scan, model fit or new
qualifying score is established by these tests. Source-bound worker/watchdog
integration, actual data/MPS/memory/disk admission, full15+15 fitting and separate
saved-model full-query replay remain pending. Original native1200 is unchanged.
