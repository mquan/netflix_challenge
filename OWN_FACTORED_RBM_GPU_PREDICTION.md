# Factored bipolar RBM prediction prototype

This extends the separately named [GPU implementation plan](OWN_FACTORED_RBM_GPU_PLAN.md).
It adds prediction equations, not a production data loader or fitting runner.
The original gradient/oracle prototype and its recorded verification remain
unchanged. No new native or GPU training process has been launched.

`scripts/upstream_factored_rbm_prediction.py` computes hidden expectations from
all supplied training ratings and deduplicated conditional item addresses,
then emits five-category probabilities and expected ratings in query order.
Cold users retain hidden bias and conditional addresses, matching the original
`rx.rs` fallback. Known training-day slots must belong to the query user;
an unseen date is explicitly `-1` and adds no training-day bias. No probability
or score calibration, query labels, RNG, clipping or selection is involved.
Every history/conditional/query row is visited through bounded tiles.

Five new CPU tests compare f64/f32 predictions and probabilities with separate
literal scalar equations; check known/unseen-day behavior, cold users,
empty histories, zero day slots, query order, duplicate queries, empty queries
and invalid cross-user day references. These join the original eight CPU
gradient/update tests. The retained test transcript and source-hash records
identify the actual scope of this check.

The prediction API expects independently validated support coordinates and
conditional deduplication. It validates coordinate bounds and query-day
ownership, but does not itself establish full input provenance or train/holdout
disjointness. Actual MPS support, throughput, complete data admission, model
initialization, the full15+15 training runner, durable actual models, resource
ownership and independent full-query replay remain pending. None of these CPU
checks produces a new prediction candidate or qualifying score.
