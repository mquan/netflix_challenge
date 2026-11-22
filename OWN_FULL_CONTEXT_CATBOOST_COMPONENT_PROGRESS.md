# Full-context tree follow-up implementation

The numerical backend, bounded full-width input handoff, and known-only fold
schedule are implemented. Fifteen combined CPU checks pass, including actual
CatBoost inner selection and fresh positive refits on synthetic data. They
also cover every 713/714/863/864-column stream, all ten eligible ridge
populations, the 8,192-row boundary, zero-seed denominators, model reload
identity, and unchanged fitting when held-user outcomes are perturbed.

The schedule completes all three independent inner selections before fresh
full refits. Each fit receives only its eligible training rows; validation
receives only eligible odd inner groups. Held-user rows enter prediction.
The ordered correction is a float64 sum of each seed's prediction divided by
three, including zero slots. Full outer predictions are clipped to [1.05,4.95]
and stored as float32. The final raw matrix column receives the matching
population's ridge baseline; the remaining raw columns stay unchanged.

These are component checks, not a production study or a new score. The live
RBM400 data/neural/qualifying-bag/own17 chain retains priority. Before this
follow-up can run, implement and verify its owned source-authenticated durable
producer and Torch-isolated consumer, complete both five-fold three-seed
studies, independently reconstruct all raw streams and actual saved models,
select the arm using complete known-only canonical scores, replay its full
qualifying bag, and verify the predeclared own18 blend before fixed audits.
The original resource baseline, full budgets and existing learned state must
remain intact.

Evidence:

- [Backend checks](experiments/upstream-full-context-catboost-backend-protocol/verification.json)
- [Full-width stream checks](experiments/upstream-full-context-catboost-stream-protocol/verification.json)
- [Fold schedule checks](experiments/upstream-full-context-catboost-fold-protocol/verification.json)
- [Complete follow-up plan](OWN_FULL_CONTEXT_CATBOOST_FOLLOWUP_PLAN.md)
