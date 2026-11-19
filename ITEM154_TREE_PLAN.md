# Match the tree partner to all154 learned contexts

The current best half mean uses an own107/50 tree and an own107/154 neural
model. Train the unchanged original tree recipe on the already checked full
own107/item154 feature bank:397 features plus nested ridge for398 tree inputs.
The183-column ridge, all107 prediction columns, every row/fold identity and
all154 learned contexts are retained exactly. Share the existing input matrices
by hardlink; no large input copy is needed.

Keep the original three seeds2026/7/13, five nested user folds, inner selection,
2000 maximum rounds, patience75,31 leaves,1000 minimum leaf rows, learning
rate0.03, original bagging/feature sampling and two CPU threads. Refit each seed
on its full outer training users for its selected round count. Require all15
actual saved-tree held-query replays and independent full-known scoring.

Reuse the current verified item154 MPS neural model and its original actual
saved-model receipts/replays. Predeclare the equal tree/neural mean and compare
against0.8592309819864167. No qualifying outcomes, subset/weight search or
automatic qualifying production. Keep the full24GiB total-memory guard and an
8GiB disk reserve, appropriate to shared input matrices rather than the older
duplicate-cache allocation. Partial failures are preserved.

Controller60069 is queued at a shared epoch boundary. The item-oriented raw
model retains its original50/50 budgets and fixed epoch5/10 follow-up.
Evidence: experiments/upstream-item154-tree/plan.json.
Implementation: scripts/upstream_item154_tree.py, reusing the unchanged
nested tree fitter and independent score/model replay verifier.
