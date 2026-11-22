# Tree producer and consumer handoff

The bounded binary handoff is implemented. Twenty-seven combined tree-component
checks pass. An actual isolated-process fixture imports Torch in the producer,
passes data over a pipe, and verifies that the receiver remains Torch-free.
This matters because importing the complete original context adapter in a tree
fitting process imports Torch and violates the planned process separation.

The producer streams 8,192-row blocks and all ten eligible ridge baselines;
only the consumer retains the full raw matrix. The wire binds exact original
row counts, all 713/714/863/864-column layouts, coefficient hashes, both CPU
operator digests and the source admission digest. Source verification runs
before and after streaming. Each ridge system retains its separate vector-dot
order. The receiver independently checks every raw feature and baseline byte,
the exact footer and EOF. Partial reads/writes work; wrong declarations are
rejected before allocating a full matrix. Truncation, trailing bytes, modified
features and changed source/weight declarations cannot complete the handoff.

This is a component, not an owned production study or new score. In particular,
EOF is insufficient evidence that the producer succeeded: after receiving all
bytes, the future supervisor must separately verify the original producer's
identity and zero exit before authorizing any CatBoost fit. Both processes
remain subject to the unchanged original baseline, the complete dedicated
16GiB group budget and the24GiB aggregate limits. Keep both original lanes,
all full study budgets, full remaining-output reservation and same-process
resource suspension. No automatic partial replay or lower-row retry is allowed.

The remaining work is the owned full-data producer/consumer controller and its
source/runtime gates, both complete five-fold three-seed studies, independent
full operator/saved-model/canonical replay and known-only selection, the full
qualifying bag and the predeclared own18 calibration/fixed audits. The existing
RBM400 data/neural/bag/own17 chain retains priority.

[Handoff evidence](experiments/upstream-full-context-catboost-wire-protocol/verification.json)
and [earlier components](OWN_FULL_CONTEXT_CATBOOST_COMPONENT_PROGRESS.md).
