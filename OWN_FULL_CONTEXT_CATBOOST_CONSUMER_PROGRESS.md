# Isolated tree consumer implementation

The Torch-free consumer component now authenticates the complete original
source metadata, source-aligned CPU proof and completion, three known-data
files, transferred raw matrix and all ten eligible ridge baselines. Before
fitting it requires a frozen producer-exit receipt for the captured identity
and exact handoff. A receipt claiming zero exit cannot authorize a producer
whose PID and start time are still live; this remains true if exec changed
its bootstrap command. EOF alone remains insufficient.

Thirty-five combined tree-component checks pass. The new consumer fixtures
use actual temporary files and a genuinely reaped child process, while their
CPU proof and reduced row counts are explicitly synthetic. They verify all
five fold dispatches and rejection of changed source bytes, transferred inputs,
exit status, population paths, widths and parent-proof flags. Actual CatBoost
selection/refitting/saved-model replay remain covered by earlier small CPU
fixtures. None of these fixtures admits the original full dataset or a score.

The full consumer preserves one raw matrix, replacing only the matching
population's baseline column before calling the original three-seed fold
schedule. Every held row must be produced exactly once, and every source is
rehashed and its identity checked before and after fitting. Model, population
and fold records must be durably retained by callbacks from a future owned
supervisor. The consumer's result explicitly remains pending independent
full replay and owned production admission.

Still required: the owned producer/consumer controller, authenticated actual
role/exit/resource gates, both complete original five-fold studies and their
independent full operator/model/canonical replay, known-only arm selection,
full qualifying bag, own18 calibration and fixed recovered audits. The running
RBM400 data/neural/bag/own17 chain retains priority. The original resource
baseline, complete budgets,16GiB group/24GiB aggregate caps, both lanes and
same-process resource continuation remain unchanged.

[Consumer checks](experiments/upstream-full-context-catboost-consumer-protocol/verification.json)
and [handoff implementation](OWN_FULL_CONTEXT_CATBOOST_HANDOFF_PROGRESS.md).
