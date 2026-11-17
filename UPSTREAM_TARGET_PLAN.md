# Plan to approach the upstream score independently

The next phase focuses on measured missing contributions rather than another broad increase in model size. Our completed fresh fourteen-column bank scores 0.867711936 on five fixed user folds. The upstream 188-column fitted reference is 0.856167156, using downloaded predictions. Both results remain development evidence.

## Gap attribution completed

The diagnostic in `scripts/upstream_gap_audit.py` uses the same ProbeX rows, final-layer user folds, ridge lambda 1000 and clipping rules. It reproduces the two existing controls before accepting ablations.

| Input bank | Columns | Fitted final ridge | Final-layer user CV |
| --- | ---: | ---: | ---: |
| Our fresh bank | 14 | 0.867690108 | **0.867711936** |
| Released base bank | 137 | 0.860345322 | **0.860485311** |
| Released base + GBM outputs | 155 | 0.857465496 | 0.857614872 |
| Released base + MLP outputs | 160 | 0.857108695 | 0.857242909 |
| Released base + FWLS outputs | 147 | 0.856485945 | 0.856610757 |
| Full released stack | 188 | 0.856167156 | **0.856323241** |

The final-ridge fitting versus CV difference is only 0.000156 for the full released stack. It explains little of our deficit. The staged diagnostic gaps are approximately 0.007227 to the released base bank and a further 0.004162 to its complete stack.

This is not end-to-end validation of the released stack: its intermediate learned columns were supplied already fitted/cross-fitted according to upstream's historical procedure. Changing only final-layer folds does not remove their prior label use. Also, some FWLS outputs themselves depend on GBM/MLP outputs. Their strong group-level performance does **not** show that those underlying stages can be omitted. Trace dependencies before assigning training cost or causal importance.

## 1. Use the predictions already generated

We saved 20 ProbeX epoch predictions from the 60-factor model, 20 from the 400-factor model, five from the wider DNN, three temporal component outputs (bias/MF/NSVD1), and two DNN components. These are available without retraining the base models.

Freeze their hashes and provenance, remove exact duplicate prediction vectors, and compare the original fourteen against snapshots, components and their combination. Preserve upstream's distinction between clipped rating predictions and unclipped component features. Refit weights inside the outer user folds; do not reuse published epoch-blend coefficients. Keep the corrected neighborhood-support variant as a separate explicitly declared diagnostic arm rather than silently changing the fourteen-model control.

The original fourteen-column score must reproduce before any gain is accepted.

## 2. Reproduce the blending strategy on our fresh bank

Train a context-dependent feature-weighted linear blend first, using simple user/movie/time features derived from known training ratings and query metadata. Audit feature construction to exclude ProbeX and qualifying outcomes. Fit all target-dependent transformations and weights using only the corresponding training folds.

Compare it with the current ridge using the same outer user folds. If it helps, test source-aligned GBM and MLP blends and a cross-fitted final stack. Intermediate out-of-fold predictions must be created without exposing the final scoring fold's targets; a row being out-of-fold at just one layer is insufficient.

These are new local fits. Released higher-level outputs remain diagnostic controls and never enter the independent model's feature matrix.

## 3. Select the missing base recipes by measured contribution and cost

Use the released vectors to screen which missing models or coherent model groups improve the current fresh bank most. Label this explicitly as exploratory development analysis, not an independently achieved score.

For each promising column, resolve its complete upstream dependency graph. Count unique parent training jobs, reusable existing artifacts, expected memory and measured family training time. Rank recipe bundles by marginal improvement and actual execution cost; multiple useful epoch/component columns can share one expensive parent.

Train the selected bundles locally and replace their diagnostic vectors with fresh predictions. Independently compare outputs where published fixtures exist. Retain unresolved fidelity differences rather than attributing them to numerical noise without evidence.

Do not commit to all 137 base columns or all 201 pipeline jobs. Some upstream capacities are expensive or impractical on the current 32 GB machine. The released-vector screen should establish whether a compact feasible bank is promising before spending days on its parents.

## 4. Integrate and measure against the same target

At each checkpoint publish:

- Number of genuinely fresh base, snapshot/component and learned-blend columns.
- Fixed-fold development RMSE, paired fold deltas and frozen-control parity.
- Reference comparisons and unresolved discrepancies.
- Native training time and memory evidence for the next proposed jobs.

The 0.860485 base-bank and 0.856323 full-stack controls are diagnostic targets, not promises that any particular compact fresh bank will match them. A fresh result near 0.856 requires reproducing enough of both the base bank and learned stack. We have not demonstrated that yet.

## Snapshot/context checkpoint completed

The six saved-output ablations are complete; see [SNAPSHOT_BLEND_PROGRESS.md](SNAPSHOT_BLEND_PROGRESS.md). The original fourteen reproduced at 0.867711936. Adding 42 nonduplicate saved snapshots and five unclipped components reaches 0.867287793. Adding a single-layer context-weighted blend reaches **0.866320074**, improving over fourteen in every user fold. Context uses known trainx identities/dates and query metadata only, with fold-local regression coefficients. No released prediction features or published epoch coefficients enter these fits. The implementation suite passes 24 tests.

## Immediate next work

Screen the missing released predictors against the new fresh snapshot/context control, with explicit diagnostic labeling. Resolve each promising recipe's complete dependency bundle and rank gain against actual native training/RAM cost before launching additional training. Optional GBM/MLP evaluation requires full isolation of final scoring folds through all layers. No additional base-model training has started.

Evidence: [gap-audit.json](experiments/upstream-expansion/gap-audit.json). Existing fresh results: [evaluation.json](experiments/upstream-expansion/evaluation.json).
