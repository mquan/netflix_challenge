# Local collaborative context: cumulative research toward 10%

Started 2026-10-18 on `research/local-context`, from `round4-combined` at
4db5fc4. Renamed to **`codex/local-context`** to identify the Codex harness.
This remains one long-lived, iterative research branch; completed experiments
are commit checkpoints, not separate research restarts. User objective: reach
at least a 10% RMSE improvement over Cinematch.

## Target and evidence standard

- Cinematch test RMSE: 0.9525. A 10% reduction is **0.85725 test RMSE**.
- Promoted reference: **R9-LAT-A2h 0.8668706425 original-probe development CV**.
- Round 12's rehearsal, including Bayes B4 and A2h, retains an approximate
  dev-to-test offset of 0.0074. The working dev target is **0.86465**, leaving
  **0.0022206425 RMSE**, approximately **0.51% of current development MSE**.
- Current calibrated test estimate is 0.8594706425, or about 9.77% below
  Cinematch. This is an estimate, not a measured hidden-test result.
- Reaching 0.86465 is a development milestone. Freeze the recipe, rehearse its
  transfer, and execute the final train+probe pipeline before claiming a
  calibrated 10% result. Hidden qualifying labels are unavailable; a verified
  competition win cannot be inferred from development CV. The historical
  winning test score is a separate, stricter comparison than the 10% threshold.

Round 12 evidence lives in
`.claude/worktrees/claude-stage-c-offset/ROUND12.md`; earlier baseline context
is in [PLAN_TO_TARGET.md](PLAN_TO_TARGET.md). That older plan's score and
experiment-status sections are historical.

## Hypothesis and work sequence

The current ensemble may discard useful relationships between the target
user's history and other users' ratings of the target movie. Learn when
agreement on particular movies transfers to the query, retaining dates and
rating-scale differences. Keep one reusable retrieval and evaluation pipeline.

1. **Additional information (LC1).** Add existing pooled/path summaries to the
   frozen raw ensemble bank under its unchanged nested blending recipe.
2. **Representation.** Compare a small structure-preserving neighbor encoder
   with a matched pooled encoder and a measured rewiring ablation. Use P2 for
   development; reserve original-probe evaluation for declared milestones.
3. **Scaling.** Vary training targets, retrieved context and model capacity
   separately. Record held-out and ensemble learning curves plus runtime.
   Invest in the dimension that continues to improve measured generalization.
4. **Integration.** Incorporate useful collaborative context into the existing
   masked-history family. Preserve matched controls as the model evolves.

The LP2 gain of 0.00568649 is against a weak pooled ridge predictor on P2.
It is not an expected ensemble gain. The histogram does not retain neighbor
identity. A failed histogram experiment alone does not disprove structural
encoders, but it also does not justify unlimited scaling.

Freeze further isolated Bayes/RBM/A2 settings sweeps for this program. Every
new experiment must name a bottleneck, its matched control, and the next
decision supported by either outcome. Failed runs retain their evidence.

## LC1: predeclared full-bank comparison

Before observing any LC1 blend score, fix these conditions:

| Arm | Inputs |
|---|---|
| Reference | Saved R9-LAT-A2h: 79 prediction columns and existing support/meta/latent features |
| Pooled | Reference plus 27 pooled neighborhood, support/date and matched kNN features |
| Paths | Pooled plus 125 rating-triplet histogram cells and 8 agreement summaries |

Retrieval is LP2's fixed recipe: 32 history movies nearest query date,
256 sampled candidate raters, up to 32 overlapping neighbors, seed 20261013.
It uses only training ratings. Query labels never enter extraction; every
column covers all 1,408,395 original-probe rows in their original order.

Both new arms rerun the original contextual ridge, nested LightGBM and NN2
recipes. Three seeds (2026, 7, 13), the original inner/outer user splits,
training settings and fixed equal LightGBM/NN2 average are retained. Only the
additional feature columns change; fold parallelism is an execution setting.
Do not fit a correction directly to saved cross-fitted ensemble predictions.

The cached reference is independently rescored. All 125 recorded input hashes
match, as do dataset, meta-context cache and LightGBM/PyTorch versions. The
current tree script differs from the reference only in an unused compact-bank
branch and directory creation when a cache is absent; neither affects LC1.
Each stage archives its source, records inputs, and checks frozen artifacts.

Primary comparisons: **paths vs pooled** (incremental path information) and
**paths vs reference** (practical ensemble gain). Also report pooled vs
reference, all five fold deltas, user-clustered paired MSE intervals, sparse
user slices, runtime, and saved prediction hashes. Intervals are descriptive:
they do not correct repeated development selection or shared fitted weights.
Adding features changes feature sampling and neural input dimension, so a
gain is evidence for the feature package, not proof of a topology mechanism.

Promotion retains the all-five-fold improvement rule against the promoted
reference. A gain of at least 0.0003 plus consistent incremental path value
supports substantial investment; it is a decision threshold, not a prediction.
Research continuation is separate: smaller informative gains or improving
learning curves can justify a bounded diagnostic follow-up. If neither the
summaries nor the matched structural test adds value beyond strong controls,
close this direction and record why; do not keep changing settings indefinitely.

## Execution and artifacts

`scripts/local_paths_features.py` exports the full feature bank. Its tests
check byte-identical serial/parallel output, row order, numeric agreement with
the pilot, probe-label invariance and overwrite refusal. The existing pilot
tests cover retrieval, target exclusion and label isolation.

`scripts/local_context_chain.py` validates the frozen bank and exported
features, runs both blend arms, then independently scores saved float32
predictions. It records an audit, stage receipts and failure/completion status.
`--resume` only reuses successful stages whose recorded outputs still match;
it does not silently overwrite or retry failed experiments.

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 \
  /Users/quan/.cache/netflix-venv/bin/python scripts/local_context_chain.py --check

OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 \
  /Users/quan/.cache/netflix-venv/bin/python scripts/local_context_chain.py
```

- Extraction: `experiments/LC1-features.{json,log,events.jsonl}`.
- Features: `artifacts/local-context/LC1/features/manifest.json`.
- Frozen inputs and commands: `experiments/LC1-audit.json`.
- Progress: `experiments/LC1-chain.status.json`.
- Final comparison: `experiments/LC1-comparison.json`.
- Models' reports and prediction columns: `artifacts/local-context/LC1/`.

No LC1 blend result or target achievement is claimed until the chain completes.

### Initial execution, 2026-10-18

- Seven targeted tests passed (`python -m unittest discover -s tests -p
  'test_local*.py'`); `git diff --check` passed.
- Full extraction completed: 1,408,395 rows, 160 features, 237.5 seconds
  (~5,929 queries/second with four workers), 99.9987% path coverage. Exported
  histogram mass is zero for no-path rows and one within float32 tolerance
  otherwise. The matched kNN feature stays within [1, 5]. The copied manifest
  is `experiments/LC1-features-report.json`.
- Preflight independently rescored the saved reference at 0.8668706424233039,
  matching its recorded 0.8668706424555479 within float32 serialization error.
- The four-stage chain launched at 17:14 UTC under `caffeinate` and niceness 10.
  Initial stage: `LC1-pooled-s3`; three fold workers. It proceeds automatically
  through pooled NN2, path LightGBM, path NN2, and paired comparisons, stopping
  on failure. Prior smaller feature additions took about 85 minutes per arm;
  this larger comparison may take several hours.

### LC1 completed: pooled gain is small; raw path summaries regress

All four stages exited successfully. The chain finished 2026-10-18 at
19:35 UTC (12:35 PDT), about 2 hours 21 minutes after launch. Comparison output
and report hashes were reverified when reviewing the results.

| Fixed 50/50 blend | Development RMSE | Change vs reference | Improved folds |
|---|---:|---:|---:|
| Promoted R9 reference | 0.8668706424 | — | — |
| + pooled summaries | 0.8667483642 | -0.0001222782 | 4/5 |
| + pooled and path summaries | 0.8671679530 | +0.0002973106 | 0/5 |

Paths are worse than pooled summaries by 0.0004195888, on all five folds and
in both sparse/dense user slices. The paired user-clustered 95% MSE interval
for paths minus pooled is [0.0006572991, 0.0007977647], entirely above zero.
Both LightGBM and NN2 individually regress with paths, so the result is not
caused only by their fixed average. It does not establish why they regress.

Pooled summaries improve the aggregate; fold 1 worsens by only 0.0000053143.
Under the predeclared all-five-fold rule, neither arm is promoted. The best
promoted score remains 0.8668706425 and the estimated target gap is unchanged.
The LP2 histogram benefit against a weak ridge control did not translate into
an ensemble gain with this direct feature-addition recipe. Further scaling of
the same raw histogram addition is unsupported. A structural encoder remains
untested; this experiment neither validates it nor rules it out. Any follow-up
must distinguish representation/regularization limitations from redundant
information, with a matched control, before committing to a large run.

Full evidence: `experiments/LC1-comparison.json` and the four stage manifests.

### Next checkpoint: matched structural comparison

Continue with step 2 of the original sequence: a small encoder preserving
individual neighbors' shared ratings versus a matched pooled encoder. Both
produce compact predictions under the same retrieval and training budget.
Include the measured rewiring ablation and evaluate on P2 before any full-bank
confirmation. A separately trained histogram-compression diagnostic was
considered after LC1, but is not an additional research phase: compact output
belongs in both encoder arms. The central test remains whether preserving the
relationships adds value. Failure against the matched control substantially
downgrades this direction; it does not automatically trigger another settings
sweep. No follow-up training has launched yet.
