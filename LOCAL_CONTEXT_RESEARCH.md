# Local collaborative context: cumulative research toward 10%

Started 2026-10-18 on `research/local-context`, from `round4-combined` at
4db5fc4. Renamed to **`codex/local-context`** to identify the Codex harness.
This remains one long-lived, iterative research branch; completed experiments
are commit checkpoints, not separate research restarts. User objective: reach
at least a 10% RMSE improvement over Cinematch.

Workflow: continue iterating on `codex/local-context`, committing and pushing
completed checkpoints on this same branch. Do not create pull requests for
this work. The initial draft PR was closed at the user's request.

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
sweep.

## LC2: matched structural encoder pilot

Implementation started after LC1 was committed. This is step 2 of the original
research sequence, on the same `codex/local-context` branch. It is a P2
development pilot, not a new original-probe score or an ensemble promotion.

### Fixed recipe, before evaluating LC2

- Same deterministic 200,000-query P2 sample as LP2, with seed 20261013 and
  32 history movies / 32 neighbors / 256 candidate raters. P2 labels are absent
  from the training context. Store rating and query-relative date payloads in
  a padded 32-by-32 cross-rating matrix, plus the history and target-neighbor
  ratings and the 27 common pooled features.
- Eleven frozen P2 base predictions (core, Bayes B4, A2h) and one support column
  are hash-checked against the Round 12 rehearsal report and frozen in
  `experiments/LC2-spec.json`. These models were trained without P2 ratings;
  some base epochs were selected using P2, so this remains development evidence.
  This is a stronger pilot control than LP2's ridge baseline, but a smaller
  bank than the final 79-column ensemble.
- Fit users are folds 2--4; fold 1 selects epochs; fold 0 is scored after all
  arms are fixed. Refit each selected model on folds 1--4. Fit the ridge starting
  prediction and feature normalization only on the appropriate fit users.
  Do not use existing P2-fitted blender outputs as supposedly held-out inputs.
- Each model predicts a residual correction to the same 11-column ridge
  starting prediction. The head also sees the same base columns, support and
  pooled features. Two seeds (2026, 7), width 16, batch 256, AdamW at 0.0003,
  weight decay 0.01, maximum 20 selection epochs, patience 4, and fixed-epoch
  refits. Epoch zero is allowed if no trained checkpoint improves selection.

| Arm | Aggregation |
|---|---|
| Pooled | Encode each rating/date triplet; use global pooled messages in place of neighbor/movie grouping |
| Structured | Same encoder parameters; one message-passing step retains shared-neighbor and shared-movie grouping, then pools learned neighbor representations |
| Shuffled | Structured architecture, with cross-rating/date payloads permuted among observed neighbors separately for each shared movie |

The pooled arm retains the same triplet information, rather than comparing
against a weaker model that cannot see triplets. Architecture, parameter count
and epoch caps match; selecting different epochs is allowed under the same
selection rule. Both neighbor and movie index permutations leave predictions
unchanged when the corresponding context is permuted consistently.

The shuffle keeps the entire missingness mask, both degree sequences and
each shared movie's rating/date multiset exactly. It changes per-neighbor
rating distributions and which neighbor supplied each rating. Export reports
measure these changes and check the preservation invariants across all rows.
It is a **rating-correspondence ablation**, not a claim that mask topology alone
causes a difference. Common pooled features, including the original kNN scalar,
are kept in every arm, so residual benefit must exceed that shared signal.

Primary comparisons are structured minus pooled and structured minus shuffled,
with paired user-clustered MSE intervals and per-seed results. Consistent gains
against both controls justify further measurement/scaling; a P2 gain alone
does not meet the 10% target or establish a final ensemble contribution. Failure
against pooled downgrades this representation; any additional investment must
be tied to a diagnosed limitation, not another open-ended settings sweep.

### Validation and execution

Eleven local-context tests pass, including target-label isolation, selection
and preprocessing isolation from evaluation labels, correct numeric context,
shuffle invariants, parameter matching, grouping sensitivity, and permutation
invariance. `scripts/local_structure_data.py` exports the context;
`scripts/local_structure_train.py` implements the three matched arms.

Context extraction is recorded as `LC2-context`. A training-only throughput
benchmark will determine the runtime estimate before the full pilot launches.
The benchmark does not score selection/evaluation users. Experiment manifests,
timestamped logs and source archives preserve the exact code for each run.

### Throughput and full-pilot launch

Context extraction completed in 48 seconds. All 200,000 query indices and
pooled feature vectors match LP2 exactly after float32 encoding. Across
114,717,630 observed edges, the shuffle has zero mask, per-movie rating or
per-movie date multiset mismatches. It changes 65.69% of edge ratings and
94.74% of edge dates; 99.24% of neighbor rating histograms change. This is a
strong correspondence perturbation, with the limitations above.

Both encoders have 8,081 parameters. The training-only MPS benchmark processed
7,680 examples in 2.95 seconds including setup, projecting about 46 seconds
per fit epoch. A 32-query CPU/MPS forward check with nonzero output weights
has maximum absolute prediction difference 2.38e-7 (tolerance 1e-5). Evidence:
`experiments/LC2-throughput-report.json` and `LC2-device-check.json`.

The full pilot is `LC2-structure-p2`, with outputs under
`artifacts/local-context/LC2/pilot/`. Its experiment manifest reports completion
or failure; timestamped logs show each arm, seed, selection/refit phase and
epoch. On completion it also writes `experiments/LC2-comparison.json`.
Estimated runtime is 2--4 hours, depending on early stopping and machine load.
The first result is the matched P2 comparison; no full-ensemble result is
promised on this schedule.

### LC2 result and decision

Completed successfully on 2026-10-18 at 18:12 Pacific (2026-10-19 01:12 UTC),
after 62 minutes. Evidence: `experiments/LC2-comparison.json` and the
`LC2-structure-p2` manifest, log, events and source archive. Evaluation covers
39,887 ratings from 33,109 held-out users; these P2 scores are not comparable
to the original-probe target above.

| Two-seed averaged model | P2 evaluation RMSE |
|---|---:|
| Frozen-column ridge baseline | 0.845304781 |
| Pooled encoder | **0.844647493** |
| Structured encoder | 0.844655319 |
| Correspondence-shuffled encoder | 0.844662261 |

Structured minus pooled is **+0.000007826 RMSE**, slightly worse. Its paired
user-clustered MSE 95% interval is [-0.000001282, +0.000027723], including zero.
Seed 2026 is worse by 0.000012309 RMSE; seed 7 is effectively tied (-4.44e-9).
Structured beats shuffled by only 0.000006942 RMSE. The averaged paired MSE
interval excludes zero [-0.000020507, -0.000002948], but seed 7's interval
includes zero and neither seed establishes a useful advantage over pooled.

All arms selected 14 epochs for seed 2026 and 10 for seed 7, stopping before
the 20-epoch cap. There is no observed justification for simply extending the
epoch budget. Similar training trajectories do not prove why structure adds
so little; the mechanism remains unresolved.

The pooled model improves the smaller ridge baseline by 0.000657288 RMSE.
That comparison includes the neural head and common features, so it cannot
attribute the gain to local context or establish a gain over the full ensemble.
Only evaluation fold 0 is populated in this screening run; the report's
`all_folds_improve: false` is not a five-fold promotion test result.

**Decision: do not scale this structural recipe or promote an ensemble change.**
It failed the declared requirement to beat both matched controls. LC1 rejected
raw path summaries in the full ensemble, and LC2 now substantially downgrades
this structural representation. Further work must diagnose a concrete limitation
before another training sweep; the current evidence does not establish a
scalable route to 10%. Preserve the retrieval, controls and evaluation harness
on the same branch. The promoted score remains 0.8668706425, with the same
0.0022206425 gap to the working development target.

Post-run verification matched all 18 output SHA-256 hashes plus the frozen
spec and context-manifest hashes. Independent rescoring of the four saved
float32 prediction files agrees with the report within 1e-8 RMSE. No code
changed during this result checkpoint.

### LC2 mechanism diagnosis

Following the user's request to diagnose the limit, audited the saved pooled
and structured checkpoints for both seeds. No new predictor was trained or
selected. `scripts/local_structure_diagnose.py` reproduces the audit;
`experiments/LC2-diagnosis.json` records the initial interventions and
`experiments/LC2-diagnosis-redundancy.json` adds a label-free reconstruction
probe and representation variance analysis. The source archive preserves both
script versions. All four checkpoint replays match their saved predictions
exactly, and cached-head predictions also match full-model inference.

The audit uses a fixed, label-independent sample of 8,192 refit users' rows
to estimate replacement means and reconstruction coefficients, plus all
39,887 evaluation rows. Replacement means and reconstruction fits never use
evaluation labels. These are post-hoc inference interventions, not new
independent performance estimates or substitutes for retraining an ablated
model. No result here changes the promoted score.

| Structured model intervention | Seed 2026 RMSE | Seed 7 RMSE |
|---|---:|---:|
| Frozen-column ridge | 0.845304781 | 0.845304781 |
| Original trained model | 0.844858831 | 0.844704748 |
| Replace local vector by its refit-sample mean | 0.844905691 | 0.844630156 |
| Exchange local vectors between evaluation queries | 0.844993764 | 0.844679381 |
| Switch to pooled aggregation with trained weights fixed | 0.844838072 | 0.844704461 |
| Shuffle correspondence with trained weights fixed | 0.844874958 | 0.844706869 |

Findings, in order of evidential strength:

1. **The encoder is connected and learning.** Relative weight changes from
   initialization are 48--55% for the edge encoder, 69--76% for the relation
   layer and 32--42% for the neighbor encoder. Setting the local vector to a
   constant changes predictions by 0.0114--0.0147 ratings RMS. This rules out
   a disconnected or entirely frozen branch; it does not prove good learning.
2. **The learned structural effect is tiny.** Switching trained weights to
   pooled aggregation changes predictions by only 0.000396--0.001362 ratings
   RMS and does not degrade evaluation RMSE in either seed. This also agrees
   with the separately trained pooled controls in LC2. The current model is
   largely insensitive to the grouping it was built to exploit.
3. **Most of the pilot gain survives without query-specific local vectors.**
   Mean replacement retains 89.5% of seed 2026's ridge improvement and improves
   seed 7's score slightly. On the refit sample, mean replacement hurts by
   0.000257 and 0.000313 RMSE; on evaluation it hurts by only 0.000047 and helps
   by 0.000075. The learned local contribution does not generalize consistently.
4. **The representation has little variation in most directions.** One
   principal component explains 96.7--96.8% of the structured local vector's
   refit-sample variance; two explain 99.3--99.6%. Pooled models behave similarly.
   This is descriptive, not proof that small-variance directions contain no
   signal. A fixed linear reconstruction from the common pooled columns explains
   only 23--30% of structured representation variance on evaluation; all global
   columns explain 26--33%. The stronger claim that local features simply copy
   existing features is **not established**.
5. **Missing context is not the immediate bottleneck.** Across audited rows,
   median context has 32 neighbors and 604 cross-ratings; only 2 of 48,079
   queries have no cross-ratings. This does not establish retrieval quality:
   candidate raters are sampled, then ranked by overlap and date, not learned
   relevance. More raw context has no demonstrated benefit yet.

There is a concrete optimization concern, but it remains a hypothesis. At
initialization the median per-coordinate local standard deviation is about
0.010--0.012 versus about 1.0 for standardized global inputs. Multiple learned
transformations and averaging steps separate structure from supervision, while
global features enter the head directly. After training, the local contribution
to the first head layer has only 12--17% of the global contribution's RMS.
This makes weak initial structural sensitivity plausible; substantial encoder
weight movement means it is not an explanation based on absent gradients.
The network also has no movie/user identity or latent features, so it can learn
generic rating/date patterns but not identity-specific transfer relationships.
Neither observation proves that fixing it will improve validation.

**Diagnosis:** LC2 learns a useful general correction to the smaller ridge
baseline, but its local branch learns little generalizable structural correction.
The failure is already visible within this pilot, before full-ensemble blending.
We have not separated an optimization/representation limitation from insufficient
incremental information in the retrieved context. The evidence does not justify
calling the whole local-context hypothesis impossible, or calling scale the cure.

The next bounded discriminator would change only local-vector normalization
before the shared head, in both pooled and structured arms, with the original
split, seeds, features and training budget. Measure structural sensitivity and
the matched evaluation difference together: improved sensitivity without an
accuracy advantage would reject input scale as the practical bottleneck. Any
positive result would still need new confirmation data because these evaluation
rows are now part of diagnosis. No normalization experiment or larger training
run was launched as part of this audit.

Verification: all 11 local-context tests pass, `git diff --check` passes, both
diagnostic script hashes match their archived sources, and all four checkpoint
replays plus cached-head evaluations pass their numerical consistency checks.

## LC3: normalize the local vector, with matched controls

Next experiment, declared before training: test the measured across-query
coordinate-scale imbalance by inserting `BatchNorm1d(16, affine=False,
eps=1e-4, momentum=0.1)` immediately before the shared prediction head. The
choice normalizes each coordinate across training examples; per-example
LayerNorm would address a different axis. Normalization changes optimization
and introduces batch dependence during training, so this tests the specific
normalization intervention, not a pure scalar-rescaling causal claim.

All three LC2 arms receive the same normalization: pooled, structured and
correspondence-shuffled. No additional learned parameters (still 8,081).
Running means/variances update only on fitting batches. Selection and final
evaluation use frozen statistics; refits start from scratch on refit users.
The default `none` option preserves the LC2 architecture and checkpoint layout.

Keep the LC2 context/spec, 200,000 queries, user split, seeds 2026/7, width 16,
batch 256, learning rate 0.0003, weight decay 0.01, epoch cap 20, patience 4,
and equal seed averaging. No retrieval, input, head, loss, or epoch-budget
changes. The code will reject singleton training batches for this option;
neither LC2 fitting partition produces a singleton batch at batch size 256.

Decision rule fixed before results: normalized structure must beat normalized
pooling in both seeds, improve the seed-averaged RMSE by at least 0.0001,
have a user-clustered paired MSE 95% interval below zero, and beat the normalized
shuffled control. Smaller gains can inform diagnosis but do not justify scaling.
Also compare normalized structured performance to LC2's structured result so
that weakening the control is not mistaken for progress. This is exploratory
P2 evidence: the evaluation users have already informed diagnosis. Any positive
result requires fresh confirmation before a full-ensemble promotion claim.

The completed run will automatically execute the checkpoint audit with frozen
batch statistics. Measure local-vector scale, local-to-global head contribution,
constant-local replacement, and fixed-weight pooling/shuffle sensitivity.
If sensitivity improves without accuracy, input scale is not the practical
explanation supported by this intervention. If sensitivity stays weak, the
normalization intervention has failed to establish that it fixes the suspected
mechanism. Neither outcome automatically triggers a larger model or data sweep.

Training experiment: `LC3-normalized-p2`; summary:
`experiments/LC3-comparison.json`; artifacts:
`artifacts/local-context/LC3/pilot/`. Follow-up audit experiment:
`LC3-normalized-diagnosis`; summary: `experiments/LC3-diagnosis.json`.
Expected runtime is 1--2 hours based on LC2's 62 minutes, subject to early
stopping and machine load. Completion manifests, not this estimate, establish
when results are available.

Pre-launch validation: all 12 local-context tests pass, including normalized
training's evaluation-label isolation, unchanged learned parameter initialization,
and evaluation-batch independence of running statistics. The training-only MPS
smoke run completed 30 steps in 3.14 seconds (about 2,447 examples/second);
no selection or evaluation scores were read. Evidence is archived under
`experiments/LC3-normalized-throughput.*`. `git diff --check` passes.

### LC3 result: normalization fixes scale, not accuracy

Training and automatic audit both completed with exit code 0 on 2026-10-18
at 21:47 Pacific, in approximately 67 minutes. All 18 saved output hashes and
four saved prediction scores were independently verified (RMSE within 1e-8).

| Two-seed averaged model | LC3 P2 RMSE |
|---|---:|
| Ridge baseline | 0.845304781 |
| Normalized pooled | 0.844679533 |
| Normalized structured | 0.844675089 |
| Normalized shuffled | 0.844685773 |

Structured minus pooled is -0.000004445 RMSE, far below the predeclared
0.0001 gate. Seed 2026 improves by 0.000014694; seed 7 regresses by
0.000005766. The paired MSE 95% interval [-0.000030996, +0.000015978]
includes zero. Structured minus shuffled is -0.000010684 RMSE, also with a
MSE interval including zero. LC3 structure is 0.000019770 worse than LC2
structure. **No promotion or structural scaling.**

The intervention did affect the diagnosed mechanism. Median local-coordinate
standard deviations rose from 0.051/0.037 to 0.823/0.869. The local-to-global
first-head-layer contribution ratios rose from 0.168/0.118 to 0.447/0.487.
For seed 2026, the first principal component's share fell from 96.8% to 46.6%.
This improved scale and representation spread without establishing a useful
structural accuracy advantage. Input scale is not a sufficient practical
explanation of LC2's failure. The matched tests now argue against investing in
this structural recipe, rather than merely against its unnormalized version.

The active 10% goal remains unmet: promoted development CV is 0.8668706425,
and the working calibrated target remains 0.86465. The previous goal work
made progress by launching a controlled test; this completed result changes
the next action by rejecting normalization as the proposed rescue.

### Attribution check before any pooled-column integration

LC2/LC3 pooled gains compare against an eleven-column ridge, not a matched
neural head with all common features. The checkpoint audit shows most of the
gain survives constant local vectors, but that is an inference intervention,
not a retrained control. The adjacent honest-residual worktree independently
suggests trying a pooled output in the full bank; its HR2/HR4 results also warn
that generic corrections often disappear in the full ensemble.

Run one cheap matched control with the local vector fixed to zero throughout
training and inference. Retain the identical 39 common features, residual
baseline, head initialization, dropout, optimizer, partitions, seeds and epoch
selection/refit budget from LC2. Encoder parameters remain allocated but unused;
effective trained capacity is smaller, so this is a test of whether the local
branch earns its cost, not a matched-capacity structural comparison. No local
normalization, matching LC2. Use the original two seeds, not a new sweep.

Compare the saved LC2 pooled predictions against this control using user-clustered
paired MSE intervals. Require a pooled advantage of at least 0.0001 RMSE,
both seeds improving, and averaged interval below zero before investing in
probe-order export and full-bank integration. If it fails, close the current
pooled/structural recipe rather than treating the ridge comparison as new signal.
This is exploratory attribution on reused development users, not promotion.
