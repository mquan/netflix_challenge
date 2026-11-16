# How we crossed the Netflix Prize 10% target

The verified result is **0.8571903481 RMSE**, a **10.00626267% reduction**
from the historical Cinematch reference of 0.9525. The target was 0.85725.
Both recovered qualifying groups pass: source0 scores **0.8561542082** and
source1 scores **0.8571903481**.

The few-hour advance came from changing the unit of work: assembling,
auditing, and combining existing prediction assets. It rests on the earlier
local modeling work and substantial external model training. The final
threshold-crossing model is a regularized linear blend with context-dependent
weights.

This is a **community recovered-label result**, with independent numerical
verification. It is not official Netflix scoring or an untouched private-test
result. Those distinctions matter when explaining both the achievement and
the apparent speedup.

## What changed from the earlier month of work?

The earlier work built the ingredients: joint factor models, neighborhood
corrections, restricted Boltzmann machines, Bayesian models, masked-history
networks, a sequence model, and increasingly capable blenders. Developing,
training, validating, and retaining those models takes time. Some historical
prediction banks also lived in temporary paths and became unavailable, which
made reproduction and production refits harder.

The later run changed the workflow:

| Earlier emphasis | Successful run |
| --- | --- |
| Implement and train additional local predictors | Reuse public development/qualifying prediction pairs and completed local predictors |
| Improve one model or a small bank at a time | Combine several model families and already-trained nonlinear blends |
| Track original-Probe development CV and estimate a retraining benefit | Produce full qualifying predictions and measure frozen artifacts directly |
| Expensive training before learning whether a new direction helps | Cheap blending experiments over reusable prediction columns |
| Recover historical checkpoints and dependencies as needed | Bind datasets, row mappings, predictions, and source code with hashes |

The previous work was still useful. The final ensemble includes five local
predictors and a locally fitted neural blend built from the 385-public-column
bank. The new run could use that accumulated work immediately.

The recorded optimization goal took approximately **8 hours 38 minutes**.
That excludes the earlier project work and the training that produced the
public prediction store. Within the final phase, the first audited public-blend
import started at **02:11 UTC on November 16, 2026**; the winning qualifying
prediction file was frozen at **02:42 UTC**, before its separate audit. That roughly
31-minute interval was integration and blending time over existing assets,
not the time required to train hundreds of predictors from raw ratings.

Sources: [earlier development plan](PLAN_TO_TARGET.md),
[research-branch record](MERGE_RECORD.md),
[first blend import](experiments/QUAL-audited-probe-blender-import-20261116.json),
[successful fit/export](experiments/QUAL-public619-meta-ridge-freeze-20261116.json).

## Breakthrough 1: reuse prediction pairs instead of retraining every model

The public
[gtakacs/netflix_prize prediction store](https://github.com/gtakacs/netflix_prize/tree/49c9b3676e6bab90fdf77e9a2da984edbfe4fe5c)
provided an existing collection of model predictions. We pinned the source
commit and verified downloads against the published index.

The useful asset is a **paired prediction column**:

- Development predictions from a predictor whose corresponding held-out
  ratings were excluded from its fitting data.
- Qualifying predictions from its matching production version.

This lets us test combinations without paying for every base-model training
run again. A predictor with only a development vector is insufficient for
building a qualifying submission.

We eventually had **536 public CF prediction columns**: 122 from the old
split, 263 from split3 with qualifying counterparts, and 151 from the newer
split. These are columns, not 536 independent algorithms; some are seeds,
snapshots, or residual-model variants.

There was measurable value before the final breakthrough. In a matched
contextual-ridge comparison on the original Probe, adding 122 public columns
to the local bank lowered RMSE from **0.8685312898 to 0.8660701636**, a gain
of approximately **0.002461**. This comparison holds the blending method
constant.

Integration required reconstructing row mappings and upstream splits.
Matching vector lengths was not enough: user IDs, movie ordering, and
development membership had to agree. Split3 holds out the original Probe
plus 352,099 further ratings. The newer split promotes 280,231 original
Probe ratings into its known training data and leaves **1,128,164 ProbeX
ratings** held out. We reconstructed these memberships without selecting by
rating values.

That work made the external predictions usable on our data. It also prevented
us from calling a ProbeX score an original-Probe result.

Evidence: [public-bank comparisons](PUBLIC_BANK.md),
[raw prediction importer](scripts/public_probe_bank.py),
[split3 mapping reconstruction](src/public_split3_map.cpp),
[ProbeX reconstruction](scripts/public_newsplit.py).

## Breakthrough 2: distinguish Probe-fitted blenders from Quiz-fitted ensembles

This was the decisive source-audit finding.

The initial imports deliberately excluded learned blend outputs. That avoided
accidentally importing a final ensemble whose weights had been fitted on the
recovered qualifying ratings, but it also excluded useful components.

Reading the source revealed distinct cases:

| Component | Fitting targets | Treatment |
| --- | --- | --- |
| Individual CF predictor | Its known training ratings | Import matching development/production predictions |
| Intermediate GBM, MLP, or FWLS blender | Probe or ProbeX ratings | Accept selected, audited outputs as pretrained **blenders** |
| Final Quiz-fitted ensemble | Qualifying Quiz ratings | Exclude |

For the selected GBM and MLP paths, the source cross-fitting functions pass
development features and development labels into the model fitter.
Qualifying labels enter the final printed diagnostics. The FWLS path builds
its regression matrices from development labels. The legacy epoch-blending
path fits its epoch weights on the designated development split.

Consequently, seeing qualifying labels loaded in a source program did not,
by itself, establish that its weights were fitted on them. Following the
actual fitting calls let us admit a useful subset without importing the
Quiz-fitted final ensembles.

We first imported **52** such blender prediction pairs, then expanded to
**77**: 24 from split3, 51 from the newer split, and two old-split FWLS outputs.
Other old jobs remained excluded because their historical dependencies were
insufficiently documented.

These outputs contain already-learned nonlinear combinations and
feature-dependent model weights. Reusing them added a different kind of
information from simply adding more factor-model snapshots.

This audit does not establish that upstream research was blind to qualifying
feedback. The recorded policy explicitly acknowledges public-Quiz-driven
hyperparameter/model selection and post-fit diagnostics. It also acknowledges
that upstream row crossfits and higher blending layers are not nested within
our final user folds.

Evidence: [audited importer and policy](scripts/public_probe_blenders.py),
[77-column manifest](artifacts/public-probe-blenders-expanded/manifest.json),
[source cross-fitting code](artifacts/public-source-audit/src/blend.rs),
[source FWLS fitter](artifacts/public-source-audit/src/bin/fwls.rs).

## Breakthrough 3: retain individual predictors alongside nonlinear summaries

The compact ensemble improved accuracy, but the final step showed that its
summaries had discarded useful information.

The compact bank contained 83 prediction inputs:

| Input type | Count |
| --- | ---: |
| Audited public Probe-fitted blenders | 77 |
| Matched NN385 blend | 1 |
| Local predictors: A2h checkpoint mean, MT3, MT8, MT1, MH3 | 5 |
| **Total** | **83** |

Its contextual ridge scored **0.8575942364** on recovered source1. Retaining
the **536 individual CF prediction columns** alongside those 83 inputs
produced the **619-input** bank and scored **0.8571903481**.

Both ridge variants used the same development population, the same five
context values, and the same selected regularization strength, **alpha
0.001**. The improvement was **0.0004038883 RMSE**. This is the clearest
matched comparison in the final stage.

The evidence supports a specific conclusion: the intermediate blends did
not make their individual predictors redundant. The final learner benefited
from both the learned summaries and access to individual prediction
differences.

The final model lets predictor weights vary with user rating count, movie
popularity, the rating date relative to the user's mean activity date, and
same-day activity. Its five context values include a constant. Regularization
limits how aggressively it can fit the many correlated inputs. The strength
was selected using known development ratings.

The feature bank also carries 238 side features for neural experiments.
**The successful ridge uses the 619 prediction columns and five context
values, not all 857 bank columns directly.** Some side information is
already encoded inside the pretrained blender outputs.

A separate 619-input NN2 completed and froze predictions, but its audit was
canceled after the ridge had independently met the target. No qualifying
score is claimed for that neural candidate.

Evidence: [compact ridge](artifacts/private-audit/probe-meta83-ridge-20261116/report.json),
[619-input ridge](artifacts/private-audit/public619-meta-ridge-20261116/report.json),
[bank construction](scripts/extend_raw_with_meta.py),
[weight selection and freeze](scripts/freeze_meta_ridge.py).

## Breakthrough 4: make wide-bank experiments cheap enough to iterate

The statistical method was established. The engineering made it practical
on the local machine.

With 619 predictions and five contexts, the ridge design has
**3,100 coefficients**. Materializing its complete float64 development
matrix for 1,128,164 rows would take approximately **26 GiB**. Doing the
same for 2,817,131 qualifying rows would take approximately **65 GiB**,
before other arrays or models.

Instead, we accumulated regression sufficient statistics, the matrices
needed to solve the regression, in chunks and ten user groups. The grouped
Gram matrices occupy approximately **0.72 GiB**. They support development
selection and final fitting without retaining the full expanded design.

Memory-mapped prediction banks and verified column overlays also let us
reuse qualifying features without copying the entire bank for every
candidate. Training, inference, and scoring ran as separate steps with
explicit input hashes.

Local production refits were another required piece. Development predictions
alone could not enter the final submission. The attention component used a
mean of completed checkpoints 4 through 8 from a full-known-data fit.
Its GPU inference was split into complete-user length bands to preserve each
user's query context. Padding changes and their effect on the stochastic
training stream were recorded rather than described as identical training.

These were enabling changes, not isolated RMSE improvements. They increased
the number of experiments we could complete and made the successful result
replayable.

Evidence: [chunked regression](scripts/streaming_blend.py),
[verified matrix overlays](scripts/qual_matrix.py),
[snapshot inference](scripts/predict_snapshot_ensemble.py).

## The measured path to the target

These are scores on the **same recovered source1 qualifying group**.
The rows are successive pipeline variants, not isolated estimates of each
component's causal contribution.

| Candidate | Source1 RMSE | Target status |
| --- | ---: | --- |
| Earlier 385-public-column NN2 control | 0.8587390514 | Above |
| 385 public columns + mid attention mean + MT3/MT8, NN2 | 0.8582647011 | Above |
| Compact 58-input stack, selected ridge | 0.8578789679 | Above |
| Compact 58-input stack, NN2 | 0.8578011759 | Above |
| Expanded compact 83-input stack, selected ridge | 0.8575942364 | Above |
| Expanded compact 83-input stack, NN2 | 0.8575337426 | Above |
| **536 individual predictions + 83 compact inputs, selected ridge** | **0.8571903481** | **Below 0.85725** |

The final reduction from the earlier qualifying control is about
**0.001549 RMSE**. The last ridge-to-ridge step supplied approximately
**0.000404**, enough to cross the line.

Audit timestamps place the compact-58 neural result at **02:22 UTC**,
compact-83 ridge at **02:31 UTC**, and successful 619-input audit at
**02:42 UTC** on November 16. The full experiment history is in
[QUALIFYING_PROGRESS.md](QUALIFYING_PROGRESS.md).

## What did not make the breakthrough

Several plausible methods failed to improve qualifying transfer:

| Experiment | Source1 RMSE | What it established |
| --- | ---: | --- |
| Average exact-duplicate qualifying predictor families during development | 0.8582828078 | A real development/production difference existed, but this treatment did not improve the then-best result |
| Drop the 220 latent/voting side features from the early 388-input neural blend | 0.8590299320 | Those side features were useful overall |
| Suppress every neural side feature | 0.8591567167 | Predictions-only neural inputs lost accuracy |
| Fit the native split3 neural blender on all 1,760,494 held-out rows | 0.8590145817 | More development rows alone did not solve transfer |
| Weight original-Probe training toward unlabeled qualifying covariates | 0.8587665894 | The tested domain-weighting approach did not improve the blend |

User/day residual calibration also showed a development improvement under
a changed information-availability protocol, then failed qualifying transfer.
It reinforced the need to compare the same available information at
development and production time.

The local Bayesian refit and CatBoost experiment were still unfinished when
the target was verified. They were stopped and **did not contribute to the
successful submission**. A restarted attention run was also stopped; the
winner uses the earlier completed checkpoint mean.

## Why the apparent speedup needs careful interpretation

Three effects are easy to conflate.

**Reuse changed the compute bill.** Much of the expensive model training had
already happened, locally or upstream. We spent the successful phase choosing
and combining outputs.

**The scoring populations differ.** The earlier promoted R9 result,
0.8668706425, is original-Probe development CV. The later
0.8645337528 result is also an original-Probe development result. The final
0.8571903481 is a qualifying-production audit after production predictors
use all 100,480,507 known ratings, including the original Probe. Subtracting
these scores does not measure a pure algorithmic gain. The table above
provides the qualifying-to-qualifying comparison.

**The evidence standard differs.** Independent verification here means a
second scorer agrees and saved inference can be replayed. It does not turn
an adaptively reused label set into an untouched test. The final stack's
upstream dependencies also prevent an independent nested-development-CV claim.

The recovered labels come from a pinned community repository, not an
official Netflix scoring service. Its two source masks have counts that
conflict with published Quiz/Test sizes. We preserved the masks and required
**both** groups to pass, so renaming them cannot change the numerical result.

## What we should carry forward

1. **Inventory reusable predictions and their production counterparts early.**
   Stored development vectors alone leave an expensive production gap.
2. **Audit fitting data flows before accepting or rejecting whole model classes.**
   Probe-fitted intermediate blenders and Quiz-fitted final ensembles have
   different roles.
3. **Compare the same rows, information availability, and blending method.**
   A change of metric or production data is not a controlled model improvement.
4. **Test summaries alongside their underlying predictors.**
   The final matched comparison showed that both were valuable.
5. **Preserve source, mappings, checkpoints, and hashes as durable artifacts.**
   This reduces repeated work and makes a threshold-crossing claim reviewable.
6. **Separate numerical target achievement from independent generalization.**
   Keep the limitations attached to the result wherever it is reported.

The reusable lesson is to make existing research composable and measurable.
That change in workflow, supported by the prior modeling work, made this
particular final advance possible in hours.

## Frozen result and verification

The improvement calculation is:

```text
100 × (1 − 0.8571903480630092 / 0.9525) = 10.00626267055%
```

The worse recovered group is below the target by approximately
**0.000059651937 RMSE**, so the margin is small.

- [Frozen model report](artifacts/qualifying/public619-meta/ridge-final/report.json)
- [Independent scoring and bitwise replay receipt](artifacts/qualifying/public619-meta/ridge-final/target-verification.json)
- [Final experiment receipt](experiments/QUAL-target-final-result-20261116.json)
- [Netflix-format submission](artifacts/submissions/netflix-0.8571903481-qualifying.txt.gz)
- [Exact export verification](artifacts/submissions/netflix-0.8571903481-qualifying.txt.gz.json)
- [Stopped optional jobs and 17 passing checks](experiments/QUAL-target-met-optional-job-cancellation-20261116.json)

The export contains 2,817,131 predictions and 17,470 movie headers. All
predictions round-trip exactly. No submission was uploaded. The checksummed
prediction, weights, and submission artifacts were preserved; code and
documentation remain in the workspace.
