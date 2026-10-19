# Final-production readiness audit

2026-10-19; current promoted reference remains R9-LAT-A2h at 0.8668706425
development CV. This audit supports the final retraining step in
`PLAN_TO_TARGET.md`; it neither changes that score nor establishes the 10% goal.
LC6 is closed without promotion. N1 is running in the other research worktree;
the frozen SO1 comparison is queued on this branch behind its full chain.

Recipe associations now cover **all 79 retained columns (77 unique hashes)**.
The digest-based audit is `experiments/production-recipe-coverage.json`.
Validation strength varies by family: this is complete recipe coverage, not
completed production retraining or a fully verified qualifying pipeline.

### Frozen latent-feature transfer (PF1)

The retained bank uses `scripts/round9_latent.py` and its **22** columns under
`artifacts/r9/latent`, including a saved eight-dimensional `basis.npy`. The
separate `round9_latent_features.py` exporter has a different schema and is not
the producer of these retained columns.

The blender must retain the meaning of these coordinates between development
and qualifying inference. A newly trained K1 and newly fitted PCA are not
guaranteed to preserve that meaning; canonicalizing component signs alone
does not establish alignment. The production policy is therefore to keep the
**original K1 checkpoint and original saved basis as the feature encoder**.
K1 retraining in the user-kNN recipe remains necessary for its prediction
column; that retrained model must not silently replace the feature encoder.

`scripts/production_latent.py` binds the original checkpoint and dataset to
the retained feature manifest. It checks that production keeps the same dense
IDs, date range and all implicit interactions, then requires each qualifying
user/movie/date triple to exist in that original context. The original
implicit arrays already include qualifying identities and dates, so their
session indices are available without qualifying ratings. The production
dataset supplies the query rows, while the encoder and projection stay frozen.

PF1 reproduced all 22 retained development arrays **byte for byte** across
1,408,395 rows, then exported all 22 arrays for **2,817,131 qualifying rows**
in the production dataset's query order. Evidence and independent verification
are `experiments/PF1-frozen-latent.json` and
`experiments/PF1-frozen-latent-verification.json`. The verification rehashes all
inputs and outputs, checks every output's length and finiteness, and compares
2,048 qualifying rows against the unmodified original exporter, including its
original-model PCA calculation. All sampled float32 values match exactly.
The projection and original feature manifest are preserved under `experiments`.

Three targeted tests cover complete fixture export and row permutation,
query-label independence, refusal to overwrite, changed checkpoint/basis
rejection, and incompatible contexts or unknown query triples. The historical
exporter and SO1's frozen dependencies remain unchanged.

This completes the latent-feature export only. PF2 below covers meta/context
arrays and movie metadata. Regenerated neighborhood support arrays, the fitted
production blender and retrained prediction bank remain pending. PF1 provides
no new development or qualifying score.

### Metadata and context transfer (PF2)

`scripts/production_metadata.py` reproduces the retained **11 metadata and five
context columns** from `artifacts/r4/meta-context.npz` byte for byte. The cache
and original feature-function sources are checked against LC6's frozen hashes.
It also reproduces all **16 movie-tag columns** against their retained blend
input hashes. The seven extra metadata columns use the unchanged, hash-verified
original `extra_meta` function; no separately saved historical extra-metadata
array exists, so byte-exact historical replay is not claimed for that block.

The qualifying export contains all **39 scalar features** (11 metadata, five
contexts, seven extra metadata, 16 movie tags) for all **2,817,131 queries**.
User/item rating statistics use train+probe labels, while implicit histories
and the five context formulas' fixed scales remain unchanged. Query ratings
are never used. Movie categories and tags are frozen. This differs from PF1's
learned latent coordinates: these statistical features retain their formulas
and update their values to reflect the larger production training set.
Any learned blender clipping/standardization must still be fitted on development
rows and applied unchanged to qualifying rows.

Evidence: `experiments/PF2-metadata.json`,
`experiments/PF2-metadata-verification.json`, and
`experiments/production-metadata-recipe.json`. The independent verifier checks
all output hashes, shapes and finiteness. Direct per-user reductions and item
moments reproduce all 23 numerical features exactly on 2,048 qualifying rows.
The original movie generator's pure table construction reproduces all 16 movie
features across **every qualifying row**. This includes 700 qualifying movies
with no development probe rows, which cannot be covered by copying values from
probe columns. The original table and generator are preserved in
`experiments/production-movie-metadata.zip`.

Three targeted tests cover merged training statistics and row order, query-label
independence and training-label sensitivity, categorical mappings, refusal to
overwrite, changed-cache rejection, and movie-ID coverage. The original feature
functions and SO1's frozen dependencies remain unchanged. This completes these
feature exports, not the final fitted blender or the score goal.

### Persisted final blender and ordered bank (PB0)

`scripts/production_blend_plan.py` builds an ordered bank manifest from the
retained tree/NN reports, PF1, and PF2. The current
`experiments/production-blender-plan.json` contains **79 prediction columns,
64 auxiliary features, and five contextual-ridge columns**, preserving the
duplicate prediction aliases. All development values were compared with the
unmodified `round8_nn_blend.bank` over all 1,408,395 rows and match exactly.
Evidence: `experiments/PB0-bank-verification.json`.

The fixed final-refit rule is the median selected budget **per seed across the
five completed development folds**. For seeds 2026/7/13, this gives tree rounds
**904/969/1106** and neural epochs **15/14/14**. This rule is chosen before any
qualifying result and introduces no new budget search. Other retained tree/NN
hyperparameters and the 50/50 mixture are unchanged. A newly promoted bank must
generate a new manifest from its own completed fold reports.

`scripts/production_blend.py` provides `check`, `fit`, and `infer` commands.
The `fit` command uses all original development probe labels, fits contextual
ridge and the NN's clipping/scaling on those rows, then saves ridge weights,
normalizer arrays, LightGBM text models, neural state dictionaries, ordered
schemas, versions and input/artifact hashes. Inference requires a verified fit
binding and the same column order; it applies the saved transforms and clips
the final predictions to [1,5]. It preserves the retained implementation's
float32 tree prediction before combining with the float64 neural estimate.
Qualifying labels never enter fitting or inference calculations.

The initial combined-library smoke run aborted with OpenMP Error 15: the
installed LightGBM and Torch runtimes conflict when both initialize training
in one process. `scripts/production_blend_tree.py` now runs tree fitting and
prediction in an isolated subprocess, with no duplicate-runtime override.
Three targeted tests pass: budget/normalizer compatibility, component and
saved-model inference compatibility, and complete CLI fit/reload with query
labels changed from zero to five. They also cover duplicate aliases, schema
and artifact rejection, unknown-label training rejection, and missing inputs.

**No full-probe final blender has been trained.** The qualifying manifest has
87 pending entries: 79 prediction columns and eight support/count aliases.
The 61 remaining qualifying feature entries are available from PF1/PF2. An
attempt to load the incomplete qualifying bank is verified to fail before
inference. Final model fitting and production-bank regeneration remain gated
on freezing the bank after the development target/promotion checks; this work
does not establish a score gain or complete the production pipeline.

## Initial artifact inventory

`scripts/production_inventory.py` initially scanned 2,646 JSON records from the primary
and relevant research worktrees and verified the promoted bank against its
frozen input hashes. Results: `experiments/production-bank-inventory.json`.

| Evidence | Prediction columns |
|---|---:|
| Included in promoted bank | 79 |
| Present and hash-matched | 79 |
| Unique prediction hashes | 77 |
| Explicit matching output-digest record found | 33 |
| Matching output-digest record also contains a command | 21 |
| No explicit output-digest record found by this scan | 46 |

The last row does **not** mean those columns are irrecoverable or that training
commands do not exist. Some historical records have only paths, some selected
epoch files lack historical digests, and some recipes live in shell scripts or
git history. Subsequent recovery is recorded below and in the coverage audit.
Input-hash mentions in
blender reports are deliberately not counted as evidence of how a column was
produced. A command record alone does not establish that every dependency is
available or that fixed-epoch qualifying inference is implemented.

Two duplicate prediction pairs are already in the promoted bank:

- `other-branch/joint256-snap.f32` and `artifacts/joint256-reg01.f32`.
- `other-branch/knn64-j256snap.f32` and `artifacts/r4/knn-J6-p1.f32`.

Preserve them in score comparisons: removing duplicate input columns can alter
regularization and nonlinear blending even though their values are identical.
They represent potential shared production work, not a new score claim.

## Required before a final production claim

1. Reach the development milestone under the declared promotion protocol, then
   freeze the retained bank and training settings. The working 0.86465 target
   remains calibrated from the approximate 0.0074 transfer offset.
2. Recover and verify each retained prediction's full training recipe, source,
   fixed epoch/sweep count, and dependencies, including graph/base-model inputs.
   Inventory support features and their train+probe transformations as well.
3. Use the verified train+probe data and qualifying-order mapping described
   below. Query ratings are unknown; their zero sentinels cannot choose epochs
   or produce meaningful RMSE.
4. Implement and verify fixed-budget production paths for each retained family,
   plus the full-probe-fitted blender and matching qualifying meta features.
   Rehearsals cover important families but do not prove the entire 79-column
   production pipeline is ready.
5. Run the frozen train+probe pipeline, validate prediction coverage/order,
   finiteness and [1,5] bounds, and preserve complete input/output provenance.
6. Report the result at the level the evidence supports. Qualifying labels are
   unavailable here, so a calibrated estimate cannot be called a verified
   hidden-test score or competition victory.

The inventory is a starting point for the column registry. Production readiness
is incomplete; no final retraining or qualifying-score claim is made.

## Recipe recovery checkpoint

`PR0-J6-knn64` replayed the documented J6 kNN command against the retained base
model and graph. Both predictions and support are byte-identical to the frozen
bank artifacts (maximum absolute difference zero). Evidence is in
`experiments/PR0-J6-knn64-verification.json`, with the command, input hashes and
source archive in the accompanying experiment records. This verifies inference
reconstruction; rebuilding the base model and graph on train+probe remains open.
`scripts/recover_knn_recipes.py` completed ten related fixed recipes in PR1:
seven matched exactly and three graph-256 J6 variants did not. The original
`ROUND4.md` base-model table records time decay 0.003 for that graph; filenames
omit it. PR2 replayed only those three variants with that documented decay.
All three predictions and the retained support column then matched exactly.
Both the failed initial reconstructions and corrected commands are preserved in
`experiments/PR1-knn-verification.json` and
`experiments/PR2-knn-decay-verification.json`. Output hashes were independently
rechecked. Together with PR0, eleven recipes now reproduce retained predictions
exactly, along with all four corresponding support columns used in the bank.

`scripts/recover_neural_recipes.py` recovered fixed-epoch command templates for
all 14 retained primary-worktree neural columns and all five recorded source
versions from git, with exact source SHA-256 matches. The registry is
`experiments/production-neural-recipes.json`; the source bytes are preserved in
`experiments/production-neural-sources.zip`. Eight columns also have historical
matching selected-output digests. The other six have current bank hash matches
and recorded epoch histories, without equivalent historical output digests.
Retained snapshot filenames take precedence over a later final training epoch.
Archive and member hashes were independently verified, and all five recipe and
inventory tests passed. These are recovered recipes, not verified retraining
results or a complete production pipeline.

## Production dataset prepared and verified

`scripts/production_dataset.py` created
`data/production/stage-c-v1/train-probe-qualifying.bin`: 100,480,507 labeled
train+probe ratings and 2,817,131 unlabeled qualifying queries for 480,189 users
and 17,770 movies. It recovers the original dense-user mapping by first
appearance in ascending raw movie files, preserves qualifying-file query order,
and writes both `dense-to-raw-user.npy` and `qualifying-order.npy`.

The labeled rows and qualifying queries exactly partition the original
103,297,638 implicit interactions. An independent inversion check locates all
1,408,395 original probe rows in production training, checks their labels, removes
them, and recovers all 99,072,112 original training items, dates and ratings.
User offsets, unchanged implicit arrays, and all output/source hashes also pass.
Evidence: `experiments/production-dataset-manifest.json` and
`experiments/production-dataset-verification.json`. Three fixture tests cover
mapping/order, label preservation, no overwrite, invalid dates, and missing or
duplicate queries. The development dataset remains unchanged.

Reproduce into a new output directory:

```sh
python scripts/production_dataset.py \
  --source data/processed/netflix.bin \
  --training-dir data/raw/extracted/download/training_set/training_set \
  --qualifying data/raw/extracted/download/qualifying.txt \
  --output-dir data/production/stage-c-v1
```

This is production-only data. Query ratings are zero sentinels, never labels;
existing development selection/scoring commands must not be used on them.
The full retained bank still lacks a complete production pipeline, and no final
model training has been launched.

## Fixed-budget neural adapter

`scripts/production_neural.py` supports the fourteen recovered primary-worktree
neural recipes. It verifies the production manifest and retained column hashes,
extracts the recipe's exact archived sources into an isolated directory, and
calls the archived fixed-epoch refit path directly. It bypasses the development
scorer, rejects selection data, early-selection settings and partial epochs,
and preserves raw column predictions for the final blender. There is no
qualifying RMSE in its report. The default only prepares a run; `--train`
executes the retained epoch budget.

All fourteen recipes passed preparation against the full production dataset:
`experiments/PN0-neural-preflight.json`. This confirms compatible imports,
arguments, configuration and verified data, not completed full-model training.
Two test methods cover all four distinct archived trainer versions plus archive
tampering. Each trainer ran two complete fixture epochs twice, with zero versus
poisoned query labels: trained parameters and predictions were identical.
Final-epoch checkpoint contents, query ordering and selection rejection pass.
The adapter writes predictions itself because the oldest archived trainer does
not expose the newer prediction-writing helper.

Example preparation, with a fresh output directory:

```sh
python scripts/production_neural.py \
  --column A2h-refit.refit.e12.f32 \
  --dataset-manifest data/production/stage-c-v1/manifest.json \
  --output-dir artifacts/production/A2h
```

Remaining work includes other retained model families and their dependencies,
production support features, the final blender, and full-budget training after
the development milestone and retained bank are frozen.

## Native fixed-budget gaps

Code inspection found that `neighborhood` and `integrated` still saved the best
query-RMSE checkpoint, unlike the existing `joint` and `rbm --fixed-epoch`
paths. They now accept `--fixed-epoch`: save each epoch's state and predictions
and disable early stopping, preserving the last requested epoch. Default
development behavior is unchanged.

An isolated binary at `artifacts/production-replay/bin/netflix-native-fixed`
passed four tests: both models run six epochs with zero versus poisoned query
labels and produce identical parameters and predictions; saved epochs and
reload predictions match; existing neighborhood and integrated temporal-isolation
tests still pass. `build/netflix` remains byte-identical to the PR2 replay binary.
Build/source hashes and checks are in `experiments/PN1-native-fixed-epochs.json`.

This closes checkpoint-selection support for these two native families, not
their entire production recipes. Native commands still print query-RMSE
diagnostics; those are meaningless on the production dataset's zero sentinels
and cannot establish a score. Bayes uses a fixed sweep budget already; its
recovered initialization/resume chain and averaging window are recorded below.
The time baseline, residual graph, warm-start checkpoints and support features
must be rebuilt from production inputs rather than reused from development.

## AutoRec fixed budgets and retained trajectories

`scripts/autorec.py --fixed-epoch` now retains the last requested epoch and
disables early stopping. All-zero query labels require that mode and produce
`query_labels=unknown` instead of a query RMSE. Invalid or mixed unknown labels
are rejected. `--save-epoch N` additionally exports a predeclared trajectory
snapshot; it requires fixed-budget mode and an epoch within that budget.

The retained artifacts have unique RMSE matches, at the historical logs'
printed precision, to these epochs:

| Column | Training group | Epoch |
|---|---|---:|
| `autorec512-a2.f32` | A2-autorec512-norm | 1 |
| `autorec512-a3-snap.f32` | A3-autorec512-reg | 14 |
| `autorec512-a3.f32` | A3-autorec512-reg | 44 |

The A3 columns can share a single 44-epoch production run with
`--fixed-epoch --epochs 44 --save-epoch 14`. Commands, hashes and epoch evidence
are in `experiments/production-autorec-recipes.json`. Only final A3 has a matching
historical output digest in its producer record. Those records did not hash
the Python source, so this is weaker provenance than the fourteen archived
masked-neural recipes; it is not a claim of byte-identical full retraining.

Three fixture tests pass: zero versus poisoned query labels produce identical
predictions at a fixed six epochs; unknown labels cannot enter development
selection; an exported epoch-two snapshot equals a separate two-epoch run.
The unmodified development path also matches the previous source's predictions
and stopping epoch exactly on a fixture, recorded in
`experiments/PN2-autorec-compatibility.json`. The new native fixed-epoch tests
are included in `make test`. No full AutoRec production training was launched.

## Bayes recipes and posterior verification

`scripts/recover_bayes_recipes.py` reads the retained version-one checkpoints
and reproduces all three retained prediction columns byte for byte from their
stored posterior sums and kept-sample counts. It verifies the historical binary
and command records, including B4's sweep-32 checkpoint. Results and production
command templates are in `experiments/production-bayes-recipes.json`.

| Column | Rank / seed | Sweeps | Burn-in / kept | Thread schedule |
|---|---|---:|---|---|
| B2 | 64 / 2026 | 128 | 16 / 112 | 4 |
| B3 | 128 / 7 | 160 | 16 / 144 | 4 |
| B4 | 128 / 13 | 160 | 16 / 144 | 4 through 32; resume with 5 |

B4 enables item-bin and drift factors. Its production resume must consume a
**new production sweep-32 checkpoint**, not the retained development checkpoint.
The exact historical binary is present and hash-verified at
`artifacts/r8/bin/netflix-bayes-0c1f29d`.

Three tests pass against that binary: single-thread zero/poisoned query labels
leave parameters and predictions identical; a 4-to-5-thread resume preserves
averaging counts and serialized posterior predictions; malformed checkpoint
formats are rejected. `experiments/PN3-bayes-production-verification.json`
records those checks and an identical-input parallel repeat. The historical
source explicitly documents Hogwild races in bag updates, so parallel retrains
are not bitwise reproducible. This was also documented in the adjacent Bayes
research worktree's Round 11 notes. The initial parallel equality test failed
for that reason; exact cross-run equality is tested only with one thread.

The small repeated fixture's prediction differences are not a full-data score
estimate or evidence that changing the sampler improves the ensemble. Stored
posterior reproduction verifies the existing artifacts, not a complete retrain.
No full Bayes production training was launched.

## RBM checkpoint budgets and resume chains recovered

`scripts/recover_rbm_recipes.py` reconstructs twelve training stages needed for
the eight retained primary-worktree RBM columns. Each final epoch comes from the checkpoint
header and has a matching training-log entry. Ten stages also match historical
checkpoint digests. All eight retained checkpoints were replayed through the
unchanged native evaluator with one thread; all eight prediction files match
the bank byte for byte, including the two oldest columns without historical
output digests. Evidence: `experiments/production-rbm-recipes.json` and
`experiments/PR3-rbm-verification.json`.

| Retained column | Required fixed training stages (absolute epochs) |
|---|---|
| rbm100 | R1: 1–14 |
| rbm200 | R3: 1–30 |
| R4D conditional | R3: 1–30 → R3R: 31–75 → R4D: 76–129 |
| R4C conditional | R3: 1–30 → R3R: 31–75 → R4C: 76–129 |
| P2 conditional | P2: 1–72 |
| R2C conditional | R2: 1–60 → R2C: 61–130 |
| P3C conditional | P3: 1–58 → P3C: 59–118 |
| P4C conditional | P4: 1–54 → P4C: 55–108 |

The stage boundaries matter: warm starts reset momentum and user-order RNG.
Preserve each recorded stage, thread count and learning schedule rather than
flattening the chain into one uninterrupted run. Templates use fixed epochs,
disable patience, and reference newly trained production predecessors. They
contain no development artifact paths. Historical source hashes are recorded,
but matching fixed-epoch builds and full production retraining remain open.
Inference replay establishes checkpoint-to-column identity, not a score gain.

The initial exact-path registry missed a ninth RBM column copied into the
reference bank: `other-branch/rbm100-fast.f32`. Recovery now matches prediction
digests across paths. `experiments/production-rbm-recipes-v2.json` includes all
nine retained RBM columns and the same twelve training stages. The extra R2
column requires epochs 1–60 and has been replayed byte for byte as well.
`experiments/PR4-rbm-reference-verification.json` combines that new replay with
the eight prior verified replays; their checkpoint hashes, budgets and resume
chains are unchanged. The original eight-column report is preserved as history.

## Joint and integrated checkpoint recovery

`scripts/recover_joint_recipes.py` inventories twelve retained joint/integrated
checkpoints covering thirteen bank columns (J6 appears twice). All twelve
checkpoint inference replays reproduce the frozen columns byte for byte.
Evidence: `experiments/production-joint-recipes.json` and
`experiments/PR5-joint-verification.json`. Integrated models also record the
graph and `.bin.nbr` sidecars required for inference and resumption.

| Stage | Fixed epochs | Predecessor |
|---|---|---|
| J1 | 1–6 | — |
| J2 | 1–40 | — |
| J3 | 1–9 | — |
| J4 | 1 only, numbering restarts | J3; add 128 day factors |
| J6 | 1–17 | — |
| T2 | 1–37 | — |
| T3 | 1–13 | — |
| T4 | 1–21 | — |
| T6 | 1–11 | — |
| T6R | 12–24 | T6 |
| I2 | 1–17 | — |
| I2R | 18–23 | I2, including its neighbor sidecar |

The current J1–J4 files do not match their original producer-record hashes,
although their predictions match the frozen bank. Their historical commands
therefore initially remain candidate recipes. A fresh six-epoch J1 training
run with the current frozen native binary now reproduces **both the checkpoint
and all predictions byte for byte**; all five fold score differences are zero.
See `experiments/PR5-J1-retraining-verification.json` and the accompanying full
experiment record/source archive. The old hash discrepancy is preserved; this
new run supplies direct recipe evidence for the currently retained J1.

PR6 completed the remaining checks sequentially: J3 (9 epochs), J4 (one epoch
from the newly reproduced J3), then J2 (40 epochs). All three newly trained
checkpoints and all 1,408,395 predictions per model match the retained files
byte for byte. The immutable commands and reference hashes are in
`experiments/PR6-legacy-joint.plan.json`; terminal status and individual stage
verification records are preserved. The independent final audit in
`PR6-legacy-joint-final-verification.json` rechecks all thirteen frozen inputs,
each saved epoch, successful command completion, checkpoint hashes and finite
prediction arrays.

Together with PR5, this supplies exact full original-data retraining evidence
for the currently retained J1–J4 recipes. The differing older producer digests
remain explicitly recorded. Production commands must still use new production
predecessors and graphs. These checks do not establish a production score or
reduce the target gap. SO1 remains queued behind N1 and its follow-on chain.

## Rebuilt graph and time-baseline dependencies

PR7 retrained the original fixed 40-iteration time baseline and rebuilt G2's
256-neighbor graph with history cap 256. Both match their retained artifacts
byte for byte. The native graph ranks each row by descending similarity and
movie ID. `scripts/production_graph.py` validates this ordering, payload,
neighbor IDs and fingerprint, then retains each row's first 64 entries. Applied
to the new graph, this exactly reproduces the retained 64-neighbor graph.

Inference using the retained epoch-4 learned-neighborhood weights and the
**newly rebuilt** baseline/graph reproduces all 1,408,395 frozen predictions
byte for byte (RMSE 0.9100421231). Evidence is in
`experiments/PR7-dependency-verification.json` and its three experiment records.
The graph transformation's two tests pass. This verifies dependency recipes
on the original dataset; neighborhood weights have not been retrained here,
and production-data graphs/baselines must still be built on the production
dataset. No score gain is claimed.

PR8 subsequently retrained the learned-neighborhood weights for exactly four
epochs using those rebuilt dependencies and the tested isolated PN1 binary.
The saved epoch, dataset/graph fingerprints and zero time decay agree with
the retained checkpoint. Historical bit identity is **not** achieved: the
current model has a version-2 header, and 433,299 of 2,274,560 parameters differ
(maximum absolute difference 7.153e-7). Of 1,408,395 predictions, 146,584 differ;
maximum absolute difference is 9.537e-7, mean absolute difference 2.994e-8.
Overall RMSE changes by +9.741e-11, with absolute fold deltas below 3e-10.
No tolerance was selected to claim a score promotion or historical bit identity.

Reloading the newly trained checkpoint reproduces its complete prediction
column byte for byte. The full numerical comparison, all five fold scores,
and reload evidence are in `experiments/PR8-neighborhood-retraining-verification.json`.
`experiments/production-neighborhood-recipe.json` binds the fixed budget,
hyperparameters, binary and dependency requirements. The current fixed-budget
recipe is now verified on original data; production-data training remains to
be done, and the retained development bank is unchanged.

## Remaining item-kNN replay batch

PR9 replays the seventeen item-kNN columns not covered by PR0–PR2: the
temporal baseline and J1–J4 G1 variants, T2/T3/T4/T6 G1 variants, the two
original power-2 variants, and six time-decayed G2 variants. It checks retained
prediction digests and the relevant retained support column, using new output
paths. The two original power-2 hypotheses use the native defaults (power 2,
shrink 0.05); other settings come from RESULTS.md/ROUND4.md and filenames.
A mismatch remains unresolved; the script does not tune against scores.

The frozen commands are in `experiments/PR9-knn-remaining-plan.json`, with live
status in `PR9-knn-remaining.status.json`. The runner and original binary are
hashed before each stage. This is recipe recovery, not a new score candidate.

PR9 completed all seventeen stages, with sixteen exact prediction matches.
The original `knn64-j64-p2` column did not use the assumed shrink 0.05. PR10
recovered its shrink directly from the retained predictions, the J1 baseline,
and the PR9 replay's neighbor-weight sums, using
`s = (replay-base)*(support+0.05)/(retained-base)-support` (with float32 0.05).
Across 8,743 informative rows, the inferred median was 1.000000101 and the
full range was 0.9999443–1.0000662. A single follow-up at shrink **1** reproduced
the retained column byte for byte. No rating labels or score search were used;
the original mismatch remains in the PR9 record. The inference and exact check
are preserved in `PR10-knn-shrink-recovery-plan.json` and its verification file.

`scripts/production_knn_registry.py` consolidates PR0/PR1/PR2/PR9/PR10 and checks
the current data, binary, model, graph and prediction digests against their
successful execution records. `experiments/production-knn-recipes.json` now
covers **all 29 retained item-kNN columns (28 unique prediction columns)** and
**six item-kNN support inputs (three unique arrays)**. Duplicate bank aliases
remain explicit. The registry rejects missing columns or changed artifacts.
This closes item-kNN inference recipe recovery. User-kNN is separate, and all
production runs still need freshly trained production base models and graphs.

## User-kNN and its K1 dependency

PR11 replayed the original user-kNN command with its retained binary and K1
checkpoint. All 1,408,395 predictions, support weights and neighbor counts
match the retained files byte for byte. This covers the separate `C-uknn-K1`
bank column and its support input.

K1's checkpoint header selects **epoch 41** at rank 256. Its checkpoint and
prediction hashes agree with the historical training record. All six compiled
training sources were recovered by their recorded SHA-256 hashes. Its source
already supports `--fixed-epoch`; the production recipe preserves learning
rate .008, regularization .02, decay .97 and six training threads, but fixes
41 epochs and disables patience. Six-thread Hogwild training is nondeterministic,
so historical weight identity is not promised. K1 has not been retrained here.

User-kNN source was recovered from commit `2419104`, named in the original
run strategy. Its run record had captured the main worktree's sources rather
than that binary's sources, so a separate rebuild check was needed. The
recovered binary passes three existing user-kNN contracts. An additional
integration test trains the recovered K1 source for three fixed epochs with
query labels 0 versus 5, then runs both original and rebuilt user-kNN binaries.
Parameters after the fingerprint, base predictions, kNN predictions, support
and counts all match. Training uses one thread in this isolation test.

On real data, the rebuilt and original user-kNN binaries also agree byte for
byte on a fixed subset of 1,405 queries (users divisible by 1000), including
support/counts. All output rows are compared, but unselected rows retain their
base prediction; this is bounded source compatibility, not a full replay of
the rebuilt binary. The original binary's full replay is separate evidence.

Sources, build commands, tests and recipe are recorded in
`experiments/production-user-knn-sources.zip`, `PR11-user-knn-verification.json`
and `production-user-knn-recipe.json`. Production K1 training and the K1-based
latent blender features remain outstanding. No score gain is claimed.

## GP/local recovery and complete column coverage

PR12 recovered the original GP entry point by its manifest hash, plus the
recorded GP2/local source set. The original GP's shared readers use that GP2
source set; full output matching verifies their compatibility. Three isolated
binaries were built without changing the working native binary. Two contracts
cover analytic GP solves and query-label 0-versus-5 isolation for both GP
variants and six-epoch global/local residual training.

All six retained outputs reproduce **byte for byte**: three static GP noise
variants, the persistent GP's noise-0.1 variant, and the global/local residual
models at epoch 6. The local checks include complete retraining from the J3
base. The independent final audit rechecks all fifteen frozen inputs and four
successful execution records. No score tuning or bank replacement occurred.

`experiments/production-kernel-recipes.json` records commands, parameters,
sources, binary hashes and the requirement for a newly trained production J3
base. GP outputs are direct inference. The legacy local trainer emits each
epoch's predictions but does not persist its residual factors; the production
recipe therefore reruns its fixed six-epoch schedule.

`scripts/production_recipe_coverage.py` associates the frozen bank by digest
with all family registries, rejects missing recipes and rechecks every retained
column hash. It covers 13 joint/integrated, 29 item-kNN, nine RBM, three AutoRec,
four GP, two local residual, fourteen neural, three Bayesian, one learned
neighborhood and one user-kNN column. The two duplicate pairs are preserved.
Its per-family validation notes explicitly retain the weaker historical
AutoRec/neural provenance, unrepeated full training, and numerical drift in
the learned-neighborhood retraining. Coverage does not upgrade those claims.

Outstanding work still includes the target development score, production
latent/context features, complete train+probe retraining, full-probe blender
fitting, qualifying-order export and end-to-end verification. None is proved
by the 79-column recipe count.
