# Final-production readiness audit

2026-10-19; current promoted reference remains R9-LAT-A2h at 0.8668706425
development CV. This audit supports the final retraining step in
`PLAN_TO_TARGET.md`; it neither changes that score nor establishes the 10% goal.
The LC6 score experiment continues independently on the same branch.

## Verified current artifacts

`scripts/production_inventory.py` scanned 2,646 JSON records from the primary
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
git history. They need further provenance recovery. Input-hash mentions in
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

J2–J4 initially needed equivalent retraining checks. All production commands must use
new production predecessors and graphs. These development-data replays and J1
retraining do not establish a production score or reduce the target gap.

PR6 has started those remaining checks sequentially: J3 (9 epochs), J4 (one
epoch from the newly reproduced J3), then J2 (40 epochs). The immutable commands
and reference hashes are in `experiments/PR6-legacy-joint.plan.json`; live status
is `experiments/PR6-legacy-joint.status.json`. It uses one CPU thread at niceness
15 and stops on any checkpoint or prediction mismatch. SO1 remains unlaunched
while N1 and its follow-on chain retain GPU priority.

PR6 has now completed J3 and J4: both newly trained checkpoints and both
prediction columns match the retained artifacts byte for byte. J4 used the
newly reproduced J3 predecessor. Their individual `PR6-legacy-joint-J3` and
`PR6-legacy-joint-J4.verification.json` reports are final; J2 remains running.

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
