# Our own Netflix training progress

## November 26, 2026

Source0 reached 0.856442 RMSE (10.0849% below Cinematch). Source1 reached 0.857521 (9.9715%), leaving 0.000271 RMSE to the 0.857250 target. These are community-recovered qualifying scores from models trained by the project on Netflix ratings. An independent refit reproduced all 2,817,131 qualifying predictions byte for byte.

| Checkpoint | Source1 RMSE |
| --- | ---: |
| November 24 Own17x + CF96 | 0.860599 |
| History models and corrections from other observed ratings | 0.859566 |
| Blend weights adjusted to each viewer’s context | 0.859017 |
| CF96 with inferred ratings and temporal attention | 0.858073 |
| Related-title neighborhood added | 0.857693 |
| Prespecified comparison omitting the 202-forecast neural stack | 0.857551 |
| Exploratory blend retaining forecast differences with similar variance on both splits | 0.857521 |

Each row measures a cumulative blend. The steps show when gains appeared. Models added together share credit. The prespecified primary blend including the neural stack scored 0.858241 on source1. The exploratory result followed inspection of earlier recovered scores. [Final audit](experiments/own-progress-20261126/OWN-final-overnight-20261126.json) · [Intermediate scores](experiments/own-progress-20261126/intermediate-scores.json).

## November 24, 2026

The CF96 refit with context and CatBoost corrections improved both recovered groups to **0.85956670 / 0.86059917 RMSE**. Claude Code ran the remote CF96 experiments and Own17x refit. Codex fitted the final correction weights on known ratings and verified the frozen predictions. The harder group remains **0.003349 RMSE** above the 10% target. [CF96 audit](experiments/own-progress-20261124/OWN17X-cf96-context-tree-20261124.json).

The CF48 residual correction improved Own17x to **0.85971372 / 0.86072426 RMSE**. Adding the CatBoost correction lowered both groups further to **0.85965272 / 0.86067024 RMSE**. Both weights were fitted on known ratings, and the harder group remains **0.003420 RMSE** above the 10% target. [CF48 audit](experiments/own-progress-20261124/own17x-cf48.json) · [CatBoost audit](experiments/own-progress-20261124/own17x-cf48-tree.json).

The Own17x global blend combines 17 existing families with 65 predictors trained on the remote Mac. The frozen predictions score **0.85984307 / 0.86084907 RMSE** on the two complete community-recovered groups. That improves the matched Own17 global control by **0.000488 / 0.000486**. The harder group remains **0.003599 RMSE** above the **0.85725** target. The remote known-row blend comparison used reused out-of-fold inputs and was non-nested. [Qualifying audit evidence](experiments/own-progress-20261124/own17x.json).

The earlier [Own17 simplex audit](experiments/own-progress-20261124/own17.json) scored **0.86032634 / 0.86132655**.

## Earlier progress

Updated November22 UTC /November21 Pacific,2026.

November22 07:33 UTC: the full selected-tree qualifying controller is now
live as owner75543/watchdog75552, waiting the complete original two-arm known
selection under its separate383-file snapshot after119 combined checks. It
covers preparation, full qualifying inference, independent raw/ridge/model/bag
replay and final pair checks, preserving the original resource policy and all
fold/seed/row budgets. The original known-tree owner66664/watchdog66674 remains
unchanged under364 frozen files, waiting completed own17 primary. Scalar
neural replay27425/owner24039 is active after all15 models finished fitting;
latent, selected NN bag and own17 remain pending. No tree model or new score is
admitted. Development0.8588630884 and recovered0.8603409488/0.8613420939 remain
unchanged; strict0.85725 on both groups remains unmet. Full own18 is still pending.
[Reviewed queued full tree pair](OWN_FULL_CONTEXT_CATBOOST_OWNED_BAG_PROGRESS_V1.md).

The newer remote pushed reportaa3eac9, observed06:34 UTC, reports TX30012/24,
RBM80012/25 and RBM4006/15 in known train_3->probe_3. No completed receipt,
export, archive or upload exists. This is a Git snapshot, not a live remote
process poll; no remote control was issued and no returned pair is admitted.
[Remote snapshot](experiments/upstream-remote-weekend-progress-0634/verification.json).

November22 07:07 UTC: the full scalar neural fit completed at06:59,
with all five original outer folds and fifteen fresh saved MPS models. Every
fold receipt and its model/preprocessing/prediction payloads was rehashed.
Separate actual-model replay worker27425 is active under owner24039. Canonical
scalar scoring, the full latent arm, selected NN qualifying bag and own17 are
still pending. No new development or recovered score is admitted.
The selected tree source gate, full pair assembly and separate ridge/model/bag
replay components pass100 combined checks, including real saved models/pipes,
ten full-width ridge systems and rehashed one-ULP mutation rejection. Their
owned qualifying controller and actual full-data run remain pending. The live
known-tree owner66664/watchdog66674 retains its original364-file/89-check plan
and waits the complete own17 primary condition. Development0.8588630884 and
recovered0.8603409488/0.8613420939 remain unchanged; strict0.85725 remains unmet.
[Complete scalar fit and live replay](experiments/upstream-rbm400-complete-scalar-fit-live-replay/verification.json).
[Selected tree pair components](OWN_FULL_CONTEXT_CATBOOST_SELECTED_BAG_PROGRESS_V1.md).

November22 06:31 UTC: a fresh remote-branch inspection finds report
9c9c1c7, observed05:34 UTC, after the remote-side resume commit. It reports
TX300 at8/24 epochs, RBM800 at9/25 and RBM400 at4/15, all still in the known
train_3->probe_3 phase. No completed receipt, export, archive or upload exists.
This is a pushed snapshot, not a current remote process poll; no remote job
control was issued here. No returned pair or new score is admitted.
[Original snapshot and scope](experiments/upstream-remote-weekend-progress-after-resume/verification.json).

November22 06:21 UTC: four of five scalar folds have completed,
with twelve fresh saved MPS models. The fourth selected epochs are[9,9,5];
every output byte in its immutable receipt has been rehashed. Worker32127
continues on the final scalar fold. Full scalar/latent replays and canonical
comparison, selected NN qualifying bag and own17 audits remain pending.
The tree owner66664/watchdog66674 are now live, waiting for the completed own17
primary under364 frozen files after89 checks. It starts both full tree arms
only if that primary misses the target. No tree model or new score is admitted.
Development0.8588630884 and recovered0.8603409488/0.8613420939 are unchanged;
strict0.85725 on both groups remains unmet.
[Fourth completed scalar fold](experiments/upstream-rbm400-fourth-scalar-fold-complete/verification.json).
[Reviewed live tree queue](OWN_FULL_CONTEXT_CATBOOST_OWNED_KNOWN_PROGRESS_V1.md).

November22 05:48 UTC: three of five scalar outer folds are complete,
with nine fresh saved MPS models. Selected epoch counts are[6,11,7], [9,5,5]
and[5,8,6]; every output byte in the completed fold receipts has been rehashed.
Worker32127 continues under owner24039. Two scalar folds, the full latent arm,
independent replays/canonical comparison, selected qualifying bag and own17
audits remain pending. Development0.8588630884 and recovered0.8603409488/
0.8613420939 are unchanged; strict0.85725 on both groups is still unmet.
Tree follow-up components now pass76 combined checks, including actual isolated
producer-to-fit orchestration, same-PID group stop/continue and inherited lane
lock retention. Resource/source/ownership fixtures remain synthetic; external
full controller and role integration are still pending. The current neural
chain retains priority and the original conditional tree-start rule remains.
[Third completed scalar fold](experiments/upstream-rbm400-third-scalar-fold-complete/verification.json).

November22 05:18 UTC: the scalar neural study has completed two of five
outer folds and saved six fresh MPS models. Their selected epoch counts are
[6,11,7] and [9,5,5]; all frozen model, preprocessing and prediction output
bytes in both receipts have been rehashed. Worker32127 remains active under
owner24039. Complete scalar/latent replay and canonical comparison, the full
selected qualifying bag, and own17 calibration/audits are still pending.
Development0.8588630884 and recovered0.8603409488/0.8613420939 remain unchanged;
the strict0.85725 target on both recovered groups is unmet. The planned tree
follow-up now has full-study replay/canonical, actual producer-exit, bounded
query receiver and saved-model block-inference components. Its61 combined
checks plus two additional real-model inference checks pass on synthetic
sources; the owned full tree run remains pending under its original condition.
[Second completed scalar fold](experiments/upstream-rbm400-second-scalar-fold-complete/verification.json).

November22 04:41 UTC: both actual neural arm manifests now bind the completed
47c137... replay proof and the correct714/864 inputs with225/375 contexts.
Owned preparation completed at04:27; the full scalar training worker32127 is
live under owner24039, with no complete fold or new score yet. The latent arm,
qualifying replay25553 and own1726473 follow their complete predecessor gates.
All original five folds/three seeds/max40epochs/guards/baseline remain. The new
full-arm tree wrapper passes49 component checks with actual small saved models
in synthetic scheduling fixtures; its owned full study remains pending.
Development0.8588630884 and recovered0.8603409488/0.8613420939 are unchanged;
strict0.85725 on both recovered groups remains unmet.
[Active scalar phase](experiments/upstream-rbm400-first-scalar-live-progress/verification.json).

November22 04:23 UTC: the full data proof47c137... remains complete and
unchanged. Neural preparation15925 stopped before fitting because its manifest
builder still read the old data directory for the separate replay proof. The
failed preparation and its empty partial directories are preserved. The new
single proof locator is used for admission, preparation and loading; full
fixtures now keep different actual/stale proof files in separate directories.
New neural owner24039/worker24317 is preparing under238 frozen files after39
checks. New qualifying-bag25553 and own1726473 wait under265/299 frozen files
after17/16 checks. Original math, budgets, guards and baseline are unchanged.
No neural model or new score is admitted; own16 remains0.8603409488/0.8613420939,
and0.85725 is unmet. The versioned future tree producer passes45 combined
component checks, including the separate-proof regression.
[Failure and active corrected chain](experiments/upstream-rbm400-proof-locator-recovery-runtime/verification.json).

November22 04:12 UTC: the complete corrected data replay passed at04:09.
Proof47c137371b06c02f7c840c3a8bd0c31755e6807ecd8bad3e5b5d65f19f3751d9
covers both full query sets, all scalar/context operators, all group statistics,
2815 ridge equations and every held ridge prediction. All seven original
prepared outputs remain unchanged; the earlier failure stays preserved.
The neural successor is now active inprepare, worker15925; qualifying-bag
owner99302 and own17owner99996 wait for its complete proofs. Full source
contracts221/239/264, original budgets, baseline and resource limits remain.
No new neural or recovered score is admitted. Development best0.8588630884
and own16 recovered0.8603409488/0.8613420939 are unchanged; target unmet.
[Completed replay and active successor evidence](experiments/upstream-rbm400-complete-context-replay-runtime/verification.json).

November22 03:49 UTC: the separate data verifier stopped at03:38 because
it authenticated two additional original code files without updating the source
declaration passed into its query reader. The strict full-width fixture now
reproduces that failure and passes with aligned declarations. All seven original
assembled outputs remain unchanged. New replay owner92207/worker92399 is live
under the original3GiB/24GiB/baseline/lane limits after7 recovery checks. New
neural owner94683 waits its complete proof after39 checks and221 frozen files;
no neural fit has started. Older waiting namespaces and the failed verifier
remain intact. The tree producer adapter passes30 combined component checks;
its full owned study and the versioned bag/own17 successors still require work.
No new score: own16 remains0.8603409488/0.8613420939 and target0.85725 is unmet.
[Failure, regression and live recovery evidence](experiments/upstream-rbm400-context-source-alignment-runtime/verification.json).

November22 03:27 UTC: full original-budget data assembly has completed.
The scalar arm retains714 neural inputs/225 contexts and the learned-context
arm864 inputs/375 contexts. All seven prepared outputs were independently
rehashed against their assembly records. The separate full-data verifier77143
is live under owner46613; neural52731, bag60568 and own1762038 continue waiting
for complete predecessor proofs. No neural fit or new score is admitted.
Tree follow-up components now pass22 CPU checks, including durable actual-model
saves and independent synthetic-fold reconstruction; their production study
and complete source admission remain pending.
[Assembly and live verification evidence](experiments/upstream-rbm400-reference-assembly-runtime/verification.json).

November22 03:15 UTC: the full saved RBM400 models and extracted learned
features now pass independent recovery replay on both query sets. The original
verifier stopped because it rejected 18 pre-existing source-path aliases;
models, cache arrays and the original failed execution remain preserved.
A versioned reader accepts only those exact aliases with matching canonical
hashes. Recovery pair proof:485edc77ff298b96bae12019df755a7a123ecc08d3e3d95633fce5eaf1695a0b.
Actual CPU data assembly worker55097 is running under owner46613. The new
neural52731, qualifying-bag60568 and own17 blend62038 owners are live and
waiting for complete predecessor verification. Full study budgets and the
original resource baseline are unchanged. No new neural model or recovered
score is admitted. Own16 remains0.8603409488/0.8613420939; target unmet.
[Verified recovery and live chain](experiments/upstream-rbm400-reference-chain-runtime/verification.json).

November22 02:03 UTC: the completed own16 simplex primary independently
scores0.8603409488199526/0.8613420938772931. Both groups improve by only
about0.0000003; this is a tiny gain and still misses0.85725. All65535
simplex supports,17 ridge equations and six frozen vectors pass the preaudit;
separate full-source canonical reconstruction and compensated scores agree.
Development best remains0.8588630884. Learned-context extraction worker12216
is now active under the original owner87109; the complete CPUdata, neural,
qualifying-bag and own17 queues remain in place.
[Own16 result](OWN_LOCAL_RBM400_RESULT.md).

November22 01:50 UTC: both original fifteen-epoch RBM400 models are saved
and independently replay every full native query and the known projection
on MPS. The complete pair proof is3e8d2ecdbb40d0c953f98b24601c2e84a9353bb45ca23d74fea26797a814199a.
The own16 known-only fit and all six vectors are frozen; independent coefficient/
vector preaudit and fixed recovered audits remain pending. The new own17
owner4412 is live, waiting for the fully verified selected NN-only pair, after
all16 CPU checks and review. All173 frozen files and the byte-identical original
baseline verify. No new numerical score or plot point.
See [own17 plan](OWN_RBM400_OWN17_CALIBRATION_PLAN.md) and
[complete RBM pair evidence](experiments/upstream-factored-rbm-gpu400-deterministic-metal/paired-verification.json).

November22 01:38 UTC: RBM400 qualifying training has completed 14/15 epochs
in the original worker19951. The selected neural qualifying-bag owner84612
is now live, waiting for both full neural arms and known-only canonical
selection. All16 CPU implementation checks and independent review pass;
163 frozen files and the byte-identical original baseline verify. The bag
and replay preserve every qualifying row, all five folds and three seed
slots. The NN-only family remains separate from the fixed-half ranking
and control. Actual bag inference/replay and own17 calibration are pending.
No new score or plot point. See [selected bag plan](OWN_RBM400_SELECTED_BAG_PLAN.md).

November22 01:16 UTC: qualifying RBM400worker19951 has completed
11/15originalepochs; its known15model and allpredictionbits alreadyreplay.
Neuralowner37638/watch37712 isnowlivewaitingthecompleteCPUdataproof, after
all38CPUchecks,cleanreview,157frozenfiles andoriginalbaselineverification.
BothfullH128arms retain5folds/3seeds/max40/pat4/b8192; actualneuralfits,
canonicalknownscores/selection,selectedwholeQbag andown17auditsremainpending.
No numericalscoreorplotpointchanges. See
[ownedneuralqueue](OWN_RBM400_NEURAL_QUEUE_PROGRESS.md).

November22 00:18 UTC: the full15-epoch known RBM400 model is saved and
independently replays all1760494 native predictions and its known projection
bitwise on MPS. Qualifying worker19951 is active with1/15 epochs complete.
Owned extraction87109/watch87113 and CPUdata30414/watch30424 wait for their
complete predecessor proofs. The full3GiB CPUassembly/independent replay
protocol passes24checks and review; all131source files and the exact original
baseline verify. Actualcache/data admission and both neural studies remain
pending. No new development or recovered score. See
[full data pipeline status](OWN_RBM400_CONTEXT_DATA_PROGRESS.md).

November21 23:20 UTC: seven full known epochs are complete in the same
deterministic-Metal worker51842;15known+15qualifying and the original CD/batch/
source/baseline remain. The learned-context extraction/operator/H128factory
prototypes pass31CPU checks and independent review. Real-source/cache/MPS
feature admission and both neural studies have not started. Known user IDs
match the canonical model coordinates on all1128164rows. No new score yet.
[Follow-up implementation status](OWN_RBM400_LATENT_COMPONENT_PROGRESS.md).

November21 22:46 UTC: the full deterministic Metal producer completed its
FIRST known epoch, all480189nonemptyusers and961batches. Same worker51842
continues the remaining14known epochs; qualifying still requires all15.
Fresh full-data admission and the repeated actual production MPS preflight
both passed. Source/config/counter/baseline/live-worker checks pass. No full
terminal model or new score yet; sourceepoch evidence is
experiments/upstream-rbm400-metal-production-protocol/first-epoch-verification.json.

November21 22:37 UTC: the actual FULL deterministic Metal MPS probe passes;
strict determinism, original500user/400hidden stress, original tolerances and
baseline/caps remain. Prior v2full63-input admission completed, but its original
Torch indexed-add MPS preflight failed before training. All prior attempts are
retained. New full-budget producer41101/watch41170 and data41606 are active;
consumer41441 waits for complete15+15model/paired replay. Newsource/handoff/
baseline checks independently pass. No completed model or score gain yet.
[Actual device proof and new production](OWN_RBM400_DETERMINISTIC_METAL_PROGRESS.md).

November21 22:04 UTC: the first native observer still produces false resource
pauses from transient OS descendants. A reviewed companion retires only exact
stopped readonly ps child PIDs; same data worker7939, original limits and
baseline remain. A new observer candidate restores the original accounting
scope and passes4tests plus20 actual scans with zero failures, all under0.08s.
It is not connected to production; explicit successor integration/retirement
is next. No MPS/model/score completion is claimed. See
[scope correction](OWN_RBM400_ORIGINAL_SCOPE_CORRECTION.md).

November21 21:51 UTC: reviewed metadata-only RBM400 recovery completed; old
owner/data/idle-consumer processes are terminal with original records retained.
New owner7652/watch7664 and data worker7939 use the fast native observer and
the exact original baseline, limits and full15+15 recipe. Data validation has
made substantial CPU progress; no MPS preflight/model/score proof exists yet.
Consumer7867/watch7878 waits for full paired verification. Resource suspension
retains the same worker and requires original stable readmission. See
[actual recovery result](OWN_RBM400_OBSERVER_RECOVERY_RESULT.md).

November21 21:08 UTC: own15 full known-only fit/freeze/preaudit and all three
fixed audits now have separate canonical verification. New recovered primary
0.8603412617/0.8613423892 improves both slightly;10% remains unmet. The original
RBM400 coordinator31460 has begun data admission. The proposed learned-context
follow-up is design only and does not alter or restart that producer.

November21 20:32 UTC: the full15-model trajectory study and separate canonical
score checks are complete. New development best0.8588630883643809; the original
qualifying-bag coordinator58980 has begun full inference. Qualifying scores
remain unchanged until the complete bag, own15 known-only freeze and audits.

November21 19:58 UTC scientific update: the complete519-stream input/ridge
study now passes independent reconstruction of every full operator, statistic,
coefficient and held prediction. Ridge0.861725344068627 improves the matched391
control0.8618508817095664 and prior expanded CF ridge0.861770514815263. This is
a component diagnostic, not a new overall development or recovered score.
Original coordinator50842 completed; original neural coordinator56323 has begun
preparing the full713-input H128 study. No new neural score exists yet.
[Complete trajectory ridge result](OWN_COMPLETE_TRAJECTORY_RIDGE_RESULT.md).

November21 19:57 UTC storage update:236 original manifest-SHA256/index-MD5
downloaded public qualifying-prediction inode groups and382 fully accounted
aliases were retired under the existing public-bank cleanup authorization.
Separate absence/provenance/protected-proof verification passes; measured
free gain2.47677GiB leaves about22.29GiB. Seven mismatched groups and unselected
files remain. All own numerical state, data, archives and processes are retained.
[Exact public-cache cleanup](PUBLIC_QUAL_CACHE_RETIREMENT.md).

November21 19:44 UTC execution update: the full519-stream input admission is
independently verified. Its fixed five-fold ridge fit reports0.8617253441
against the matched391 control0.8618508817; independent fit replay is running.
This is a known-fold component diagnostic, not a new development best or a
qualifying score. Exact paired aliases leave486 raw streams and713 neural
inputs without dropping any declared candidate before alias verification.
The subsequent full neural/qualifying/own15 stages remain queued.

The separate owner-override remote-return adapter is implemented, independently
reviewed and declared with24 passing CPU/metadata/private-process tests. No
numerical return has been admitted. It preserves all67 remote pairs, original
scientific validators, immutable source/helper pins and unchanged local guards.
Only exact idle original return coordinators are eligible for handoff; no
native training process is signalled. See
[return protocol](REMOTE_WEEKEND_OWNER_OVERRIDE_RETURN_PLAN.md).
Exact idle-only handoffs have now retired original return owners87173/87178/87185
and their watchdogs. Successors15521/15537/15551 and live watchdogs wait for
the original own15 full independent completion; no return scientific namespace
or first-scientific baseline exists.
Remote commit76758c0e29805fd74636ebe1f33aadda364f18bd reports all three remote
native jobs paused at the owner's request at19:07:27 UTC/11:07:27 Pacific;
this supersedes the older running snapshot. No remote resume is performed.

Best independently verified known-ProbeX development CV: **0.8588630883643809**.
The complete713-input H128 NN/tree154 half improves the matched391 benchmark
0.8589133967520619 by0.0000503083876811. All15 actual original five-user-fold,
three-seed models and complete eligible statistics/scalers/normalized inputs/
held predictions independently replay. Separate canonical verification confirms
NN0.8590060211471172 and the exact fixed-half vector. No qualifying outcome
entered training or selection. The neural provider matches pairedB on both
queries; tree154 retains its historical A/B realization caveat.
[Complete trajectory neural result](OWN_COMPLETE_TRAJECTORY_NEURAL_RESULT.md).

The preceding matched391 result0.8589133967520619 improved own352 by only
0.0000026602614702; its original source/model/continuation evidence is retained.
[Preceding matched391 result](OWN391_PAIRED_NEURAL_RESULT.md).

Current independently verified recovered qualifying benchmark:

- source0: **0.8603412616628533**,1,408,789 rows.
- source1: **0.8613423892425344**,1,408,342 rows.

The predeclared own15 known-only simplex improves own14 by only
0.0000384724/0.0000257621. All32767 supports,actual KKT/objective conditions,
16ridge equations and six full frozen vectors independently verify before
labels. The preceding full qualifying-bag replay reopens all15 actual new
models; separate canonical verification rehashes the source closure and
reconstructs/scores all three candidates. Apparent known-fit0.8586688835
is not nested CV. Both fixed controls are worse than the primary on both
groups and remain controls. [Complete own15 result](OWN15_COMPLETE_TRAJECTORY_RESULT.md).

The preceding own14 primary0.8603797340/0.8613681513 and its complete source/
model/16383-support/frozen-vector/canonical proofs remain preserved.
[Preceding own14 result](OWN14_QUALIFYING_RESULT.md).

The strict target is below **0.85725 on BOTH groups**; it is still unmet.
The worse-group gap is0.0040923892. Development and recovered qualifying
use different rows. Audits are adaptive community evaluation, not official
or untouched Test. No downloaded pretrained numerical features or
qualifying outcomes enter fitting or candidate selection.

The own352 qualifying fold bag is now complete: all15 original saved models,
all qualifying inputs and each of five fold vectors independently replay,
and the uniform bag matches every stored bit. Its seven-family primary
scores0.8604024389/0.8613852654, a negligible gain against own330; the
fixed-half control worsens both groups. Both candidates were frozen; the
seven-family fit passes all127 simplex supports and complete arithmetic.
The apparent known calibration0.8587651555 is not nested CV.

The separate complete13-family known-only simplex/ridge comparison is
frozen and independently verified: all8191 simplex supports, the full14-
dimensional ridge equations and every stored vector pass. Both studies
completed preaudit before either new recovered audit; both audits and
separate canonical checks are complete.
A reviewed queue then runs the declared context-weighted
FWLS study: all13 own component families,14 distinct original context gates,
lambda30, unpenalized intercept and five held-user models. Original public
source/configuration is checked against immutable pinned Git blobs; only
our own fresh numerical features and weights are used. This is a new
stacking method. Its complete independent replay verifies conditional
meta-crossfit0.8588517253, not fully nested end-to-end CV. Its recovered
scores0.8603967792/0.8613960266 worsen both groups against the primary;
this negative result is retained without a parameter revision.
[Frozen comparison declaration](ALL_COMPLETE13_CALIBRATION_PLAN.md) ·
[Context-weighted stacking declaration](OWN13_FEATURE_WEIGHTED_STACKING_PLAN.md).

All48 returned native snapshot pairs from the original DNN24/attention32/
RBM200e producers now pass independent full known-row projection and paired
alias replay. There are45 distinct pairs,36 beyond the earlier nine compact
returns; no score selected a snapshot. The guarded larger FWLS study now
admits all413 declared entries: original352 scalars, all45 native pairs,
all three completed TX60o pairs and13 nonlinear components. It separately
replays both full input streams and the complete exact-pair alias partition.
There are404 distinct signals and5657 interaction/bias coefficients. It is
finished all five held-user fits with the published lambda30. Full
independent system/model/prediction replay passes, but conditional meta-CV
0.8602208791 and recovered0.8613295003/0.8623423168 miss the primary.
This negative result is retained; no coefficients or parameters changed
after audit. The four-GiB worker and24GiB research guards held throughout;
no large interaction cache or Gram files were written.
[Complete negative result](OWN404_FWLS_RESULT.md).
[All native snapshots](REMOTE15H_ALL_NATIVE_SNAPSHOTS_PLAN.md) ·
[Full own-bank feature-weighted stacking](OWN_FULL_FEATURE_WEIGHTED_STACKING_PLAN.md).

The matched paired B391 follow-up independently admits every original
330 parent bit,22 temporal inputs and all39 new distinct native/TX pairs
on both query populations. Complete467/617 operators, all group statistics,
eligible ridge solves and held prediction bits pass. Known B352/B391 ridge
changes only0.8618576840 to0.8618508817. This tiny diagnostic gain is not a
neural or qualifying claim. The reviewed H128/rank32 three-seed/fivefold
original40pat4 neural study is now independently complete, with a negligible
new fixed-half development best. The predeclared H512 control is independently complete and worse:
NN0.859315326191788 / fixed-half0.8590481464872496. All15 models and
full preprocessing/prediction/canonical checks pass; retain H128 using
known results only. [H512 negative result](OWN391_H512_CONTROL_RESULT.md).
[Matched input result](OWN391_PAIRED_DATA_RESULT.md).

The full temporal100r trajectory now supplies22 independently verified
snapshot pairs after one exact complete-pair alias. Adding all22 to own330
improves the same-row ridge0.8619710828 to0.8618611087, with all five folds
improving. Every original/new operator, group statistic, eligible solve and
stored prediction independently matches. That ridge diagnostic alone made
no neural or qualifying claim; the later full579-input neural and qualifying
results are independently verified above.
[Complete ridge result](OWN352_TEMPORAL_RIDGE_RESULT.md).

The full NN-pool audit is complete. All five new three-seed refits passed
independent serialized-pipeline replays. All three candidates were frozen
before outcomes, then independently reconstructed; the known-only
seven-family fit passed all127 simplex supports. Primary own7 scores
**0.8605628330 /0.8615658884** essentially tie with a slight group tradeoff.
The equal pool and fixed25%tree/75%pool worsen both groups. No meaningful qualifying gain at that audit; the later own330 result above
now advances the benchmark. All negative records remain. Apparent known-fit0.8590445175 is not nested CV.
[Complete audit](NNPOOL6_QUALIFYING_RESULT.md).

The complete own330 neural study passed all five folds/three seeds:
406ridge/556features/557NN inputs, with every retained/new bit admitted.
Its fixed tree154 half **0.8589474951** is the preceding development best;
the full qualifying result is now separately verified above. Own176's fixed half0.8591336855 had
narrowly missed the previous0.8591267885 best. Both full studies retain all
original inputs, architecture and epoch budgets. Both original CF96
training phases now complete95/95, with raw export and the original finalizer
complete. All196 original prediction files independently pass admission:
98 pairs reduce to96 exact distinct paired inputs, without fitting or
qualifying-label reads. [Complete CF96 bank](CF96_COMPLETE_BANK_RESULT.md).
The remote original22/22 TX50n compact pair also passes local independent
admission; neither addition has yet been incorporated into a new score.
The separate complete-CF delta queue will compare every96 distinct CF pair
and TX50n against all391 existing scalar inputs on both queries, retain
every exact novel pair, and run the same fixed-lambda1000 known ridge
diagnostic with full independent replay. It entered source admission, then
the compressor-above5GiB guard suspended only its own actual worker in RAM.
No fit began. After original1600 completed, recovery continued the same
intake, but its guard stopped it again. Its original declaration remains;
complete novelty is unfinished. After H128 proof, a serialized continuation briefly resumed the intake,
but its unchanged guard stopped it again during remaining diagnostic memory
pressure. The same worker is preserved. Reviewed waiting owner30570 defers the same intake until full original1200
and own14 independent completion, then requires the original primary lane
and stable resource admission. No intake restart or new score is claimed.
Original failures, source expectations and original3/24/5GiB guards remain.
[Frozen delta plan](OWN_COMPLETE_CF_DELTA_PLAN.md) ·
[Deferred exact continuation](OWN_CF_NATIVE_DEFERRED.md).
Original1600 source-pinned build/admission and full native execution are
complete after a verified completion-boundary handoff
from the restored childless legacy scheduler; no native fit was interrupted.
The childless native supervisor later exited through cleanup for full
own330 production; its successor subsequently acquired both lanes for the
exact1600 recipe and released them normally after independent completion.
No native model was interrupted.
[Full neural execution](OWN330_NEURAL_EXECUTION.md).

Original1600 now passes independent verification of BOTH original2-epoch
phases and all six query vectors/native log. Rehashing all outputs finds
two distinct paired snapshots: terminal exactly aliases epoch2 on both
queries. Complete canonical single-model known ProbeX error0.9349703049
does not advance the benchmark; it is not blend CV. The qualifying-phase
native diagnostics use zero sentinels and are not recovered qualifying
scores. Matched391 H128 is now independently complete after exact-process
continuation under its original guards. The separate H512 control is fully verified and worse; other scientific
groups remain paused while original1200 runs.
[Complete original1600 result](ORIGINAL1600_RESULT.md).

The paired391 H128 qualifying bag is independently complete. Every original
five-fold/three-seed model slot, scaler, ridge and query operator replays,
including all normalization, seed, clipped-fold and final bag bits. No
model was refitted and no qualifying labels were read by that study.
[Frozen inference-only plan](OWN391_FOLD_BAG_PLAN.md) ·
[Full independent bag replay](experiments/upstream-own391-fold-bag/independent-verification.json).

The new own14 comparison adds that bag to all13 previous own families.
Its original known-only simplex and ridge10 coefficients are complete;
all three known and three qualifying candidate vectors are frozen.
The independent all16383-support/15-equation/six-vector preaudit completed
in its original process21221 without restart. The original frozen audit26049
subsequently hit the unchanged compressor guard and remains preserved in RAM.
Reviewed waiting owner35292 requires full original1200 both23-epoch phases,
all58 numeric outputs plus its log and independent completion proof before
continuing that SAME audit. No recovered audit result or canonical score
exists yet. Inputs, coefficients and mathematics are unchanged; original
resource limits remain, with a stricter20GiB total ceiling. Apparent known
fitting error is not nested CV.
[Original complete comparison plan](OWN14_MATCHED391_CALIBRATION_PLAN.md) ·
[Same-audit continuation](OWN14_POST_NATIVE_AUDIT.md).

A separate full-budget CPU tree follow-up now has reviewed code, bounded
binary feature transport, four transport tests and explicit Torch/LightGBM
runtime isolation. It retains all617 features, original31-leaf/min1000/
max2000pat75/three-seed/five-user-fold recipe, full held-user exclusions
and a fixed own352 NN partner whose label/user/group hashes exactly match.
Its first declaration was suspended by the unchanged compressor guard while
hashing a large inherited legacy cache, before any feature matrix or tree
fit. The actual process is preserved; the original native job continued during that earlier guard event.
Private read-cache-control tests pass byte/metadata/prior-policy restoration,
but a controlled33-inode real-input trial still exceeded the compressor
limit. All prior settings were restored and the same declaration is again
preserved in RAM, with no fit or numerical recipe change. After both1600
and primary391H128 completed independently, the reviewed deferred controller
continued that same declaration, which completed. The original full CPU fit
then started, and its guard preserved the actual fit group and binary-input
producer in RAM during overlap. No complete tree fold exists yet. Further
continuation will serialize against the other large jobs; all original
model budgets and resource limits remain unchanged.
[CPU tree plan](OWN391_CPU_TREE_PLAN.md) ·
[Actual guard and recovery evidence](OWN391_CPU_TREE_RECOVERY.md).

The separate original1200 preflight now accounts for every58 numeric output
of the unchanged1,200-factor/100-neighbor/23+23-epoch preset:48 ordinary/epoch
query arrays,8 component-query arrays and2 complete training arrays. Their
payload alone is1.156GiB; output storage must be reserved above the native
18.75GiB disk floor before building or launching. Authorized retirement of
190 downloaded public prediction inode groups recovered0.799GiB; dataset
arrays, current own inputs/models/checkpoints and earlier exact archives
were preserved. After independently completed1600 and paired391H128, separate exact
original1200 source/data preparation completed. Isolated compilation began
but its guard preserved the actual build group94817 in RAM during overlap.
The same original build subsequently completed under exact recovery, and
independent source/data/binary admission passed. Native16010 is now running
the unchanged23+23 phases; its first four trainx epochs completed at0.95079/0.92846/0.91667/0.90925. This
early single-model diagnostic is not a qualifying or blend result. Full
training and separate independent58-output verification remain.
Unused completed1600 build intermediates were losslessly archived and fully
verified before removal, recovering209MiB without changing its executable
or numerical outputs. Admitted1200 unused build intermediates were also
fully archived before removal, recovering another~210MiB without changing
the running binary or scientific inputs. [Archive proof](BUILD_CACHE_ARCHIVE_RESULT.md). Its disk guards reserve the
full1.25GiB output allowance through preparation/build and preserve the
original18.75GiB continuous floor during native writes.
[Original1200 requirements](ORIGINAL1200_EXECUTION_PLAN.md) ·
[Frozen execution queue](experiments/upstream-original1200-queue/plan.json).

All149 unused completed local native pairs now passed a complete own330
ridge evaluation and independent replay. Same canonical rows/folds:
**0.8622279137 to0.8619710828**, with all five folds improving. The full
406-column design retains every own181 input bit; every group statistic,
eligible solve and actual stored prediction matches. Bounded file reads
kept the diagnostic within its1.5GiB worker guard while temporal training
continued. That initial ridge admission made no neural/global promotion;
the complete neural result above is now separately verified.
[Complete result](OWN330_NATIVE149_RIDGE_RESULT.md).

The separate own181 diagnostic adds all fifteen recent own pairs, including
new verified BKNBRX100w and MFRBMX60h returns. Full input/statistic/ridge
replay gives **0.8622279137**, versus own176 **0.8622734678** and original166
**0.8624919426**. The last five add only0.0000455541 ridge improvement.
No own181 neural fit or qualifying promotion. [Result](OWN181_RECENT15_RIDGE_RESULT.md).
Missing original1,200/1,600-factor coverage is now audited; memory lower
bounds are not safe launch estimates. [Next capacity work](HIGH_CAPACITY_ORIGINAL_PLAN.md).
The isolated1600-factor preparation now binds439 source/data files and
proves both full training/query populations disjoint. Its reviewed childless supervisor was retired through cleanup; the
successor admitted its full22GiB reservation and acquired both lanes.
An isolated exact executable now passes its source/build/data admission;
the earlier shared-binary receipt mismatch was preserved. The original
1600-factor/2epoch-perphase native run is independently complete; its raw
known ProbeX diagnostic is not a new blend or qualifying score.
[Guarded execution](HIGH_CAPACITY_ORIGINAL_EXECUTION.md).

New own176 input/ridge admission is complete. All ten recent own scalar
predictors add nonduplicate canonical inputs to the original own166/all90
bank. Its ridge diagnostic improves **0.8624919426 to0.8622734678**; every
original/new input bit, full group statistic and actual stored ridge
prediction independently matches. This is not a neural/global promotion.
The full five-fold,three-seed403-input H128/rank32 study has completed,
including all15 saved models and full original preprocessing replays.
NN0.8592799296; fixed tree154 half0.8591336855. No global promotion or
qualifying refit. [Complete neural result](OWN176_RECENT10_NEURAL_RESULT.md).
[Complete diagnostic and next study](OWN176_RECENT10_RIDGE_RESULT.md).

The own167/context218 study is complete and negative: NN **0.8598066347**,
fixed half with tree154 **0.8594786621**. All five folds,15 actual saved
models, both eligible scalers and ridge statistics were independently
reconstructed. No qualifying refit follows this result. The original
supervisor's post-fit zombie-process accounting failure is preserved;
a separate replay supervisor verified the completed fits without retraining.

The matched comparison using the original full split3 holdout is complete.
Its1,760,494 legitimate known rows include632,330 additional ratings, all
verified against original known data and excluded from native training.
Both arms evaluate the same1,128,164 canonical rows and exclude every held
user's additional labels. Control fits only canonical training rows;
primary fits all eligible holdout rows. Both retain all73 distinct own
native prediction columns,130 original voting features and the same full
five-fold/three-seed H128/rank32 recipe. Five control and two expanded folds
completed before a resource guard stopped its neural worker. Recovery fitted
only the three missing expanded folds; all ten pipelines and30 actual saved
models passed independent replay. Expanded NN0.8722568222 improves matched
control0.8727302875; fixed tree halves0.8622259940 vs0.8624405419. Both miss
the global best, so no qualifying refit. [Result](SPLIT3_MATCHED_RESULT.md).
[Matched study](SPLIT3_FULL_HOLDOUT_PLAN.md).
The guarded recovery completed after temporal22/22 verification, preserving
all seven original pipelines and independently replaying all ten in full.
[Recovery plan](SPLIT3_MATCHED_RESUME_PLAN.md).

Both original voting-generation phases completed. All130 full-holdout
feature payloads are preserved in a full-byte-verified lossless archive.
The idle RAM owner released after completed study replay, freeing0.878GiB
so own330 could retain its full16GiB reservation. [Exact preservation](SPLIT3_VOTING_ARCHIVE_RESULT.md).
129/130 qualifying feature
payloads match our earlier own outputs; ordinal-SVD differs. That exact
parity failure is preserved. Development uses the new complete holdout
features; any production must capture all130 corresponding qualifying
features rather than substitute the earlier table.
[Feature admission](SPLIT3_FEATURE_ADMISSION.md).

Native CFNADE96 resumed from independently verified86/81 checkpoints;
its same original X and qualifying processes subsequently completed both
original95/95 budgets and independent full-output verification. Temporal100r
completed22/22; all49 outputs and the original source/data/query projection
passed independent verification. Its measured
15.9GiB footprint exceeded the8GiB admission estimate and caused the matched
study interruption and CF qualifying disk-guard stop. Separate identity-checked
memory and monitor-loss guards preserve native state and serialize recovery.
The first serial guard suspended the same TX process when the own181 CPU
diagnostic overlapped it. That diagnostic has completed. A separate v2
monitor continued the same original process with unchanged24GiB total,
128MiB swap-growth and18.75GiB disk limits; its own monitor-loss watchdog
and signal handling preserve the original native state on interruption.
After fitting finished, the v2 monitor reached its disk floor just before
verification completed. A separately verified handoff rechecked the complete
native proof, all168 CF checkpoint arrays and recovered disk reserve before
resuming CF and releasing the waiting neural recovery. No temporal restart.
Free disk recovered to about20.4GiB after retiring270 additional exact
downloaded public latent caches under the existing cleanup authorization.
[Resource recovery](TX100_MEMORY_RECOVERY.md).

Original split3 RBM200 completed25/25 and passed
separate source/binary/data/output and complete known-projection checks.
The item-oriented autoencoder completed50/50;
all64 final user-context columns and every query gather are independently
verified. Remote DNN24 completed8/8 and attention32 completed12/12; both
returned two verified own pairs. Remote exponential RBM200 is now complete
25/25, with five own pairs imported and locally verified against the frozen
plan/helper, both full epoch traces, canonical known map/training disjointness,
complete finite vectors and exact original native qualifying NPY bytes.
All ten recent own pairs, including local RBM200, passed separate complete
lineage/finite/exact-duplicate checks. Raw known scores are diagnostics;
no new ensemble or qualifying score is claimed. The remote compact returns
do not contain the full1,760,494-row Probe3 vectors; keep them out of the
expanded-population bank until the original full vectors return. Eight
claimed remote extension producers retain the original training deadline.

[Own qualifying result](OWN330_QUALIFYING_RESULT.md) ·
[Exact state/storage recovery](CF96_DISK_RECOVERY.md) ·
[Current development verification](experiments/upstream-own352-bounded-replay/independent-canonical-verification.json).

The remote TX60o extension completed both25-epoch phases; all three compact
own prediction pairs passed original recipe/data/map/native-header checks.
The full native archive of the first three remote producers is also now
locally admitted: both compressed-part hashes and all125 plaintext members
verify, all96 full query vectors are retained, and allnine earlier compact
known/qualifying pairs replay exactly. Complete1,760,494-row native holdout
training disjointness is separately checked. No new ensemble score follows
from transport. [Native admission](experiments/upstream-remote15h-native-stream/independent-verification.json).

The next full neural iteration is now predeclared: retain every paired B391
input/context bit and append all genuinely novel complete CF95/TX50n and
original1600/1200 snapshot/component pairs. No epoch/error-based selection;
complete both-query clipped aliases only. Full original H128/rank32, all
five folds/three seeds/max40pat4, full numerical replay and qualifying bag,
then a known-only15-family simplex with all32767 support checks before
recovered outcomes. Implementation/review remain; no new result claimed.
[Full trajectory model plan](OWN_COMPLETE_TRAJECTORY_NEURAL_PLAN.md).

The full trajectory input/ridge implementation and separate verifier now
pass review and eight operator/resource checks. Waiting owner50842 requires
full original1200 and complete CF95/TX50n proofs before acquiring primary
resources. It keeps all519 original/candidate scalar streams before complete
paired alias checks and verifies all expanded operators, original-width391
control, full group systems/solves and every held prediction. No neural or
qualifying score is claimed. [Guarded data execution](OWN_COMPLETE_TRAJECTORY_DATA_EXECUTION.md).

The weekend handoff is pushed at8afae5a: full original factorizedRBM400,
temporal300 and bipolarRBM800, serially on the confirmed M3Max/96GiB until
Monday08:00Pacific. Every epoch/native output is required; transfer after
each producer. Remote execution checkpoint6710d4d now records verified r2
preparation on the confirmed hardware with190GiB free; the first400pf
producer waits original compressor admission. No native launch/completion
or new result is established by that checkpoint. The earlier BLAS linker
failure remains preserved. [Pinned startup observation](experiments/upstream-remote-weekend/coordinator-startup-observation.json).
[Weekend instructions](REMOTE_WEEKEND_HANDOFF.md).

The complete-trajectory neural and original-fold qualifying-bag stages now
pass separate prelaunch review. Waiting owners56323/58980 form the next
dependency chain. H128/rank32 retains every novel column/all225 contexts,
all5folds/3seeds/max40pat4/batch8192 and original AdamW. The width-generalized
scaler matches original618 and independent746-column fixture statistics;
the zero-addition model initializes bit-identically to established H128.
The bag preserves zero-budget slots and original float32 seed division/
float64 accumulation, with a separate full query/model/stream replay.
Fourteen combined invariant checks pass; no full-fit memory/score or new
qualifying result is claimed. The separately declared15-family calibration and recovered-audit queue
is now implemented and reviewed below.
[Neural execution](OWN_COMPLETE_TRAJECTORY_NEURAL_EXECUTION.md) ·
[Qualifying bag](OWN_COMPLETE_TRAJECTORY_FOLD_BAG_PLAN.md).

Final waiting owner68008 now implements known-only15-family calibration,
all six-vector freezing, independent all32767-support/16-equation/arithmetic
preaudit, three fixed recovered audits and separate canonical scoring. It
retains all14 existing families plus the new verified full-trajectory NN.
Equivalent simplex optima are validated by objective/KKT rather than
requiring identical coefficients for duplicate predictors. Six new checks
pass, including alias-provenance and one-bit vector corruption rejection.
No new known fit or recovered score has run; the target remains unmet.
[Complete final assembly plan](OWN15_COMPLETE_TRAJECTORY_CALIBRATION_PLAN.md).

The local weekend return intake is now implemented, independently reviewed,
and queued separately for all three producers. Owners87173/87178/87185
wait the current own15 canonical completion, then require complete pinned
remote transfers before acquiring primary resources. Every compressed and
decoded byte, all native training/factor shapes and values, full local
source/data/split identities, complete training disjointness and every
compact/native query bit must pass. Eleven receiving-specific checks and
68 combined checks pass. Large side arrays remain in verified original
remote archives; all complete query and compact pairs are retained locally.
No new remote output or score has been admitted.
[Complete return admission](REMOTE_WEEKEND_RETURN_PLAN.md).

Five obsolete public-bank statistics caches have now been retired under
the existing public-result cleanup authorization, recovering2.508GiB. All11
current scientific proof hashes and12 preserved metadata/source hashes pass
a separate read-only replay. Models, predictions, checkpoints, data and
processes remain unchanged. Free space is about22.91GiB for pending native
outputs and weekend returns.
[Bounded public-cache cleanup](PUBLIC_STAT_CACHE_RETIREMENT_RESULT.md).

A separate full weekend learning iteration is now implemented, reviewed and
queued behind all complete original producers. It examines all586 original
and candidate scalar streams (391+128local+67remote) before exact paired
alias checks, with up to813 neural inputs and every225 original context
position retained. Waiting owners6054/6062/6067/6075 cover complete data/ridge
replay, original H128 neural/actual-model/canonical replay, full qualifying
bag/replay and16-family known-only calibration/freeze/audit/canonical checks.
All65535 simplex supports and17 ridge equations are required. Both the old391
and preceding local trajectory known references are proof-bound, including
all original saved model expectations; no recovered outcome selects inputs.
Ten new checks/82 combined checks pass. No full new fit, remote return or
new score has been measured. All three weekend producers must complete;
missing/partial work cannot silently shrink this declaration.
[Full weekend iteration](WEEKEND_COMPLETE_TRAJECTORY_PLAN.md).

Remote hourly checkpointcf93bb4 at06:45UTC confirms original400pf native
training24357, started21:20Pacific, elapsed1:25:19/CPU80:57.55, under the SAME
run-job73979 and watchdog24356. Two of fifteen first-phase epochs are logged,
with standalone diagnostic0.92531. The original compressor admission passed;
the snapshot records unchanged swap376837242.88bytes and
55.24training hours remaining. Full launch/epoch/source/output verification,
export and transfer remain pending. This is a pinned checkpoint, not a
coordinator live remote-process poll; no completed producer or score claim.
[Earlier pinned training checkpoint](experiments/upstream-remote-weekend/coordinator-hourly-observation-20261121T064538.json).

Later execution commitee38c3e supersedes that training-only snapshot:
following the owner-recorded switch to parallel execution, all three remote
producers hit the unchanged12GiB compressor guard at08:14UTC. The receipts
report original native RAM preserved for24357(400pf),68411(800p) and68498(TX300r),
with compressor12.91/12.67/13.07GiB and unchanged recorded swap. These are
guard stops, not completed producers. Current stopped identities and recovery
must be checked on that host; the coordinator has no direct process-control
session there. No completed export, archive, upload or new score is verified.
The earlier progress JSON remained stale after the failures. Recovery must
preserve the same native processes, original baselines, recipes and guards,
with verified adoption ownership/watchdogs before any continuation.
[Pinned guard receipts](experiments/upstream-remote-weekend/coordinator-parallel-guard-observation.json) ·
[Remote same-process recovery instruction](REMOTE_WEEKEND_GUARD_RECOVERY_NOTE.md).

Execution checkpoint3d9a2a0 at09:13UTC separately confirms all three original
native PIDs in`Ts`(stopped) state. No recovery, complete producer or transfer
was established there. The coordinator's latest fetch still finds that same
checkpoint; it is a pinned observation, not a current remote process poll.
[Stopped native process checkpoint](experiments/upstream-remote-weekend/coordinator-hourly-observation-20261121T091331.json).

The preserved original own14 preaudit21221 completed successfully at01:09UTC:
all16383 simplex supports,15 ridge equations and all six complete vectors
independently match before outcomes. Its original source/coefficients and
RAM continued without restart. The subsequent audit26049 hit the unchanged
global compressor guard (about8.72GiB versus5GiB) and remains stopped in RAM.
No audit/canonical result or new qualifying score exists. Reviewed waiting
owner35292 will continue that SAME audit only after full original1200
23+23/all58numeric+log independent completion and original admission checks;
there is no repeated overlap attempt or baseline reset. Native16010 remains
active after completing all23 first-phase epochs and the final probex
diagnostic0.87775; its second fulltrain→qual phase has completed19/23 epochs.
Phase-two diagnostic errors use zero-filled qualifying targets and do not
measure accuracy. The full second23-epoch phase and independent all-output
verification remain required; no new ensemble or qualifying score is implied.
All89 relevant checks pass.
[Completed preaudit](experiments/upstream-own14-matched391-calibration/independent-preaudit-verification.json) ·
[Exact same-audit continuation](OWN14_POST_NATIVE_AUDIT.md).

[Live epoch13 process snapshot](experiments/upstream-original1200-exact-build-recovery/coordinator-epoch13-observation.json).

[Live epoch18 process snapshot](experiments/upstream-original1200-exact-build-recovery/coordinator-epoch18-observation.json).

[Completed first23-epoch phase snapshot](experiments/upstream-original1200-exact-build-recovery/coordinator-first-phase23-observation.json).

[Same native process at second-phase epoch3](experiments/upstream-original1200-exact-build-recovery/coordinator-second-phase3-observation.json).

[Same native process at second-phase epoch10](experiments/upstream-original1200-exact-build-recovery/coordinator-second-phase10-observation.json).

Disk subsequently declined to about20.03GiB. Ten inactive legacy build caches
have now been preserved in lossless archives, recovering427.6MiB net measured
space and restoring about20.46GiB free above the unchanged18.75GiB floor.
All10,085 retained payloads, gzip integrity, original absence and54 protected
file hashes pass a separate replay. The original native PID/start/command
remains live; scientific sources, executables, models, data, predictions and
checkpoints were preserved.
[Verified legacy-cache archives and restore instructions](BUILD_CACHE_ARCHIVE_RESULT.md).

[Same native process at second-phase epoch13](experiments/upstream-original1200-exact-build-recovery/coordinator-second-phase13-observation.json).

The separately named local factored bipolar RBM now has a reviewed full-budget
execution controller and worker, committed and pushed at49191de. Its real owner
31460 and watchdog31465 are alive, waiting for complete original1200 and own15
verification before any numerical dispatch. Both15-epoch fits, saved-model
replays and paired assembly pass on small synthetic CPU fixtures; all64 relevant
checks pass. They include real-process inherited-lane retention after journal
failure, independent physical/RSS admission, preserved worker measurement
failures and rejection of altered authenticated predictions.

The recipe retains H400/rank50/500 users/seed128, every original row and all15+15
epochs. Full original63 payload/header/semantic admission, actual fresh-MPS
full-width stress qualification and original resource guards remain mandatory.
No production arrays have been scanned by this new implementation, no actual
MPS work or production fit has occurred, and no new score is claimed. Resource
stops preserve the same process, RNG/model/write stack, owner/watchdog, lanes and
original baseline; they never restart or reduce the numerical recipe.
[Owned execution and resource protocol](OWN_FACTORED_RBM_GPU_EXECUTION.md) ·
[Live waiting-owner observation](experiments/upstream-factored-rbm-execution/coordinator-waiting-observation.json) ·
[64 CPU checks and independent review](experiments/upstream-factored-rbm-kernel/execution-verification.json) ·
[Full original-data admission](OWN_FACTORED_RBM_GPU_ADMISSION.md) ·
[Fresh actual-MPS qualification](OWN_FACTORED_RBM_GPU_PREFLIGHT.md) ·
[Complete saved predictor](OWN_FACTORED_RBM_GPU_TERMINAL.md).

[Same native process at second-phase epoch16](experiments/upstream-original1200-exact-build-recovery/coordinator-second-phase16-observation.json).

The local RBM also has a complete reviewed path into score verification, pushed
at aa528ca. Its separate sixteen-family calibration queue is live with owner
45713 and a watchdog, waiting for complete own15 and the full local RBM producer.
All original15 families remain; the new full RBM is the sixteenth. Its declared
primary uses known-only nonnegative sum-one weights. The released ridge10 control
retains all17 equations, and the preceding own15 primary is copied exactly as a
control. All six vectors must be frozen and independently verified before a new
recovered audit; outcomes never select the primary. The small CPU integration
passes all65,535 simplex supports, full ridge equations, exact vector arithmetic,
label-access gates and separate synthetic audits. Together with the RBM checks,
all73 relevant CPU tests pass. No production calibration, new recovered audit or
score has occurred. Original source/model/data hashes and unchanged resource
limits remain mandatory through canonical verification.
[Complete known-only local RBM blend plan](OWN_LOCAL_RBM_BLEND_PLAN.md) ·
[73 synthetic CPU checks and review](experiments/upstream-local-rbm-blend-protocol/verification.json) ·
[Live calibration waiting-owner observation](experiments/upstream-local-rbm-blend-queue/coordinator-waiting-observation.json).

The weekend16 calibration freeze had a copied15-family validation despite
its declared full16-family known fit. A small fixture reproduces the rejection;
the separately reviewed correction now retains all16 families and the original
coefficients, arithmetic, fixed half control and independent verifier. All79
relevant CPU checks pass, including exact six-vector arithmetic, original known
hash preservation, rejection of subsets/invalid populations and a real-process
idle-owner handoff test. The original frozen modules remain unchanged.

The original idle coordinator6075 and watchdog6077 were retired only after
proving no scientific namespace, worker or swap baseline existed, then freezing
and rechecking the exact control owner. No scientific process was signaled or
refitted. Its reviewed successor59231 and a watchdog are alive, waiting for the
same original own15 preaudit and complete weekend bag proof. All original67
remote snapshots and full586-stream weekend neural requirements remain intact.
No new production calibration or score exists. The original native1200 remains
active at23/23 first-phase and19/23 second-phase epochs.
[Full16 freeze correction and safe idle handoff](WEEKEND_OWN16_FREEZE_REPAIR.md) ·
[79 CPU checks and independent review](experiments/upstream-weekend-own16-freeze-repair/protocol-verification.json) ·
[Actual idle-only handoff](experiments/upstream-weekend-own16-freeze-repair/idle-handoff.json) ·
[Corrected waiting-owner observation](experiments/upstream-weekend-own16-trajectory-queue-v2/coordinator-waiting-observation.json) ·
[Original native second-phase epoch17](experiments/upstream-original1200-exact-build-recovery/coordinator-second-phase17-observation.json).

At the epoch18 checkpoint, the same native1200 process had18/23 second-phase
epochs and18 full-size qualifying epoch-vector files. Its five recent intervals were about25.8–25.9
minutes each. Their median projects the epoch23 vector around09:33 Pacific;
that estimate excludes final exports, independent verification, calibration and
new score reporting. It is not an ETA for the10% goal. The memory, swap and
disk samples at that checkpoint were within unchanged next-audit limits; its owner
still waits for full native completion. About20.35GiB is free. A readonly storage
check retained dense partial numerical state and made no cleanup changes.
[Native epoch18, measured cadence and handoff readiness](experiments/upstream-original1200-exact-build-recovery/coordinator-second-phase18-observation.json).

The same original native has now reached19/23 second-phase epochs, with all19
full-size qualifying epoch-vector files observed. The original23/23 first
phase remains complete. No recovered/ensemble score or producer completion is
claimed before the remaining epochs and independent all-output verification.
[Original native epoch19 checkpoint](experiments/upstream-original1200-exact-build-recovery/coordinator-second-phase19-observation.json).

A reviewed remote same-process adoption package is now pushed on
`research/remote-weekend-recovery` at39ac3cc. Its13 targeted CPU checks pass,
including24 immutable pre-recovery control expectations, baseline rewrite
rejection, real test-owned pause/continuation and inherited-lock checks,
explicit post-stop requests, full synthetic15+15/all-native-output validation,
and truthful unknown orphan exit status. A continuation intent cannot count as
resumed work; separate actual running evidence gates the400-first ordering.
The helper requires the original96GiB host, exact original stopped processes,
terminal old owners/watchdogs and unchanged original source/data/binary,
environment, first baseline, capacities and deadline. It never launches a model.
The original frozen training/export/return helpers remain unchanged.

The remote agent must fetch that branch into a separate worktree and run the
host-side validation/adopter. This coordinator has no direct session on that
host; no remote adoption or continuation is established by these CPU tests.
The latest pinned remote execution branch still ends at3d9a2a0 with stopped
process observations. The local native and guarded queues continue separately.
[Complete remote recovery instruction](REMOTE_WEEKEND_SAME_PROCESS_ADOPTION.md) ·
[13 CPU protocol checks and review](experiments/upstream-remote-weekend-adoption-protocol/verification.json).

The same original native reached21/23 second-phase epochs at08:41 Pacific on
November21; its23/23 first phase remains complete. A header/length inventory
finds50 of the required58 arrays present with the planned shapes and float32
format. The eight remaining outputs are epochs22–23 and the six final exports.
This structural check is not numerical admission. Recent saved-vector cadence
still projects the epoch23 vector around09:33 Pacific, excluding final exports,
independent verification and calibration. The native, supervisor and watchdog
remain live, with no recorded guard failure; all scores remain unchanged.
[Native epoch21 and complete structural inventory](experiments/upstream-original1200-exact-build-recovery/coordinator-second-phase21-observation.json).

The original1200 producer has completed both23-epoch phases and exited with
return code0. Independent verification passed the original source/data/build
admission, all58 full-size finite numeric outputs and the persisted native log
(59 output/log hashes). Its completion receipt matches the independent proof.
The final epoch vector was saved at09:32:35 Pacific; the final full-training
export then completed before verification. At the recorded handoff, the same
original own14 audit process26049 had resumed under its existing owner35292,
with the original baseline and unchanged guards. No new recovered/canonical
score is established by this checkpoint; scores and plots remain unchanged.
[Independent original1200 completion](experiments/upstream-original1200/independent-verification.json) ·
[Completion receipt](experiments/upstream-original1200-exact-build-recovery/completion.json) ·
[Actual same-audit handoff observation](experiments/upstream-original1200-exact-build-recovery/coordinator-completion-handoff-observation.json).

Own14 now passes the separate full canonical verification at
0.8603797340342072/0.8613681513408222 after continuation of the same original
frozen audit. Its14-family primary improves both complete13 groups only
slightly; the10% target remains unmet. The original1200 producer is fully
verified complete. The same CF intake then finished all488 pairs, finding66
new signals and31 exact aliases; independent novelty replay is running before
the original fixed ridge fit. Full trajectory and GPU queues remain guarded.
[Own14 complete result](OWN14_QUALIFYING_RESULT.md) ·
[Canonical own14 proof](experiments/upstream-own14-matched391-calibration/independent-canonical-verification.json).

The newly fetched remote checkpoint444fa33 records a reported owner override
of remote memory/RSS/compressor/swap guards and fresh r8/r9/r10 preparations.
A separate executable policy check confirms an exact controller-only plan
change and exact capacity/compressor patch; every numerical recipe, dataset
expectation, thread/seed/epoch budget and all67 required pairs remain original.
Deadline, watchdog and disk checks remain in the source. The old stopped
attempts were terminated according to that checkpoint. No new launch receipt
or completed numerical return appears in the pinned commit, so no current
remote process or score is claimed. All coordinator guards and frozen intake
expectations remain unchanged. A separately declared return adapter is needed
for this new policy variant.
[Remote policy variant and full-return requirements](REMOTE_WEEKEND_OWNER_OVERRIDE_RETURN_PLAN.md).

The completed CF novelty inventory now passes the separate independent replay
of all488 pairs:66 new signals and31 exact aliases. The original deferred
controller30570 has advanced to the declared fixed-lambda1000 ridge fit,
with process23654 live. No fit completion or new score is yet established.
[Independent novelty proof](experiments/upstream-complete-cf-delta/input-verification.json) ·
[Actual fixed-fit handoff](experiments/upstream-complete-cf-delta-native-deferred/coordinator-novelty-to-fit-observation.json).

The expanded CF95/TX50n fixed-lambda study is independently complete. All
operator/statistic/weight/prediction bits and canonical known scores verify.
Its known ridge0.8617705148 improves its matched parent391 ridge0.8618508817
by0.0000803669; this is not the current best development score and no new
qualifying predictions were generated. Full519-candidate trajectory admission
is now live in the original queue before its complete ridge/neural stages.

Three additional obsolete public-dependent statistic archives were retired
under the existing public-bank cleanup authorization after exact full SHA,
536/387/122 manifest-input matches, closed/single-link/untracked checks and
a durable pre-deletion journal. Independent scope/absence/provenance and ten
scientific-proof/twelve metadata-hash replays pass. Measured free-space gain
is0.93722GiB; about20.02GiB is free. Every independent model, checkpoint,
prediction, dataset and own working state remains intact. These three public
cache contents were intentionally discarded and are no longer restorable
through the older public-cache archive helper.
[Independent public-cache retirement](experiments/upstream-retire-remaining-public-stats-1/independent-verification.json).

The separate remote-override context library passes six metadata fixture
checks with the original scientific-validator bytecode and default module
globals unchanged. Its private plan configuration preserves all67 pairs and
adds fresh-preparation/binary/deadline/lineage checks. This is a library
prototype only: production declaration, supervision and actual numerical
return admission still need completion. No new score is claimed.
[Library prototype checks](experiments/upstream-remote-weekend-override-context-prototype/verification.json).
