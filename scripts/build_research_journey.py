#!/usr/bin/env python3
"""Build the self-contained research journey from documented milestones and run events."""
import base64
import datetime as dt
import hashlib
import json
from html import escape
import pathlib
import re
from zoneinfo import ZoneInfo


ROOT = pathlib.Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"

# These are editorial milestones, not a claim that every experiment was a new best.
# Calendar dates before timestamped run events are documentation upper bounds.
MILESTONES = [
    ('FRESH176 recent10 ridge diagnostic', '0.8622734677900217', '2026-11-20T10:24:09.517265+00:00', 'Fresh training', 'fresh_cv', 'codex', 'screen', 'Ten own predictors add verified linear information; neural study queued', 'Same canonical1128164 rows and original user folds. Ridge improves original166 control0.8624919426 to0.8622734678 after all ten independently trained published pairs are added. Complete252/402-column operators, every full group statistic and actual ridge prediction bit independently replay; complete canonical fingerprints show no exact duplicate additions to old166. This is a ridge diagnostic, not a neural/global promotion or qualifying gain. Full403-input H128/r32 fivefold/three-seed study waits complete original temporal and matched recovery. Development best remains0.8591267885.', 'experiments/upstream-own176-recent10-data/independent-verification.json', 'complete original/new input bits, full eligible-user ridge replay and canonical compensated score; no neural or qualifying fit'),
    ('FRESH167 context218 tree half', '0.8594786621236125', '2026-11-20T08:06:33.769245+00:00', 'Fresh training', 'fresh_cv', 'codex', 'screen', 'Larger own latent context: complete negative development study', 'All five original user folds and15 actual saved NN models independently replayed. Both complete eligible original scalers and ridge statistics independently recomputed. NN0.8598066347; fixed tree154 half0.8594786621, above existing0.8591267885. Original post-fit supervisor accounting failure preserved; separate recovery verified actual completed fits without retraining. No qualifying fit or outcomes. Next matched study tests full original split3 holdout on the same canonical evaluation users.', 'experiments/upstream-own167-context218-replay-recovery/verification.json', 'complete original fits, independent preprocessing statistics and every actual saved-model held prediction; negative result retained'),
    ('NNPOOL6 own7 qualifying', '0.8615658883962303', '2026-11-20T07:29:36.418872+00:00', 'Fresh training', 'fresh_qualifying', 'codex', 'audit', 'Own7 full audit: essentially tied, six-family benchmark retained', 'Five new full three-seed neural refits plus reused baseNN154; all complete serialized ridge/scaler/model query replays and original-bank gates pass. All three candidates frozen before outcomes; independent127-support known-only calibration and exact arithmetic pass. Primary groups0.8605628330 /0.8615658884 slightly trade off versus established0.8605620602 /0.8615662023. No meaningful qualifying gain, target still missed. Apparent known-fit0.8590445175 is not nested CV; CV best stays0.8591267885.', 'experiments/upstream-nn-pool6-assembly/independent-recovered-audit.json', 'predeclared primary; own full models, known-only weights, frozen vector and complete independent adaptive recovered audit'),
    ('NNPOOL6 tree25 qualifying', '0.8616658311435437', '2026-11-20T07:29:36.418872+00:00', 'Fresh training', 'fresh_qualifying', 'codex', 'audit', 'Fixed tree25/pool75: development gain does not carry to qualifying', 'The independently verified development0.8591267885 fixed ratio is reproduced with actual full models and matching paired production inputs. Both full recovered groups0.8606639172 /0.8616658311 worsen versus the six-family benchmark. All inputs and saved pipelines verified, complete fixed float32 vector frozen and reconstructed before outcomes. Negative qualifying result retained, no coefficient changes.', 'experiments/upstream-nn-pool6-assembly/independent-recovered-audit.json', 'predeclared fixed control; actual serialized-pipeline replay and exact frozen arithmetic'),
    ('NNPOOL6 equal qualifying', '0.8617966292621777', '2026-11-20T07:29:36.418872+00:00', 'Fresh training', 'fresh_qualifying', 'codex', 'audit', 'Equal six-NN full audit: no qualifying promotion', 'Equal pool of all six declared own neural variants, five newly refitted and the actual base full NN reused. Full model/preprocessing replay and original-source gates pass. Fixed complete vector frozen before outcomes. Recovered groups0.8607645056 /0.8617966293 are worse than the established six-family benchmark. Matched known NN improvement remains a distinct development result.', 'experiments/upstream-nn-pool6-assembly/independent-recovered-audit.json', 'predeclared equal-pool control; no recovered-outcome fitting or model selection'),
    ('FRESH native NN pool6 tree31', '0.8591267885072646', '2026-11-20T05:47:47.086792+00:00', 'Fresh training', 'fresh_cv', 'codex', 'promoted', 'Fixed75% native NN pool /25% tree: independently improved development best', 'One separately declared fixed3:1 pool-to-tree follow-up after matched NN-family improvement; no weight grid/coefficient optimization or qualifying outcomes. All six original complete neural studies and existing tree154 retain35 actual source fold receipts,90 saved NN and15 saved tree models plus actual replays. Every weightedfloat32 bit and complete canonical known score independently agree.0.8591267885 improves prior0.8592012938. Adaptive known-CV result; full refits/streaming qualifying production remain pending and no new qualifying score is claimed.', 'experiments/upstream-native-nn-pool6-tree31/verification.json', 'all actual original models/replays, separately fixed weights and independent every-bit canonical full-known scoring'),
    ('FRESH109 CF96 prefix20/30 mean screen', '0.8592548691507627', '2026-11-20T05:31:43.008812+00:00', 'Fresh training', 'fresh_cv', 'codex', 'screen', 'Own original CF96 epoch20/30: complete negative neural study', 'Own full33/31 source package,84 state arrays per phase,original data/native arithmetic parity and both20/30 native query pairs reverified. Original95/95 source model remains incomplete. Full400-input H128/r32 neural study: all five nested user folds,15 actual saved models and both independent exact mean/canonical scores pass. NN0.8594077567, tree50 mean0.8592876737, tree154 mean0.8592548692; both means exceed current0.8592012938 best. No qualifying refit.', 'experiments/upstream-item154-tree-cf96-prefix2030-compact-mps/verification.json', 'own package source proof,full data/operator parity,actual saved models and separate exact mean scoring'),
    ('FRESH native NN pool6', '0.8592546775639587', '2026-11-20T05:36:26.559042+00:00', 'Fresh training', 'fresh_cv', 'codex', 'promoted', 'Six fixed native-neural variants improve matched NN; global CV best retained', 'Equal pool of all six completed native-append own107/item154 MPS NNs improves matched NN0.8593837349 to0.8592546776,with four of five folds lower. Original35 source fold receipts and90 actual saved NN models plus15 tree models/replays bound. Independent compensated six-way arithmetic,every float32 vector bit and canonical known scores pass. Fixed tree154 half mean0.8592096046 does not beat global0.8592012938. NN-family promotion only; full-refit/seven-family known-only calibration plan frozen before new qualifying outcomes. Initial unclipped score reports preserved; conventional float64 clipping correction changes no vector or decision.', 'experiments/upstream-native-nn-pool6/independent-canonical-verification.json', 'all original full model/replay evidence and independent complete-vector compensated canonical scoring'),

    ('FRESH115 split3 eight mean screen', '0.8592426521393607', '2026-11-20T05:18:11.108948+00:00', 'Fresh training', 'fresh_cv', 'codex', 'screen', 'All eight completed split3 additions: verified negative neural study', 'All eight unused own original DNN/NADE,attention64 andKNN3 predictions combined with full own107/item154. All five nested user folds,15 actual saved-model replays with original reference normalization and both independent exact means/canonical scores pass. NN0.8594358913; tree50 mean0.8592734660 and tree154 mean0.8592426521 both exceed current0.8592012938 best. Full1,128,164x406 input/scaler bit parity; H128/r32,all rows/folds/seeds/optimizer/stopping unchanged. Supervision-only helper-import recovery preserved; no qualifying refit or promotion.', 'experiments/upstream-item154-tree-split3eight-compact-mps/verification.json', 'full actual data/operator parity, all original saved-model replays and independent complete stored-mean scoring'),

    ('FRESH109 KNN3 mean screen', '0.859293141998', '2026-11-20T04:53:39.778082+00:00', 'Fresh training', 'fresh_cv', 'codex', 'screen', 'Both original residual KNN3 chains: verified negative full neural study', 'Both original own full native KNN3 producers and every known projection verified. All five nested user folds,15 actual saved-model replays with original reference normalization and separate exact mean/canonical score checks pass. NN0.8594597171; tree50 mean0.8593260080 and tree154 mean0.8592931420 both exceed current0.8592012938 best. Complete actual-data bit parity validates memory-only input/scaler execution changes; full H128/r32,400 inputs,data/folds/seeds/optimizer/stopping remain fixed. No qualifying refit or promotion.', 'experiments/upstream-item154-tree-knn3pair2-compact-mps/verification.json', 'complete real transformation parity, all actual full saved-model replays and separate canonical stored-mean checks'),
    ('FRESH109 attention mean screen', '0.8592708192436762', '2026-11-20T03:33:58.312796+00:00', 'Fresh training', 'fresh_cv', 'codex', 'screen', 'Original attention64 pair: verified negative development study', 'Both own original attention64 terminal/epoch5 pairs after full12/12 source training. Complete five nested user folds and15 actual saved-neural-model replays pass. NN0.8594307511; predeclared tree50 mean0.8593032269 and tree154 mean0.8592708192 both worsen the declared0.8592309820 reference and current0.8592012938 best. Separate exact means and full canonical scoring pass. Original capacity/data/folds/seeds/stopping unchanged. No qualifying fit or promotion.', 'experiments/upstream-item154-tree-attention2-virtual-mps/verification.json', 'complete actual saved-model replays and separate stored-mean/canonical scoring for both declared tree partners'),
    ('FRESH107 own6 qualifying', '0.8615662023089099', '2026-11-20T03:28:01.064658+00:00', 'Fresh training', 'fresh_qualifying', 'codex', 'audit', 'Six eligible own families: independently verified qualifying best', 'New actual full tree154 joins the five preceding own families. One known-only nonnegative sum-one fit; complete qualifying vector frozen before canonical recovered audit. Independent compensated statistics, all63 simplex supports, original actual saved-family replays and exact known/full vectors pass. Both groups improve to0.8605620602 /0.8615662023. Apparent known-fit0.8590482079 is not nested CV; CV best remains0.8592012938. First optimizer precision failure occurred before qualifying label access; separately declared ftol-only correction preserves original evidence and model objective. Target remains unmet.', 'experiments/upstream-own6-tight-calibration/independent-verification.json', 'known-only convex weights, complete actual saved-family replays and independent full canonical group scoring'),
    ('FRESH107 tree154 qualifying', '0.8617189998770529', '2026-11-20T03:03:55.662803+00:00', 'Fresh training', 'fresh_qualifying', 'codex', 'audit', 'Full tree154 fixed half mean: matched control improves, overall best retained', 'Actual full tree154 three-seed fit and saved-model replays, unchanged actual full NN154/ridge, and original half mean. Frozen vector independently reconstructs bit for bit; full canonical groups score0.8607366969 /0.8617189999. Both improve the matched prior tree50/NN154 half mean0.8607516602 /0.8617378857; both remain worse than the five-family convex best0.8605716962 /0.8615832225. CV-selected tree budgets760/968/880 unchanged; no recovered outcome fitting or selection. Target remains unmet.', 'experiments/upstream-item154-tree-production-recovered/independent-verification.json', 'complete actual full-family model replays, exact frozen mean and separate canonical complete-group scoring'),
    ('FRESH109 itemAE mean screen', '0.8593159372374096', '2026-11-20T02:27:53.929793+00:00', 'Fresh training', 'fresh_cv', 'codex', 'screen', 'Fixed item-autoencoder snapshots: verified negative study', 'Original epoch5/10 raw item-autoencoder pair added to the current own model context. Complete five nested user folds and15 actual saved-neural-model replays pass. NN0.8594477768; original tree50 half mean0.8593159372 worsens its declared0.8592309820 reference and current0.8592012938 best. Separate exact mean and canonical scoring pass. No checkpoint/capacity/budget change, qualifying refit or promotion.', 'experiments/upstream-rbm50-tree-itemae-virtual-mps/verification.json', 'complete held-user predictions, actual saved-model replay and independent negative fixed-mean result'),
    ('FRESH107 tree154 mean', '0.8592012938198756', '2026-11-20T02:23:02.006312+00:00', 'Fresh training', 'fresh_cv', 'codex', 'cv', 'Full item154 context in both families: new independently verified CV best', 'Original predeclared50/50 mean of full-context tree154 and unchanged NN154. All five original user folds,15 actual saved-tree replays and original neural replay lineage pass. Exact family/fold bindings, complete rounded mean reconstruction and separate canonical scoring give0.8592012938, improving0.8592309820. The completed full fixed-half audit improves its matched control but remains above the overall calibrated qualifying best; no qualifying outcome fits or selects the model.', 'experiments/upstream-item154-tree-item154-mps/verification.json', 'complete five-fold held-user predictions, actual saved-model replays and separate exact mean scoring'),
    ('FRESH107 quadratic audit', '0.8615935718033368', '2026-11-20T01:18:38.009163+00:00', 'Fresh training', 'fresh_qualifying', 'codex', 'audit', 'Negative quadratic calibration: convex best retained', 'Same five replayed own families; one fixed21-coefficient quadratic residual ridge with penalty1000, known-only fitting and freeze before recovered audit. Independent direct SVD rank21 and exact vector reconstruction pass. Qualifying0.8605784947 /0.8615935718 worsens both groups; preceding convex best retained. Apparent known-fit0.8590477906 is not nested CV. Serialization-only checker recovery preserves original fitting and vector.', 'experiments/upstream-own5-quadratic/independent-verification.json', 'fixed known-only quadratic study; independent recovered audit, no promotion'),
    ('FRESH107 own5 qualifying', '0.861583222505447', '2026-11-20T00:46:23.233602+00:00', 'Fresh training', 'fresh_qualifying', 'codex', 'audit', 'Five own families: independently verified qualifying best', 'One convex blend of our replayed tree and four neural branches. Weights fitted only on known OOF predictions, complete qualifying vector frozen before audit. Both canonical groups improve to0.8605716962 /0.8615832225. Apparent known-fit0.8590715586 is not nested CV; CV best remains0.8592309820. Independent simplex solve and exact stored-vector reconstruction pass. Target remains unmet.', 'experiments/upstream-own5-calibration/independent-verification.json', 'known-only convex weights, actual saved-family replays and independent canonical complete-group scoring'),
    ('FRESH107 calibrated qualifying', '0.8616635956343016', '2026-11-19T23:24:49.787911+00:00', 'Fresh training', 'fresh_qualifying', 'codex', 'audit', 'One known-fit coefficient: new independently verified qualifying best', 'Same actually replayed own full tree and item154 neural models, with one convex coefficient fitted only to known OOF predictions: tree0.3223099946 /NN0.6776900054. Known-fit0.8591642749 is not nested CV; CV best remains0.8592309820. Frozen before canonical recovered-label access. Independent coefficient refit, exact known/full vectors and complete-group scores0.8606524657 /0.8616635956 pass. Worse-group9.5366304%; target still missed. CSV-header-only audit recovery preserves original weights and frozen vector.', 'experiments/upstream-item154-weight-calibration/independent-verification.json', 'locally fitted known-only coefficient, frozen own full models and independent canonical complete-group scoring'),
    ('FRESH107 item186 mean screen', '0.8593522338040437', '2026-11-19T22:56:36.811704+00:00', 'Fresh training', 'fresh_cv', 'codex', 'screen', 'Rank1000 item context: existing best retained', 'All five nested user folds and15 actual saved-model replays pass. Fixed32 own rank1000 item components extend current154 contexts to186. Expanded-tree half mean0.8593522338 is worse than0.8592309820; separate exact mean/full-known scoring passes. A separately declared evaluator-only recovery fixes the preceding-best path while preserving all training and original failed evidence. No qualifying refit.', 'experiments/upstream-rbm50-tree-item186-virtual-mps-recovered/verification.json', 'actual saved-model replays and separate fixed-mean scoring after evaluator-reference recovery'),
    ('FRESH111 split3 mean screen', '0.8592838017183226', '2026-11-19T22:32:55.519443+00:00', 'Fresh training', 'fresh_cv', 'codex', 'screen', 'Four original split3 additions: existing best retained', 'All five nested user folds and15 actual saved-model replays pass. Own111 raw predictions plus154 learned context, fixed expanded-tree half mean0.8592838017, above0.8592309820. Separate exact mean and full-known compensated scoring pass. Virtual input tensors are completely fingerprinted and reconstructed for each worker/replay. No qualifying refit or promotion.', 'experiments/upstream-rbm50-tree-split3-virtual-mps/verification.json', 'complete held-user predictions, actual saved-model replay and independent fixed-mean scoring'),
    ('FRESH107 user218 mean screen', '0.8594486747386042', '2026-11-19T21:34:59.819959+00:00', 'Fresh training', 'fresh_cv', 'codex', 'screen', 'Own epoch20 user context: existing best retained', 'All five nested user folds and15 actual saved neural-model replays pass. The fixed half mean with the expanded tree verifies0.8594486747, higher than0.8592309820. The64 user components separately passed third saved-encoder inference, independent CSR/covariance and every-user projection checks. No promotion or qualifying refit.', 'experiments/upstream-rbm50-tree-user218-mps/verification.json', 'complete held-user predictions, actual saved-model replay and independent fixed-mean scoring'),
    ('FRESH107 item154 qualifying', '0.8617378856802073', '2026-11-19T21:19:34.547042+00:00', 'Fresh training', 'fresh_qualifying', 'codex', 'audit', 'Own64 item context: independently verified qualifying improvement', 'Known-development-selected expanded-tree/item154 neural fixed half mean. New full398-input NN7/7/8 fit and actual saved-model query replay; original actual replayed full tree729/706/868 reused. Float32 mean frozen before label access. Separate canonical complete-group compensated scores0.8607516602 /0.8617378857 improve both preceding groups; worse-group improvement9.5288309%. Both still miss0.85725; no qualifying outcomes in fitting/selection.', 'experiments/upstream-item154-stream-production/independent-verification.json', 'complete frozen mean, actual full model replay and independent canonical group scoring'),
    ('FRESH AutoRec consensus qualifying', '0.8618006321174798', '2026-11-19T21:09:20.189306+00:00', 'Fresh training', 'fresh_qualifying', 'codex', 'audit', 'Own fixed consensus: verified qualifying improvement', 'Known-development-selected fixed equal consensus of the two rounded family means. Original replayed full tree50/NN90 reused; new own109 NN7/9/8 fitted on all known labels and actually replayed on every qualifying row. Final float32 vector frozen before canonical-label access. Separate complete-group compensated scores0.8608202878 /0.8618006321 improve both preceding groups, but still miss0.85725. No qualifying outcomes in fitting/selection.', 'experiments/upstream-autorec-consensus-stream-production/independent-verification.json', 'complete frozen consensus, actual model replay and independent canonical group scoring'),
    ('FRESH AutoRec consensus', '0.8592432880327434', '2026-11-19T20:49:30.654060+00:00', 'Fresh training', 'fresh_cv', 'codex', 'development', 'Own fixed consensus: new verified development best', 'Post-study fixed equal average of the verified own107/all90 and own109/all90 expanded-tree ensembles. Weights declared after component development scores, before this mean; no weight search or qualifying-based selection. Exact float32 mean and separate compensated score verify0.8592432880. Qualifying recipe frozen separately.', 'experiments/upstream-autorec-consensus/verification.json', 'complete source/model rehash and separate mean verification'),
    ('FRESH107 item154 mean', '0.8592309819864167', '2026-11-19T21:02:27.793803+00:00', 'Fresh training', 'fresh_cv', 'codex', 'development', 'Own64 item context: new verified development best', 'Own107/all90 plus64 fixed unsupervised item components from our freshly trained RBM150 and temporal-SVD400 factors. All five nested folds and15 actual saved-model replays pass. Disk-guard recovery preserved the original recipe, first four receipts and incomplete fifth-fold archive. Predeclared expanded-tree half mean and separate compensated score verify0.8592309820, improving the preceding consensus0.8592432880. No qualifying score yet.', 'experiments/upstream-rbm50-tree-item154-mps/verification.json', 'complete same-recipe recovery, actual saved-model replay and separate mean verification'),
    ('FRESH109 AutoRec mean screen', '0.859274524524327', '2026-11-19T20:43:34.949782+00:00', 'Fresh training', 'fresh_cv', 'codex', 'screen', 'Own AutoRec additions: existing best retained', 'Fixed own raw autoencoder snapshots5/10 added to own107/all90. All five nested user folds and15 actual saved-model replays pass; separate fixed mean and compensated scoring agree0.8592745245, above0.8592720613. No new global record or qualifying refit. The original50/50 raw trajectory continues.', 'experiments/upstream-rbm50-tree-autorec109-mps/verification.json', 'complete held-user predictions, actual saved-model replay and separate mean verification'),
    ('FRESH110 GPU mean screen', '0.8592996370615437', '2026-11-19T19:38:19.008309+00:00', 'Fresh training', 'fresh_cv', 'codex', 'screen', 'Whole110 GPU mean: existing best retained', 'Complete whole110 neural fitting and all15 actual saved-model replays. Post-study fixed half mean with the expanded tree, declared after the neural result; no preregistration before that result is claimed. Score0.8592996371 is higher than0.8592720613. No new global record, full-known refit or qualifying audit.', 'experiments/upstream-rbm50-tree-whole110-post-study/verification.json', 'complete held-user predictions and actual saved-model replay'),
    ('FRESH166 GPU mean screen', '0.8592892332444937', '2026-11-19T19:38:21.642546+00:00', 'Fresh training', 'fresh_cv', 'codex', 'screen', 'Own166 GPU mean: existing best retained', 'Fixed all59 captured own snapshot additions, same five nested known-user folds and all15 actual saved-model MPS replays. Expanded-tree half mean0.8592892332 is higher than0.8592720613. Ridge diagnostic improvement did not carry through to this nonlinear recipe. No qualifying refit.', 'experiments/upstream-rbm50-tree-warm166-mps/verification.json', 'complete held-user predictions and actual saved-model replay'),
    ('FRESH166 H512 mean screen', '0.8593417716067698', '2026-11-19T19:49:02.891488+00:00', 'Fresh training', 'fresh_cv', 'codex', 'screen', 'Larger512/rank64 GPU model: existing best retained', 'Same own166/all90 data and393 input partition; larger hidden512/rank64 architecture, unchanged optimizer and nested stopping. All five folds and15 actual saved-model replays pass. Fixed expanded-tree half mean0.8593417716 remains above0.8592720613, so it is not a new global record or qualifying candidate.', 'experiments/upstream-rbm50-tree-warm166-h512-mps/verification.json', 'complete held-user predictions and actual saved-model replay'),
    ('FRESH107 all90 mean qualifying', '0.8618605526405846', '2026-11-19T19:21:58.833120+00:00', 'Fresh training', 'fresh_qualifying', 'codex', 'audit', 'Expanded tree plus all90 GPU NN: new verified qualifying best', 'Selected solely from completed known-ProbeX development. Full-known tree budgets729/706/868 and MPS NN7/8/7; actual full saved-model inference replay passes for both. Fixed float32 half mean frozen before recovered-label access. Separate canonical full-group scoring verifies source0 0.8608774626 and source1 0.8618605526. Worse group improves9.5159525%, both still miss0.85725. Exact family schemas294/334 and prior paired-parent differences remain explicit. No downloaded numerical predictions or qualifying-outcome fitting/selection; adaptive recovered audit, not official/untouched Test.', 'experiments/upstream-rbm50-tree-latent90-production/frozen-mean-score-verification.json', 'actual saved-family replays and separate frozen-mean/canonical-score verification'),
    ('FRESH107 all90 GPU mean', '0.8592720612630157', '2026-11-19T18:40:35.813754+00:00', 'Fresh training', 'fresh_cv', 'codex', 'development', 'Expanded tree plus all90 GPU NN: new verified development best', 'Fixed half mean declared while the all90 GPU neural study was fitting. All15 saved MPS neural models replay bitwise; original expanded-tree replay and all ten actual fold receipts/models rehashed. Same1,128,164 rows and five nested user folds. Independent stored-mean and compensated scoring agree0.8592720613, improving0.8593661639. Tree has294 inputs and NN334; original family schemas remain explicit. Full-known fitting/replay and new qualifying audit are pending. No qualifying-outcome fitting or selection.', 'experiments/upstream-rbm50-tree-latent90-mps/verification.json', 'complete fixed-combination and separate stored-mean verification'),
    ('FRESH107 RBM50 mean', '0.8593661638722666', '2026-11-19T18:32:00+00:00', 'Fresh training', 'fresh_cv', 'codex', 'development', 'Expanded tree plus context NN: new verified development best', 'Fixed half mean of own107 plus50 learned RBM context tree and NN on all1,128,164 known-ProbeX rows and five nested user folds. All15 saved tree models replay; the reused NN retains all15 actual saved-model replays. Independent compensated scoring confirms0.8593661639, improving standalone NN0.8594554093. Four folds improve and one worsens; lower aggregate score is promoted. No new qualifying score; qualifying outcomes are excluded from fitting and selection.', 'experiments/upstream-rbm50-tree/independent-verification.json', 'complete independent score and saved-model replay observation'),
    ('FRESH107 RBM50 qualifying', '0.8619141823430468', '2026-11-19T16:54:57.918725+00:00', 'Fresh training', 'fresh_qualifying', 'codex', 'audit', 'Own107 plus50 RBM context: new verified qualifying best', 'Known-CV-selected standalone neural candidate, weights1 and budgets14/14/13 frozen before fitting and new qualifying outcomes. Actual saved-model inference replay, exact frozen family vector, canonical labels, full counts and separate compensated scoring verify source0 0.8609266298 /source1 0.8619141823. Worse group improves9.5103%; both still miss0.85725. Own freshly trained predictors and learned user components only; no downloaded numerical features or qualifying-based selection. Adaptive recovered audit, not official Test.', 'experiments/upstream-rbm50-production/verification.json', 'coordinator complete independent-verification observation'),
    ('FRESH107 RBM50 NN', '0.8594554093063749', '2026-11-19T16:15:57.477292+00:00', 'Fresh training', 'fresh_cv', 'codex', 'development', 'Own107 plus50 original RBM context: new development best', 'Standalone neural residual blender on107 freshly trained predictors plus50 original learned user components. Same five nested user folds,1,128,164 rows and462,858 users. All15 saved neural models replay and separate compensated scoring verifies0.8594554093 versus0.8594692564. Lower aggregate score promoted without a minimum-gain or all-fold veto. Fixed tree91 mean0.8594773263 is higher. No new qualifying outcome yet; recovered target remains unmet.', 'experiments/upstream-rbm50-context/independent-verification.json', 'coordinator complete-verification observation'),
    ('FRESH103 mean qualifying', '0.8620220770129196', '2026-11-19T13:59:33+00:00', 'Fresh training', 'fresh_qualifying', 'codex', 'audit', 'Own103 fixed combination improves qualifying audit', 'Original tree91 plus fresh full-fit contextNN103, fixed float32 half mean. Full NN replay and exact original tree provenance/replay bindings verify; separate mean and compensated scoring confirm source0 0.8610694977 and source1 0.8620220770. Both miss0.85725. No qualifying labels fit or select weights. Adaptive recovered audit, not official Test.', 'experiments/upstream-cf48-103-production/frozen-mean-score-verification.json', 'separate frozen-mean and compensated-score verification'),
    ('FRESH103 NN qualifying', '0.8619938614020105', '2026-11-19T14:03:00+00:00', 'Fresh training', 'fresh_qualifying', 'codex', 'audit', 'Own103 standalone NN: new verified qualifying best', 'NN-only candidate selected from the completed known-ProbeX comparison and committed before either new qualifying outcome. Exact own full-fit NN family vector frozen after actual saved-model replay. Canonical label blob, complete group counts, source/model/scaler hashes and independent compensated scores verify source0 0.8609916544 and source1 0.8619938614. Worse group improves9.5020%; both miss0.85725. No downloaded numerical features or qualifying-fitted weights. Adaptive recovered audit, not official/untouched Test.', 'experiments/upstream-cf48-103-nn-audit/verification.json', 'predeclared standalone frozen-vector and compensated-score verification'),
    ('FRESH103 context', '0.8595400254184109', '2026-11-19T13:39:58+00:00', 'Fresh training', 'fresh_cv', 'codex', 'development', 'Own tree91 plus contextNN103: improved fixed combination', 'Same five nested user folds and known-ProbeX rows. All15 neural models replay and independent scoring verifies the fixed half mean0.8595400254. The standalone neural component is stronger at0.8594692564 and is also promoted. Full-known refitting and separately predeclared qualifying audits are pending; no recovered labels enter fitting or candidate choice.', 'experiments/upstream-cf48-103-context/independent-verification.json', 'independent five-fold scoring and actual saved-model replay'),
    ('FRESH103 NN', '0.8594692563654075', '2026-11-19T13:39:59+00:00', 'Fresh training', 'fresh_cv', 'codex', 'development', 'Own103 context NN: new verified development best', 'Standalone context NN on103 freshly trained predictors. Five nested user folds,1,128,164 known-ProbeX rows and462,858 users. All15 saved neural models replay; compensated scoring verifies0.8594692564 versus prior best0.8599596978. No minimum-gain or all-fold veto. NN-only qualifying candidate explicitly declared before either new qualifying outcome. Adaptive development evidence, not official Test or qualifying success.', 'experiments/upstream-cf48-103-context/independent-verification.json', 'independent five-fold scoring and actual saved-model replay'),
    ('FRESH91 context qualifying', '0.8624166617824088', '2026-11-19T12:37:38.739288+00:00', 'Fresh training', 'fresh_qualifying', 'codex', 'audit', 'Context91 qualifying audit: preceding best retained', 'Own full-fit contextNN91 plus exact full-fit tree91 reuse. Both actual saved-family replays, fixed float32 mean, canonical labels, exact group counts and separate compensated scoring verify source0 0.8615005553 and source1 0.8624166618. Both miss0.85725 and are slightly worse than the preceding own qualifying best0.8624021195. Development best0.8599596978 remains a separate population/metric; no qualifying outcomes fit or select inputs, budgets or weights. Adaptive recovered audit, not official/untouched Test.', 'experiments/upstream-context91-production/frozen-mean-score-verification.json', 'separate frozen-mean and compensated-score verification'),
    ('FRESH91 context', '0.8599596977902322', '2026-11-19T11:55:08.035073+00:00', 'Fresh training', 'fresh_cv', 'codex', 'development', 'Own tree91 + contextNN91: verified development best', 'Fixed half mean on the same1,128,164 known-ProbeX rows and462,858 users. Both families pass saved-model replay; ten actual originating fold receipts and model artifacts are rehashed, shared row identities checked, and compensated scoring confirms0.8599596978 versus0.8599745622. No minimum-gain or all-fold veto. Own trained features only; qualifying production is pending. Adaptive development, not official or untouched Test.', 'experiments/upstream-tree91-context91/verification.json', 'independent fixed-combination verification recorded'),
    ('FRESH91 tree + NN86', '0.8599745622436025', '2026-11-19T08:55:33.473120+00:00', 'Fresh training', 'fresh_cv', 'codex', 'development', 'Own trained predictors: tree91 + contextNN86', 'Fixed half mean of the verified tree91 and unchanged contextNN86 held-user vectors. Independent compensated RMSE 0.8599745622 over the same 1,128,164 known-ProbeX rows and 462,858 users. All ten actual originating fold receipts and model artifacts rehashed; both families previously passed full saved-model replay. Fixed91 plain mean scored 0.8601526901 and was not promoted. Separate family schemas remain explicit; a new full qualifying production refit is planned. Adaptive development, no qualifying-label fitting or selection.', 'experiments/upstream-tree91-context86/verification.json', 'independent fixed-combination verification recorded'),
    ('FRESH91 qualifying', '0.8624021195492101', '2026-11-19T09:56:46.408790+00:00', 'Fresh training', 'fresh_qualifying', 'codex', 'audit', 'Own tree91 + exact full-fit contextNN86: qualifying audit', 'Fresh full-known-ProbeX tree91 with its own ridge and three actual nested budget medians; exact own full-fit contextNN86 reused only after source/bank/config/budget/model/replay checks. Both family replays, the fixed float32 half mean and separate compensated scoring verify. Source0 0.8614776641; source1 0.8624021195. Both still miss 0.85725; worse group improves9.4591% over the Cinematch reference. Adaptive community-recovered audit; no official/untouched Test claim and no qualifying outcomes in fitting/selection.', 'experiments/upstream-tree91-context86-production/frozen-mean-score-verification.json', 'separate frozen-mean and compensated-score verification'),
    ('FRESH86 qualifying', '0.8624355592570154', '2026-11-19T09:32:32.418194+00:00', 'Fresh training', 'fresh_qualifying', 'codex', 'audit', 'Own matching86 bank: qualifying audit', 'Full matching86 tree/context-NN fit. Both saved families replay, their fixed float32 mean verifies, and a separate compensated score check covers both exact recovered groups. Source0 0.8615073690; source1 0.8624355593. Both miss 0.85725; the worse group improves 9.4556% over Cinematch. Own freshly trained features only; no qualifying outcomes in fitting or selection. The paired production bank uses declared own retrains, with source-vector differences from the development realization explicitly quantified. Adaptive community-recovered audit, not official or untouched Test.', 'experiments/upstream-cf12-matching-production/frozen-mean-score-verification.json', 'separate frozen-mean and compensated-score verification'),
    ('FRESH85 qualifying', '0.8628067765436453', '2026-11-19T06:49:44.709186+00:00', 'Fresh training', 'fresh_qualifying', 'codex', 'audit', 'Our 85-predictor bank: first qualifying audit', 'Both independent saved-family replays and the fixed float32 mean pass. Source0 scores 0.8619187880 on 1,408,789 rows; source1 scores 0.8628067765 on 1,408,342 rows. Both miss 0.85725; the worse group improves 9.4166% over the Cinematch reference. No downloaded pretrained features or qualifying-label fitting/selection. Community-recovered audit, not official or untouched Test. Different population from known-ProbeX CV.', 'experiments/upstream-first85-completion-audit/verification.json', 'independent full-model replay and compensated scoring'),
    ('FRESH86 matched', '0.8600300601883686', '2026-11-19T05:26:23.094064+00:00', 'Fresh training', 'fresh_cv', 'codex', 'development', 'Own trained predictors: matching tree + context NN', 'Complete five-user-fold known-ProbeX result: all five folds improve over 0.8600637148. All 15 new trees replay bit-for-bit; the preceding verified NN vector is unchanged. Independent full-row scoring verifies 0.8600300602. Own trained predictions only; qualifying score pending. Separate evaluation population from the historical original-Probe series.', 'experiments/upstream-cf12-tree/independent-verification.json', 'independent complete comparison recorded'),
    ("FRESH86 context", ".8600637147575518", "2026-11-19T02:08:38+00:00", "Fresh training", "fresh_cv", "codex", "development", "Own trained predictors + context NN", "Our best fully verified fresh-training development result. Five user folds over 1,128,164 known-ProbeX rows; 15 neural models replay bit-for-bit and independent scoring verifies the fixed tree/NN average. Public source code is reused, with no downloaded pretrained prediction features. Adaptive development; qualifying evaluation is pending. This population differs from the earlier original-Probe series.", "experiments/upstream-cf12-context/independent-verification.json", "verified result recorded"),
    ("C0", ".98256528", "2026-09-27T08:45:41-07:00", "Biases", "single", "codex", "promoted", "User and movie bias", "The first project model proved ingestion and probe isolation, but it missed most user–movie interaction.", "RESULTS.md", "commit upper bound"),
    ("C1", ".93491601", "2026-09-27T08:45:41-07:00", "Time", "single", "codex", "promoted", "Temporal effects model", "Day effects, frequency and drift delivered the largest early jump.", "RESULTS.md", "commit upper bound"),
    ("C2", ".90267216", "2026-09-27T09:10:33-07:00", "Factors", "single", "codex", "promoted", "32-factor SVD++", "Collaborative factors captured interactions that biases could not.", "RESULTS.md", "commit upper bound"),
    ("B1", ".8936900925", "2026-09-27T09:23:40-07:00", "Blend", "cv", "codex", "promoted", "First user-fold blend", "User-disjoint fitting gave a cleaner development estimate for combining diverse factor columns.", "RESULTS.md", "log time proxy"),
    ("B4", ".8866646513", "2026-09-27T16:08:00-07:00", "Neighborhood", "cv", "codex", "promoted", "Learned item neighbors", "A weaker standalone neighbor model added complementary local movie signal.", "RESULTS.md", "log time proxy"),
    ("B7", ".8858460517", "2026-09-27T16:22:00-07:00", "Neighborhood", "cv", "codex", "promoted", "128-neighbor graph", "More neighbor capacity lowered the five-fold blend score again.", "RESULTS.md", "log time proxy"),
    ("B6 tree", ".8804020935", "2026-09-27T18:08:50-07:00", "Blend", "cv", "claude", "promoted", "LightGBM over 15 columns", "Residual kNN, RBM and AutoRec columns plus support metadata made the nonlinear blender useful.", "RESULTS.md", "observed run finish"),
    ("Union", ".8782199562", "2026-09-27T18:30:47-07:00", "Blend", "cv", "codex", "promoted", "Two branches, fixed average", "A 50/50 average of complementary banks and blenders gained without fitting a second-level weight.", "COMBINED_BRANCHES.md", "commit upper bound"),
    ("R3", ".8773860815", "2026-09-27T19:28:07-07:00", "Blend", "cv", "codex", "promoted", "Residual tree correction", "Predicting residual error directly beat the previous equal blend.", "ROUND3.md", "commit upper bound"),
    ("R4-C", ".8754534170", "2026-09-27T20:20:03-07:00", "Neighborhood", "cv", "claude", "promoted", "Wider neighborhood bank", "Time-decayed and 256-neighbor kNN columns plus converged models improved every fold.", "ROUND4.md", "commit upper bound"),
    ("R5-meta", ".8748094089", "2026-09-27T21:56:15-07:00", "Metadata", "cv", "claude", "promoted", "Training-derived context", "Seven user and movie context features sharpened the residual tree.", "ROUND5.md", "commit upper bound"),
    ("R6-RBM5-s3", ".8731159148", "2026-09-28T13:45:25-07:00", "RBM", "cv", "claude", "promoted", "Six conditional RBM columns", "Different hidden sizes and seeds paid more than further continuation of one RBM.", "ROUND6.md", "commit upper bound"),
    ("C5 parallel", ".8759027924", "2026-09-30T21:52:13-07:00", "GP", "cv_parallel", "codex", "parallel", "Local GP compact blend", "The parallel Codex branch rebuilt a compact 17-column bank and added a Gaussian-process prediction. C5 was its best reproducible local result at that checkpoint, but it did not supersede the main score frontier.", "GP_COMPACT.md", "run-start upper bound"),
    ("R7-MH3", ".8725848", "2026-09-30T22:31:12-07:00", "Sequence", "cv", "claude", "promoted", "Masked rating history", "A categorical masked-history predictor brought sequence signal into the bank.", "ROUND7.md", "commit upper bound"),
    ("R7-MT56-s3", ".8703893", "2026-10-01T02:07:55-07:00", "Sequence", "cv", "claude", "promoted", "Masked history variants", "Multiple temporal masked models and three tree seeds pushed below 0.871.", "ROUND7.md", "commit upper bound"),
    ("C12 parallel", ".8747780516", "2026-10-01T21:12:19-07:00", "Session context", "cv_parallel", "codex", "parallel", "Fixed session-context average", "The parallel Codex branch combined C10 and C11 after GP, kNN, RBM and training-day context experiments. C12 narrowly beat historical R5 overall, but only three of five folds improved and the main Round 7 bank was already lower.", "SESSION_META.md", "commit upper bound"),
    ("R8 Bayes", ".8692319", "2026-10-03T14:49:11-07:00", "Factors", "cv", "claude", "promoted", "Bayesian timeSVD++", "A rank-128 Bayesian factor snapshot added a distinct posterior-mean column.", "ROUND8.md", "commit upper bound"),
    ("R8 NN2", ".8676503", "2026-10-04T19:50:33-07:00", "Blend", "cv", "claude", "promoted", "Neural residual blend", "A fixed 50/50 LightGBM and NN2 average improved all five user folds.", "ROUND8.md", "commit upper bound"),
    ("R8 A2e", ".8674131", "2026-10-06T19:26:12-07:00", "Sequence", "cv", "claude", "promoted", "Longer masked sequence", "The 768-step history model and drift factors advanced the promoted result.", "ROUND8.md", "commit upper bound"),
    ("R8 A2h", ".8670984524", "2026-10-09T04:26:14-07:00", "Sequence", "cv", "codex", "promoted", "1536-step A2h + NN2", "A2h improved every fold over the previous promoted ensemble.", "ROUND8.md", "observed run finish"),
    ("R9 LAT", ".8668706425", "2026-10-10T10:46:00-07:00", "Metadata", "cv", "codex", "promoted", "K1 latent and bias features", "Twenty-two features enriched the existing 79-column bank; every fold improved.", "ROUND9.md", "documented completion"),
    ('R10 B5', '.8668212', '2026-10-11T01:42:00-07:00', 'Factors', 'cv', 'claude', 'promoted', 'Bayesian item-frequency screen', 'Completed mean RMSE 0.8668212 improves the documented R9 control despite two worse folds. Accepted retrospectively under the revised overall-score policy; original rejection preserved in the source report.', 'ROUND10.md', 'documented completion'),
    ("PUB466 NN2", ".864533752805203", "2026-11-15T19:14:51+00:00", "Public bank", "cv", "codex", "development", "466-base local/public NN2", "The entire original Probe reached 0.8645337528 with a 466-base local/public bank and nested user-fold NN2 development evaluation. Probe-guided selection remains; this is separate from the recovered qualifying audit.", "PUBLIC_BANK.md", "observed run finish"),
    ("QUAL619 ridge", ".8571903480630092", "2026-11-16T02:42:52+00:00", "Public bank", "qualifying", "codex", "audit", "619-input contextual ridge", "Expanded public prediction bank: 536 public individual predictors, 77 public pretrained ensembles and six local inputs, combined by contextual ridge. Frozen predictions score 0.8571903481 on recovered source1 and 0.8561542082 on source0. Both pass 0.85725. Adaptive community audit, not official Netflix scoring or an untouched private Test; this marker is not connected to the Probe CV series.", "experiments/QUAL-target-final-result-20261116.json", "frozen prediction audit"),
    ('HR2a', '0.8668444070534381', '2026-10-18T19:11:48.291851+00:00', 'Residuals', 'cv', 'claude', 'promoted_candidate', 'Honest A2h residual blend', 'Completed mean improves R9. Promoted retrospectively as a candidate; above the earlier B5 score, so it does not advance the chronological record.', 'experiments/journey-history/HR2a-blend.status.json', 'report timestamp'),
    ('HR2b', '0.8668187324114364', '2026-10-19T01:13:09.403156+00:00', 'Residuals', 'cv', 'claude', 'promoted', 'Honest Bayesian residual blend', 'Completed mean RMSE 0.8668187324 improves R9 and the preceding recorded B5 score. Promoted retrospectively; two worse folds no longer veto the aggregate gain.', 'experiments/journey-history/HR2b-blend.status.json', 'report timestamp'),
    ('LC6', '0.8668264382456018', '2026-10-19T07:28:32.408983+00:00', 'Local context', 'cv', 'codex', 'promoted_candidate', 'Training-trajectory snapshots', 'Independent full-Probe verification gives 0.8668264382, better than its R9 control. Promoted retrospectively as a candidate; HR2b had already recorded 0.8668187324, so this is not a new overall record.', 'experiments/journey-history/LC6-final-verification.json', 'report timestamp'),
    ('N1 full', '0.8667006276575644', '2026-10-31T05:26:49-07:00', 'Sequence', 'cv', 'codex', 'promoted', 'Choice-token transformer blend', 'Independent full-Probe scoring verifies 0.8667006277. Promoted retrospectively on lower aggregate RMSE; the single worse fold remains a diagnostic.', 'experiments/journey-history/N1-full-independent-verification-20261031.json', 'verification recording upper bound'),
    ('H8 full', '0.8664848394547376', '2026-10-31T09:24:23-07:00', 'Blend', 'cv', 'codex', 'promoted', 'Matched headroom-feature blend', 'Independent matched full-Probe comparison verifies 0.8664848395 and all folds improve. Promoted retrospectively; the former 0.0003 minimum-gain gate no longer vetoes the lower score.', 'experiments/journey-history/H8-full-independent-comparison-20261031.json', 'verification recording upper bound'),
    ('PUB201 NN2', '0.8651051433018259', '2026-11-15T18:52:33.377678+00:00', 'Public bank', 'cv', 'codex', 'screen', '201-base local/public NN2', 'The first public-bank neural blend improved the development mean; no formal promotion is asserted.', 'experiments/journey-history/PUB1-union-NN2-20261115.events.jsonl', 'observed run finish'),
    ('PUB201 tree', '0.865322345270635', '2026-11-15T18:54:32.926288+00:00', 'Public bank', 'cv', 'codex', 'screen', '201-base residual tree', 'Completed nested user-fold tree comparison on the original Probe; a measured development candidate, not a promoted checkpoint.', 'experiments/journey-history/PUB1-union-tree-20261115.events.jsonl', 'observed run finish'),
    ('PUB466 ridge', '0.8655424521021198', '2026-11-15T18:51:41.178076+00:00', 'Public bank', 'cv', 'codex', 'screen', '466-base contextual ridge', 'Adding strict-training-subset public predictions improved the ridge screen; the subsequent 466-base NN2 was lower.', 'experiments/journey-history/PUB3-union-ridge-screen.json', 'observed run finish'),
    ('PUB201 calibration', '0.8659848000118202', '2026-11-15T18:34:34.892615+00:00', 'Public bank', 'cv', 'codex', 'screen', 'Nested movie calibration', 'A user-disjoint inner split chose the per-movie correction penalty; all folds improved against the matched ridge. This is a separate screen.', 'experiments/journey-history/item-calibration.json', 'observed run finish'),
]

RUNS = [
    ("T2-f64-lr005-reg015-d97", "Joint factors", "probe_rmse", "epoch", "Time-aware factor training", "claude"),
    ("R2-rbm100-fast", "RBM", "probe_rmse", "epoch", "Conditional RBM learning", "claude"),
    ("P2-rbm200-condbias-scratch", "RBM + context", "probe_rmse", "epoch", "Contextual RBM continuation", "claude"),
    ("B3-full-r128", "Bayesian factors", "mean_probe_rmse", "sweep", "Posterior-mean factor learning", "claude"),
]

# Selected public Quiz leaderboard observations, not a continuous reconstruction.
# Dates for annual prizes are the dated announcement/report, not necessarily submission time.
COMPETITION = [
    dict(date="2006-10-02", score=.9514, team="Cinematch", title="The public baseline",
         note="Netflix launched the challenge with Cinematch at 0.9514 RMSE on the public Quiz set.",
         source="https://gregstoll.com/~gregstoll/netflixprize/leaderboard20061019.html"),
    dict(date="2006-10-18", score=.9084, team="How low can he go?", title="First weeks: 0.9084",
         note="A preserved October 2006 Netflix leaderboard lists this leading submission on October 18.",
         source="https://gregstoll.com/~gregstoll/netflixprize/leaderboard20061019.html"),
    dict(date="2007-11-13", score=.8712, team="KorBell / BellKor", title="2007 Progress Prize",
         note="BellKor blended 107 results; the 0.8712 Quiz score earned the first $50,000 progress prize. Date is the prize announcement.",
         source="https://pzs.dstu.dp.ua/DataMining/recom/bibl/ProgressPrize2007_KorBell.pdf"),
    dict(date="2008-11-25", score=.8616, team="BellKor in BigChaos", title="2008 Progress Prize",
         note="The combined BellKor and BigChaos solution reached 0.8616. Date is the team's report, not its submission.",
         source="https://www2.seas.gwu.edu/~simhaweb/champalg/cf/papers/ProgressPrize2008_BigChaos.pdf"),
    dict(date="2009-06-26", score=.8558, team="BellKor's Pragmatic Chaos", title="The 10% line falls",
         note="The new combined team crossed the public Quiz threshold and started the final 30-day period.",
         source="https://www.stat.berkeley.edu/~jon/stat-151b-spring-2012/netflix_overview.pdf"),
    dict(date="2009-07-26", score=.8553, team="The Ensemble", title="Final public leaderboard",
         note="The Ensemble led the public Quiz board at 0.8553. BellKor's Pragmatic Chaos had 0.8554 on Quiz; both rounded to 0.8567 on hidden Test, and the earlier submission won.",
         source="https://people.csail.mit.edu/dsontag/courses/ml12/slides/lecture12.pdf"),
]

# Score or run-recording commits with an explicit Claude co-author trailer.
CLAUDE_RECORDS = {
    "B6 tree": "325366559b2e4d6395e160d8c47ac8386b805730",
    "R4-C": "909ff0305d0093b115acdd0c9be9ba501c224eab",
    "R5-meta": "1314dfd4304ee1626b200363e543999f06ef7a3f",
    "R6-RBM5-s3": "5113022c841f8f877f2a28762ee9809b26ef5065",
    "R7-MH3": "3b6d035796afa003dc36defc5ebd131d13ac0a15",
    "R7-MT56-s3": "f2d13c52ffc2c59308d28f8b5960407b0a791e64",
    "R8 Bayes": "14b2540dd4488aefe1721804ab9aa42126833a88",
    "R8 NN2": "e90adad1d222afe0b7d120c683c5083c81cff2c5",
    "R8 A2e": "16f4eb4a385aa5a0347abf54af2eb5edd2b80fda",
    "R10 B5": "d9796da1b826e5ed5e919e01a18823d63d300d0a",
    "T2-f64-lr005-reg015-d97": "bcdf4a27b389337d243ed53a2c7ee20e3a9f0d6b",
    "R2-rbm100-fast": "325366559b2e4d6395e160d8c47ac8386b805730",
    "P2-rbm200-condbias-scratch": "2424aaee6773927b1992ecd8590f9fed6d8196a8",
    "B3-full-r128": "2d4488ed023f15b23784292784b96a00611e0aac",
}

# Quan-only score commits; the user confirmed these were Codex sessions.
CODEX_RECORDS = {
    "C0": "4e7f65636a8e3f448e591cb8490ebcb56d513d34",
    "C1": "4e7f65636a8e3f448e591cb8490ebcb56d513d34",
    "C2": "66591d002dbaa4a45524457e162ce49ffa255c29",
    "B1": "cffd6bfe8406ba3332511e7ba26da0c0ca23b874",
    "B4": "cffd6bfe8406ba3332511e7ba26da0c0ca23b874",
    "B7": "700fbeb9af146e73c4d4c0f7387c0e0d5a9581e2",
    "Union": "bfb875f26de3de7895ab118615e8156651f1e28a",
    "R3": "43da9c70ffae34a0c33c8643799f671ea396a271",
    "R8 A2h": "f1639454e15b0637f61537b993f7f9de8151b3d7",
    "R9 LAT": "0e8171bff83bb63d9beae1d9fece06ffba199dae",
    "C5 parallel": "c368d4bfca0f56ed02dfad5a41dc97e10fa9be85",
    "C12 parallel": "c368d4bfca0f56ed02dfad5a41dc97e10fa9be85",
}

RECORD_COMMITS = {**CLAUDE_RECORDS, **CODEX_RECORDS}


def run_curve(name, label, metric, step_key, description, credit):
    manifest = json.loads((EXPERIMENTS / (name + ".json")).read_text())
    start = dt.datetime.fromisoformat(manifest["started_utc"])
    points = []
    for raw in (EXPERIMENTS / (name + ".events.jsonl")).read_text().splitlines():
        event = json.loads(raw)
        line = event["line"]
        score = re.search(r"(?<!\w)" + metric + r"=([0-9.]+)", line)
        step = re.search(r"(?<!\w)" + step_key + r"=(\d+)", line)
        if score and step:
            hour = (dt.datetime.fromisoformat(event["observed_utc"]) - start).total_seconds() / 3600
            points.append([round(hour, 4), round(float(score.group(1)), 7), int(step.group(1))])
    if not points:
        raise ValueError(f"No runtime score events for {name}")
    return dict(id=name, label=label, metric=metric, step=step_key, description=description,
                credit=credit, creditCommit=RECORD_COMMITS.get(name),
                elapsedHours=round(manifest["elapsed_seconds"] / 3600, 3), points=points,
                source=f"experiments/{name}.events.jsonl")


def validate_fresh_status():
    record=json.loads((EXPERIMENTS/'fresh-reproduction-status.json').read_text())
    if record['qualifying_status']!='audited_miss':
        raise ValueError('Review the site narrative before publishing a changed result')
    candidate=EXPERIMENTS/'upstream-rbm50-tree-latent90-mps/verification.json'
    development=json.loads(candidate.read_text())
    check_path=candidate.parent/'independent-mean-check.json';checked=json.loads(check_path.read_text())
    if (hashlib.sha256(candidate.read_bytes()).hexdigest()!=record['preceding_independent_verification_sha256'] or
        development['state']!='complete_verified_fixed_combination' or not development['improves_previous'] or
        development['rmse']!=.8592720612630157 or not development['fixed_average_bitwise_equal'] or
        not development['original_saved_model_replays_verified'] or
        checked['verification_sha256']!=hashlib.sha256(candidate.read_bytes()).hexdigest() or
        not checked['fixed_mean_bitwise_equal'] or abs(checked['rmse']-development['rmse'])>1e-12):
        raise ValueError('The selected fixed mean requires exact complete evidence and separate stored-mean scoring')
    for relative in ['experiments/upstream-rbm50-tree/tree-replay.json','experiments/upstream-latent90-mps-context/nn-replay.json']:
        path=ROOT/relative;replay=json.loads(path.read_text())
        if (hashlib.sha256(path.read_bytes()).hexdigest()!=development['source_sha256'][relative] or
            [row['fold'] for row in replay['records']]!=list(range(5)) or
            not all(row['bitwise_equal'] for row in replay['records'])):
            raise ValueError('Actual saved-family replay evidence is incomplete')
    current_path=ROOT/record['development_verification_path'];current=json.loads(current_path.read_text())
    independent_path=ROOT/record['development_independent_check_path'];independent=json.loads(independent_path.read_text())
    kind=current['state']
    expected_weights=[.5,.5]
    if kind=='complete_fixed_post_study_consensus':
        checked_state='complete_independent_consensus_and_score';checked_flag='stored_consensus_bitwise_equal'
    elif kind=='complete_verified_fixed_combination':
        checked_state='complete_separate_stored_mean_and_score_check';checked_flag='fixed_mean_bitwise_equal'
    elif kind=='complete_fixed_tree25_pool75_development':
        checked_state='complete_separate_fixed_tree25_pool75_and_canonical_score';checked_flag='every_weighted_float32_bit_equal'
        expected_weights=dict(tree=.25,nn_pool=.75)
        pool_relative='experiments/upstream-native-nn-pool6/independent-canonical-verification.json'
        pool_path=ROOT/pool_relative;pool=json.loads(pool_path.read_text())
        verifier=ROOT/'scripts/verify_upstream_native_nn_pool6_tree31.py'
        if (not current['actual_origin_model_and_replay_proofs_bound'] or
            hashlib.sha256(pool_path.read_bytes()).hexdigest()!=current['source_sha256'][pool_relative] or
            pool['state']!='complete_independent_canonical_fixed_six_native_nn_pool' or
            not pool['all_original_source_models_and_complete_replay_receipts_rehashed'] or
            not pool['every_pooled_float32_bit_recomputed_with_compensated_sum'] or
            independent['verifier_sha256']!=hashlib.sha256(verifier.read_bytes()).hexdigest()):
            raise ValueError('Fixed weighted pool requires original actual source/model replay and canonical arithmetic evidence')
    else:raise ValueError('Unrecognized verified development combination')
    if (hashlib.sha256(current_path.read_bytes()).hexdigest()!=record['independent_verification_sha256'] or
        current['rmse']!=record['best_development_rmse'] or current['weights']!=expected_weights or
        current['rmse']>=development['rmse'] or current['qualifying_labels_read'] is not False or
        independent['state']!=checked_state or not independent[checked_flag] or
        independent['verification_sha256']!=record['independent_verification_sha256'] or
        abs(independent['rmse']-current['rmse'])>1e-12):
        raise ValueError('Current development result requires exact independent mean/score evidence')
    for relative in record['development_required_replay_paths']:
        path=ROOT/relative;replay=json.loads(path.read_text())
        if (hashlib.sha256(path.read_bytes()).hexdigest()!=current['source_sha256'][relative] or
            [r['fold'] for r in replay['records']]!=list(range(5)) or not all(r['bitwise_equal'] for r in replay['records'])):
            raise ValueError('Current development requires all original actual saved-model replays')
    proof=EXPERIMENTS/'upstream-first85-completion-audit/verification.json'
    verified=json.loads(proof.read_text())
    if (hashlib.sha256(proof.read_bytes()).hexdigest()!=record['first85_completion_verification_sha256'] or
            verified['state']!='complete_independent_completion_audit' or verified['goal_achieved'] or
            not verified['all_saved_models_independently_replayed'] or
            not verified['fixed_equal_average_bitwise_verified'] or
            verified['groups']['test']['rmse']!=.8628067765436453):
        raise ValueError('The published first qualifying miss requires its actual independent proof')
    latest_path=ROOT/record['latest_qualifying_verification_path'];latest=json.loads(latest_path.read_text())
    if latest['state']=='complete_independent_frozen_consensus_and_scores':
        mean_ok=latest['exact_rounded_consensus_bitwise_equal']
        replay_ok=latest['actual_new_full_neural_replay_verified'] and latest['original_full_model_replays_reused_and_verified']
    elif latest['state'] in {'complete_independent_own5_calibration_and_frozen_group_scores', 'complete_independent_own6_calibration_and_frozen_group_scores'}:
        family_count=6 if latest['state']=='complete_independent_own6_calibration_and_frozen_group_scores' else 5
        cal_path=latest_path.parent/'calibration.json';freeze_path=latest_path.parent/'frozen.json'
        cal=json.loads(cal_path.read_text());freeze=json.loads(freeze_path.read_text())
        for path in [cal_path,freeze_path]:
            if hashlib.sha256(path.read_bytes()).hexdigest()!=latest['bindings'][str(path.relative_to(ROOT))]:
                raise ValueError('Own-family calibration/freeze binding changed')
        weights=latest['weights']
        if (len(weights)!=family_count or len(cal['families'])!=family_count or weights!=cal['weights'] or weights!=freeze['weights'] or min(weights)<0 or
            abs(sum(weights)-1)>1e-15 or not latest['independently_solved_full_simplex'] or
            max(abs(a-b) for a,b in zip(weights,latest['independent_simplex_weights']))>1e-5 or
            latest['known_score_is_nested_cross_validation'] is not False or cal['known_score_is_nested_cross_validation'] is not False or
            freeze['qualifying_labels_read'] is not False or cal['qualifying_labels_used_for_fitting_or_selection'] or
            freeze['prediction_sha256']!=latest['prediction_sha256'] or not latest['both_groups_improved']):
            raise ValueError('Own-family known-only simplex fit and frozen vector evidence required')
        mean_ok=latest['exact_known_and_full_stored_vectors_bitwise']
        replay_ok=latest['all_original_development_and_full_family_replays_verified']
    elif latest['state']=='complete_independent_known_calibration_and_frozen_group_scores':
        cal_path=latest_path.parent/'calibration.json';freeze_path=latest_path.parent/'frozen.json'
        cal=json.loads(cal_path.read_text());freeze=json.loads(freeze_path.read_text())
        for path in [cal_path,freeze_path]:
            if hashlib.sha256(path.read_bytes()).hexdigest()!=latest['bindings'][str(path.relative_to(ROOT))]:
                raise ValueError('Original known-only calibration/freeze binding changed')
        weights=latest['weights_tree_nn']
        if (weights!=cal['weights_tree_nn'] or weights!=freeze['weights_tree_nn'] or min(weights)<0 or
            abs(sum(weights)-1)>1e-15 or abs(weights[0]-latest['independently_refitted_coefficient'])>1e-11 or
            latest['known_score_is_nested_cross_validation'] is not False or cal['score_is_nested_cross_validation'] is not False or
            freeze['qualifying_labels_read'] is not False or cal['qualifying_labels_used_for_fitting_or_selection'] or
            freeze['prediction_sha256']!=latest['prediction_sha256'] or not latest['both_groups_improved']):
            raise ValueError('Separate apparent calibration score, known-only weights and original frozen vector required')
        mean_ok=latest['exact_known_and_full_stored_vectors_bitwise'];replay_ok=latest['actual_full_family_replays_reused']
    elif latest['state']=='complete_frozen_mean_and_score_verification':
        mean_ok=latest['fixed_float32_mean_bitwise_equal'];replay_ok=latest['saved_family_replay_receipts_verified']
    else:raise ValueError('Unrecognized independently verified qualifying result')
    if (hashlib.sha256(latest_path.read_bytes()).hexdigest()!=record['latest_qualifying_verification_sha256'] or
        record['qualifying_verification_sha256']!=record['latest_qualifying_verification_sha256'] or
        latest['groups']!=record['latest_qualifying_groups'] or not mean_ok or not replay_ok or
        latest['label_git_blob']!='47ed4788ffba11aad029854a8e60d4dee04fc899' or
        latest['qualifying_labels_used_for_fitting_or_selection'] or latest['both_groups_below_target'] or
        [latest['groups'][k]['rows'] for k in ['source0','source1']]!=[1408789,1408342]):
        raise ValueError('Exact frozen-model replay and independent canonical group evidence required')
    return record


def overview_selection(milestones, record):
    """Keep complete evidence in the ledger and representative points on busy days."""
    ids = {'H8 full'}
    days = {'2026-10-31'}
    for key in ['development_verification_path', 'latest_qualifying_verification_path']:
        matches = [row for row in milestones if row[9] == record[key]]
        if len(matches) != 1:
            raise ValueError('One exact evidence-bound overview milestone required: ' + key)
        row = matches[0]
        ids.add(row[0])
        days.add(dt.datetime.fromisoformat(row[2]).astimezone(ZoneInfo('America/Los_Angeles')).date().isoformat())
    return dict(ids=sorted(ids), days=sorted(days))


def own_training_milestones(rows):
    """Keep archived public-bank records out of the current site's data."""
    return [row for row in rows if row[3] != 'Public bank' and row[4] != 'qualifying']


def main():
    items = []
    published = own_training_milestones(MILESTONES)
    for ident, score, time, strategy, metric, credit, status, title, note, source, basis in published:
        if not (ROOT / source).exists():
            raise FileNotFoundError(source)
        items.append(dict(id=ident, score=float(score), time=time, strategy=strategy,
                          metric=metric, credit=credit, creditCommit=RECORD_COMMITS.get(ident), status=status, title=title,
                          note=note, source=source, basis=basis))
    items.sort(key=lambda item: dt.datetime.fromisoformat(item["time"]))
    frontier = [item["score"] for item in items if item["metric"] == "cv" and item["status"] == "promoted"]
    if any(before <= after for before, after in zip(frontier, frontier[1:])):
        raise ValueError("Main journey must descend; document screens separately")
    if any(before["score"] <= after["score"] for before, after in zip(COMPETITION, COMPETITION[1:])):
        raise ValueError("Selected competition leaderboard milestones must descend")
    record = validate_fresh_status()
    payload = dict(milestones=items, runs=[run_curve(*run) for run in RUNS], competition=COMPETITION,
                   overview=overview_selection(published, record))
    template = (ROOT / "research_journey.template.html").read_text()
    marker = "__JOURNEY_DATA__"
    if template.count(marker) != 1:
        raise ValueError("Expected one data insertion marker")
    output = template.replace(marker, json.dumps(payload, separators=(",", ":")))
    # The public overview omits the operational dashboard; provenance is still
    # validated above and the complete status remains in the linked records.
    output = output.replace('__FRESH_REPRODUCTION__','')
    for name, filename, color in (("claude", "claude-code.svg", "#D97757"), ("codex", "codex.svg", "#b6a5ff")):
        icon = (ROOT / "assets" / filename).read_text().replace('fill="currentColor"', f'fill="{color}"')
        data_url = "data:image/svg+xml;base64," + base64.b64encode(icon.encode()).decode()
        output = output.replace(f"__{name.upper()}_LOGO__", data_url)
    (ROOT / "research_journey.html").write_text(output)
    print(f"research_journey.html: {len(items)} milestones, {len(payload['runs'])} runtime curves")


if __name__ == "__main__":
    main()
