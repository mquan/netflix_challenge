#!/usr/bin/env python3
"""Build the self-contained research journey from documented milestones and run events."""
import base64
import datetime as dt
import hashlib
import json
import math
from html import escape
import pathlib
import re
from zoneinfo import ZoneInfo


ROOT = pathlib.Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"

# These are editorial milestones, not a claim that every experiment was a new best.
# Calendar dates before timestamped run events are documentation upper bounds.
MILESTONES = [
    ('FRESH complete trajectories H128 tree154 half', '0.8588630883643809', '2026-11-21T20:32:49.027774+00:00', 'Fresh training', 'fresh_cv', 'codex', 'promoted', 'Complete trajectories slightly improve verified development CV', 'All519 declared streams were checked before33 exact paired aliases;486 raw predictors yield713 neural inputs with all225 contexts. All15 original five-user-fold/three-seed H128rank32 models, complete eligible statistics/scalers/normalized inputs and held predictions independently replay. Separate canonical fixed half0.8588630884 improves0.8589133968 by0.00005031; NN0.8590060211. Historical tree154 A/B caveat retained. This is known-ProbeX development, not a recovered qualifying result or target achievement.', 'experiments/upstream-complete-trajectory-h128-mps/independent-canonical-verification.json', 'full original15-model/preprocessing replay, separate canonical arithmetic and publication source/file admission'),
    ('FRESH complete trajectories H128 neural control', '0.8590060211471172', '2026-11-21T20:32:49.027774+00:00', 'Fresh training', 'fresh_cv', 'codex', 'audit', 'Neural-only control improves its matched predecessor', 'Full713-input neural-only CV0.8590060211 improves matched391 NN0.8590533486. It remains above the predeclared fixed half0.8588630884. All five folds and15 saved models independently replay; no recovered score is claimed from this development result.', 'experiments/upstream-complete-trajectory-h128-mps/independent-verification.json', 'same complete actual-model replay and separate full canonical neural score'),
    ('FRESH own14 qualifying', '0.8613681513408222', '2026-11-21T17:46:19.142031+00:00', 'Fresh training', 'fresh_qualifying', 'codex', 'promoted', 'Matched391 fold bag adds a small verified qualifying gain', 'The predeclared14-family known-only simplex scores0.8603797340/0.8613681513, improving complete13 by only0.00001234/0.00000564. All16383 supports,15ridge equations,six frozen vectors,full original source closure and15 actual new qualifying models independently verify. Apparent known-fit0.8587148868 is not nested CV. Both groups still miss0.85725. Adaptive recovered evaluation, not official or untouched Test.', 'experiments/upstream-own14-matched391-calibration/independent-canonical-verification.json', 'all14 ordered families, full16383-support preaudit, original frozen arithmetic, complete15-model query replay and separate compensated canonical scores'),
    ('FRESH own14 ridge10 qualifying', '0.8613781986131858', '2026-11-21T17:41:55.749100+00:00', 'Fresh training', 'fresh_qualifying', 'codex', 'audit', 'Declared own14 signed ridge control trades off recovered groups', 'Known-only ridge10 with unpenalized intercept scores0.8603762149/0.8613781986. It improves source0 but worsens source1 relative to the predeclared simplex primary. All15 equations and frozen predictions verify. No control promotion or outcome-driven revision.', 'experiments/upstream-own14-matched391-calibration/independent-recovered-audit.json', 'predeclared signed ridge control, full known normal equations and separate frozen-group canonical audit'),
    ('FRESH matched391 fixed-half qualifying', '0.8615957340466804', '2026-11-21T17:41:55.749100+00:00', 'Fresh training', 'fresh_qualifying', 'codex', 'audit', 'Matched391 fixed half remains a negative qualifying control', 'The unchanged half of the full fifteen-model matched391 query bag and historical tree154 scores0.8606336976/0.8615957340, worse on both groups than the preceding complete13 primary. Known development0.8589133968 remains a different evaluation. All original actual models and frozen arithmetic pass; no coefficient or fold selection follows outcomes.', 'experiments/upstream-own14-matched391-calibration/independent-recovered-audit.json', 'complete original15-model qualifying bag, exact declared fixed half and separate canonical recovered audit'),
    ('FRESH paired391 H512 control', '0.8590481464872496', '2026-11-20T21:22:36.756866+00:00', 'Fresh training', 'fresh_cv', 'codex', 'audit', 'Wider matched391 control is fully verified and worse', 'Original618 inputs/5held-user-folds/3seeds/max40pat4 H512rank32 control completes all15 actual models, full eligible preprocessing/normalized input and held prediction replay. Separate canonical NN0.8593153262/fixed-half0.8590481465 both miss H1280.8590533486/0.8589133968. Same paused worker continued without restart or numerical/guard changes. Historical tree154 A/B caveat retained. No H512 qualifying inference; H128 remains selected using known results only.', 'experiments/upstream-own391-paired-h512-mps/independent-canonical-verification.json', 'all15 actual models, original full preprocessing/prediction bits and separate canonical control score'),
    ('FRESH paired391 H128 tree154 half', '0.8589133967520619', '2026-11-20T20:36:27.360415+00:00', 'Fresh training', 'fresh_cv', 'codex', 'promoted', 'Matched391 neural study verifies a negligible development gain', 'All618 inputs,393 signals/225 contexts and original five-user-fold/three-seed H128rank32/max40pat4 recipe retained. All15 actual models, complete eligible scalers/normalized inputs and held predictions independently replay; separate canonical half0.8589133968 improves own352 by only0.0000026603. NN alone0.8590533486 is slightly worse. Original paused worker continued without restart or guard changes. Neural provider matches pairedB on both queries; historical tree154 A/B realization caveat retained. No new qualifying score; target remains unmet.', 'experiments/upstream-own391-paired-h128-mps/independent-canonical-verification.json', 'all15 actual models, complete original preprocessing/prediction bits and separate compensated canonical scores'),
    ('FRESH paired391 ridge diagnostic', '0.8618508817095664', '2026-11-20T16:46:03.380540+00:00', 'Fresh training', 'fresh_cv', 'codex', 'screen', 'Matched paired inputs pass; added scalar ridge gain is tiny', 'Both complete matchedB input streams retain the original330parent bits plus22temporal and all39 new eligible native/TX predictors. All467design/617nonlinear operators, ten-group statistics, eligible lambda1000 solves and stored held predictions independently replay. Known pairedB352/B391 ridge0.8618576840 to0.8618508817 gains only0.0000068. This is a data/ridge diagnostic, not a neural or qualifying promotion. Original-budget618-input H128 neural follow-up is queued behind original1600 coverage.', 'experiments/upstream-own391-paired-data/independent-verification.json', 'every complete matched paired input/statistic/eligible solve/stored prediction and canonical known score'),
    ('FRESH full404 context FWLS qualifying', '0.8623423167927773', '2026-11-20T16:26:31.810223+00:00', 'Fresh training', 'fresh_qualifying', 'codex', 'audit', 'Full-bank feature-weighted stacking is a negative result', 'All413 source entries retained before exact paired aliases yield404 signals and5657coefficients. Original14contextgates/lambda30/unpenalizedbias,5held-user models and uniform clippedQbag. Independent all interaction/system/coefficient/prediction bits replay; frozen scores0.8613295003/0.8623423168 worsen both groups. Conditional metaCV0.8602208791 is not fully nested end-to-end CV. No parameter revised or promotion made.', 'experiments/upstream-full-fwls/independent-recovered-audit.json', 'complete original source inventory, all five actual models and every canonical frozen qualifying prediction'),
    ('FRESH complete13 context FWLS qualifying', '0.8613960265963411', '2026-11-20T16:07:31.642829+00:00', 'Fresh training', 'fresh_qualifying', 'codex', 'audit', 'Context-weighted component stack narrowly misses the constant-weight primary', 'All13 own components and14 deduplicated original public context gates, lambda30 and unpenalized intercept. All five held-user models, complete interactions/group systems and every known/Q prediction replay. Uniform clipped qualifying bag scores0.8603967792/0.8613960266, worsening both groups against complete13 simplex. Conditional meta-crossfit0.8588517253 is not fully nested end-to-end CV. No promotion or parameter revision.', 'experiments/upstream-own13-fwls/independent-recovered-audit.json', 'all actual five models, independent complete group systems and full frozen qualifying arithmetic/canonical audit'),
    ('FRESH complete13 qualifying', '0.8613737959073772', '2026-11-20T16:06:03.588087+00:00', 'Fresh training', 'fresh_qualifying', 'codex', 'promoted', 'All completed component families add a small verified qualifying gain', 'All13 eligible own families retained after one exact base alias. Known-only simplex checked across all8191 supports; signed ridge10 control and all four stored vectors independently verify. Both studies froze and passed preaudit before either new recovered audit. Separate full canonical scores0.8603920746/0.8613737959 improve preceding own330 by only0.0000105/0.0000143. Apparent known-fit0.8587382 is not nested CV. Target remains unmet; larger context-weighted studies are running. Adaptive recovered groups, not official or untouched Test.', 'experiments/upstream-complete13-calibration/independent-canonical-verification.json', 'all original model/source closure, all8191 supports, exact frozen arithmetic and separate compensated canonical scores'),
    ('FRESH complete13 ridge10 qualifying', '0.8613839925824824', '2026-11-20T16:00:15.582674+00:00', 'Fresh training', 'fresh_qualifying', 'codex', 'audit', 'Declared signed ridge control trades off groups against primary', 'The original released ridge lambda10 with unpenalized intercept is fitted only on complete known OOF features. All14 normal equations and every stored bit independently verify. Frozen control scores0.8603876739/0.8613839926; source0 is better and source1 worse than the predeclared simplex primary. No outcome-driven coefficient revision or control promotion.', 'experiments/upstream-complete13-calibration/independent-recovered-audit.json', 'predeclared original ridge control, full independent normal equations and exact complete frozen-group audit'),
    ('FRESH352 own7 qualifying', '0.8613852653620429', '2026-11-20T16:06:02.699421+00:00', 'Fresh training', 'fresh_qualifying', 'codex', 'promoted', 'Complete own352 fold bag adds negligible qualifying improvement', 'All15 original models, all qualifying operators and uniform five-fold bag independently replay. The declared seven-family known-only fit passes all127 supports and every candidate bit. Separate canonical scores0.8604024389/0.8613852654 improve own330 by only0.0000001/0.0000029. Known apparent-fit0.8587652 is not nested CV. The subsequent complete13 primary advances both groups slightly further; target remains unmet.', 'experiments/upstream-own352-fold-bag-assembly/independent-canonical-verification.json', 'actual15-model full replay, all127 supports, every frozen vector and separate compensated group scores'),
    ('FRESH352 fixed-half qualifying', '0.8616038796640847', '2026-11-20T15:55:58.486581+00:00', 'Fresh training', 'fresh_qualifying', 'codex', 'audit', 'Declared full own352 fold-bag half misses qualifying benchmark', 'The unchanged fixed half of the independently replayed own352 bag and tree154 scores0.8606564518/0.8616038797, worsening both groups against the preceding own330 primary. All models and arithmetic pass; this negative control remains preserved. No coefficients or fold choices changed after audit.', 'experiments/upstream-own352-fold-bag-assembly/independent-recovered-audit.json', 'predeclared fixed half, actual fifteen-model bag and independent complete frozen-group audit'),
    ('FRESH352 full neural tree154 half', '0.8589160570135321', '2026-11-20T14:27:42.659125+00:00', 'Fresh training', 'fresh_cv', 'codex', 'promoted', 'Complete temporal addition improves the verified development best', 'All579 inputs,354signals/225contexts and original five-user-fold/three-seed H128rank32/max40pat4 recipe retained. All15 actual models completed. Compressor guard stopped only the original independent verifier; exact bounded full-column/ordered-row statistics matched every actual scaler and all normalized/prediction bits. Separate canonical half0.8589160570 improves0.8589474951; NN0.8590361083. No own352 qualifying score; allfold qualifying bag declared before outcomes.', 'experiments/upstream-own352-bounded-replay/independent-canonical-verification.json', 'all original actual15 models, exact full eligible statistics/normalized inputs and separate canonical arithmetic'),
    ('FRESH330 own7 qualifying', '0.8613881328035494', '2026-11-20T13:41:41.235187+00:00', 'Fresh training', 'fresh_qualifying', 'codex', 'promoted', 'Own330 seven-family blend improves both recovered groups', 'All three full557-input models, original7/7/8 budgets, complete ridge/scaler/normalized bits and both full-query saved-model replays pass. Exactly two candidates frozen before outcomes. Independent127-support known-only weights and every frozen vector bit verify. Separate canonical scores0.8604025394/0.8613881328 improve own6 on both groups. Apparent known-fit0.8587992 is not nested CV; target remains unmet. Adaptive recovered audit, not official or untouched Test.', 'experiments/upstream-own330-assembly/independent-canonical-verification.json', 'full original saved-model/preprocessing replay, all127 simplex supports and separate complete frozen-group scores'),
    ('FRESH330 fixed-half qualifying', '0.8615599700738508', '2026-11-20T13:41:41.235187+00:00', 'Fresh training', 'fresh_qualifying', 'codex', 'audit', 'Development-selected fixed half trades off recovered groups', 'Original full own330 and tree154 pipelines independently replay. Predeclared frozen half scores0.8606104179/0.8615599701; source0 worsens against own6 and source1 improves slightly. Primary known-only seven-family calibration improves both; no outcome-driven coefficient revision.', 'experiments/upstream-own330-assembly/independent-recovered-audit.json', 'predeclared frozen control and separate complete canonical score'),
    ('FRESH352 temporal22 ridge diagnostic', '0.8618611086522293', '2026-11-20T13:42:35.002116+00:00', 'Fresh training', 'fresh_cv', 'codex', 'screen', 'Complete temporal trajectory improves every known ridge fold', 'All22 original temporal100r snapshot pairs retained after one exact complete-pair alias. Original own330 bits retained in428design/578nonlinear columns. Full canonical input/statistic/solve/prediction replay verifies0.8618611087 versus0.8619710828 on the same userfolds. Ridge diagnostic only; full579-input neural follow-up remains pending.', 'experiments/upstream-own352-temporal-data/independent-verification.json', 'every original/new input bit, all group statistics, eligible solves and actual stored ridge predictions'),
    ('FRESH330 full neural tree154 half', '0.8589474951024617', '2026-11-20T12:59:58.878245+00:00', 'Fresh training', 'fresh_cv', 'codex', 'promoted', 'Own330 full neural/tree half becomes the verified development best', 'All330 own scalar predictors/all90 learned contexts retained. Complete557-input H128rank32 fivefold/three-seed40pat4 fitting,15 actual saved models, every original eligible scaler/statistic/normalized input and held prediction replay. NN0.859109418608456; fixed tree154 half0.8589474951024617 improves prior0.8591267885072646. Separate final arithmetic reconstructs every float32 bit and full compensated canonical score. Adaptive known CV; no qualifying score or refit claimed,10% target still unmet.', 'experiments/upstream-own330-mps/independent-canonical-verification.json', 'all15 actual models/full original eligible preprocessing plus separate exact final-average/canonical scoring'),
    ('FRESH176 full neural fixed-half screen', '0.8591336855012253', '2026-11-20T12:44:30.094177+00:00', 'Fresh training', 'fresh_cv', 'codex', 'screen', 'Recent-ten full neural study narrowly misses global best', 'All five userfolds/three seeds,15 actual models, both full eligible scalers, original group statistics and every held prediction bit replay. NN0.8592799295700446 improves NN166/154 controls. Fixed tree154 half0.8591336855012253 exceeds global0.8591267885072646 by0.000006897. No global or qualifying promotion; full557-input own330 fitting now active.', 'experiments/upstream-own176-recent10-mps/independent-verification.json', 'all actual saved models and complete eligible-statistic/scaler/prediction replay'),
    ('FRESH_SPLIT3 expanded-known fixed-half diagnostic', '0.8622259939637812', '2026-11-20T12:44:30.094177+00:00', 'Fresh training', 'fresh_cv', 'codex', 'screen', 'Extra known training labels help the matched bank, which remains below best quality', 'Identical canonical1128164-row evaluation and held users. Both arms retain full73raw/130voting inputs and H128rank32/5fold/3seed40pat4. NN0.8727302875 control versus0.8722568222 expanded; fixed tree halves0.8624405419 versus0.8622259940. All ten complete pipelines/30 actual models and both original scalers replayed. More632330 legitimate labels help, but both miss global; no qualifying refit.', 'experiments/upstream-split3-matched-resume/independent-verification.json', 'all complete actual ten-fold saved models, original eligible preprocessing and canonical prediction bits'),
    ('FRESH330 unused149 ridge diagnostic', '0.8619710828137934', '2026-11-20T11:47:32.052557+00:00', 'Fresh training', 'fresh_cv', 'codex', 'screen', 'All149 unused own predictors improve every ridge fold', '247 completed native pairs audited;93 existing canonical matches and5 exact pair aliases removed. All149 fixed additions improve same-row own181 ridge0.8622279136776032 to0.8619710828137934, with allfive folds improving. Full406-column operators retain every control bit; separate bounded/sequential readers replay all group statistics, eligible solves and stored predictions. Diagnostic only, not a neural/global or qualifying promotion. All556 features/557 inputs still require complete neural admission and fitting.', 'experiments/upstream-own330-native149-data/independent-verification.json', 'all original/new operator bits, full eligible-user ridge statistics and actual prediction replay; no neural or qualifying model'),
    ('FRESH181 recent15 ridge diagnostic', '0.8622279136776032', '2026-11-20T11:15:05.868204+00:00', 'Fresh training', 'fresh_cv', 'codex', 'screen', 'Fifteen own additions: modest linear gain; no overall promotion', 'Same canonical1128164 rows and original user folds. Full257/407-column operators, eligible group statistics and actual ridge predictions independently replayed. Original166ridge0.862491942587002; own176ridge0.8622734677900217; own181ridge0.8622279136776032. Last five additions improve only0.0000455541. All fifteen canonical payloads distinct from original166. No own181 neural or qualifying fit; development and recovered bests unchanged.', 'experiments/upstream-own181-recent15-data/independent-verification.json', 'full input and group-statistic replay, actual stored ridge bits; diagnostic only'),
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


def validate_own330_development(record,current_path,current,model_path,model):
    """Require the full557-input pipelines and separate final arithmetic proof."""
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path=EXPERIMENTS/'upstream-own330-mps/manifest.json'
    manifest=json.loads(manifest_path.read_text())
    records=model['records']
    if (digest(current_path)!=record['independent_verification_sha256'] or
        current['rmse']!=record['best_development_rmse'] or current['rows']!=1128164 or
        not current['all15_actual_saved_models_rehashed'] or not current['every_fixed_average_float32_bit_equal'] or
        current['qualifying_labels_read'] is not False or not current['improves_previous'] or
        current['complete_model_preprocessing_verification_sha256']!=digest(model_path) or
        current['verifier_sha256']!=digest(ROOT/'scripts/verify_upstream_own330_canonical.py') or
        model['state']!='complete_actual_own330_all_fits_original_statistics_scalers_and_saved_model_replays' or
        model['verifier_sha256']!=digest(ROOT/'scripts/verify_upstream_own330_mps.py') or
        model['manifest_sha256']!=digest(manifest_path) or current['manifest_sha256']!=digest(manifest_path) or
        model['fixed_half_tree154_rmse']!=current['rmse'] or model['fixed_half_sha256']!=current['prediction_sha256'] or
        [r['fold'] for r in records]!=list(range(5)) or sum(r['actual_models_reopened'] for r in records)!=15 or
        not all(r['actual_saved_pipeline_bitwise_equal'] and r['both_original_eligible_scalers_recomputed'] and
                r['all_original_normalized_input_bits_equal'] for r in records) or
        manifest['nn_inputs']!=557 or manifest['raw_predictions']!=330 or manifest['context_columns']!=90 or
        manifest['fixed_tree_weight']!=.5 or manifest['config']['nn_hidden']!=128 or manifest['config']['context_rank']!=32 or
        manifest['config']['seeds']!=[2026,7,13] or manifest['config']['nn_epochs']!=40 or manifest['config']['nn_batch']!=8192):
        raise ValueError('Own330 publication requires all15 actual557-input models and separate exact canonical scoring')
    for relative,expected in manifest['code_sha256'].items():
        if digest(ROOT/relative)!=expected:raise ValueError('Original full own330 numerical source changed')
    for r in records:
        receipt=ROOT/r['receipt_path']
        if digest(receipt)!=r['receipt_sha256']:raise ValueError('Actual original own330 fold receipt changed')
        for name,expected in r['outputs'].items():
            path=ROOT/'artifacts/upstream-own330-mps/nn'/str(r['fold'])/name
            if digest(path)!=expected:raise ValueError('Actual original model/scaler/vector changed')
    path=ROOT/'artifacts/upstream-own330-mps/fixed-half-tree154.f32'
    if digest(path)!=current['prediction_sha256']:raise ValueError('Actual canonical fixed-half vector changed')


def validate_own352_development(record,current_path,current,model_path,model):
    """Require the full579-input pipelines and separate final arithmetic proof."""
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path=EXPERIMENTS/'upstream-own352-temporal-mps/manifest.json'
    manifest=json.loads(manifest_path.read_text())
    records=model['records']
    if (digest(current_path)!=record['independent_verification_sha256'] or
        current['rmse']!=record['best_development_rmse'] or current['rows']!=1128164 or
        not current['all15_actual_saved_models_rehashed'] or not current['every_fixed_average_float32_bit_equal'] or
        current['qualifying_labels_read'] is not False or not current['improves_previous'] or
        current['complete_model_preprocessing_verification_sha256']!=digest(model_path) or
        current['verifier_sha256']!=digest(ROOT/'scripts/verify_upstream_own352_temporal_canonical.py') or
        model['state']!='complete_actual_own352_bounded_original_statistics_scalers_and_saved_model_replays' or
        model['verifier_sha256']!=digest(ROOT/'scripts/verify_upstream_own352_bounded_scalers.py') or
        model['manifest_sha256']!=digest(manifest_path) or current['manifest_sha256']!=digest(manifest_path) or
        model['fixed_half_tree154_rmse']!=current['rmse'] or model['fixed_half_sha256']!=current['prediction_sha256'] or
        [r['fold'] for r in records]!=list(range(5)) or sum(r['actual_models_reopened'] for r in records)!=15 or
        not all(r['actual_saved_pipeline_bitwise_equal'] and r['both_original_eligible_scalers_recomputed'] and
                r['all_original_normalized_input_bits_equal'] for r in records) or
        manifest['nn_inputs']!=579 or manifest['raw_predictions']!=352 or manifest['context_columns']!=90 or
        manifest['fixed_tree_weight']!=.5 or manifest['config']['nn_hidden']!=128 or manifest['config']['context_rank']!=32 or
        manifest['config']['seeds']!=[2026,7,13] or manifest['config']['nn_epochs']!=40 or manifest['config']['nn_batch']!=8192):
        raise ValueError('Own352 publication requires all15 actual579-input models and separate exact canonical scoring')
    for relative,expected in manifest['code_sha256'].items():
        if digest(ROOT/relative)!=expected:raise ValueError('Original full own352 numerical source changed')
    for r in records:
        receipt=ROOT/r['receipt_path']
        if digest(receipt)!=r['receipt_sha256']:raise ValueError('Actual original own352 fold receipt changed')
        for name,expected in r['outputs'].items():
            path=ROOT/'artifacts/upstream-own352-temporal-mps/nn'/str(r['fold'])/name
            if digest(path)!=expected:raise ValueError('Actual original model/scaler/vector changed')
    path=ROOT/'artifacts/upstream-own352-temporal-mps/fixed-half-tree154.f32'
    if digest(path)!=current['prediction_sha256']:raise ValueError('Actual canonical fixed-half vector changed')

    recovery_path=EXPERIMENTS/'upstream-own352-bounded-replay/plan.json'
    recovery=json.loads(recovery_path.read_text())
    if (model['recovery_plan_sha256']!=digest(recovery_path) or
        current['source_sha256'][str(recovery_path.relative_to(ROOT))]!=digest(recovery_path) or
        model['bounded_scaler_sha256']!=digest(ROOT/'scripts/upstream_exact_streamed_scaler.py') or
        not model['complete_original_statistics_bitwise_equal'] or not model['all_original_models_preserved_without_refitting'] or
        not recovery['all15_original_fits_rehashed'] or recovery['original_model_fitting_restarted'] or
        not all(r['actual_models_reopened']==3 for r in records) or sum(r['rows'] for r in records)!=1128164 or
        manifest['config']['nn_patience']!=4 or manifest['design_columns']!=428 or manifest['nonlinear_columns']!=578 or
        len(manifest['signal_indices'])!=354 or len(manifest['context_indices'])!=225 or
        sorted(manifest['signal_indices']+manifest['context_indices'])!=list(range(579))):
        raise ValueError('Own352 requires original complete statistics, all actual models and explicit bounded-replay provenance')
    for key,expected in recovery['source_sha256'].items():
        if key.endswith('.py') and digest(ROOT/key)!=expected:
            raise ValueError('Original bounded replay implementation changed')



def validate_own391_development(record,current_path,current,model_path,model,require_improvement=True):
    """Require complete matched391 original fits and separate canonical proof."""
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    variant=current['variant']
    if variant not in {'h128','h512'}:raise ValueError('Only originally declared matched391 arms permitted')
    report=EXPERIMENTS/('upstream-own391-paired-'+variant+'-mps')
    run=ROOT/'artifacts'/report.name
    manifest_path=report/'manifest.json';manifest=json.loads(manifest_path.read_text())
    records=model['records'];cfg=manifest['config']
    reference_path=EXPERIMENTS/'upstream-own352-bounded-replay/independent-canonical-verification.json'
    reference=json.loads(reference_path.read_text())
    data_path=EXPERIMENTS/'upstream-own391-paired-data/manifest.json'
    data_proof_path=data_path.parent/'independent-verification.json';data_proof=json.loads(data_proof_path.read_text())
    if (current_path!=report/'independent-canonical-verification.json' or model_path!=report/'independent-verification.json' or
        digest(current_path)!=record['independent_verification_sha256'] or
        record['verified_development_report_sha256']!=digest(current_path) or
        current['fixed_half_tree154_rmse']!=record['best_development_rmse'] or
        current['complete_actual_model_replay_sha256']!=digest(model_path) or
        current['verifier_sha256']!=digest(ROOT/'scripts/verify_upstream_own391_paired_canonical.py') or
        not current['all_five_original_folds_verified'] or not current['every_nn_and_fixed_half_float32_bit_equal'] or
        not current['neural_inputs_matched_B_on_both_queries'] or current['qualifying_labels_read'] is not False or
        current['goal_complete'] is not False or
        current['tree_control_retains_historical_A_B_realization_caveat'] is not True or
        current['actual_models_rehashed']!=sum(r['actual_models_reopened'] for r in records) or
        current['improves_previous']!=(current['fixed_half_tree154_rmse']<current['declared_global_reference_rmse']) or
        (require_improvement and not current['improves_previous']) or
        current['source_sha256']!=manifest['source_sha256'] or
        model['state']!='complete_independent_matched_B391_actual_neural_models_full_eligible_statistics_and_predictions' or
        model['variant']!=variant or manifest['variant']!=variant or
        manifest['state']!='declared_original_budget_matched_B391_neural_arm' or
        current['manifest_sha256']!=digest(manifest_path) or model['manifest_sha256']!=digest(manifest_path) or
        model['verifier_sha256']!=digest(ROOT/'scripts/verify_upstream_own391_paired_mps.py') or
        not model['complete_original_statistics_bitwise_equal'] or not model['all_original_models_preserved_without_refitting'] or
        not model['matched_B_known_and_qualifying'] or model['qualifying_labels_read'] is not False or
        model['production_started'] is not False or model['goal_complete'] is not False or
        model['nn_rmse']!=current['nn_rmse'] or model['fixed_half_tree154_rmse']!=current['fixed_half_tree154_rmse'] or
        model['fixed_half_sha256']!=current['prediction_sha256'] or
        [r['fold'] for r in records]!=list(range(5)) or sum(r['rows'] for r in records)!=1128164 or
        not all(0<=r['actual_models_reopened']<=3 and r['both_complete_eligible_scalers_independently_recomputed'] and
            r['all_normalized_input_bits_equal'] and r['actual_saved_pipeline_bitwise_equal'] for r in records) or
        manifest['raw_predictions']!=391 or manifest['design_columns']!=467 or manifest['nonlinear_columns']!=617 or
        manifest['nn_inputs']!=618 or manifest['fixed_tree_weight']!=.5 or not manifest['matched_B_known_and_qualifying'] or
        manifest['qualifying_labels_read'] is not False or len(manifest['signal_indices'])!=393 or
        len(manifest['context_indices'])!=225 or sorted(manifest['signal_indices']+manifest['context_indices'])!=list(range(618)) or
        cfg['nn_hidden']!={'h128':128,'h512':512}[variant] or cfg['context_rank']!=32 or
        cfg['seeds']!=[2026,7,13] or cfg['nn_epochs']!=40 or cfg['nn_patience']!=4 or cfg['nn_batch']!=8192 or
        manifest['declared_global_reference_rmse']!=reference['rmse'] or current['declared_global_reference_rmse']!=reference['rmse'] or
        (require_improvement and current['fixed_half_tree154_rmse']>=reference['rmse']) or
        manifest['source_sha256'][str(reference_path.relative_to(ROOT))]!=digest(reference_path) or
        manifest['input_manifest_sha256']!=digest(data_path) or
        manifest['source_sha256'][str(data_proof_path.relative_to(ROOT))]!=digest(data_proof_path) or
        data_proof['state']!='complete_independent_matched_B391_all_operators_statistics_ridge_and_stored_predictions' or
        data_proof['manifest_sha256']!=digest(data_path) or
        not data_proof['every_full_operator_statistic_solve_and_prediction_bit_equal'] or
        not data_proof['matched_B_known_and_qualifying'] or data_proof['qualifying_labels_read'] is not False):
        raise ValueError('Matched391 promotion requires every original fit/input/model replay and separate canonical score')
    for relative,expected in manifest['code_sha256'].items():
        if digest(ROOT/relative)!=expected:raise ValueError('Original matched391 numerical source changed')
    for relative,expected in manifest['source_sha256'].items():
        if relative.endswith('.py') and digest(ROOT/relative)!=expected:
            raise ValueError('Inherited matched391 numerical source changed')
    for row in records:
        path=ROOT/row['receipt_path'];receipt=json.loads(path.read_text())
        if (digest(path)!=row['receipt_sha256'] or receipt['outputs']!=row['outputs'] or
            receipt['fold']!=row['fold'] or receipt['manifest_sha256']!=digest(manifest_path) or
            len(receipt['chosen_iterations'])!=3 or not all(type(c)==int and 0<=c<=40 for c in receipt['chosen_iterations']) or
            sum(c>0 for c in receipt['chosen_iterations'])!=row['actual_models_reopened']):
            raise ValueError('Complete original matched391 selected budgets and receipts required')
        for name,expected in row['outputs'].items():
            if digest(run/'nn'/str(row['fold'])/name)!=expected:raise ValueError('Actual matched391 model/scaler/prediction changed')
    if len(manifest['controls'])!=1 or manifest['controls'][0]['id']!='tree154':
        raise ValueError('Original fixed tree154 control required')
    tree_path=manifest['controls'][0]['prediction']
    if digest(ROOT/tree_path)!=manifest['source_sha256'][tree_path]:raise ValueError('Original matched391 tree control changed')
    for name,expected in [('nn.f32',model['nn_sha256']),('fixed-half-tree154.f32',current['prediction_sha256'])]:
        if digest(run/name)!=expected:raise ValueError('Actual complete matched391 canonical prediction changed')


def validate_own330_qualifying(latest_path, latest):
    """Require original full models, known-only simplex and both frozen checks."""
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    for key, expected in latest['bindings'].items():
        if digest(ROOT/key)!=expected:
            raise ValueError('Own330 canonical qualifying binding changed')
    report=latest_path.parent
    cal=json.loads((report/'calibration.json').read_text())
    freeze=json.loads((report/'frozen.json').read_text())
    pre=json.loads((report/'independent-preaudit-verification.json').read_text())
    audit=json.loads((report/'independent-recovered-audit.json').read_text())
    primary='own7-own330-known-calibration'
    weights=latest['weights']
    if (latest['primary']!=primary or audit['primary']!=primary or
        cal['family_ids']!=['tree50','nn50','nn90','nn109','nn154','tree154','nn330'] or
        len(cal['families'])!=7 or len(weights)!=7 or min(weights)<0 or abs(sum(weights)-1)>1e-12 or
        weights!=cal['weights'] or weights!=pre['weights'] or
        max(abs(a-b) for a,b in zip(weights,latest['independent_simplex_weights']))>1e-5 or
        not latest['independently_solved_full_simplex'] or latest['simplex_supports_examined']!=127 or
        pre['simplex_supports_examined']!=127 or pre['qualifying_labels_read'] is not False or
        pre['state']!='complete_independent_own330_127_support_and_all_frozen_arithmetic_before_audit' or
        pre['freeze_sha256']!=digest(report/'frozen.json') or
        pre['calibration_sha256']!=digest(report/'calibration.json') or
        pre['source_sha256']!=cal['source_sha256'] or
        audit['preaudit_verification_sha256']!=digest(report/'independent-preaudit-verification.json') or
        audit['freeze_sha256']!=digest(report/'frozen.json') or
        audit['qualifying_outcomes_used_for_fitting_or_selection'] or
        freeze['plan_sha256']!=digest(report/'plan.json') or
        freeze['calibration_sha256']!=digest(report/'calibration.json') or
        freeze['qualifying_labels_read'] is not False or cal['qualifying_labels_read'] is not False or
        latest['known_score_is_nested_cross_validation'] is not False or
        cal['known_score_is_nested_cross_validation'] is not False or
        latest['prediction_sha256']!=freeze['predictions'][primary] or not latest['both_groups_improved']):
        raise ValueError('Original own330 known-only coefficient/freeze/audit chain required')
    if set(freeze['predictions'])!={'known-fit','fixed-half-own330-tree154',primary}:
        raise ValueError('All original own330 frozen known and qualifying vectors required')
    for name, expected in freeze['predictions'].items():
        if digest(ROOT/'artifacts/upstream-own330-assembly'/(name+'.f32'))!=expected:
            raise ValueError('Actual frozen own330 candidate changed')
    for key, expected in cal['source_sha256'].items():
        if key.endswith('.py') and digest(ROOT/key)!=expected:
            raise ValueError('Original own330 fitted implementation changed')
    path=EXPERIMENTS/'upstream-own330-full-refit/independent-verification.json'
    full=json.loads(path.read_text())
    if (digest(path)!=cal['source_sha256'][str(path.relative_to(ROOT))] or
        full['state']!='complete_original_own330_full_known_preprocessing_and_both_actual_model_queries' or
        not full['complete_known_statistics_scaler_normalized_bits_replayed'] or
        full['qualifying_labels_read'] is not False or full['full_iterations']!=[7,7,8] or
        [r['query'] for r in full['records']]!=['probex','qual'] or
        [r['rows'] for r in full['records']]!=[1128164,2817131] or
        not all(r['actual_models_reopened']==3 and r['all_prediction_bits_equal'] for r in full['records']) or
        latest['actual_new_full_models_reopened']!=3 or
        not latest['complete_known_statistics_scaler_normalized_bits_replayed']):
        raise ValueError('All three actual original full own330 models and preprocessing replays required')
    for name, expected in full['outputs'].items():
        if digest(ROOT/'artifacts/upstream-own330-full-refit'/name)!=expected:
            raise ValueError('Actual full own330 saved pipeline changed')
    return (latest['exact_known_and_full_stored_vectors_bitwise'] and pre['all_stored_candidate_arithmetic_bitwise'],
            latest['all_original_development_and_full_family_replays_verified'])


def validate_complete13_qualifying(latest_path,latest):
    """Require all13 original families, frozen fits and complete actual replay."""
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    for key,expected in latest['bindings'].items():
        if digest(ROOT/key)!=expected:raise ValueError('Complete13 original source/model/proof changed')
    report=latest_path.parent
    cal=json.loads((report/'calibration.json').read_text());freeze=json.loads((report/'frozen.json').read_text())
    pre=json.loads((report/'independent-preaudit-verification.json').read_text())
    audit=json.loads((report/'independent-recovered-audit.json').read_text());plan=json.loads((report/'plan.json').read_text())
    ids=['tree50','nn50','nn90','nn109','nn154','tree154',
        'native:upstream-split3-virtual-mps-context','native:upstream-attention2-virtual-mps-context',
        'native:upstream-knn3pair2-compact-mps-context','native:upstream-split3eight-compact-mps-context',
        'native:upstream-cf96-prefix2030-compact-mps-context','nn330','nn352-fold-bag']
    primary='complete13-simplex';weights=latest['weights']
    if (latest['state']!='complete_independent_complete13_calibration_and_frozen_group_scores' or
        latest['primary']!=primary or audit['primary']!=primary or plan['primary']!=primary or
        cal['family_ids']!=ids or [r['id'] for r in cal['families']]!=ids or
        len(weights)!=13 or min(weights)<0 or abs(sum(weights)-1)>1e-12 or
        weights!=cal['simplex_weights'] or
        max(abs(a-b) for a,b in zip(weights,latest['independent_simplex_weights']))>1e-5 or
        latest['independent_simplex_weights']!=pre['independent_simplex_weights'] or
        latest['simplex_supports_examined']!=8191 or pre['simplex_supports_examined']!=8191 or
        pre['state']!='complete_independent8191_support_and_ridge10_all_frozen_arithmetic_before_audit' or
        pre['ridge_equations']!=14 or pre['ridge_lambda']!=10. or cal['ridge_lambda']!=10. or
        not pre['all_stored_candidate_arithmetic_bitwise'] or pre['qualifying_labels_read'] is not False or
        pre['source_sha256']!=cal['source_sha256'] or latest['source_sha256']!=cal['source_sha256'] or
        pre['freeze_sha256']!=digest(report/'frozen.json') or
        pre['calibration_sha256']!=digest(report/'calibration.json') or
        pre['verifier_sha256']!=plan['verifier_sha256'] or
        freeze['plan_sha256']!=digest(report/'plan.json') or freeze['calibration_sha256']!=digest(report/'calibration.json') or
        audit['preaudit_verification_sha256']!=digest(report/'independent-preaudit-verification.json') or
        audit['freeze_sha256']!=digest(report/'frozen.json') or
        freeze['qualifying_labels_read'] is not False or cal['qualifying_labels_read'] is not False or
        audit['qualifying_outcomes_used_for_fitting_or_selection'] is not False or
        latest['known_score_is_nested_cross_validation'] is not False or
        latest['prediction_sha256']!=freeze['predictions'][primary] or not latest['both_groups_improved'] or
        not latest['independently_solved_full_simplex'] or not latest['all_original_source_and_fitted_model_files_rehashed']):
        raise ValueError('Original all13-family all8191-support freeze and audit chain required')
    if set(freeze['predictions'])!={'known-simplex','known-ridge10','complete13-simplex','complete13-ridge10'}:
        raise ValueError('Every original complete13 frozen known/control/primary vector required')
    for name,expected in freeze['predictions'].items():
        if digest(ROOT/'artifacts/upstream-complete13-calibration'/(name+'.f32'))!=expected:
            raise ValueError('Actual frozen complete13 candidate changed')
    full_path=EXPERIMENTS/'upstream-own352-fold-bag/independent-verification.json';full=json.loads(full_path.read_text())
    if (digest(full_path)!=cal['source_sha256'][str(full_path.relative_to(ROOT))] or
        full['state']!='complete_independent_all15_original_own352_fold_qualifying_models_and_uniform_bag' or
        full['original_models_reopened']!=15 or latest['actual_own352_qualifying_models_reopened']!=15 or
        not full['all428_578_input_operators_independently_reconstructed'] or not full['uniform_five_fold_bag_bitwise_equal'] or
        full['qualifying_labels_read'] is not False or [r['fold'] for r in full['records']]!=list(range(5)) or
        not all(r['actual_models_reopened']==3 and r['rows']==2817131 and r['every_qualifying_prediction_bit_equal'] for r in full['records'])):
        raise ValueError('All15 actual original qualifying models and full operators must replay')
    return (latest['exact_known_and_full_stored_vectors_bitwise'] and pre['all_stored_candidate_arithmetic_bitwise'],
            latest['all_original_development_and_full_family_replays_verified'])


def verify_own14_publication_sources(root,bindings):
    """Check recorded code and the new family's actual model/scaler/bag bytes."""
    prefixes=('artifacts/upstream-own391-paired-h128-mps/',
              'artifacts/upstream-own391-fold-bag/',
              'experiments/upstream-own391-paired-h128-mps/',
              'experiments/upstream-own391-fold-bag/')
    counts={name:0 for name in ('code',*prefixes)}
    for name,expected in bindings.items():
        relative=pathlib.Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Relative original source paths required')
        group='code' if relative.suffix=='.py' else next((p for p in prefixes if name.startswith(p)),None)
        if group is None:continue
        if hashlib.sha256((root/relative).read_bytes()).hexdigest()!=expected:
            raise ValueError('Recorded own14 code/model/scaler/bag source changed: '+name)
        counts[group]+=1
    if not all(counts.values()):raise ValueError('Complete code and new-family artifact/proof coverage required')
    return counts


def validate_own14_qualifying(latest_path,latest):
    """Admit the predeclared own14 primary through its complete frozen chain."""
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    report=latest_path.parent
    names=['plan.json','calibration.json','known-completion.json','frozen.json',
           'independent-preaudit-verification.json','independent-recovered-audit.json']
    if set(latest['bindings'])!={str((report/name).relative_to(ROOT)) for name in names}:
        raise ValueError('Complete own14 metadata binding set required')
    for key,expected in latest['bindings'].items():
        if digest(ROOT/key)!=expected:raise ValueError('Own14 frozen evidence changed')
    plan,cal,known,freeze,pre,audit=[json.loads((report/name).read_text()) for name in names]
    base_path=EXPERIMENTS/'upstream-complete13-calibration/independent-canonical-verification.json'
    base=json.loads(base_path.read_text());validate_complete13_qualifying(base_path,base)
    base_cal=json.loads((base_path.parent/'calibration.json').read_text())
    candidates=['own14-simplex','own14-ridge10','matched391-fixed-half']
    weights=cal['simplex_weights']
    if (latest['state']!='complete_independent_own14_full_source_models_coefficients_and_frozen_group_scores' or
        any(v['primary']!=candidates[0] for v in (latest,plan,freeze,audit)) or
        plan['families']!=14 or plan['candidates']!=candidates or
        cal['family_ids']!=base_cal['family_ids']+['nn391-matched-fold-bag'] or
        [r['id'] for r in cal['families']]!=cal['family_ids'] or cal['families'][:13]!=base_cal['families'] or
        len(weights)!=14 or any(not math.isfinite(w) or w<0 for w in weights) or abs(sum(weights)-1)>1e-12 or
        max(abs(a-b) for a,b in zip(weights,pre['independent_simplex_weights']))>1e-5 or
        len(pre['independent_simplex_weights'])!=14 or
        latest['simplex_supports_examined']!=16383 or pre['simplex_supports_examined']!=16383 or
        latest['ridge_equations']!=15 or pre['ridge_equations']!=15 or
        any(v['ridge_lambda']!=10. for v in (plan,cal,pre)) or
        pre['state']!='complete_independent_all16383_supports_ridge15_and_six_vectors_before_audit' or
        not pre['all_six_candidate_vectors_bitwise'] or not latest['all_six_vectors_independently_verified'] or
        not latest['full_inherited_source_closure_rehashed'] or latest['actual_new_qualifying_models_reopened']!=15 or
        latest['source_sha256']!=freeze['source_sha256'] or pre['source_sha256']!=freeze['source_sha256'] or
        any(freeze['source_sha256'].get(k)!=v for k,v in cal['source_sha256'].items()) or
        any(v['qualifying_labels_read'] is not False for v in (plan,cal,freeze,pre)) or
        any(v['qualifying_labels_used_for_fitting_or_selection'] is not False for v in (latest,audit)) or
        any(v['known_score_is_nested_cross_validation'] is not False for v in (latest,plan,cal,pre)) or
        latest['official_or_untouched_test'] is not False or audit['official_or_untouched_test'] is not False or
        pre['freeze_sha256']!=digest(report/'frozen.json') or pre['calibration_sha256']!=digest(report/'calibration.json') or
        pre['verifier_sha256']!=plan['verifier_sha256'] or
        freeze['plan_sha256']!=digest(report/'plan.json') or freeze['calibration_sha256']!=digest(report/'calibration.json') or
        freeze['known_completion_sha256']!=digest(report/'known-completion.json') or
        audit['preaudit_verification_sha256']!=digest(report/'independent-preaudit-verification.json') or
        dt.datetime.fromisoformat(pre['observed'])>=dt.datetime.fromisoformat(audit['observed']) or
        audit['state']!='complete_all_three_original_own14_frozen_candidate_audits' or
        [v['candidate'] for v in latest['results']]!=candidates or latest['results']!=audit['results'] or
        latest['groups']!=latest['results'][0]['groups'] or
        latest['prediction_sha256']!=freeze['predictions'][candidates[0]] or
        not all(latest['groups'][k]['rmse']<base['groups'][k]['rmse'] for k in ('source0','source1'))):
        raise ValueError('Full original own14 primary, preaudit and separate canonical evidence required')
    expected={'known-simplex','known-ridge10','known-fixed-half',*candidates}
    if set(freeze['predictions'])!=expected:
        raise ValueError('All six original own14 frozen vectors required')
    for name,expected_hash in freeze['predictions'].items():
        if digest(ROOT/'artifacts/upstream-own14-matched391-calibration'/(name+'.f32'))!=expected_hash:
            raise ValueError('Own14 frozen candidate vector changed')
    verify_own14_publication_sources(ROOT,freeze['source_sha256'])
    bag_path=EXPERIMENTS/'upstream-own391-fold-bag/independent-verification.json'
    bag=json.loads(bag_path.read_text())
    if (digest(bag_path)!=freeze['source_sha256'][str(bag_path.relative_to(ROOT))] or
        bag['state']!='complete_independent_matched391_actual_fold_models_all_query_streams_and_bag' or
        bag['actual_models_reopened']!=15 or bag['seed_slots']!=15 or bag['rows']!=2817131 or
        not all(bag[k] for k in ('all_original_model_and_input_streams_replayed','uniform_five_fold_bag_bitwise_equal',
                                'matched_B_raw_operators_on_both_queries','known_OOF_and_qualifying_bag_constructions_distinct')) or
        bag['qualifying_labels_read'] is not False):
        raise ValueError('Complete original own391 fifteen-model qualifying bag replay required')
    return True,True


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
    control_path=EXPERIMENTS/'upstream-own391-paired-h512-mps/independent-canonical-verification.json'
    control_rows=[r for r in MILESTONES if r[9]==str(control_path.relative_to(ROOT))]
    if control_rows:
        control=json.loads(control_path.read_text());model_path=control_path.parent/'independent-verification.json'
        proof_sha=hashlib.sha256(control_path.read_bytes()).hexdigest()
        if len(control_rows)!=1 or float(control_rows[0][1])!=control['fixed_half_tree154_rmse'] or control_rows[0][6]!='audit':
            raise ValueError('Negative H512 milestone must retain its independently verified score and control status')
        subject=dict(independent_verification_sha256=proof_sha,verified_development_report_sha256=proof_sha,
            best_development_rmse=control['fixed_half_tree154_rmse'])
        validate_own391_development(subject,control_path,control,model_path,json.loads(model_path.read_text()),require_improvement=False)
    expected_weights=[.5,.5]
    own330=kind=='complete_separate_original330_tree_half_arithmetic_and_full_canonical_score'
    own352=kind=='complete_separate_bounded_original352_tree_half_arithmetic_and_full_canonical_score'
    own391=kind=='complete_separate_matched_B391_canonical_neural_and_fixed_half_scores'
    complete_trajectory=kind=='complete_separate_complete_trajectory_canonical_neural_and_fixed_half_scores'
    specialized=own330 or own352 or own391 or complete_trajectory
    if own330:
        validate_own330_development(record,current_path,current,independent_path,independent)
    elif own352:
        validate_own352_development(record,current_path,current,independent_path,independent)
    elif own391:
        validate_own391_development(record,current_path,current,independent_path,independent)
    elif complete_trajectory:
        from complete_trajectory_publication import validate
        validate(ROOT,record,current_path,current,independent_path,independent)
    elif kind=='complete_fixed_post_study_consensus':
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
    if not specialized and (hashlib.sha256(current_path.read_bytes()).hexdigest()!=record['independent_verification_sha256'] or
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
    elif latest['state']=='complete_independent_own330_calibration_and_frozen_group_scores':
        mean_ok,replay_ok=validate_own330_qualifying(latest_path,latest)
    elif latest['state']=='complete_independent_complete13_calibration_and_frozen_group_scores':
        mean_ok,replay_ok=validate_complete13_qualifying(latest_path,latest)
    elif latest['state']=='complete_independent_own14_full_source_models_coefficients_and_frozen_group_scores':
        mean_ok,replay_ok=validate_own14_qualifying(latest_path,latest)
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
    """One daily minimum per evaluation series/author; retain the complete ledger."""
    preferred = set()
    for key in ['development_verification_path', 'latest_qualifying_verification_path']:
        matches = [row for row in milestones if row[9] == record[key]]
        if len(matches) != 1:
            raise ValueError('One exact evidence-bound overview milestone required: ' + key)
        preferred.add(matches[0][0])
    daily = {}
    for row in own_training_milestones(milestones):
        timestamp = dt.datetime.fromisoformat(row[2])
        day = timestamp.astimezone(ZoneInfo('America/Los_Angeles')).date().isoformat()
        key = (day, row[4], row[5])
        # Resolve equal scores deterministically, keeping evidence-bound labels.
        rank = (float(row[1]), row[0] not in preferred, -timestamp.timestamp(), row[0])
        if key not in daily or rank < daily[key][0]:
            daily[key] = (rank, row)
    return dict(ids=sorted(value[1][0] for value in daily.values()),
                days=sorted({key[0] for key in daily}),
                rule='Lowest recorded RMSE per Pacific day, evaluation series and author')


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
    for filename in ['research_journey.html','index.html']:
        (ROOT / filename).write_text(output)
    print(f"research_journey.html: {len(items)} milestones, {len(payload['runs'])} runtime curves")


if __name__ == "__main__":
    main()
