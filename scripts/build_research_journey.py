#!/usr/bin/env python3
"""Build the self-contained research journey from documented milestones and run events."""
import base64
import datetime as dt
import hashlib
import json
from html import escape
import pathlib
import re


ROOT = pathlib.Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"

# These are editorial milestones, not a claim that every experiment was a new best.
# Calendar dates before timestamped run events are documentation upper bounds.
MILESTONES = [
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
    ("FRESH86 context", ".8600637147575518", "2026-11-19T02:08:38+00:00", "Fresh training", "fresh_cv", "codex", "development", "Own trained predictors + context NN", "Our best fully verified fresh-training development result. Five user folds over 1,128,164 known-ProbeX rows; 15 neural models replay bit-for-bit and independent scoring verifies the fixed tree/NN average. Public source code is reused, with no downloaded pretrained prediction features. Adaptive development; qualifying evaluation is pending. This population differs from the earlier original-Probe/public-bank series.", "experiments/upstream-cf12-context/independent-verification.json", "verified result recorded"),
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


# Same recovered-label population throughout; never mix these with Probe CV.
QUALIFYING = [
    ("388-input ridge", "01:03", .8582647011016761, .8574231002840432),
    ("58-input ridge", "02:19", .8578789678524765, .8567927057358324),
    ("58-input NN2", "02:22", .8578011759258087, .8567189009050679),
    ("83-input ridge", "02:31", .8575942363558479, .8565118571755685),
    ("83-input NN2", "02:33", .8575337426262905, .8564604689103762),
    ("619-input ridge", "02:42", .8571903480630092, .8561542082012507),
]


def qualifying_svg():
    """One audited score sequence, reused by the site and README preview."""
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 440" role="img" aria-labelledby="audit-title audit-desc">',
             '<title id="audit-title">Recovered qualifying audit: both groups cross 0.85725</title>',
             '<desc id="audit-desc">Six candidate audits on November 16 UTC. Source1 falls from 0.8582647 to 0.8571903; source0 from 0.8574231 to 0.8561542. Adaptive community evaluation, not official Netflix scoring. Horizontal positions are candidate order.</desc>',
             '<rect width="960" height="440" fill="#181d29"/>']
    def text(x, y, value, color="#a8b0c2", size=14, anchor="start"):
        parts.append(f'<text x="{x}" y="{y}" fill="{color}" font-family="Arial,Helvetica,sans-serif" font-size="{size}" text-anchor="{anchor}">{escape(value)}</text>')
    def y(score):
        return 325-(score-.8560)/.0025*255
    text(80, 30, "Recovered qualifying RMSE · lower is better", "#f2f1ed", 18)
    text(80, 53, "Candidate order · audit times: Nov 16 UTC / Nov 15 Pacific", size=13)
    for tick in (.8560, .8565, .8570, .8575, .8580, .8585):
        yy=y(tick)
        parts.append(f'<path d="M80 {yy} H875" stroke="#4b5364" opacity=".5"/>')
        text(70, yy+5, f"{tick:.4f}", size=13, anchor="end")
    parts.append(f'<path d="M80 {y(.85725)} H875" stroke="#baaaff" stroke-dasharray="6 5"/>')
    text(875, y(.85725)-9, "10% target: 0.85725", "#baaaff", 14, "end")
    for group, color, label in ((2, "#70d9d0", "source1"), (3, "#f6c26b", "source0")):
        coords=[(100+i*150, y(row[group])) for i,row in enumerate(QUALIFYING)]
        points=" ".join(f"{x},{yy}" for x,yy in coords)
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
        for (x,yy),row in zip(coords,QUALIFYING):
            parts.append(f'<circle cx="{x}" cy="{yy}" r="5" fill="{color}"><title>{escape(row[0])} · {label} · {row[group]:.10f} · {row[1]} UTC</title></circle>')
        text(850, coords[-1][1]+24, f"{label}: {QUALIFYING[-1][group]:.7f}", color, 15, "end")
    for i,(name,time,*_) in enumerate(QUALIFYING):
        text(100+i*150, 358, name, "#f2f1ed", 13, "middle")
        text(100+i*150, 378, time+" UTC", size=12, anchor="middle")
    text(80, 413, "Adaptive recovered-label audit; different population and training protocol from Probe CV.", size=13)
    parts.append('</svg>')
    return "\n".join(parts)


def fresh_status_html():
    record=json.loads((EXPERIMENTS/'fresh-reproduction-status.json').read_text())
    if record['best_development_rmse']!=.8592720612630157 or record['qualifying_status']!='audited_miss':
        raise ValueError('Review the site narrative before publishing a changed result')
    candidate=EXPERIMENTS/'upstream-rbm50-tree-latent90-mps/verification.json'
    development=json.loads(candidate.read_text())
    check_path=candidate.parent/'independent-mean-check.json';checked=json.loads(check_path.read_text())
    if (hashlib.sha256(candidate.read_bytes()).hexdigest()!=record['independent_verification_sha256'] or
        development['state']!='complete_verified_fixed_combination' or not development['improves_previous'] or
        development['rmse']!=record['best_development_rmse'] or not development['fixed_average_bitwise_equal'] or
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
    proof=EXPERIMENTS/'upstream-first85-completion-audit/verification.json'
    verified=json.loads(proof.read_text())
    if (hashlib.sha256(proof.read_bytes()).hexdigest()!=record['first85_completion_verification_sha256'] or
            verified['state']!='complete_independent_completion_audit' or verified['goal_achieved'] or
            not verified['all_saved_models_independently_replayed'] or
            not verified['fixed_equal_average_bitwise_verified'] or
            verified['groups']['test']['rmse']!=.8628067765436453):
        raise ValueError('The published first qualifying miss requires its actual independent proof')
    latest=EXPERIMENTS/'upstream-rbm50-production/verification.json'
    qualified=json.loads(latest.read_text())
    if (hashlib.sha256(latest.read_bytes()).hexdigest()!=record['qualifying_verification_sha256'] or
            qualified['state']!='complete_standalone_nn_frozen_vector_and_score_verification' or
            qualified['groups']['source1']['rmse']!=.8619141823430468 or
            not qualified['standalone_vector_matches_saved_family_bitwise'] or
            qualified['qualifying_labels_used_for_fitting_or_selection'] or qualified['both_groups_below_target']):
        raise ValueError('The new own107/RBM50 qualifying best requires its exact predeclared frozen-vector evidence')
    rows=''.join(f'<tr><td>{escape(row["work"])}</td><td>{escape(row["state"])}</td></tr>' for row in record['work'])
    latest_path=EXPERIMENTS/'upstream-rbm50-production/verification.json'
    latest=json.loads(latest_path.read_text())
    if (hashlib.sha256(latest_path.read_bytes()).hexdigest()!=record['latest_qualifying_verification_sha256'] or
            latest['groups']!=record['latest_qualifying_groups'] or latest['both_groups_below_target']):
        raise ValueError('Review the latest qualifying audit and target narrative before publishing')
    return f'''<section id="fresh" aria-labelledby="fresh-title">
      <div class="section-head"><div><div class="section-num">Current work / own trained predictions</div><h2 id="fresh-title">Reproducing the result ourselves</h2></div></div>
      <p>Snapshot: <time datetime="{escape(record['observed'])}">{escape(record['display_time'])}</time>. These are recorded training states, not a live dashboard.</p>
      <p>The verified development best improved from <strong>0.8594554 to0.8593662, then0.8592721 RMSE</strong>. The latest fixed half mean combines our expanded107/50 tree with the larger107/all90 GPU context NN. All15 actual saved neural models replay bitwise on the Apple GPU; all15 saved trees retain their actual replays. Both families' ten originating fold receipts and models were rehashed, and separate stored-mean and compensated scoring checks agree on all1,128,164 known-ProbeX rows and462,858 users. Four folds improve and one worsens; lower aggregate RMSE is promoted. The full-known NN fit and actual saved-model replay are complete; the CPU tree refit is active, preserving the exact294-input tree and334-input NN schemas. <strong>The fresh-training 10% qualifying target remains unmet.</strong></p>
      <p>The new best independently verified own-training qualifying result is the <strong>standalone own107+50 context NN:0.8609266 on source0 and0.8619142 on source1</strong>, improving the preceding NN1030.8609917 /0.8619939. The NN-only choice and epoch budgets14/14/13 were fixed from known-ProbeX evidence before fitting and new qualifying outcomes. Actual saved-model inference replay passed, then the exact family vector was frozen before canonical-label access. Full group counts, model/scaler/source bindings and separate compensated scores verify both groups. The worse group improves <strong>9.5103%</strong> over the Cinematch reference, leaving0.0046642 RMSE to the strict0.85725 target. No qualifying outcomes fit or select weights. The paired parent retains previously disclosed differences from cached development. Both original latent models are now complete20/20 and all90 support components are verified. The larger hidden128/rank32 GPU neural study and its selected tree combination are now complete in development. The selected qualifying refit is active and the whole110 raw-predictor study is queued; neither has a new qualifying score.</p>
      <div class="ledger-wrap"><table class="ledger"><thead><tr><th>Work</th><th>Recorded progress</th></tr></thead><tbody>{rows}</tbody></table></div>
      <p>Public source implementations are reused; all prediction features in this effort are trained by us. Qualifying outcomes are excluded from fitting and model selection. The final vector must be frozen and replayed before both community-recovered groups are independently scored below 0.85725. These are not official Netflix or untouched Test results.</p>
      <p><a href="PROGRESS.md">Full progress record</a> · <a href="RBM50_QUALIFYING_RESULT.md">Latest own-training qualifying result</a> · <a href="FIRST85_QUALIFYING_RESULT.md">First own-training qualifying result</a> · <a href="experiments/upstream-first85-completion-audit/verification.json">Independent full-model audit</a> · <a href="CFNADE12_MATCHING_TREE_PROGRESS.md">Verified development result</a> · <a href="REMOTE_RECOVERY_PROGRESS.md">Remote recovery</a> · <a href="READY_OUTPUT_COVERAGE.md">Public feature coverage</a> · <a href="SPLIT3_PILOT_PLAN.md">Original split3 pilot</a> · <a href="CONTEXT91_PRODUCTION_PLAN.md">Context91 production</a> · <a href="CONTEXT91_QUALIFYING_RESULT.md">Preceding context91 audit</a> · <a href="CF48_103_CONTEXT_RESULT.md">Own103 context result</a> · <a href="LATENT90_SUPPORT_RESULT.md">All90 original support components</a> · <a href="LATENT90_MPS_CONTROL_PLAN.md">Larger GPU neural study</a> · <a href="LATENT90_MPS_RESULT.md">Latest development result</a> · <a href="WARM59_SNAPSHOT_PLAN.md">Next59 own snapshot features</a> · <a href="WARM166_MPS_PLAN.md">Own166 GPU study</a> · <a href="experiments/fresh-reproduction-status.json">Snapshot evidence</a></p>
    </section>'''


def main():
    items = []
    for ident, score, time, strategy, metric, credit, status, title, note, source, basis in MILESTONES:
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
    payload = dict(milestones=items, runs=[run_curve(*run) for run in RUNS], competition=COMPETITION)
    template = (ROOT / "research_journey.template.html").read_text()
    marker = "__JOURNEY_DATA__"
    if template.count(marker) != 1:
        raise ValueError("Expected one data insertion marker")
    output = template.replace(marker, json.dumps(payload, separators=(",", ":")))
    output = output.replace('__FRESH_REPRODUCTION__',fresh_status_html())
    output = output.replace("__QUALIFYING_GRAPH__", qualifying_svg())
    (ROOT / "assets/qualifying_progress.svg").write_text(qualifying_svg() + "\n")
    for name, filename, color in (("claude", "claude-code.svg", "#D97757"), ("codex", "codex.svg", "#b6a5ff")):
        icon = (ROOT / "assets" / filename).read_text().replace('fill="currentColor"', f'fill="{color}"')
        data_url = "data:image/svg+xml;base64," + base64.b64encode(icon.encode()).decode()
        output = output.replace(f"__{name.upper()}_LOGO__", data_url)
    (ROOT / "research_journey.html").write_text(output)
    print(f"research_journey.html: {len(items)} milestones, {len(payload['runs'])} runtime curves")


if __name__ == "__main__":
    main()
