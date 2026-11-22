---
status: paused-for-session-restart
branch: research/full-upstream-reproduction
timestamp: 2026-11-22T07:44:32.822094+00:00
files_modified:
  - FULL_REPRODUCTION_PROGRESS.md
  - experiments/full-upstream-reproduction/inventory.json
  - experiments/upstream-cfnade-parallel/build.log
  - experiments/upstream-production-bank/inventory.json
---

## Working on: Netflix independent ten-percent improvement

### Summary

User requested a good stopping point to restart with another model. Agent goal
is being paused for that restart; detached scientific jobs remain running.
Resume only when the user resumes the work. This file is the handoff, not proof
of goal completion. Reinspect live processes and current artifacts first.

Actual repository: `/Users/quan/workspace/netflix_upstream_repro`.
User's original cwd `/Users/quan/workspace/netflix_prize` is an older workspace.
Branch: `research/full-upstream-reproduction`.
Remote: `git@github.com:mquan/netflix_challenge.git`.
Python: `.venv/bin/python` (3.10), macOS MPS.
Numerical-component checkpoint commit: `c1a7bce`.
Four pre-existing dirty tracked files listed above are unrelated: NEVER edit,
stage, revert or commit them. There are ~1300 untracked experiment artifacts;
use explicit staging, never git add -A. No new subagents without user request.

### Goal and verified scores

Achieve and independently verify primary RMSE strictly BELOW **0.85725 on BOTH**
community-recovered qualifying groups using freshly trained own predictions and
known-only locally fitted weights. No downloaded pretrained numerical features
or qualifying outcomes in fitting/model selection. Public source/schema allowed.
Recovered audits are adaptive, not official or untouched Test. Preserve the
full data, family count, folds, seeds and mathematical/scientific budgets.

Best verified independent primary remains own16:
**0.8603409488199526 / 0.8613420938772931** (about9.57038% on the worse group).
`experiments/upstream-local-rbm-blend-deterministic-metal/independent-canonical-verification.json`
SHA256 `d646e5b449052126fdfae87978578d741a4e58a85ce0a8ec524406a2caa9d1e0`.
Development best **0.8588630883643809**, NN alone0.8590060211471172.
Development canonical SHA256
`aa9385d900f2c8a4844c1fc4d2413fddb418340c8e07c37472624ba4912f23b8`.
No new development/recovered score has been admitted this session. Target unmet.

### Live jobs at last actual ps poll (2026-11-22 around07:42 UTC)

- Neural owner24039, watchdog24046; **worker27425 actively running replay_scalar**
  (about42minutes elapsed and36minutes CPU, stateRs). All five scalar folds
  and15 actual MPS models completed at06:59UTC. Counts selected per fold:
  [6,11,7], [9,5,5], [5,8,6], [9,9,5], [7,8,8].
  Control `experiments/upstream-rbm400-neural-execution-reference-v4`;
  phase wrappers `.../upstream-rbm400-neural-execution-results-reference-v4`.
  Scalar report `experiments/upstream-own-rbm400-scalar-h128-mps-reference-v4`.
  Full independent replay then canonical scalar, full latent fit/replay/canonical,
  and known-only selection are automatic. Original fivefold/threeseed/max40epoch
  arms714/864inputs,225/375contexts, preserved. Frozen238source snapshot.
- Selected NN bag owner25553/watchdog25567 waits complete both-arm studies and
  original selection: `experiments/upstream-rbm400-bag-execution-reference-v4`.
  Frozen265sources,17checks; all2817131Q rows/5fold3slots, NN-only matched pair.
- Own17 owner26473/watchdog26484 waits the NN-only pair:
  `experiments/upstream-rbm400-own17-execution-reference-v4`.
  Frozen299sources,16checks; all131071supports/18equations/sixvectors/3audits.
- Known CatBoost owner66664/watchdog66674 waits completed original own17 primary:
  `experiments/upstream-full-context-catboost-execution-v1`.
  Frozen364files,89checks. Starts both complete full-budget known tree arms ONLY
  if primary own17 misses strict target on either group; does not use comparison
  candidates to trigger or Q outcomes to select arms/hyperparameters.
- NEW full selected-tree Q pair owner75543/watchdog75552 is LIVE WAITING:
  `experiments/upstream-full-context-catboost-bag-execution-v1`.
  Frozen383files,119checks. Four phases prepare/infer/replay/admit; no scientific
  tree pair root has started. Both known tree arms, canonical proofs and
  known-only selection must finish first. If own17 primary already passes both
  targets, it verifies original not-needed/full audit and skips the tree pair.

No jobs were killed, restarted or signaled for the session restart. Detached
owners are PPID1 and watchdogs hold inherited lane descriptors. Revalidate PIDs
AND start times/commands; do not trust old status files alone. Observation
expiry is not terminal and never justifies restarting a scientific job.

### Resource and immutable-source rules

Original scientific baseline must remain byte-identical:
`experiments/upstream-factored-rbm-execution-deterministic-metal/first-scientific-baseline.json`
SHA `689232383581ed51cbdffc8a1da5ae047ee94d690ea51b64bccbbab20b447168`.
Compressor baseline2526543872, swap baseline2073097666.56 bytes.
Retain both primary/GPU lanes,24GiB aggregate physical AND RSS independently,
16GiB MPS/tree-family caps,3GiB CPU calibration caps, compressor<5GiB
(admission<4.5GiB), swapgrowth<=128MiB,18.75GiB disk floor +complete remaining
outputs +reserve. Same PID/group/frames/RNG/batch/arrays on pause/resume;
three consecutive full reservations10sec apart before readmission. No baseline
reset, relaxed limits, observer expansion, automatic model retries or shortened
budgets. Original observer `upstream_research_footprints_native_original_scope.py`.
Free disk about21GiB. Preserve all own models/data/predictions/archives/scratch.
Prior public-only cleanup already completed; do not repeat it.

Every reviewed/frozen source/doc/protocol expected hash must remain unchanged.
Use NEW versioned files if an actual failure requires a correction. In particular
all existing full-context CatBoost components and both live owner source sets
are frozen. Check live trees via `check_sources(read(REPORT/'plan.json'))`.
DO NOT regenerate the old known owner's declaration/source_files/frozen_plan:
its glob would now see later additions and invalidate its historical review.
The same saved-plan rule is preferred for the new pair owner.

### Completed prerequisites and corrections (do not redo)

Deterministic Metal RBM400 known15/Q15 pair and actual saved-model replay:
`experiments/upstream-factored-rbm-gpu400-deterministic-metal/paired-verification.json`
SHA `3e8d2ecdbb40d0c953f98b24601c2e84a9353bb45ca23d74fea26797a814199a`.
Recovered retained-cache pair with only18 approved exact aliases,5205references:
`experiments/upstream-rbm400-reference-replay/paired-verification.json`
SHA `485edc77ff298b96bae12019df755a7a123ecc08d3e3d95633fce5eaf1695a0b`.
Complete CPU context manifest SHA
`3119fb2bc14de9703876c87022b0fa8a53cc42d38ca7bee7aedfcc3deba90ecd`.
Original CPU verifier source-dictionary key mismatch is already corrected by a
separate versioned source-alignment replay (no array changes):
`experiments/upstream-rbm400-context-source-alignment-replay/independent-verification.json`
SHA `47c137371b06c02f7c840c3a8bd0c31755e6807ecd8bad3e5b5d65f19f3751d9`.
Both fullquery operators,2815ridge equations and all original held bits verify.
Neuralv3 stale proof-path failure happened BEFORE fitting. V4 uses one correct
CPU proof locator throughout prepare/load/admission and producer_v2. Old failed
namespaces/logs/learned state are preserved, not edited or retried.

### Full-context CatBoost and owned pair already implemented

Original plan `OWN_FULL_CONTEXT_CATBOOST_FOLLOWUP_PLAN.md` is frozen.
CPU CatBoost1.2.10, depth8,lr.03,l2reg10,max2000/pat100,border64,
Bernoulli.8,rsm.9,threads2, fivefold/threeseed2026/7/13+fold, innergroups,
ridge1000 eligible baseline. Training-only quantization, all inner selections
before fresh full refits, zero slots divided by3. Whole family16GiB.
Both full raw arms/10ridge solves/all models/OOFbits/separate known CV scores
must finish before known-only tree selection, scalar tie within1e-12.

New pair owner plan `OWN_FULL_CONTEXT_CATBOOST_OWNED_BAG_PLAN_V1.md`, review
`experiments/upstream-full-context-catboost-owned-bag-protocol-v1/verification.json`,
actual launch/runtime proof in same directory.119combined checks pass91.407sec.
Output reserve768MiB =128vectors+512metadata+64staging+64reserve. All7vectors
are about69MiB, so64MiB vector inventory would have been insufficient; corrected
BEFORE freeze. Source bootstrap registration race also fixed BEFORE freeze:
admission waits in the same frame for the exact still-live captured source.

Sources stay isolated/Torch-free in tree receiver roots. Each fresh child waits
for byteG before Torch/import/data; actual parent captures/admission; complete
wire footer+EOF THEN actual parent zero-exit reaping before publication. Metadata
file SHA (newline) and logical encoded metadata SHA are distinct. Do not conflate.
Q inference8192blocks, all5fold/15slots, ordered f64(seedpred/3),baseline+correction,
clipf32 fold, orderedf64 fivefoldsum/5->clipf32bag. Matched known is TREE-ONLY.
Separate replay solves10ridges, opensallmodels, uses separate native loops and
compares every known/fold/bag bit. Finalchecks link exact source requests,
authorizations, lifetime/parent/group, receipts and all inputs/outputs. Standalone
component proofs remain unadmitted until complete external owner phase/resource
receipts and terminal-family release. No recovered outcomes are read here.

### Own18 current implementation and next work

Frozen own18 provider gate (commit063886e):
`scripts/upstream_tree_own18_sources_v1.py`, matching test,
`OWN_TREE_OWN18_SOURCE_PROGRESS_V1.md`,
`experiments/upstream-tree-own18-sources-protocol-v1/verification.json`.
It retains all17 original families in order, appends matched full tree pair,
uses the same tree pair as standalone control, requires all4phase/4resource
proofs + owned completion/release + no live scientific roles. Four focused
20/23row fixtures pass; upstream owner/data/numerical/resource admission mocked.
Its387source expectations are FROZEN. Do not edit it or its reviewed doc/test.
Existing own17 proof loader imports Torch; the later blend-provider process may
therefore import Torch. It does NO CatBoost train/predict. Do not add a blanket
no_torch assertion there; tree inference roots retain their separate isolation.

JUST CHECKPOINTED (c1a7bce), components NOT YET OWNED/PRODUCTION-ADMITTED:
- `scripts/upstream_tree_own18_calibration_v1.py`
- `scripts/verify_upstream_tree_own18_arithmetic_v1.py`
- `scripts/verify_upstream_tree_own18_calibration_v1.py`
- `scripts/test_upstream_tree_own18_calibration_v1.py`
Four focused numerical checks pass8.931sec, including real full18-family
known-only fit, ALL262143supports,19ridge equations, sixfrozenvectors,
preaudit-before-label gate, two separate fixed/canonical audits on80/61synthetic
rows, coefficient KKT/add/remove/nonunique cases, reclip lowerbound, FDidentity,
and rehashed vector mutation rejection. These tests mock only upstream sources
and label delivery; no production data/model/score is admitted.
Command:
`env OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m unittest discover -s scripts -p test_upstream_tree_own18_calibration_v1.py -v`
Log `/tmp/netflix-own18-calibration-tests.log`.

The numerical component uses existing source/IO rules, SLSQP ftol1e-17 unchanged,
active-face KKT polish, ridge10 unpenalized intercept. Independent compensated
products, allsupports and every output bit are checked, with guard callbacks
inside products/support enumeration/output blocks. New tree control is RAW;
prior ANN family still NPY. During drafting, AST copying omitted contextmanager
decorators; both vector/columns decorators were corrected before passing tests.
Regex dimension conversion temporarily changed ftol; restored1e-17 before tests.

NEXT on resume, after polling live jobs:
1. Reinspect these four new numerical files and add needed tests/review evidence.
2. Write `OWN_TREE_OWN18_CALIBRATION_PLAN_V1.md` (currently intentionally missing).
3. Implement `scripts/upstream_tree_own18_execution_v1.py` and owned worker;
   numerical source_declaration deliberately requires that future owned plan,
   known gate/ownership, monitor/baseline/prerequisites and review protocol
   `experiments/upstream-tree-own18-owned-protocol-v1/verification.json`.
   Do NOT call production known/freeze functions with a no-op guard to bypass it.
4. Full5phases known/freeze/preaudit/audit/canonical, CPU3GiB, original lanes,
   baseline/whole-output/same-frame policy. Freeze/review before explicit queued
   launch. Wait actual owned complete tree pair; no early numeric work.
5. Verify all18families,262143supports,19equations,sixvectors BEFORE Qlabels;
   exactly3fixed candidates, primary `full-tree-own18-simplex`, comparison
   `full-tree-own18-ridge10`, control `selected-full-context-tree-control`.
6. Continue measured own iterations/remote returns if strict primary target
   remains unmet. Never shrink scope or mark complete on tests/partial progress.

### Remote laptop and website

Remote branch `research/remote-weekend-execution`, latest inspected commit
`aa3eac9a4d278f7a819697160f05b8c09f57f664`, report06:34UTCNov22:
TX30012/24, RBM80012/25, RBM4006/15 in known train_3->probe_3. No completed
receipt/export/archive/upload or admitted pair. Snapshot only, not live SSH.
Remote side resumed its original jobs in e52830b; no control was issued here.
Evidence `experiments/upstream-remote-weekend-progress-0634/{progress-original,verification}.json`.
Remote host M3Max96GiB has its own policy; do not apply local16/24/5GiB blindly.
Training cutoffNov23 14:00UTC(Mon6amPacific), transfer/verification16:00UTC(8am).
Read-only fetch remote execution/recovery/handoff branches to inspect newer
reports; do not restart/stop remote jobs from a lagged Git snapshot.

Site https://mquan.github.io/netflix_challenge/ updated and fully verified.
Pages commit `4ffec250228d587ef54e1adb513b568b9af70649`,859explicitstaticfiles,
102ownmilestones,25dailyrepresentatives,zero public-bank scores. Logos and target
label below line retained. No new score/plot point this turn.
Whitelist `experiments/website-owned-tree-full-query-queue-publish-files.json`.
Live proof `experiments/website-owned-tree-full-query-queue-live-verification.json`
(all859+ordinaryroot match; completed07:37UTC). Do not repeat this verification
without a new publication/change. Own18 numerical component checkpoint is not
published; no production update needed until meaningful progress.
Pages worktree `/Users/quan/workspace/netflix_own_pages_update` clean detached.
Publish only explicit own static whitelist, no scripts/data/model/rawlogs/public
bank docs. Keep existing one representative per Pacific day/evaluation/author;
plot only complete verified metrics. Full own16 closure validation takes2-3min.

### Session handoff notes

User permits autonomous iteration, code fixes, commits/pushes, site updates;
no repeated permission prompts. Do not send messages to others. No new subagents
unless user/applicable instructions explicitly ask. No AGENTS found earlier.
Keep progress updates under60seconds during active work. Pause now is explicit
user direction for a model/session restart, NOT a scientific failure or goal
completion. Resume the original objective without redefining it.

Skill used for this handoff: context-save,
`/Users/quan/.agents/skills/gstack/context-save/SKILL.md`.

### Final observation superseding the earlier running-worker snapshot

At2026-11-22T07:44:50UTC, actual worker27425 was stateTs, CPU37:50.30,
with owner24039 and watchdog24046 still live. Official neural status is
`same_worker_RAM_preserved_waiting_original_readmission`, phase replay_scalar,
stable_samples0. The agent did NOT signal or stop this scientific worker.
Its07:44:39 resource request shows free_bytes20440154112 and required output
reserve319607647; original18.75GiB floor +reserve is20452266847, about11.55MiB
above sampled free disk. Compressor2580004864 and swap1972371456 bytes;
no GPU/disk accounting error. See
`experiments/upstream-rbm400-neural-execution-reference-v4/replay_scalar-worker-resource-request.json`.
The same original numerical frame/RAM is preserved. Reinspect actual status,
free disk and supervisor readmission first on resume. Do not restart the model,
relax the reserve/floor, reset the baseline or delete own learned/data artifacts.
This is an observed resource wait, not a terminal failure or target achievement.
