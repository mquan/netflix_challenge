# STATUS — start here

**Last updated:** 2026-11-08 by Codex. Update on every state change and before ending a session.

## Goal and score

| | Value |
|---|---|
| Goal | **quiz 0.8554** ≈ **0.8628 dev CV** |
| Promoted best | **R9-LAT-A2h 0.8668706 dev CV** (≈0.8595 estimated test) |
| Gap | **0.0041 dev CV** |
| Mapping | test ≈ dev CV − 0.0074; estimates only, no qualifying labels |

## Current plan

1. **Codex:** **short runs first (user, 11-08)**: finish S0 and the existing
   short CPU neural queue; hold new GPU/downstream-family launches until all
   short outputs, audits and S0 report complete. Current A2f continues.
   Then resume the original guarded production queues.
2. **Claude:** [moonshot](MOONSHOT_REBUILD.md). **M0 done 11-07**: M1 config (d128 L4, 3072 tokens, MLP head) peaks at 7.3 GB on MPS, ≈1.3 h/epoch on P2 with the GPU shared. No moonshot process running. **On hold (user, 11-08): start nothing; nothing queued.** Honest odds of M1 passing both gates ≈10–15%. Resume only on user request.
2b. **Claude:** BDS blender data-scaling diagnostic **queued** (PID 12307, starts only after production's final runner exits; ≈3 h CPU). Plan/rule: `experiments/BDS-20261108-plan.json`; result lands in `artifacts/blend-scaling/BDS-summary.json`. Production schedule review: [PRODUCTION_SPEEDUP_REVIEW.md](PRODUCTION_SPEEDUP_REVIEW.md).
3. **Codex:** complete the fixed S0 full-blender scoping experiment.

New-family pilots remain permanently closed. No cloud GPUs. No validated ETA for the target.

## In flight — recovered 2026-11-07

All old experiment processes had disappeared; logs stopped Nov 2 around 15:06 UTC
without terminal records. Interrupted outputs/logs are preserved under
`artifacts/production/P0-20261031/interrupted-20261107/`.
Recovery verified **621 distinct frozen inputs**, unchanged recipes and bash syntax.
Original frozen files remain unchanged. New processes are detached from the tool session.

| Job | Current handles/state |
|---|---|
| Production GPU | Runner **71725**, trainer **66571**: A2h completed all 12 epochs and audited (9.18 h elapsed); A2f now epoch 3/10, batch 2600. Controller **71725 held (SIGSTOP)** to prevent next GPU recipe; current A2f trainer continues. |
| S0 lane 0 | Runner **71723**, tree coordinator **65139**: final S0 arm, quality-transformer tree, 3/5 folds logged; neural stage follows. |
| S0 lane 1 | Runner **71724** exited successfully: all lane 1 commands complete. |
| Family queues | Family controllers held except the short CPU queue: CPU 71728, RBM 71730, dependencies 71732, joint 71734, item-kNN 71736, AutoRec 71738, Bayes 71740, kernel 71742, user-kNN 71744. CPU queue completed/audited MH3 and MT0; MT1 complete/audited; 3/7 short CPU stages audited. CPU controller 71728 enabled, waiting for normal memory before MT2. Other controllers SIGSTOP-held, not terminated. |
| Final bank/blender/export | Controller **71746** SIGSTOP-held as part of short-first barrier. No final fit/export. |
| S0 audits/report | Monitor **71751** confirms **24/25** receipts; reporter **71754** waits for both lane processes. |
| Idle-sleep assertion | `caffeinate -i -w 71746`, PID **71758**, releases on final runner exit. |

Full recovery plan/scripts and handles: `experiments/recovery-20261107/plan.json`
and `launch.json`. Resource guards retain original script-name signatures;
CPU/report recovery copies replace obsolete numeric lane PID checks.
Do not duplicate these processes. [P0_EXECUTION.md](P0_EXECUTION.md) holds protocols.

## Next actions

- Short-first holds applied to ten own controllers; no active trainers stopped.
  Release checker **92690** verified live/waiting under `experiments/short-first-20261108/`.
  It resumes original controllers only after all seven CPU audits, all
  25 S0 receipts, complete S0 summary, short runner exit and stable normal
  memory; controller command/start-time identity must still match.

- Latest verified progress **2026-11-08 16:42 UTC**: A2h, MH3 and MT0 complete;
  all three production receipts match required metadata and output hashes.
  A2f/MT1 and the last S0 tree arm are the three active training jobs.
  `experiments/recovery-20261107/progress-20261108.json`.
- Production completion: **4/82 audits** (A2h + MH3/MT0/MT1). Final bank still requires **79 columns
  + eight support aliases** and export of **2,817,131** raw-order queries.
- AutoRec requires primary GPU completion and successful MPS parity verification.
- S0 has **24/25** audited arms; all bound output hashes independently checked; no partial synthesis. Frozen monitor/report
  remain responsible for later receipts and complete-experiment synthesis.
- Latest power/resource check: AC, battery 80% charging, normal memory pressure
  (1). GPU A2f and final S0 tree active; short CPU queue enabled and memory-guarded.
- Runtime review supersedes the earlier uncalibrated 1–1.5 week allowance:
  historical critical-path accounting is ≈64 h before uncertainty; rough current
  schedule **3–5 days**, optimized **48–72 h target, unvalidated**. Concurrent
  independent Bayes chains offer ≈16.7 h ideal saving; overlap contention and
  peak memory must be checked. **Acceleration changes remain unimplemented; user now prioritizes short runs.** Details and
  verification plan: P0_EXECUTION.md "Runtime review" and
  `experiments/runtime-audit-20261108/evidence.json`. No ETA for target score.
- No new promoted quality result. Production predicts unlabeled qualifying rows;
  its training loss cannot establish a hidden-score gain. S0 synthesis waits
  for its final arm. Retained best remains **0.8668706 dev CV**.

## Decisions waiting on the user

None. Short-first sequencing is active; moonshot stays on hold.

## Recent results (newest first)

| Date | Result | Doc |
|---|---|---|
| 11-07 | Moonshot M0 gate met: M1 config 7.3 GB peak with opt-in pad_row_tokens + checkpoint (N1 recipe 57→7.1 GB) | MOONSHOT_REBUILD.md |
| 11-07 | Moonshot M0 step 2: opt-in MLP head + warmup/cosine schedule; N1 parity holds; 10 tests pass | MOONSHOT_REBUILD.md |
| 10-31 | Moonshot M0 step 1: N1 fork + query-chunked attention; equality and byte-for-byte parity tests pass | MOONSHOT_REBUILD.md |
| 10-31 | User: production retrain assigned to Codex; moonshot (one large sequence model) approved | MOONSHOT_REBUILD.md |
| 10-31 | E3: shared fold 0 worsens 0.00000131; five-fold gain 0.00001277 < 0.0003, two folds worse → closed; both comparisons independently reproduced | experiments/E3-independent-verification-20261031.json |
| 10-31 | L1 local low-rank: standalone 0.8924 P2, but bank RMSE worsens by 0.0000023 → closed; retained-family proxy gains do not forecast new-family gains | FAMILY_AUDIT.md |
| 10-31 | New-family audit: build order LLORMA-style local low-rank → feature GBDT base model → co-clustering mixture | FAMILY_AUDIT.md |
| 10-31 | S0 all79 reproduces retained R9 exactly: tree/NN/average max difference 0.0; saved CV 0.8668706424 | experiments/S0-full-parity-verification-20261031.json |
| 10-31 | H8 full bank 0.86648484; all five folds improve R9, but matched-control gain 0.00022028 < fixed 0.0003 → not promoted | experiments/H8-full-independent-comparison-20261031.json |
| 10-31 | Scoping: new families ≈0.0007–0.001 each; blend-per-standalone slopes Bayes 0.14, GP 0.10, A2 0.028, others ≈0 | STRUCTURAL_PLAN.md |
| 10-31 | N1 full bank −0.00017, fold 1 worse → not promoted | NEXT_EXPLORATIONS.md |
| 10-31 | Headroom map done: user shift absorbed by Stage C (A7), movie side empty, proxies 0 | HEADROOM.md |
| 10-31 | Closed: E1, E5, A7, SO1, H9 (none passed) | GOAL.md "Closed bets" |

## Where things live

- **GOAL.md:** goal, operating rules (don't interfere with the other session,
  thread caps, pre-registered gates), claims table, closed bets.
- **STRUCTURAL_PLAN.md:** current plan, timeline and scoping data.
- **NEXT_EXPLORATIONS.md / HEADROOM.md:** per-bet protocols and results.
- **Older history:** PLAN_TO_TARGET.md, ROUND*.md, RESULTS.md.
  RESUME_CHECKPOINT.md is superseded by this page.
- **Commits:** every step is committed and pushed on `codex/local-context`,
  authored as `Quan Nguyen <mquannie@gmail.com>`.
