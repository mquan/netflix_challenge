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

1. **Codex:** short-first phase **complete**: seven CPU models audited,
   S0 25/25 arms and final summary complete. Barrier released original guarded
   production queues on Nov 8 at **20:34 UTC**. Continue production retrain/export.
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
| Production GPU | Runner **71725**, trainer **55561**: A2h and A2f audited; A2e epoch 12/12, batch 1200. Controller resumed after short-run barrier. |
| S0 | Both lanes exited zero; **25/25** nonreference arms audited and complete 26-arm report verified, including all 156 bound output hashes. |
| CPU neural | Runner **71728** exited zero; all **7/7** fixed recipes complete/audited. |
| Native families | RBM runner **71730**, R3R trainer **68864** (epoch 43/75; R3 audited); dependencies runner 71732 exited zero, all four stages audited; joint runner **71734**, J6 completed/audited (first five joint stages complete); runner waits for normal memory. Remaining family controllers resumed at original guards. |
| Final bank/blender/export | Runner **71746** resumed, waits for all 82 audits. No final fit/export. |
| Short-first barrier | Checker **92690** exited successfully; all ten controller identities checked and SIGCONT completed after short-run prerequisites. `experiments/short-first-20261108/release.json`. |
| Idle-sleep assertion | `caffeinate -i -w 71746`, PID **71758**, releases on final runner exit. |

Full recovery plan/scripts and handles: `experiments/recovery-20261107/plan.json`
and `launch.json`. Resource guards retain original script-name signatures;
CPU/report recovery copies replace obsolete numeric lane PID checks.
Do not duplicate these processes. [P0_EXECUTION.md](P0_EXECUTION.md) holds protocols.

## Next actions

- Latest independent result check **2026-11-08 23:03 UTC**: **21/82** production
  audits match required metadata and all bound output hashes. Includes 2 GPU,
  7 CPU neural, 3 RBM, 4 dependency and 5 joint stages.
  `experiments/recovery-20261107/progress-20261108T2303Z.json`.
- Two active trainers: GPU A2e and RBM R3R. Joint queue waits at memory guard. Other family queues
  remain at original resource/data guards. No acceleration rewrite applied.
- S0 completed. Retained best remains **0.8668706424 dev CV**. Lowest observed
  substitution is 0.8668505622 (gain 0.00002008), but two folds worsen;
  no promotion. Random 64-column subsets average 0.86729399; 32-column
  subsets average 0.86830235. Substitution ratios are diagnostics, not future
  gain forecasts or evidence that deleting a whole family is safe.
- Production still requires 79 columns/eight support aliases, all 82 audits,
  final fit, two exact qualifying inferences and raw-order export of 2,817,131
  queries. AutoRec still needs MPS parity after the primary GPU queue.
- Latest resource check: memory pressure fluctuates normal/warning; latest 2. Heavy queues resumed
  only after short-run completion and two normal-memory checks 30 seconds apart.
- Latest ETA: **roughly 3–4 days remaining (Nov 11–12)** on the original
  schedule. Nominal critical-path accounting ≈57.5 h (RBM ≈17.6 h, Bayes
  ≈35.8 h, later stages/final allowance 4 h), with slowdown/failure uncertainty.
  Accelerated schedule remains unimplemented. BDS adds ≈3 h after production.
  No ETA for the quiz target. See P0_EXECUTION.md Runtime review.

## Decisions waiting on the user

None. Short-first sequencing is complete; moonshot stays on hold.

## Recent results (newest first)

| Date | Result | Doc |
|---|---|---|
| 11-08 | S0 all 25 arms complete; largest observed gain only 0.00002008 with two folds worse → no promotion. Seven CPU production models complete; barrier released, 19/82 production audits independently verified | experiments/short-first-20261108/results.json |
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
