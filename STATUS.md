# STATUS — start here

**Last updated:** 2026-10-31 by Claude and Codex. Every session updates this page
whenever state changes, and always before it ends. Keep it to one screen and
move detail into the linked docs.

## Goal and score

| | Value |
|---|---|
| Goal (user, 2026-10-31) | **quiz 0.8554** (winner; test 0.8567) ≈ **0.8628 dev CV** |
| Promoted best | **R9-LAT-A2h 0.8668706 dev CV** (≈0.8595 est. test, ≈9.8% over Cinematch) |
| Gap | **0.0041 dev CV** |
| Mapping | test ≈ dev CV − 0.0074 (Stage C offset, measured twice). Estimates only; no qualifying labels |

## Current plan

Current authorized work:
1. **Codex:** retrain the retained bank on production data and build the final export.
2. **Claude:** [moonshot sequence model](MOONSHOT_REBUILD.md), M0 engineering;
   its GPU pilot follows Codex's production GPU queue.
3. **Codex:** finish the fixed S0 full-blender scoping experiment.

The user permanently closed new-family pilots after L1 failed; see
[FAMILY_AUDIT.md](FAMILY_AUDIT.md) and [STRUCTURAL_PLAN.md](STRUCTURAL_PLAN.md).
Retained-family proxy gains do not forecast gains from untried families.
No cloud GPUs (user decision). There is no validated ETA for reaching the target.

## In flight

| Job | Owner | Check |
|---|---|---|
| Production retrain P0 | Codex | A2h fixed 12-epoch MPS run live: epoch 5 live; first four epochs complete; trainer 32347, runner 32318, session 91805. Seven exact archived-source GPU recipes queued serially with audits; 233 inputs frozen. `experiments/P0-GPU-launch-20261031.json` |
| Production family handoffs | Codex | Nine managed bash queue runners launched and verified live at resource guards; zero new full trainers or stage directories. CPU/RBM/dependency/joint/kNN/AutoRec/Bayes/kernel/user-kNN PIDs and sessions: `experiments/P0-managed-queues-launch-20261031.json`. Do not duplicate them. |
| Final bank/blender/export handoff | Codex | Managed bash waiter PID 75657/session 29645 live at guard; requires all 82 audits and family process exit. No final fit or export started. `experiments/P0-FINAL-launch-20261031.json`. Do not duplicate. |
| S0 full-blender scoping, independent of the ridge proxy | Codex | 10/25 complete and audited (first- and second-seed n1/n4/n16/n32/n64). Latest average 0.8673388, output hashes/five folds/score checked. Lane 0 third-seed n1 neural (43550/session 96236); lane 1 third-seed n4 neural (40399/session 83214). Full synthesis waits for all 25 arms. `experiments/S0-diversity-s20261101-n64-full-verification-20261031.json` |

## Next actions

1. **Codex:** monitor and audit the running production/S0 stages. All family
   queues and the final runner are launched behind their frozen guards; do not
   duplicate them. Full protocol, evidence and managed handles:
   [P0_EXECUTION.md](P0_EXECUTION.md).
   - Planned coverage: **79 columns + eight support aliases**.
   - Actual production completion: **0/82 stage audits**; final fit/export unstarted.
   - Final runner waits for all audits, queue exit and stable normal memory,
     then resolves the bank, fits retained medians, verifies two exact inference
     runs and exports all 2,817,131 queries in original raw order.
   - AutoRec still requires primary GPU completion and a successful MPS parity
     check. All current score/provenance limitations remain in the runbook.
   Last runtime check **2026-11-01 06:27 UTC**: A2h epoch 5/batch 3300;
   epoch 4 complete (480,189 users; 8,052.9 training seconds). S0 second-seed n64 complete and automatically audited,
   all output hashes/five folds/score checked. Both third-seed n1/n4 trees completed all
   five folds; neural successors PID 43550/40399 are live in lanes 0/1 (one/two folds complete).
   Ten fixed arms audited. All ten guarded waiters and watcher remain live;
   no completed production stage or failed runner. This check is a verified
   wait: trainer 32347 and both neural stages are live;
   all 82 required production audit paths remain absent. Memory pressure fluctuates
   between normal (1) and warning (2); latest level 2. Trainer progress continues,
   no new trainers.
   Process inventory: three active training Python processes belong to Codex
   (32347 production GPU; 43550/40399 S0 neural). Five other Python-based
   Grafana MCP service processes are infrastructure, not experiment trainers.
   System sleep interrupted progress while on battery; now on AC and all three
   trainers active again. Power/process evidence: `experiments/P0-sleep-interruption-20261101T061022Z.json`.
   Filesystem check: about 106 GiB currently free.
   Temporary idle-sleep assertion PID 82488/session 69265 is tied to final
   runner 75657 and releases on its exit. Verified on AC power;
   `experiments/P0-idle-sleep-assertion-20261031.json`.
   Latest full process/guard-log check: `experiments/P0-live-health-20261101T035640Z.json`.
1b. **Claude:** moonshot (MOONSHOT_REBUILD.md). M0 engineering on CPU now; GPU pilot M1 after Codex's E3 and production GPU work. Stop rule: close if M1 fails both gates.
2. **Codex:** monitor both S0 CPU lanes and memory, verify completed arms,
   then report all fixed diversity/substitution results. E3 and its audit are
   terminal; neither P2 gate passed, so no full-training follow-up.
   Complete-result reporter is frozen in `experiments/S0-report-binding-20261031.json`;
   five tests and real-reference/partial-experiment checks pass. Its bash
   handoff PID 94435/session 80182 is verified waiting for both lane runners;
   do not launch a duplicate report. `experiments/S0-report-handoff-20261031-launch.json`.
   Completed-arm audit watcher is also live: PID 77645/session 64149, checks
   every 60 seconds and audits only after producer exit. Its actual partial-
   experiment check leaves all five prior receipts unchanged; invalid receipt
   rejected. It never trains, retunes or synthesizes early. Do not duplicate or
   manually race its future receipt writes. `experiments/S0-arm-monitor-launch-20261031.json`.

## Decisions waiting on the user

- None. User decisions 2026-10-31: production → Codex; moonshot → go.

## Recent results (newest first)

| Date | Result | Doc |
|---|---|---|
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
