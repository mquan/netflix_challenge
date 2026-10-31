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
| Production retrain P0 | Codex | A2h fixed 12-epoch MPS run live: epoch 2 live; first epoch took 52.3 min; trainer 32347, runner 32318, session 91805. Seven exact archived-source GPU recipes queued serially with audits; 233 inputs frozen. `experiments/P0-GPU-launch-20261031.json` |
| S0 full-blender scoping, independent of the ridge proxy | Codex | 4/25 complete and audited (first-seed n1/n4/n16/n32); n16 average 0.8710975, n32 average 0.8680112. Lane 0 now first-seed n64 tree (coordinator 96739/session 96236); lane 1 second-seed n1 tree (63963/session 83214). Full synthesis waits for all 25 arms. `experiments/S0-diversity-s20261031-n16-full-verification-20261031.json` |

## Next actions

1. **Codex:** monitor production MPS and prepare remaining native families.
   All 79 columns/supports and the final fitted blender remain required.
   Prepared, not launched:
   - CPU neural: seven fixed recipes, waits for a free S0 slot and normal memory.
     `P0-CPU-neural-20261031-plan.json`.
   - RBM: 12 stages / nine retained columns, waits for both S0 lanes and all
     seven audited CPU neural stages. `P0-RBM-20261031-plan.json`.
   - Dependencies: baseline, graph256, exact graph64 prefix, learned neighborhood;
     waits for both S0 lanes. One thread; combined planned maximum is ten.
     `P0-DEPS-20261031-plan.json`: 263 inputs / 264 checksum checks pass;
     five tests and eight actual native fixture handoffs pass, including query-label isolation.
   - Joint/integrated: 12 stages / 13 columns, one thread; waits for both S0 lanes
     and all four dependency-stage audits. `P0-JOINT-20261031-plan.json`:
     297 inputs / 298 checksum checks pass; six audit/resource tests pass.
     J4 restarts epoch numbering; integrated resumes require audited parent sidecars.
   Native coverage and build provenance are in `P0-native-dependency-inventory-20261031.json`
   and `P0-joint-stage-build-map-v2-20261031.json`. Item-kNN now maps all 29 occurrences
   to 28 exact recipes, new production base/graph inputs and six support aliases (three rules):
   `P0-knn-column-map-20261031.json`. Output auditor passes six tests, including archived
   CLI defaults. All 28 recipes pass two query-label fixtures: 56 kNN stage audits and 128
   pipeline commands; predictions/support are label-independent, support is nonzero, and
   all three support rules differ (`P0-knn-fixture-verification-20261031.json`).
   Next freeze guarded kNN execution.
1b. **Claude:** moonshot (MOONSHOT_REBUILD.md). M0 engineering on CPU now; GPU pilot M1 after Codex's E3 and production GPU work. Stop rule: close if M1 fails both gates.
2. **Codex:** monitor both S0 CPU lanes and memory, verify completed arms,
   then report all fixed diversity/substitution results. E3 and its audit are
   terminal; neither P2 gate passed, so no full-training follow-up.
   Complete-result reporter is frozen in `experiments/S0-report-binding-20261031.json`;
   five tests and real-reference/partial-experiment checks pass. Its bash
   handoff PID 94435/session 80182 is verified waiting for both lane runners;
   do not launch a duplicate report. `experiments/S0-report-handoff-20261031-launch.json`.

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
