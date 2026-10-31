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

[STRUCTURAL_PLAN.md](STRUCTURAL_PLAN.md), with ridge-proxy week-0 scoping done
(independent full-blender validation in flight) and
lanes re-ranked:
1. **New model families** (unproven prospective gain; first pilot L1 failed).
2. Quality upgrades only in high-slope families (Bayes, GP/local).
3. Blend-aware selection.
4. The production pipeline.

The ≈0.0007–0.001 proxy gains describe retained families, not expected gains
from untried families; [FAMILY_AUDIT.md](FAMILY_AUDIT.md) identifies this
survivorship bias. No cloud GPUs (user decision). The proposed program costs
≈7–10 weeks on the M3 Max; this is not a validated ETA for reaching the target.
The week-3 stop check remains ≥0.0015 bank gain.

## In flight

| Job | Owner | Check |
|---|---|---|
| S0 full-blender scoping, independent of the ridge proxy | Codex | Exact all79 parity verified; runners 39539/39615 and worker groups verified live. First n4 arm has tree folds 0/1/2 complete; n1 has fold 2 complete and workers continue; neither arm complete. 25 fixed arms total; `experiments/S0-lanes-20261031-launch.json` |
| E3 (λ=0.1 CE on A2h head), fixed two-epoch P2 treatment | Codex | Epoch 1 complete (validation 0.8650174); epoch 2 live, PID 38985/session 51948; batch progress in `experiments/E3-select-20261031.log` (3900/6168 at elapsed 74m). Frozen gate handoff PID 60202/session 35306 verified waiting; both S0 worker groups also live. Handoff audits completion and runs both gates once after trainer+runner exit; `experiments/E3-gate-handoff-20261031-launch.json` |

## Next actions

1. **Claude:** L1 closed (no gain). Before building the next family (feature GBDT base model, co-clustering), re-examine the plan, since the per-family estimate is survivorship-biased (FAMILY_AUDIT.md). **Decision needed from the user** (see below).
2. **Codex:** monitor E3 and its frozen gate handoff; inspect the automatic
   `scripts/e3_candidate_gate.py` result once complete (seven fixture tests pass).
   Do not manually launch a duplicate gate or retry partial output.
3. **Codex:** monitor both S0 CPU lanes and memory, verify completed arms,
   then report all fixed diversity/substitution results; retain E3 priority.

## Decisions waiting on the user

- After L1's null result: continue with the next new-family pilots, or step back and reassess the target? (asked 2026-10-31)

## Recent results (newest first)

| Date | Result | Doc |
|---|---|---|
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
