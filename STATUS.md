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
1. **New model families** (≈0.0007–0.001 each; about 4–6 needed).
2. Quality upgrades only in high-slope families (Bayes, GP/local).
3. Blend-aware selection.
4. The production pipeline.

No cloud GPUs (user decision). The estimate is ≈7–10 weeks on the M3 Max,
with a stop check at week 3 (≥0.0015 bank gain).

## In flight

| Job | Owner | Check |
|---|---|---|
| H8 full-bank comparison (choice-affinity column) | Codex | Control 50/50 verified at 0.86670512; treatment tree verified at 0.86675819; final treatment NN live (PID 32367); `artifacts/headroom/H8-full-20261031/runner.log` |
| S0 full-blender scoping, independent of the ridge proxy | Codex | All79 reference reproduction live; three tree folds complete. Original runner will stop before NN on changed narrative-doc hash; exact original document archived, administrative continuation prepared. `experiments/S0-document-rebind-verification-20261031.json` |
| E3 (λ=0.1 CE on A2h head), queued on GPU after H8 | Codex | `experiments/E3-plan-20261031.json`; starts when no other training is live |

## Next actions

1. **Claude:** audit model families the winners and the literature used that
   are absent here. Rank them by expected error-structure novelty and pick the
   first 2–3 to build (STRUCTURAL_PLAN lane 1). Waiting for the user's go-ahead.
2. When H8 and E3 finish, record their full-bank and P2 results in GOAL.md's
   closed bets and here.
3. **Codex:** finish S0 reference parity before interpreting its 25 other
   fixed arms. E3 also waits for S0 under its no-other-training condition.

## Decisions waiting on the user

- Go-ahead for the new-family audit (asked 2026-10-31).

## Recent results (newest first)

| Date | Result | Doc |
|---|---|---|
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
