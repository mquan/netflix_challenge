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
| Production retrain P0 | Codex | A2h fixed 12-epoch MPS run live: epoch 3 live; epoch 2 complete; trainer 32347, runner 32318, session 91805. Seven exact archived-source GPU recipes queued serially with audits; 233 inputs frozen. `experiments/P0-GPU-launch-20261031.json` |
| Production family handoffs | Codex | Nine managed bash queue runners launched and verified live at resource guards; zero new full trainers or stage directories. CPU/RBM/dependency/joint/kNN/AutoRec/Bayes/kernel/user-kNN PIDs and sessions: `experiments/P0-managed-queues-launch-20261031.json`. Do not duplicate them. |
| S0 full-blender scoping, independent of the ridge proxy | Codex | 5/25 complete and audited (first-seed n1/n4/n16/n32/n64); n64 average 0.8673321. Lane 0 second-seed n4 tree (71100/session 96236); lane 1 second-seed n1 neural (56799/session 83214). Full synthesis waits for all 25 arms. `experiments/S0-diversity-s20261031-n64-full-verification-20261031.json` |

## Next actions

1. **Codex:** monitor production MPS and prepare final bank resolution/export.
   All 79 columns/supports and the final fitted blender remain required.
   Managed queues launched and waiting behind these guards:
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
   - Item-kNN: 28 recipes / 29 columns and six support aliases; waits for all
     16 joint/dependency audits. `P0-KNN-20261031-plan.json`: 278 inputs /
     279 checksum checks pass; ten audit/resource/handoff tests and all 28 native
     label-isolation fixtures pass. Ordered alias manifest waits for all outputs.
   - AutoRec: two MPS trajectories, epochs 1 and 44 with snapshot14; requires all
     seven primary GPU audits, primary process exit and a successful MPS parity check.
     `P0-AUTOREC-20261031-plan.json`: 255 inputs / 256 checksum checks pass.
     Both budgets and 512-unit/two-layer architecture pass tiny CPU training/reload audits;
     actual MPS prepare rejects the absent MPS-specific receipt. Waiting for primary GPU completion.
   - Bayes: four stages / three columns, waits for all 12 RBM audits and S0,
     CPU neural and RBM process exit, plus normal memory. Exact four/five-thread
     settings preserved; a tiny historical-thread test must pass before training.
     `P0-BAYES-20261031-plan.json`: 249 inputs / 250 checksum checks pass;
     five audit/resource tests, eight single-thread native fixture handoffs and
     all three full historical checkpoint scans pass. B4 resumes the newly
     produced sweep-32 parent, retaining 144 samples at sweep 160. Queue waiting;
     historical-thread runtime test remains pending. `P0-BAYES-preflight-20261031.json`.
   - GP/local: four invocations / six columns, waits for all 32 Bayes/item-kNN
     audits, their process exit and normal memory; consumes newly audited J3.
     `P0-KERNEL-20261031-plan.json`: 252 inputs / 253 checksum checks pass;
     five audit/resource tests and eight actual kernel fixture handoffs pass.
     Complete GP exports and all six local epochs repeat exactly; query-label
     isolation passes. Missing J3 audit rejects prepare before creating output.
     Queue waiting. `P0-KERNEL-preflight-20261031.json`.
   - User-kNN: new K1 rank256/41 epochs at six threads, then original retained
     inference binary at three threads; waits for all four kernel audits and CPU
     process exit. One prediction plus two support aliases. `P0-USERKNN-20261031-plan.json`:
     250 inputs / 251 checksum checks and five audit/resource tests pass. Ten
     native fixture commands cover full one-thread K1 budgets, exact original/
     rebuilt inference, label isolation and nonzero support/corrections. Queue
     waiting. `P0-USERKNN-preflight-20261031.json`.
   Frozen plans cover all **79/79 columns plus eight support aliases**, verified
   in original blender order by `P0-all-family-planned-coverage-20261031.json`.
   This is planning coverage: **0/82 required production-stage audit files exist**.
   Full outputs and final fitted blender/export remain outstanding. All nine
   family runners are live at guards; only primary GPU and both S0 lanes are
   currently training. Prelaunch verified 486 distinct frozen inputs and an
   acyclic runner-dependency graph. PIDs/sessions and initial waiting state are
   recorded in `P0-managed-queues-launch-20261031.json`; never launch duplicates.
   The separate latent basis remains frozen.
   Final bank resolver now requires all 82 matching audits and all bound output
   hashes, preserves original order/settings/ready latent+metadata files, and
   fills exactly 79 predictions/eight support aliases. Five synthetic complete-
   bank contract tests pass; real incomplete production bank rejects without
   creating `resolved-bank.json`. Engineering only, not production completion.
   `P0-bank-resolver-verification-20261031.json`. Final fit/export handoff pending.
   Submission serialization passes five tests, including text/float32 roundtrip
   and rejection of changed identities, labels, counts or bounds. Full real-data
   order check verifies 2,817,131 queries and 17,470 movie headers against raw
   qualifying IDs/dates, dense-user mapping and production rows. No submission
   written; model/inference provenance handoff remains required.
   `P0-submission-order-verification-20261031.json`.
   Final handoff now binds bank/model hashes before each command, audits fixed
   budgets and outputs afterward, requires two exact inference runs, then binds
   raw-order text export. Four tests pass with real tiny CLI fitting/inference/
   export and synthetic production receipts. Real final fit rejects the absent
   resolved bank before creating started metadata. Full fixed-budget final
   runner still needs freezing/launch. `P0-final-handoff-verification-20261031.json`.
   AutoRec's historical provenance limitations remain explicit; MPS engineering verification
   is still pending. Recent logs and active workers confirm all three live jobs continue.
   Full production outputs remain required; the development gap is unchanged.
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
