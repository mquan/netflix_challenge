# Standing goal: match the Grand Prize winning score

Set by the user on 2026-10-30 as "reach 10%"; **raised by the user on
2026-10-31 to the actual winning score**: "the actual winning score is .855, I
want to hit that number." This session keeps running and trying new bets
until the goal is reached or the bets run out.

## Target

- **Primary: quiz RMSE 0.8554**, BellKor's Pragmatic Chaos's winning
  quiz score. Its test score was 0.8567.
- Mapped to our metric with the measured Stage C offset (test ≈ dev CV − 0.0074,
  measured twice; [PLAN_TO_TARGET.md](PLAN_TO_TARGET.md)):

  | Target | Dev-CV equivalent | Gap from 0.8668706 |
  |---|---:|---:|
  | 10% prize threshold (test 0.85725) | ≈0.86465 | 0.0022 |
  | winner, test 0.8567 | ≈0.8641 | 0.0028 |
  | **winner, quiz 0.8554 (goal)** | **≈0.8628** | **0.0041** |

  The same 0.0074 offset is applied to quiz and test. Our offset is
  calibrated on the probe → qualifying relationship, and quiz and test are
  halves of qualifying. The winner's own quiz–test difference (0.0013) shows
  this split is itself about ±0.001 uncertain.
- How the current best maps: 0.86687 − 0.0074 ≈ 0.8595 estimated test, i.e.
  1 − 0.8595/0.9525 ≈ 9.8% better than Cinematch.
- The gap can close from either side: a better dev-CV bank, or a larger
  Stage C retraining offset (measured with the Phase 0 rehearsal). Every
  claimed gain must be measured under its own fixed protocol.
- Qualifying labels are not available locally. Any score is an **estimate**
  from dev CV plus the measured offset, never a verified hidden-test score.
  The full production retrain has not been run end to end yet.

## Operating rules

1. **Do not stop, edit or interfere with the other session's work.** That
   covers SO1 (`scripts/selection_objective_*.py`, `experiments/SO1-*`), N1
   (`.claude/worktrees/honest-residual-chains`) and any file frozen in their
   plans (`experiments/SO1-plan.json`, `SO1-full-plan.json`,
   `N1-resume-*-plan.json`). Don't rebuild the shared `build/netflix`, which
   their frozen scripts call; build separate binaries instead.
2. **The GPU belongs to the other session's queue** while it runs. This
   session uses CPU only until that queue finishes or the user says otherwise.
3. **SO1 waits for a moment with no other Python process.** Long runs from
   this session are therefore C++ driven by bash, never the Python
   `scripts/experiment.py` wrapper. Short Python jobs (blends, weight files)
   run only while an SO1 stage is actively training (`experiments/SO1-resume-*.status.json`
   shows a running stage), never in the gap between its stages.
4. **At most 3 bets run in parallel**, keeping this session's total CPU
   threads at about 10 or fewer so the other session's data loading isn't
   starved. Check `ps` and memory before each launch.
5. **Every bet has a fixed protocol and gate written down before any score**
   (in [NEXT_EXPLORATIONS.md](NEXT_EXPLORATIONS.md)). No sweeps after reading
   the probe. Promotion of a dev-CV bank still requires all five folds to
   improve.
6. Commit and push this session's own files after each step, authored as
   `Quan Nguyen <mquannie@gmail.com>`, on `codex/local-context`, without a PR.

## Strategy

**Current plan (2026-10-31): [STRUCTURAL_PLAN.md](STRUCTURAL_PLAN.md).** It has four gated lanes: winner-grade base models, a properly scaled transformer, blend-aware selection, and the end-to-end production pipeline, with week-0 scoping first. The earlier program below (headroom map, gate, ledger) is complete.

Work is chosen by the program in [STRATEGY.md](STRATEGY.md): (A) a headroom map of which information sources could close the gap, (B) one shared gate plus a ledger, (C) one long-lived sequence-model line with a shared GPU queue. New one-off bets pause until A is done.

## Claims (in progress, per session)

Production execution protocols and managed handles: [P0_EXECUTION.md](P0_EXECUTION.md).
Current runtime state remains in [STATUS.md](STATUS.md).

| Work | Owner | State |
|---|---|---|
| **Production retrain (STRUCTURAL_PLAN lane 4)** | **Codex — assigned by the user 2026-10-31** | P0 GPU queue live (A2h epoch 4, PID 32347/session 91805). Nine family bash waiters plus final bash waiter PID 75657/session 29645 are managed and live; do not duplicate (`P0-managed-queues-launch-20261031.json`, `P0-FINAL-launch-20261031.json`). All 79 columns/eight supports mapped; 0/82 production-stage audits. Final frozen plan: 500 inputs/501 checks, waits for all audits/process exit/normal memory, then resolves bank, fits retained medians, runs two exact inferences and raw-order export. Raw-order verification matches 2,817,131 rows/17,470 headers. No final fit/export started. Next: monitor and audit guarded transitions/results. Full outputs and target remain outstanding |
| Moonshot: one large sequence model (MOONSHOT_REBUILD.md) | this session (Claude) | M0 engineering started 2026-10-31 (CPU only until the GPU queue frees) |
| Lane 1 new-family audit (reading; no compute) | this session (Claude) | **done 2026-10-31**: FAMILY_AUDIT.md. Build order: LLORMA-style local low-rank → feature GBDT base model → co-clustering mixture |
| L1 LLORMA-style local low-rank pilot (CPU, C++) | this session (Claude) | **closed 2026-10-31: +0.0000023 beyond bank11+N1 (gate 0.0010)** |
| Structural plan week 0: scoping (diversity curve, quality→blend conversion) | this session (Claude) | **done 2026-10-31**: new families ≈0.0007–0.001 each; quality slopes Bayes 0.14, GP 0.10, A2 0.028, others ≈0; lanes re-ranked in STRUCTURAL_PLAN.md |
| N1 full-bank verification | Codex | complete: 0.86670063 overall, but fold 1 worsens; no promotion; retained best and target gap unchanged |
| A. Headroom map (`scripts/headroom_map.py`, HEADROOM.md) | this session (Claude) | **done 2026-10-31**: the only deployable new signal found is N1-style choice history; see HEADROOM.md conclusion |
| H8. Choice affinity beyond N1 (P2) | this session (Claude) | done 2026-10-31: borderline −0.00036 (−0.00029 vs recalibrated); a small complement only |
| H8 nested validation | Codex | complete: −0.000373 vs ridge, −0.000289 vs matched tree on P2 fold 0; full-bank value pending |
| H9 choice co-occurrence embeddings | Codex | closed: +0.000048 gain beyond H8, below 0.0003; H8 control reproduced exactly |
| H8 deployable CPU column | Codex | complete: treatment 0.86648484, all five folds improve both references; matched-control gain 0.00022028 < fixed 0.0003; not promoted; retained R9 unchanged |
| A7. Sibling-split Stage C rehearsal (K1 family) | this session (Claude) | **closed 2026-10-31: Stage C absorbs the sibling signal; pooling +0.00178 worse after retraining** |
| E2 prototype (movie-side context in A2) | other agent (Codex) | CPU components verified; deferred after headroom review; GPU queue priority preserved |
| H4 outer-fold dependency audit | Codex | complete: dependence confirmed on actual P2; evidence in HEADROOM_INTERPRETATION.md; no H4 implementation changes |
| B. Shared P2 candidate scoring command | Codex | complete: three fixture tests pass and E5 primary result independently reproduced exactly; no running trainer rewired |
| E3 gate preparation | Codex | complete: training and automatic audit exited zero; both fixed P2 gates failed; saved predictions independently reproduce comparisons exactly; E3 closed |
| Structural plan week-0 scoping | Codex | full all79 parity exact; 10/25 further fixed arms audited (first- and second-seed n1/n4/n16/n32/n64); latest n64 average 0.8673387788; lane 0 third-seed n1 tree live (PID 16110), lane 1 third-seed n4 tree live (PID 24702); full synthesis waits for all fixed arms |
| S0 complete-result reporter | Codex | live bash handoff PID 94435/session 80182 waits for both lanes, then audits all 26 arms once; five tests pass, real reference parity exact, actual partial experiment rejected without summary; no frozen training edits |
| S0 completed-arm audit monitor | Codex | Live bash PID 77645/session 64149; audits all 25 nonreference arms only after producer exit and complete files, using unchanged frozen audits. First automatic receipt (second-seed n1) independently rechecked: all hashes, five folds and scores exact; ten of 25 arms audited. No training, retuning or early synthesis. Do not duplicate or race its receipt writes. `S0-arm-monitor-first-audit-verification-20261031.json` |

## Closed bets

- **New-family pilots (STRUCTURAL_PLAN lane 1): closed for good by the user, 2026-10-31**, after L1. The feature GBDT base model and co-clustering are not to be built (FAMILY_AUDIT.md).
- E3 (CE=0.1 on A2h distribution head): closed 2026-10-31. Shared P2 fold 0 worsens by 0.00000131; five-fold gain 0.00001277 < fixed 0.0003, with folds 0/1 worse. Both gates fail; no full-training follow-up. See `experiments/E3-independent-verification-20261031.json`.
- L1 (LLORMA-style local low-rank ensemble): closed 2026-10-31. Standalone 0.8924 on P2 but +0.0000023 beyond bank11 + N1 (gate −0.0010).
- H8 (choice-affinity full-bank comparison): complete 2026-10-31. Treatment 0.86648484, gain 0.00038580 versus R9 and 0.00022028 versus matched control, all five folds better. The fixed 0.0003 matched-control minimum fails; not promoted, no threshold relaxation.

- SO1 (selection auxiliary objective): closed 2026-10-31. Fixed P2 gain 0.0000275, below its frozen 0.0005 threshold; original gate independently reproduced exactly. Queue completed without a conditional full stage.

- H9 (unlabelled-choice co-occurrence SVD): closed 2026-10-31. P2 gain 0.000048 beyond H8, below the fixed 0.0003 gate; interval includes zero.

- E5 (ordinal Bayes likelihood): closed 2026-10-31. Standalone −0.00097 on P2 but only −0.000135 beyond bank11 (gate −0.0003).
- A7 (probe-sibling pooling after Stage C): closed 2026-10-31. Stage C absorbs it (+0.00178 worse).
- E1 (P2 up-weighting in the Stage C retrain): closed 2026-10-31. w2 +0.00014 and w3 +0.00049 worse than the control on all folds; the Bayes arm was stopped by the pre-registered rule. See NEXT_EXPLORATIONS.md.

## Slots

| Slot | Bet | State |
|---|---|---|
| 1 | S0 scoping lane 1 | first-seed n4/n32 and second-seed n1/n16/n64 complete and audited; second-seed n64 average 0.86733877877; third-seed n4 tree live (PID 24702), continuing 12 exact fixed arms; no GPU |
| 2 | S0 scoping lane 0 | first-seed n1/n16/n64 and second-seed n4/n32 complete and audited; second-seed n32 average 0.86843318372; third-seed n1 tree live (PID 16110), continuing 13 exact fixed arms; no GPU |
| 3 | Production retrain P0 | A2h fixed MPS retrain live, first of seven serialized GPU recipes; all 79 retained columns and final blender remain required |

E3's fixed GPU treatment and both gate evaluations are complete and failed;
E2 (movie-side context in A2) remains deferred. The log of results is kept in
[NEXT_EXPLORATIONS.md](NEXT_EXPLORATIONS.md).

E2's CPU data component is ready: training-only indexes for P2/full and two
10,000-query sampling pilots are verified (`experiments/E2-data-verification-20261030.json`).
Exact query-conditioned co-rater codes are also defined and verified on 15
CPU pairs (`experiments/E2-query-code-verification-20261030.json`). Whole-cache
generation and the matched neural learning plan remain pending. A batched
producer/joint-attention prototype has passed CPU fixture learning and one
real-query forward checks (`experiments/E2-joint-audit-20261030.json`). No real
E2 training or GPU job has run.

A resumable exact-code exporter and strict read-only cache loader also pass
fixture and 15-real-row CPU checks (`experiments/E2-cache-audit-20261030.json`).
Whole-cache generation/MPS feasibility and the matched training runner remain
pending; the exporter has not been launched on the GPU.

The matched P2 training runner (`scripts/e2_train.py`) now passes three CPU
fixture tests for batch/RNG parity, repeatability and label-independent
prediction. No real E2 optimization has run. Its learning recipe and shared
gate binding wait for the headroom map; no GPU job was launched.

An independent [headroom interpretation audit](HEADROOM_INTERPRETATION.md)
distinguishes additive group-shift gains from general interaction headroom.
The diagnostics guide priorities but do not prove an E2 gain ceiling or that
the goal is impossible. Existing queue priority and H4's decision point stand.

A7's first K1 attempt exited before training (`training session missing`).
Read-only diagnosis in `experiments/A7-order-audit-20261030.json` confirms
82,523 decreasing date edges within user training segments after appending
A siblings. K1 requires chronological segments; the owning session must repair
and reverify the dataset/artifact provenance before continuing that attempt.
This diagnosis did not modify or restart A7.

The A7 owner repaired the ordering in v2 (`ac6871e`). Independent verification
in `experiments/A7-order-repair-verification-20261030.json` checks all
99,775,636 training tuples: no within-user date inversions, identical per-user
item/day/rating content, and unchanged offsets/probe/implicit arrays versus
v1. The owner's A7v2 graph chain is live; this session did not relaunch it.

Independent A7v2 scoring now confirms that the fixed pooling extension gate
fails: Stage C+pool is 0.0017787484 worse than Stage C on all five folds.
Evidence is `experiments/A7v2-independent-score-evidence-20261031.json`;
full results and interpretation are in NEXT_EXPLORATIONS.md. This is a
K1-family rehearsal result; the promoted best and full-bank offset stand.
