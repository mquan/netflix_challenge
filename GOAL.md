# Standing goal: reach the 10% improvement

Set by the user on 2026-10-30. This session keeps running and trying new bets
until the goal is reached or the bets run out. It does not stop at a single
experiment.

## Target

- **10% better than Cinematch's 0.9525 test RMSE = 0.85725 test RMSE.**
- Mapped to our metric with the measured Stage C offset (test ≈ dev CV − 0.0074,
  measured twice; [PLAN_TO_TARGET.md](PLAN_TO_TARGET.md)): **≈0.86465 dev CV**.
- Current promoted best: R9-LAT-A2h at **0.8668706425** dev CV, a gap of
  **≈0.0022**.
- The gap can close from either side: a better dev-CV bank, or a larger
  Stage C retraining offset (measured with the Phase 0 rehearsal). Both count,
  but every claimed gain must be measured under its own fixed protocol.
- Qualifying labels are not available locally. Reaching the target is an
  **estimate** from dev CV plus the measured offset, never a verified
  hidden-test score.

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

Work is chosen by the program in [STRATEGY.md](STRATEGY.md): (A) a headroom map of which information sources could close the gap, (B) one shared gate plus a ledger, (C) one long-lived sequence-model line with a shared GPU queue. New one-off bets pause until A is done.

## Claims (in progress, per session)

| Work | Owner | State |
|---|---|---|
| A. Headroom map (`scripts/headroom_map.py`, HEADROOM.md) | this session (Claude) | H1–H6 done 2026-10-31: user-level shift is the headroom; probe-sibling pooling −0.00186 pre-Stage C |
| A7. Sibling-split Stage C rehearsal (K1 family first) | this session (Claude) | starting 2026-10-31 |
| E2 prototype (movie-side context in A2) | other agent (Codex) | CPU model/cache/P2 runner verified; learning decision waits for headroom map and GPU queue (see E2 notes) |
| H4 outer-fold dependency audit | Codex | complete: dependence confirmed on actual P2; evidence in HEADROOM_INTERPRETATION.md; no H4 implementation changes |

## Closed bets

- E1 (P2 up-weighting in the Stage C retrain): closed 2026-10-31. w2 +0.00014 and w3 +0.00049 worse than the control on all folds; the Bayes arm was stopped by the pre-registered rule. See NEXT_EXPLORATIONS.md.

## Slots

| Slot | Bet | State |
|---|---|---|
| 1 | E5 ordinal Bayes pilot, ordinal arm (P2, rank 64) | running from 2026-10-31 (`experiments/E5-p2-r64-ordinal.log`) |
| 2 | E5 ordinal Bayes pilot, matched Gaussian control | running from 2026-10-31 (`experiments/E5-p2-r64-gaussian.log`) |
| 3 | E3 (λ=0.1 cross-entropy on A2h's distribution head) | code + CPU tests done 2026-10-31; **waits for the GPU** after the other session's queue |

GPU bets E2 (movie-side context in A2) and E3 (ordinal A2h head) are queued
behind the other session. The log of results is kept in
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
