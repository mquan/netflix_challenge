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

## Slots

| Slot | Bet | State |
|---|---|---|
| 1 | E1 stage 1: P2 up-weighting in the K1-family Stage C retrain | running since 2026-10-30 20:40 (`experiments/E1-chain.out`) |
| 2 | E1-Bayes: the same weighting as per-row noise precision in Bayes timeSVD++ (rank 64) | being implemented |
| 3 | open | — |

GPU bets E2 (movie-side context in A2) and E3 (ordinal A2h head) are queued
behind the other session. The log of results is kept in
[NEXT_EXPLORATIONS.md](NEXT_EXPLORATIONS.md).
