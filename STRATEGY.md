# Strategy: from one-off bets to a measured program

Written 2026-10-31 at the user's request ("is there a more durable strategy?").
It governs how this branch picks work toward the goal in [GOAL.md](GOAL.md).

## Why change

| Period | Best dev CV | Gain |
|---|---:|---:|
| R5 → R6 (2026-09-27/28) | 0.87481 → 0.87312 | 0.0017 |
| R7 | → 0.87039 | 0.0027 |
| R8–R9 (to 2026-10-09) | → **0.86687** | 0.0035 |
| 2026-10-09 → 2026-10-31: R10–R13, LC1–LC6, HR0–HR4, E1 | 0.86687 | **0** |

Every promoted gain came from a **new model family's first arrival** (conditional
RBM, date-aware masked history, Bayes timeSVD++, the A2 transformer). Variants
and add-on columns have paid ≤0.0001. Three weeks of individually reasonable
bets produced no promotion. The causes are structural:

1. **Bets are picked from papers, not from where our error is.** Round 11
   located the error: 67% of squared error is on rows from days without a
   same-day training rating, and it grows with the gap since the user's last
   training day. But nobody has measured how much of the 0.0022 each
   information source could remove at most.
2. **Per-bet overhead is high.** A clean negative (E1) needed new trainer
   code, tests, weight files, chains and a scorer.
3. **The paying resource (GPU, A2 family) is used serially and ad hoc**, while
   CPU slots run lower-prior bets because they are free.
4. **Two sessions on one branch duplicate work** (E1's scorer was built twice).

## The program

### A. Headroom map (first, CPU, ~1 day)

Before any further bet, estimate on P2 (training-only, the probe never read)
how much each *information source* could remove from the current bank's error.
These are oracle or upper-bound experiments. Cross-fitted residuals of the
bank11 P2 blend are pooled, leaving out the target row, within groups that
each stand for a source: the user overall, the user's day and week, the movie,
the movie's week, and the calendar day. Each pooled feature is added to the
same fixed ridge gate. A source whose oracle gain is small cannot be the route
to 0.0022, however clever the model. Sources with large oracle gain, and
features observable at test time that proxy for them, define the bets.

**Decision point.** If the reachable sources together plausibly sum to well
below 0.0022, state plainly that incremental work on this pipeline will not
reach 10%, and decide with the user whether to rebuild.

### B. One shared gate and a ledger

One command scores any new P2 prediction column against the frozen bank with a
fixed protocol: SO1's bank11 ridge comparison, plus the full-bank dev-CV gate
for promotion. Every bet, its pre-registered gate and its outcome go in one
ledger table in [NEXT_EXPLORATIONS.md](NEXT_EXPLORATIONS.md). A bet then costs
only its model, and killing it is cheap. Existing pieces (SO1 gate, E1 fixed
scorer) are reused, not rewritten.

### C. One long-lived line on the family that scales, with a shared GPU queue

The sequence model (A2 family) is the only line still paying. Treat it as a
roadmap ranked by the headroom map, not as separate bets: two-sided context
(E2, being prepared by the other agent), distribution loss (E3, ready), and
the full event stream including probe and qualifying dates (N1's direction).
GPU jobs from both sessions go through one priority queue, so the GPU neither
idles nor gets contested.

## Coordination between sessions

- Claim work in the GOAL.md slot table **before** starting it. Don't build
  something another session has claimed.
- Never stop, edit or slow the other session's jobs or frozen files (GOAL.md rules).
- New one-off bets pause until A is done, except E5 (already running, cheap,
  and a first test case for B).

## Part A outcome (2026-10-31)

See the conclusion in [HEADROOM.md](HEADROOM.md). The user-level shift is
already captured by Stage C (A7). The movie side and simple proxies are
exhausted. N1-style choice history is the only measured deployable new signal
(−0.0012 on P2 beyond A2h). The program's line C should therefore centre on
choice-history sequence models, with N1's full-bank evaluation as the next
decisive number. That evaluation belongs to the other session's queue; this
session doesn't run or duplicate it.
