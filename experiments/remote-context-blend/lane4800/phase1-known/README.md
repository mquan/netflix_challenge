# 4800lm phase 1 (trainx → probex) — known-side diagnostic, 2026-11-26

Phase 1 ended 12:25Z Nov 26 after 27 epochs (≈71 min each), probex RMSE 0.87698. The qual phase is running, ETA ≈21:00Z Nov 27.
Fixed bank v2 control, global ridge λ=1000, 5 user folds. Probex side only (`RCB_KNOWN_ONLY=1`); the release follows
when both phases verify.

| arm | pooled | Δ vs bank | folds better |
|---|---|---|---|
| bank | 0.863155 | | |
| + base | 0.862771 | −0.000384 | 5/5 |
| + snapshots 3/13/21/24 | 0.862727 | −0.000428 | 5/5 |
| + the other 23 epochs | 0.862667 | −0.000489 | 5/5 |

The base alone gains 2× the 22-job knn3d wave (−0.000185). It is the largest single-producer addition to bank v2 so far.
