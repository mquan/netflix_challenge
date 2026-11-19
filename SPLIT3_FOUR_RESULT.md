# Complete split3 native batch: fixed four-column diagnostic

Both original DNN16t/DNN32nm runs completed7/7, and NADE12s completed8/8.
Separate whole-model checks bind unchanged source/binary/data, both full epoch
traces, native outputs, finite complete queries and every original known-row
projection. Query labels are excluded from original training; qualifying
targets are zero. No public pretrained numerical features were used.

The fixed batch retains the three terminal predictors and the predeclared
published NADE epoch01 pair. No subset search is performed. Add all four to
the own107 ridge problem under the same ten user groups, five outer folds
and penalty1000. Ridge improves **0.8625801963952328 →0.8625621609004741**;
all five folds improve. A separate solver recomputes every fold's statistics,
weights and full held-out prediction vector; maximum discrepancy5.75e-12.

This is a **ridge diagnostic**, not a new global nonlinear or qualifying score.
The verified best remains development0.8592309820 and recovered qualifying
0.8607516602 /0.8617378857. Both qualifying groups still miss0.85725.
Any nonlinear extension requires new actual fitting/replays and an independently
verified fixed mean before a separately frozen production recipe.

Evidence: experiments/upstream-split3-diversity/*-verification.json and
experiments/upstream-split3-four-screen/{plan,ridge-screen,ridge-verification}.json.
Code: scripts/upstream_split3_four_screen.py and
scripts/verify_upstream_split3_four_screen.py.

The subsequent full nonlinear study is complete: fixed expanded-tree mean
0.8592838017 is worse than the current0.8592309820. All15 actual neural
replays and separate scoring pass; no qualifying promotion. See
[SPLIT3_VIRTUAL_MPS_PLAN.md](SPLIT3_VIRTUAL_MPS_PLAN.md).
