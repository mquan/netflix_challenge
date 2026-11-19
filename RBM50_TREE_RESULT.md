# Expanded tree and context NN: verified development improvement

The fixed50/50 mean scores **0.8593661638722666** on all1,128,164 known-ProbeX
rows, improving the preceding standalone NN0.8594554093063749. Independent
compensated scoring gives0.8593661638722665. Both families use107 own freshly
trained raw predictors plus50 original learned RBM user supports; the original
five nested user folds and fitting settings are preserved.

All15 saved tree models pass actual reload/inference replay on their held-user
rows. The unchanged NN vector retains its originating five-fold receipts and
all15 actual saved-model replays. Every originating artifact remains hash-bound;
no new neural fit or synthetic training receipt is claimed. The stored fixed
float32 half mean is independently verified bitwise.

The tree alone scores0.8601340713325017. The fixed mean improves four folds and
worsens one versus the NN; lower aggregate RMSE is promoted without a minimum
gain or all-fold veto. The final actual tree budgets by seed2026/7/13 are:

| Fold | 2026 | 7 | 13 |
| --- | ---: | ---: | ---: |
| 0 | 713 | 707 | 919 |
| 1 | 771 | 706 | 810 |
| 2 | 831 | 567 | 1120 |
| 3 | 698 | 628 | 778 |
| 4 | 729 | 780 | 868 |

Full-known tree refitting should use the per-seed medians729/706/868 and exact
paired107/50 bank, and reuse the verified full-known NN with its original
ridge, scaler, models and replay evidence. Fix the50/50 mean before fitting
and new qualifying outcomes. This production step is pending.

The best verified recovered qualifying result remains the standalone NN:
0.8609266298 /0.8619141823. Both miss0.85725. This development result is adaptive
known-ProbeX evidence, not official/untouched Test or a new qualifying result.
No downloaded numerical prediction feature or qualifying outcome was used in
fitting or selection. Previously disclosed paired-parent source differences
remain explicit.

Evidence: experiments/upstream-rbm50-tree/{manifest,evaluation,
independent-verification,tree-replay}.json and tree/0.json through tree/4.json.
Mean vector SHA256:
f9b22a9591d6fe5ae203092e3c4074197ce6476883e5a0faa09205cd9fbfb02f.
