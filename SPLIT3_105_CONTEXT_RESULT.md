# Own105 context comparison: complete, no promotion

All five nested user folds completed on the same 1,128,164 known-ProbeX rows.
All 15 saved neural models replayed successfully; separate compensated scoring
and the fixed float32 mean were checked.

| Candidate | Independently recomputed RMSE |
| --- | ---: |
| Previous standalone NN103 | 0.8594692563654073 |
| Standalone NN105 | 0.8594783385089899 |
| Tree91 / NN105 fixed half mean | 0.8595509427264155 |

Own105 adds our completed split3 attention terminal and epoch8 predictions.
Its aggregate score is higher, so neither arm replaces the previous best and
no full qualifying refit is scheduled for this study. No all-fold or minimum-gain
veto is involved. Qualifying outcomes did not enter fitting or candidate choice.

Evidence: experiments/upstream-split3-105-context/evaluation.json,
independent-verification.json, nn-replay.json and nn/0.json through nn/4.json.
The qualifying best remains 0.8609916544 / 0.8619938614; the target remains unmet.
