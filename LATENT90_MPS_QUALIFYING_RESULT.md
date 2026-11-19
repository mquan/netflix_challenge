# Expanded tree/all90 MPS NN: verified qualifying improvement

The selected fixed half mean scores **0.8608774626094220 on source0** and
**0.8618605526405846 on source1**. Both improve the preceding standalone NN
0.8609266298 /0.8619141823. The worse group improves **9.5159525%** over0.9525,
leaving0.0046105526 RMSE to the strict0.85725 target. **Both groups still miss.**

Selection and weights came solely from complete known-ProbeX development:
0.8592720613 for this fixed50/50 combination. Actual five-fold median budgets
were tree729/706/868 and NN7/8/7. The full-known tree preserves294 inputs with
50 original learned supports; the float32 MPS NN preserves334 inputs with all90
supports. Both use107 own raw predictors and the same183-column paired ridge.
Previously disclosed paired-parent source differences remain explicit.

Both families completed full-known fitting and actual saved-model reload/
qualifying inference replay, passing bitwise. The exact float32 half mean was
frozen before this recovered-label audit. The separate checker verified876
source bindings,50 code files, model/scaler/output hashes, both actual replay
receipts, exact frozen mean and independently compensated full-group scores.
It rechecks replay evidence; it does not rerun inference itself.

The canonical label Git blob is47ed4788ffba11aad029854a8e60d4dee04fc899.
Source0 contains1,408,789 ratings and source1 contains1,408,342. Those counts
reverse the published Quiz/Test sizes; source names are preserved. This is an
adaptive community-recovered audit, not official Netflix or untouched Test.
No downloaded numerical prediction feature or qualifying outcome enters
fitting, candidate budgets or combination weights.

Frozen vector SHA256:
41ce516f07d2d9483cc1c92424fb22db9cda10ecc09a24f0a7f8d194173ae040.
Full-plan SHA256:
13a8338f50fee05dc3e70afabdc1022cffc31c3230cc940421a38530801314c0.

Evidence: experiments/upstream-rbm50-tree-latent90-production/{plan,nn,
nn-replay,tree,tree-replay,report,frozen-mean-score-verification}.json,
and recovered-audit/report.json. The active10% goal is not complete.
