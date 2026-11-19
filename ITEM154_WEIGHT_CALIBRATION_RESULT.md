# Known-only coefficient calibration: verified qualifying improvement

Recovered source0 RMSE **0.860652465694796** on1,408,789 rows; source1
**0.8616635956343016** on1,408,342 rows. Both improve the preceding fixed-half
0.8607516601651759 /0.8617378856802073. Worse-group improvement is9.5366304%
over0.9525; strict0.85725 gap0.0044135956 remains. The goal is not achieved.

Reuse the same own full tree and item154 neural models, their actual saved-model
replays and source lineage. Fit exactly one convex coefficient from known OOF
predictions, using the predeclared least-squares formula, without a grid or
qualifying outcomes: tree0.3223099946058576 /NN0.6776900053941424. Every source
binding is rehashed. The apparent known-fit score0.8591642749401914 is not a
nested user-fold validation score. Keep0.8592309819864167 as the verified CV best.

The qualifying vector was frozen before the new recovered audit, SHA256
`a1576516af8d71187eb10cb3239840bac952b76049cc94036a7d1cbd455c5520`.
The independent checker refits the coefficient with a separate vectorized dot
product, verifies both exact stored F32 vectors, rehashes original own model/
replay/source inputs, checks canonical label Git blob
`47ed4788ffba11aad029854a8e60d4dee04fc899`, and computes compensated scores on
both complete groups. Existing audit scores are not coefficient-fit inputs.
This is an adaptive recovered audit, not official Netflix scoring or an
untouched private Test result.

The first independent audit parser stopped at the canonical CSV header before
computing either group score. Original checker/source bindings are preserved;
a separate checker changes only the header skip. The coefficient and already
frozen predictions are unchanged. The recovery declaration records both checker
hashes and the original vector digest.

Evidence: experiments/upstream-item154-weight-calibration/{plan,calibration,
frozen,audit-recovery,independent-verification}.json.
Producer: scripts/upstream_item154_weight_calibration.py.
Original/corrected independent checkers:
scripts/verify_upstream_item154_weight_calibration{,_recovered}.py.
