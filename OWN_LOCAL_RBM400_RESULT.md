# Original RBM400 and own16: verified, but only a tiny gain

The predeclared sixteen-family known-only simplex scores **0.8603409488199526**
and **0.8613420938772931** on the two full community-recovered groups. The
preceding own15 primary scored0.8603412616628533/0.8613423892425344: improvement
is only0.000000312843/0.000000295365. Both still miss the strict0.85725 target;
the worse-group gap is0.0040920938772931. Development best remains0.8588630884.

Both original RBM400 models completed all15 epochs, with factors50 and the
original CD1/2/3 schedule, batches, seed, source and resource baseline. Actual
saved-model MPS replay verifies every1,760,494 native known prediction,
the1,128,164-row known projection, and all2,817,131 qualifying predictions.
The [complete pair proof](experiments/upstream-factored-rbm-gpu400-deterministic-metal/paired-verification.json)
has SHA256`3e8d2ecdbb40d0c953f98b24601c2e84a9353bb45ca23d74fea26797a814199a`.

All sixteen original families remain in order. Weights used known labels
only; the new scalar RBM receives weight0.0016778366438841104. All65535 simplex
supports, actual coefficient KKT/objective checks,17 ridge equations and all
six stored vectors pass independent verification before recovered labels.
Separate canonical source rehash, vector reconstruction and compensated group
scoring confirm the result. Apparent fitted blend RMSE is not nested CV.

The predeclared ridge10 comparison scores0.8603399172592097/0.8613555317652924:
it trades off the two groups and does not replace the simplex primary. The
unchanged own15 primary is retained as the third fixed control. No coefficients
or candidates were changed after seeing qualifying outcomes. These are
adaptive recovered audits, not official or untouched Test results; tree154
retains its disclosed historical A/B realization caveat.

The [canonical proof](experiments/upstream-local-rbm-blend-deterministic-metal/independent-canonical-verification.json)
has SHA256`d646e5b449052126fdfae87978578d741a4e58a85ce0a8ec524406a2caa9d1e0`.
The owned chain now extracts learned contexts from these actual models before
full CPU data replay, both original-budget H128 studies, the selected full-query
bag/replay and the [own17 calibration](OWN_RBM400_OWN17_CALIBRATION_PLAN.md).
The [high-capacity tree follow-up](OWN_FULL_CONTEXT_CATBOOST_FOLLOWUP_PLAN.md)
is planned if own17 misses the target; it has not started. The goal remains active.
