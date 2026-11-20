# Five-family own calibration

The frozen blend scores **0.8605716962 / 0.8615832225** on the two canonical recovered qualifying groups. Both improve the preceding one-coefficient blend. The worse group improves Cinematch by **9.5451%**; both groups still miss the strict **0.85725** target. These adaptive recovered audits are not official or untouched Test results.

The blend uses only our fully trained, actually replayed tree50 and neural50/90/109/154 families. In that fixed order its weights are 0.2127930147, 0.1739187754, 0.1471917052, 0.1452865438 and 0.3208099609. A single nonnegative sum-one least-squares fit uses complete known OOF vectors. The qualifying vector was frozen before recovered-label access. No recovered outcome fitted or selected these weights. Previously disclosed paired-parent source realization differences remain.

The apparent known-fit RMSE **0.8590715586** is not nested cross-validation. The independently checked development CV best remains **0.8592309820**.

The separate verifier enumerates all 31 nonempty simplex active sets, checks compensated sufficient statistics, reconstructs both stored vectors bitwise, rehashes original family/source bindings and scores both complete canonical groups. Frozen prediction SHA256: `f6dd55b967e06b018861ef29d2a27db10ee387f0366b4053db75ac419803770d`.

Evidence: [plan](experiments/upstream-own5-calibration/plan.json), [calibration](experiments/upstream-own5-calibration/calibration.json), [freeze](experiments/upstream-own5-calibration/frozen.json), [independent verification](experiments/upstream-own5-calibration/independent-verification.json).
