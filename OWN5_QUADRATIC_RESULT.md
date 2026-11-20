# Fixed quadratic own-model calibration: negative result

The frozen quadratic residual calibration scores **0.8605784947 / 0.8615935718** on the two complete canonical recovered qualifying groups. Both are slightly worse than the preceding convex blend **0.8605716962 / 0.8615832225**. Keep the convex result as the best own qualifying score. The10% target remains unmet.

This single declared study uses only the same five actually replayed own families. It fits21 coefficients: an intercept, five prediction/disagreement coordinates and their15 quadratic products. All20 nonconstant features are standardized on complete known rows, with a fixed ridge penalty1000 and no hyperparameter search. The prior known-only convex blend is its offset. The full qualifying vector was frozen before this recovered audit. No recovered outcomes fitted or selected coefficients.

Apparent known-fit RMSE **0.8590477906** improves on the convex fit, but is not nested CV and does not replace the **0.8592309820** development best. Its failure to improve the qualifying audit argues against claiming a gain from the apparent fit alone. These adaptive recovered audits are not official or untouched Test scores.

Independent verification uses a full rectangular direct SVD solve (rank21), coefficient maximum error **2.04e-16**, independent basis/population standardization checks, bitwise known/full vector reconstruction, source rehashes and compensated complete-group scoring. The original checker failed only while serializing a NumPy integer rank. A separate recovered checker casts that field to a built-in integer and binds the recovery declaration; all original numerical fitting/checking and frozen artifacts remain unchanged.

Evidence: [plan](experiments/upstream-own5-quadratic/plan.json), [fit](experiments/upstream-own5-quadratic/calibration.json), [freeze](experiments/upstream-own5-quadratic/frozen.json), [serialization recovery](experiments/upstream-own5-quadratic/verification-recovery.json), [independent verification](experiments/upstream-own5-quadratic/independent-verification.json).
