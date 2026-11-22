# Independent full-arm tree replay and canonical scoring

The complete-arm verifier now independently solves all ten original eligible
ridge1000 systems, compares every saved coefficient bit and normal-equation
residual, reconstructs every original fold population, reloads every positive
saved model and verifies every held prediction and the complete OOF join.
It authenticates all source, model, prediction and actual replay-producer
exit-receipt bytes before and after replay. Partial inventories, extra or
missing payloads and namespace reuse are rejected.

Separate canonical scoring reclips each stored float32 held prediction in
float64 and uses math.fsum over every squared known error. Selection requires
both complete arms and their linked source/model/replay proofs, chooses the
lower tree-only CV score and uses scalar for ties within1e-12. Qualifying
outcomes do not enter fitting, replay, scoring or selection.

All54 combined tree component checks passed. The new complete-study fixture
fits real small CPU CatBoost models across all five folds, retains fifteen
seed slots, independently reloads/replays all saved positive models and
computes the separate canonical score. It uses synthetic20-row source and
ownership proofs and a mocked raw-stream boundary; it does not admit full
production data. The original pinned backend and fold selection remain in
use. A one-ULP coefficient mutation and missing model inventory are rejected.

Still required before production: the owned controller, role bootstrap and
whole-family resource/exit gates; full paired studies and actual independent
raw-stream replays; selected full qualifying bag and replay; own18 calibration,
exhaustive independent verification and the three fixed recovered audits.
The live neural and own17 chain retains priority, and the existing follow-up
condition is unchanged. These component checks add no numerical score or
plot point and do not establish the target.

[Component proof](experiments/upstream-full-context-catboost-full-replay-protocol/verification.json).
