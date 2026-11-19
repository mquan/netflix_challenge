# Own raw masked-denoising model: fixed real-data experiment

Fit the declared17,770→512→17,770 tanh autoencoder on our canonical original
rating histories. Both phases run50 complete epochs, with256 users per batch,
AdamW learning rate0.0005, weight decay0.0001, seed2026 and50% input corruption.
Inputs are observed ratings minus3.6. The loss covers observed training ratings
hidden from the corrupted input. Query predictions use complete available
histories and clip to1..5. No accuracy-based early stopping or epoch selection
is used. Fixed snapshot epochs are5,10,20,30,40,50.

The trainx history contains99,352,343 ratings and excludes every one of the
1,128,164 known query pairs. Fulltrain contains100,480,507 ratings and excludes
all2,817,131 qualifying query pairs. Both private CSR representations were
independently rebuilt using lexicographic sorting and checked against every
original rating tuple, user count and complete query projection. Qualifying
targets are zero; recovered outcomes never enter these histories or the fit.

The synchronized synthetic full-width benchmark measured MPS at4.39x/4.46x CPU
throughput for batches64/256 and passed actual saved-model replay. This is not
full-data timing or accuracy evidence. A separate three-epoch, two-phase fixture
verified optimizer/RNG checkpoint resume, six actual saved-model query replays,
complete query coverage and retention of snapshot weights while older optimizer
states are pruned. Real cached history batches also match independent scalar
reconstruction bitwise.

Alternate the two phases at durable epoch boundaries. Each epoch gets the
existing shared meta fitting lock, then releases it, allowing shorter blending
work to run. Fresh epoch workers reload their actual saved optimizer and RNG
states. Keep the latest two optimizer checkpoints per phase and every declared
snapshot's model weights. Partial epochs stop on failure and remain available
for explicit recovery; an observation timeout never causes automatic restart.

Every declared snapshot must reload its saved model and reproduce the complete
query vector bitwise. A paired snapshot receipt is written only after both
phases finish that epoch. Earlier complete snapshot pairs do not imply the full
50/50 run has finished. Any use in a blend still requires source checks, nested
known-user-fold evaluation and a new frozen qualifying audit.

This is an additional own raw-model recipe. It does not claim exact original
native-preset parity or alter the ongoing original95-epoch CF96 training. No
public numerical predictions, new model accuracy or target achievement are
claimed before actual real-data fitting and verification complete.

Code: scripts/upstream_mps_autorec.py, scripts/upstream_mps_autorec_data.py,
scripts/verify_upstream_mps_autorec_data.py.
Frozen real-data configuration: experiments/upstream-mps-autorec/plan.json.

First real snapshot: epoch5 completed for both phases and actual saved-model
inference replay passed on all1,128,164 /2,817,131 query rows. Separate source/
model/query binding checks and compensated known-only scoring give a standalone
component RMSE0.9332148095. This supplies no ensemble or qualifying improvement.
The declared50/50 trajectory continues; epoch5 is not a completed50-epoch model.
Evidence: experiments/upstream-mps-autorec/{pair-00005,
snapshot-00005-verification}.json.
