# Own item-oriented raw predictor: fixed50/50 run

The implementation is complete and queued. Real-data fitting has not started:
the controller waits for actual completion of the current raw50/50 model,
preserving disk headroom and all existing source/model artifacts.

Train **480,189 user inputs →128 tanh units →480,189 outputs**,123,408,701
parameters, with one example per item and all17,770 item rows. Keep both full
original rating histories. This is an additional own recipe, not exact native
preset parity. No public pretrained numerical features or qualifying outcomes
enter training.

Freeze batch32, float32 MPS, bounded0.30 memory fraction, AdamW learning
rate0.0005/weight decay0.0001, seed2026 and50 epochs per phase. Center observed
ratings at3.6. Corrupt half of observed inputs, scale retained inputs by2,
and minimize MSE only on observed ratings hidden by corruption. Query inference
uses complete histories and clips to1..5. Snapshots5/10/20/30/40/50 are fixed;
there is no accuracy-based epoch, rank or stopping selection.

Losslessly transpose the original99,352,343 /100,480,507 tuples to item-row CSR.
Store user-column indices asuint32 to retain all480,189 user IDs. Check every
tuple/count, unique item/user pair and phase-query exclusion. A separate
lexicographic sort replays all original training tuples and full query maps.
Qualifying targets must remain zero. Both independent checks precede fitting.

Each full epoch runs in a fresh worker, restoring actual optimizer and CPU/MPS
RNG states. Alternate phases at durable boundaries under the existing GPU lock.
Keep one optimizer/RNG checkpoint per phase: fsync the new file, atomically
replace it, fsync its directory and new pointer, then retire older optimizer
states. Preserve every snapshot weight and prediction array. Every snapshot
must reload its actual saved model and reproduce the entire query vector
bitwise. Failures preserve partial outputs; observation expiry never restarts
training. Reserve10GiB before starting an epoch.

The isolated fixture passed with user IDs69,999–70,004, independently rejected
a wrong column despite an updated producer payload hash, rejected query leakage,
performed four fresh-process optimizer/RNG resumes and six actual saved-model
query replays, and independently reconstructed original-order predictions from
unsorted original triples. One-checkpoint retention, all snapshot weights and
the fsync/atomic-pointer path were exercised. This is not real-data accuracy.

The full-width benchmark passed all four CPU/MPS actual saved-model replays.
Batch32 MPS steps measured0.1493196s versus CPU0.3723881s. Synthetic throughput
is not a real-data epoch-time or predictive-quality claim.

The queue verifies actual preceding50/50 completion and all original snapshot
model/replay bindings, then prepares/checks both new histories, fits the full
new50/50 recipe and independently checks all six paired snapshots. Neither
current raw training nor original95-epoch CF96 training is altered.

Code: scripts/upstream_mps_item_autorec{,_data,_queue}.py,
scripts/verify_upstream_mps_item_autorec{_data,_snapshot}.py,
scripts/test_upstream_mps_item_autorec.py.
Queue declaration: experiments/upstream-mps-item-autorec-queue/plan.json.
Fixture: experiments/upstream-mps-item-autorec-fixture/verification.json.
