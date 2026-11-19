# Qualifying path for the verified CFNADE-12 context improvement

The complete known-ProbeX experiment improved RMSE from 0.8602161957664950 to
0.8600637147575518, with four of five folds improving. Preserve the measured
combination: the original 85-predictor tree and the new 86-predictor context NN.
No qualifying result has selected this model, its inputs, weights or budgets.

`upstream_cf12_bank.py` extends the actual completed paired 85-predictor bank with
the complete original CFNADE-12 predictor. It requires the full native receipt,
all recorded output hashes, finite paired float32 arrays, terminal/epoch-40
identity, zero qualifying training targets and known-row identities. It retains
the original 85-bank centering and feature order. Only the new NN uses the
162-column ridge design and 222 features (223 inputs after its fitted ridge).

`upstream_cf12_production.py` waits for the first frozen 85-predictor qualifying
audit. It skips if the target is already verified. Otherwise it serializes with
the source-extension fitting lock, defers to a stronger independently verified
development candidate, and builds the new paired bank from complete native jobs.
The original 85-predictor attempt and its frozen weights remain unchanged.

Reuse the first attempt's full-known-data tree component instead of fitting it
again. Verify its exact parent-bank manifest, original nested budgets, seeds,
tree settings, versions, source/code hashes, ridge, saved models, prediction and
semantic replay. Publish the reused vector with explicit original-plan and
family-receipt lineage and `fit_performed: false`. Its original 161-column ridge
and 221 features remain separate from the new neural inputs. Replay the saved
tree in a clean interpreter through its original production adapter.

Refit only the new context NN on the actual paired known ProbeX predictions.
Use the pre-audit median budgets **19, 20, 20**, all three original seeds, and
known-label-fitted ridge/scalers. The unchanged generic production implementation
fits, predicts and replays this family. It then freezes the fixed equal mean of
the two float32 family vectors before invoking the existing recovered-label
audit and independent compensated summation. Both recovered groups must be
strictly below 0.85725; this is never an official or untouched Test claim.

Three focused production tests execute real small-data neural fits and saved
model replay, actual cached-tree replay with its original schema, and origin
bank/budget validation. Both family replays reject modified predictions even
after the recorded byte hashes are updated. The neural test includes a zero
budget seed and confirms its checkpoint is not invented.

The old source-model watcher held the shared fitting lock while waiting for
unfinished native jobs. `upstream_source_extension_watch.py` changes scheduling
only: it waits outside that lock and acquires it for the unchanged original
source-data preparation. Three tests verify that waiting leaves the lock free,
active fitting blocks preparation, and actual preparation holds then releases
the lock. This lets the measured candidate qualify without waiting for unrelated
source models, while preventing overlapping heavy work. The superseded watcher
and 85-predictor context queue are stopped only while verifiably idle; no native
training worker is interrupted.

Reports: `experiments/upstream-cf12-production/`. Until the paired bank, model
replay, frozen prediction and both recovered-group checks actually complete,
the target remains unverified and the goal remains active.
