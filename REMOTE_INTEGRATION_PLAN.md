# Integrate completed remote bknbrx predictions

Branch: `research/fresh-remote-integration`. Remote assignment commit
`50c9cf2` is merged without changing its outputs or the frozen nonlinear engine.

The remote completed bknbrx-100 (about nine minutes), returning terminal,
epoch-1 and epoch-5 ProbeX vectors. Terminal and epoch 5 duplicate exactly.
CF-NADE is profiling evidence only and supplies no model features.

Before looking at the new ensemble scores, fix these comparisons:

1. Reproduce fourteen-output ridge 0.8677119362283443, snapshot/context ridge
   0.8663200739003126, latest 62-column ridge 0.866286461848646 and verify the
   existing nonlinear result's saved hashes/independent scoring receipt.
2. Add both unique raw remote rating vectors to the 62-column ridge. Clip
   them to [0,6] at design construction, preserve all original fourteen-only
   context interactions, and keep the last intercept unpenalized. Refit with
   the same five user folds, lambda 1000 and output clip [1.05,4.95].
3. Refit tree and neural residual corrections with exactly the frozen
   `upstream_nonlinear.CONFIG` and the same nested protocol. Keep all original
   68 nonlinear input features byte-identical and append the two clipped new
   predictions centered at fixed 3.6. Do not recompute existing row means.
4. Compare the same fixed 50/50 average with **0.8644717283393785**, and report
   matched per-family changes against the original 62-column fits. Run both
   families regardless of the cheap ridge screen's sign; no sweeps follow.

The receiving side verifies the pinned job, receipt SHA/bindings, all three
NPZ/content hashes, finite float32 shape, zero-qualifying-outcome data receipt,
and ratings/user/item/date file hashes against its native data. Deduplicate
by content, not filename. Fulltrain/qual files stay on the remote, where all
13 native output hashes were verified; do not claim they were rehashed here.

The implementation is a separate adapter in
`scripts/upstream_remote_integration.py`; it does not change the original
engine or its frozen manifests. Its own code, immutable prepared inputs and
remote/parent manifests are hashed before refitting. Parent models are not
retrained and remote models are not replaced by released fixture predictions.

Run targeted/full upstream checks, then:

```sh
.venv/bin/python scripts/upstream_remote_integration.py --prepare
.venv/bin/python -u scripts/upstream_remote_integration.py
```

Use detached CPU workers, two threads per family, with sleep prevention.
Reports are `experiments/upstream-remote-integration/`; cached inputs/models
are `artifacts/upstream-remote-integration/`. Reuse completed folds only with
their verified hashes; interrupted folds cannot be silently overwritten.
After completion replay each family's saved models using `--verify-family`,
then independently score with `--verify-scores`. Export final float32 vectors,
all fold deltas and negative results, and push the completion evidence.

This remains exploratory ProbeX development evaluation on historically
selected base recipes. No qualifying outcomes or downloaded prediction inputs
enter these fits; reaching a development score does not verify the qualifying
10% target. The new remote temporal plan is a separate parallel assignment.
