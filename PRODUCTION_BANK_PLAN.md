# Qualifying production path

The current fixed voting-feature experiment uses 85 prediction columns and all 130 published voting features. Before final fitting, rebuild **both** ProbeX and qualifying inputs from the same verified local execution receipts. The initial development bank contains fresh outputs trained on several machines; retraining Hogwild recipes may produce different predictions. The final blend must learn from its actual paired ProbeX outputs.

`upstream_production_bank.py` resolves each column against the original native tree, the isolated native backfill tree, or the parity-verified mapped tree. It requires a complete receipt binding both ProbeX and qualifying files to the pinned command and source. It refuses missing or modified pairs and preserves partial assembly. Published/downloaded prediction vectors are never candidate inputs. Inventory availability alone is not completion verification.

The assembler preserves the frozen feature ordering: original 62-column mean/centering; five metadata contexts; interactions on the original fourteen rating columns; 23 later rating predictions; and all 130 voting features. The ridge design has 161 columns and nonlinear inputs have 221 columns. Chunking explicitly preserves the original float32 summation order, including single-row chunks. All metadata contexts use training identities/dates and query identities/dates. Qualifying outcomes stay zero.

`upstream_qualifying_backfill.py` prioritizes the older missing RBM40, temporal KNN and neural KNN chains in an independent unmodified checkout. Only completed, hash-verified native parents are hardlinked into it. Native predictions, receipts and compilation targets remain separate, preventing concurrent directory scans from attributing another job's files. Resource checks include both existing native groups and the voting blender. Its results must be explicitly verified before any later integration into the main native directory.

After complete paired assembly, refit the fixed ridge/tree/NN recipe on all known ProbeX rows. Freeze tree/NN iteration budgets from the nested development experiment before inspecting qualifying outcomes; retain all three seeds and the fixed equal average. Fit scalers on known ProbeX only and save all models, transforms, hashes and predictions. Replay the saved models and independently verify the final vector. Only then use `audit_qualifying.py` for a recovered-label audit. Both recovered groups must be below 0.85725. Repeated audits are adaptive development evidence; they are not untouched official Test results. Keep the goal active and continue source-model/feature iterations if either group misses.

Verified progress on 2026-11-19 UTC: the paired bank is **81/85**. All six DNN
follow-on endpoints have completed and their independent paired checks and
native output hashes have been verified. The missing columns are exclusively
`tsvdx5-150o`, `tsvdx5-150o_ep06`, `tsvdx5-150o_ep16` and
`tsvdx5-150o_ep24`. The independent temporal source phase must finish all 25
epochs and its complete paired receipt before any of these are admitted.
`upstream_phase_bank.py` and `upstream_phase_production.py` own this execution
path; the final fitter remains queued. Snapshot evidence is
`experiments/upstream-phase-bank/through-bank81.json` and
`experiments/full-upstream-reproduction/dnn-backfill/completion-verification.json`.
