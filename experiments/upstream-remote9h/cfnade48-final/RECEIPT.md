# CFNADE48 resumed-training receipt (both phases complete, 95/95 epochs)

**State: verified_complete_resumed_cfnade48 — 0 failures** (`verification.json`, observed 2026-11-19T12:02:48.047024+00:00).
This was a **resumed** trajectory from the coordinator's checkpoints, not a fresh, uninterrupted `u.model_job` execution.

- Original reference command (origin-plan.json): `cargo build --release --bin cfnade-new && ./target/release/cfnade-new cfnade-48`
- Original execution (pre-checkpoint, coordinator): `nice -n 10 …/upstream-cfnade-sparse-rows/target/release/cfnade48-stage x|qual`
- Resumed execution (this laptop): `experiments/upstream-remote9h/run/launch-cf48.sh x|qual`
  → `caffeinate -i python scripts/remote12h_supervisor.py --name remote9h-cf48-{x,qual} --deadline 2026-11-19T14:33:28Z -- $PKG/bin/cfnade48-stage {x,qual}`
  (cwd `$PKG/native-source`, RAYON_NUM_THREADS=2, BLAS/OMP/Accelerate=1, CF_CHECKPOINT_DIR/SOURCE/DATA set, CF_EXIT_AFTER_CHECKPOINT/CF_PROOF_* unset)
- Binary: cfnade48-stage sha256 0a239de36f5ce926ae6463788f6c88c651be65b903c6e4a90ed0639ebfbf68b2 (identical to origin-plan `binaries`; no rebuild)
- Package: release remote9h-cfnade48-20261119 `cfnade48-resume.tar.gz` sha256 0399ae6ed06f2529e302f6f090bed8125e91cf4e51e4f445fe8390ccd1be5c63
- Checkpoint origins: trainx epoch 31 (state md5 36b3baaea82af43d41a980d4a308f9d3), fulltrain epoch 30 (48d3a8bab749be85342a41508a564f85)
- source_id e14530b5ade496efb3980a64f98442d35c9a606960add79149ce67b057677d94 · data_id 55ce479fc16447a826d013288b2ccb3e1eec39a07d7783f3d860a3f6e33e1fe0
- Config: CfNadeConfig n_hidden 160 / rank 48 / n_epochs 95 / seed 42 (unchanged)
- Resume proof: `run/RESUME_PROOF.md` (both regenerated restore-epoch vectors were byte-exact). Disclosed deviation: owner.json sidecars written from state.json identity.
- Runs: x 2026-11-19T06:05:50.501609+00:00 → 2026-11-19T11:49:24.869507+00:00 (5.726 h, exit 0, peak RSS 3841.4 MB);
  qual 2026-11-19T06:05:50.499728+00:00 → 2026-11-19T12:02:35.237614+00:00 (5.946 h, exit 0, peak RSS 3937.5 MB). No swap growth, no leftovers.
- Final probex RMSE (native log, epoch 95): 0.89496. Qual targets are zero throughout and no qualifying labels were read, so the qual "RMSE" in the log is meaningless.

## Verified outputs
| phase | final state.json md5 (epoch 95) | tensor MD5s | snapshots | terminal native NPY sha256 | NPZ sha256 | prediction bytes sha256 |
|---|---|---|---|---|---|---|
| trainx→probex | cb61d09dd20ff8eeae35b50e77116a37 | 84/84 | 95/95 | 353ecdbf96a74ec36ee71d4b6c1454901da4ecf7c5ba74b2b14d02637c842edc | 06f13dc61fd9ba7cd43800acaee653fbf885cec5b83c2ce632684d25414eabe8 | 482cdc166d8d762b4c62fa42c5fa6a01db0f82b5b4f81ebc4966d7914ef3e34b |
| fulltrain→qual | 4de313c7dd363d22edfa8baed316ce46 | 84/84 | 95/95 | 9043511f13319acf2d4d6e8136afd7f06edee904481b6dd4c9a4f56497dd8f68 | affff77a1a8087c7254868385bde717d1e768ad0785fbca360c27dbb1343b72e | 879347b014767d6065a68c94ca9931224472731976fb02ceaaca205bd13b9ebb |

Checks: every snapshot is 1-D float32, finite, with the expected rows (1,128,164 probex / 2,817,131 qual). The terminal vector is byte-equal to ep95.
The NPZ is reloaded and byte-compared. The pre-resume snapshots (31 probex, 30 qual) still match the package manifest hashes.
Swa_count is 78 in both phases, which is consistent with swa_start 18.
Row-identity hashes of data/{probex,qual}/*.npy and all 95 per-epoch hashes are in `verification.json`.
No local blend selection was done; frozen-vector auditing is for the coordinator.
