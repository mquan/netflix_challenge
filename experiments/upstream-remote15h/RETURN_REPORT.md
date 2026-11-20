# Remote15h return report (final, 2026-11-20 ~17:05Z)

Arrival 03:44:19Z · training deadline 17:44:19Z · return deadline 18:44:19Z · M3 Max / 96 GiB, on AC with caffeinate.
Native roots: `artifacts/upstream-remote15h-r2` (assigned) and `artifacts/upstream-remote15h-ext` (extension).
Draft release: **`remote15h-own-20261120`** on mquan/netflix_challenge. No qualifying labels were read, no public
numerical predictions were used, and no blends or qualifying scores were fitted remotely.

## Checklist

- [x] **Assigned: 3/3 producers complete, 9/9 verified published pairs.** Both full original phases ran, the helper
      export verification passed, and every archive is uploaded.
- [x] Assigned full native outputs: a lossless archive in 2 parts, manifest `execution/native-archive-remote15h-assigned-native.json`.
- [x] **Extension (spare window, see `../upstream-remote15h-ext/README.md`): 4 more complete producers, 9 more
      verified pairs.** Same rules and deadline. Four queued jobs were not admitted for lack of time (below).
- [x] Extension full native outputs: a lossless archive in 2 parts, manifest `../upstream-remote15h-ext/execution/native-archive-remote15h-ext-native.json`.
- [x] Bootstrap attempt 1 failed at the BLAS link step. It is preserved and documented in `execution/bootstrap-attempt1/`.

| Producer | Set | Epochs/phase | Train time | Final probe_3 RMSE | Asset (bytes, sha256 prefix) | Pairs |
|---|---|---:|---:|---:|---|---|
| s3-dnn24 | assigned | 8 | 1h33m | 0.92276 | s3-dnn24-export.tar.gz (26,850,792, 9d51366d) | terminal, ep03 |
| s3-attn32 | assigned | 12 | 35m | 0.90871 | s3-attn32-export.tar.gz (26,735,250, 8193f136) | terminal, ep06 |
| s3-rbm200e | assigned | 25 | 3h32m | 0.90313 | s3-rbm200e-export.tar.gz (66,646,136, 8c12bf2e) | terminal, ep01/05/10/17 |
| s3-bknbrx-100w | ext | 12 | 26m | 0.90083 | s3-bknbrx-100w-export.tar.gz (26,649,612, daec6eb3) | terminal, ep01 |
| s3-mfrbmx-60h | ext | 12 | 14m | 0.94455 (min 0.91045 @ep06) | s3-mfrbmx-60h-export.tar.gz (40,225,184, 76909a97) | terminal, ep06, ep08 |
| s3-tx60o | ext | 25 | 3h37m | 0.88219 | s3-tx60o-export.tar.gz (40,006,710, cb3d43a4) | terminal, ep02, ep24 |
| s3-tx50n | ext | 22 | 3h06m | 0.88137 | s3-tx50n-export.tar.gz (13,429,643, 7739d603) | terminal |

Full archive hashes and per-pair native/payload hashes are in each `execution/<job>/transfer.json` and `verification.json`.
The probe_3 RMSE figures are diagnostics read from native logs, not an admission criterion. Snapshot choice is left
to the coordinator's nested CV. The recipe overfits mfrbmx-60h after epoch 6, and its published ep06/ep08 snapshots
are included.

**Not admitted** (each needs more reserved hours than remained at 16:53Z): s3-bk4-g100, s3-rbm100p, s3-bknbrx-200t,
s3-mfrbmx-50w. The extension claim on these is **released**; the coordinator may queue them.

**Retained on the remote** (nothing deleted): both native roots, including preds_3, logs, receipts and binaries; the
exports; the per-job archives; and the native archive parts under `artifacts/native-archives/`. Everything in the
native archives and exports is uploaded. The generated train_3/probe_3 rating arrays and the linked raw data were
intentionally not transferred.

**Gotcha for future macOS handoffs:** the helper's `environment()` drops `RUSTFLAGS`, so the Accelerate link flags
must come from a `.cargo/config.toml` in the native tree's parent directory.

## Local cleanup (2026-11-20, at the user's request; disk nearly full)

Deleted local copies only after checking each against the release asset by name and byte size:
`artifacts/native-archives/` (4 parts), and the per-job `exports/` and `archives/` under both native roots.
All of them are recoverable from draft release `remote15h-own-20261120`. The native roots (preds_3, logs, receipts,
binaries) and the bootstrap attempt-1 root are kept.
