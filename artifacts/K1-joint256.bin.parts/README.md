# K1-joint256.bin (split for GitHub)

Original K1 checkpoint (`artifacts/K1-joint256.bin`, 1,173,446,408 bytes,
trained on the full training data). It is the feature encoder for the R9
latent features (`scripts/round9_latent_features.py --model` default;
PRODUCTION_READINESS.md requires this original checkpoint). Split into
95 MiB parts because GitHub rejects files over 100 MB.

Reassemble and verify (one step): `sh scripts/restore_k1_checkpoint.sh`.
Equivalent manual commands from the repo root:

```sh
cat artifacts/K1-joint256.bin.parts/K1-joint256.bin.part-* > artifacts/K1-joint256.bin
(cd artifacts && shasum -a 256 -c K1-joint256.bin.parts/K1-joint256.bin.sha256)
```

Expected: `K1-joint256.bin: OK` (SHA-256 `e4dc4609…6fb6d7`). Its prediction
column `artifacts/K1-joint256.f32` is tracked separately.
