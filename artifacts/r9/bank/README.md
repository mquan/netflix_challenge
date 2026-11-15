# R9-LAT-A2h blend input bank (79 columns)

Self-contained copy (no symlinks) of the exact inputs to the promoted
R9-LAT-A2h ensemble (0.8668706425 dev CV), materialized from
`artifacts/structural/S0-scoping-20261031/banks/all79`, which S0 verified
reproduces R9 with zero prediction difference
(`experiments/S0-full-parity-verification-20261031.json`).

- `columns/000.f32`–`078.f32`: the 79 prediction columns, in blend order.
- `support/000.f32`–`045.f32`: 46 support / meta / latent features.
- `experiments/UNION-round-manifest.json`, `BOOST-support-inputs.json`:
  the manifests `round4_boost.py` reads from `--bank-root`. Keys were
  rewritten from absolute S0 paths to bank-relative paths (`columns/000.f32`);
  order and hashes are unchanged.
- `artifacts/r4/meta-context.npz` (tracked separately, 90 MB): meta/context
  feature cache both blend scripts read; `round8_nn_blend.py` has no
  fallback. SHA-256 `cfbb5b1e…3b0c`, as recorded by S0.
- `MANIFEST.sha256`: SHA-256 of every file; each hash matches an input
  recorded in `artifacts/r9/R9-LAT-A2h-20261010-s3.json`.

Each `.f32` is float32, one value per row of `data/processed/netflix.bin`
(1,408,395 rows). `netflix.bin` itself is not in git; build or copy it
separately on a new machine.

## Reproduce the R9 blend

```sh
python scripts/round4_boost.py --data data/processed/netflix.bin \
  --output <out>/all79-s3 --bank-root artifacts/r9/bank \
  --extra --extra-support --rounds 8000 --learning-rate 0.02 \
  --leaves 63 --min-leaf 200 --processes 3 --seeds 2026 7 13 --extra-meta

python scripts/round8_nn_blend.py --data data/processed/netflix.bin \
  --like <out>/all79-s3.json --output <out>/all79-NN2 \
  --hidden 128 --layers 2 --dropout 0.2 --lr 0.0003 --weight-decay 0.01 \
  --batch 4096 --epochs 60 --patience 5 --seeds 2026 7 13 --threads 2
```

The final ensemble is `<out>/all79-NN2.avg.f32` (50/50 tree + NN). To test a
new column, append it with `--extra <file.f32>` on the tree stage.
