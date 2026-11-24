# Remote context blend — wave, bank v2 and tree arm (2026-11-24)

All numbers are known probex rows (1,128,164), out-of-fold on the original 5 user
folds (seed 20261116). No qualifying or recovered labels were read. Bank v2 and the
tree arm were declared after v1/v2 results were seen, so everything here is
**adaptive known-row evidence**; the owner-admitted primary (context v2 on bank v1,
`primary.json`) is unchanged.

## 1. knn3d wave — standalone RMSE, knn3d vs matched knn3

| base | split | knn3 | knn3d | Δ |
|---|---|---|---|---|
| s3-tx300r (first run) | probe_3 | 0.87607 | 0.87596 | −0.00011 |
| s3-tx50n | probe_3 | 0.87875 | 0.87863 | −0.00012 |
| s3-tx60o | probe_3 | 0.87738 | 0.87697 | −0.00041 |
| s3-rbm800p | probe_3 | 0.90041 | 0.89998 | −0.00043 |
| s3-dnn24 | probe_3 | 0.91425 | 0.91377 | −0.00048 |
| s3-attn32 | probe_3 | 0.89862 | 0.89788 | −0.00074 |
| s3-rbm200e | probe_3 | 0.89147 | 0.89073 | −0.00074 |
| s3-bknbrx-100w | probe_3 | 0.89451 | 0.89503 | **+0.00052** |
| s3-mfrbmx-60h | probe_3 | 0.93226 | 0.93181 | −0.00045 |
| s3-rbm400pf (LR 0.015) | probe_3 | 0.89210 | 0.89120 | −0.00090 |
| tsvdx4-60 | probex | 0.87482 | 0.87468 | −0.00014 |
| tsvdx5-150o | probex | 0.87306 | 0.87287 | −0.00019 |

The date kernel helps on 11/12 bases, most on non-temporal bases (RBMs, attention).
It hurts only on bknbrx-100w, which is itself a neighbourhood model. 22 jobs, 0 non-zero exits.
Packages: release asset `rcb-knn3d-wave.tar.gz` (818,104,142 B, sha256
`4784440bbc17fd48bd2c36909ed81e50986c21e58bf37e8e533760874ad0d5b8`). It holds per-job
native files, canonical probex/qual npz, logs, receipts and `index.json`. Member hashes are in
`knn3d/wave/package-members.sha256`.

## 2. Bank v2 (65 columns = bank v1 43 + 22 wave chains) — unchanged v2 blends

| arm | bank v1 (v2 grid) | bank v2 | Δ |
|---|---|---|---|
| global ridge λ=1000 | 0.863340 | 0.863155 | −0.000185 |
| context v2 | 0.862355 | 0.862197 | −0.000158 |
| context − global | −0.000985 (5/5) | −0.000958 (5/5) | |

Per-fold context gains on bank v2: 0.00116, 0.00093, 0.00081, 0.00092, 0.00096. λ_ctx = 1e2
(grid floor again) in all 5 folds and the final fit; the inner score still falls as λ falls, so
the shrinkage grid has not found its optimum. Summary: `blend-bank2/summary.json`.

## 3. Local CatBoost tree-residual arm on bank v2 (declared rule: replaces primary iff ≥0.001 and ≥4/5 vs context)

| fold | global | context | tree | tree − context |
|---|---|---|---|---|
| 0 | 0.862570 | 0.861414 | 0.861027 | −0.000387 |
| 1 | 0.860546 | 0.859612 | 0.859201 | −0.000411 |
| 2 | 0.863908 | 0.863093 | 0.862582 | −0.000511 |
| 3 | 0.864826 | 0.863902 | 0.863531 | −0.000371 |
| 4 | 0.863929 | 0.862968 | 0.862622 | −0.000346 |
| pooled | 0.863155 | 0.862197 | **0.861792** | −0.000405 |

The tree beats context on 5/5 folds but by only 0.00041 pooled, so **it does not replace the
primary**. Every slot early-stopped between 730 and 1017 iterations (inner gain ≈0.001 over the
ridge baseline). Receipts: `tree-bank2/*.json`, `tree-bank2/combined-summary.json`.

The three `final` (all-known → qual) units first crashed on a code bug: the ridge baseline indexed
the known matrix instead of the qual matrix (`f66e542`). For fold units the two are the same
matrix, so fold results are unaffected. The crashed outputs are kept in
`artifacts/tree-bank2/crashed-final-v1`. Only the 3 final units were rerun: they selected 785, 1093
and 991 iterations. The qual tree vector has sha256 `cf5404f5…30dc`, is finite, and is clipped to
[1.05, 4.95]. Its RMS distance from qual context-bank2 is 0.0462, against 0.0475 on known rows.

Release asset `rcb-bank2-blend-tree.tar.gz` (201,454,793 B, sha256 `d99cf925…f352a`) holds the
bank-v2 global/context known and qual vectors and coefficients, all tree units (models, corrections,
receipts) and the combined known/qual tree vectors. Members: `tree-bank2/package-members.sha256`.
`primary.json` lists these as `adaptive_followups`.

## 4. Standing

Best known-row candidate here is tree on bank v2 (0.861792), then context v2 on bank v2 (0.862197),
then the admitted primary (0.862355). Mapping to the coordinator's target (below 0.85725 on both
recovered groups) needs the coordinator's recovered audit; it cannot be done on this machine.
