# Lane 1 — CF-NADE residual neighbourhoods

Base: own CFNADE48. Training-row predictions were exported from its epoch-95 checkpoint, and the probex and qual replays
are byte-exact (plan amendment). Chains run in `artifacts/rcb-r1` (`lane1.sh`, tmux `cf-lanes:lane1`).

## Missing inputs for the preferred rank-96 base (own CF96)

None of the own CF96 vectors are on this machine. Only the upstream reference `cfnade-96*` vectors are here, and they
are excluded. To run the same four chains on CF96, upload to `remote-context-blend-20261124`:

| file (into `preds_new/`) | rows | phase | role |
|---|---|---|---|
| `cfnade-96.trainx.npy` | 99,352,343 | known (trainx→probex) | training-row predictions |
| `cfnade-96.probex.npy` | 1,128,164 | known | query predictions |
| `cfnade-96.fulltrain.npy` | 100,480,507 | qualifying (fulltrain→qual) | training-row predictions |
| `cfnade-96.qual.npy` | 2,817,131 | qualifying | query predictions |

All four must be float32 in canonical native row order, from the same terminal model (state the SWA terminal or
epoch), with sha256 for each file. Once they arrive, `lane1.sh` reruns with `cfnade-96` in place of `cfnade-48`.

## Result (2026-11-24, native held-out bank v2, 5 user folds, global ridge λ=1000)

| arm | standalone probex | pooled blend | Δ vs base | folds better |
|---|---|---|---|---|
| no-cf48 | — |  0.863574 | +0.000419 | 0/5 |
| base | 0.89496 |  0.863155 | +0.000000 | 0/5 |
| base+knn3 | 0.88265 |  0.863019 | -0.000136 | 5/5 |
| base+knn3d | 0.88171 |  0.863046 | -0.000109 | 5/5 |
| base+knn-25c | 0.92910 |  0.862888 | -0.000267 | 5/5 |
| base+knn-25m | 0.89158 |  0.862949 | -0.000206 | 5/5 |
| base+all | — |  0.862696 | -0.000459 | 5/5 |

qual sha256 (frozen, fit on all known rows):
* no-cf48: `c2ea087e404eac11d5def3f5b8a11e6ac20c8dd33191ae5fd55a58f4a51a125c`
* base: `3cb2f0f1f93ed27a7bd939025bcd89bb5095d84413f1cd3eb97faae3064ca033`
* base+knn3: `8ec45a4f72401dbeeb7dd10c763e2b3551393e084c3bdebcbc74971392b62e7e`
* base+knn3d: `776ea0f53bf4502c379256c7d04099545fef19e9fa90b5780e2cd738f72abbb4`
* base+knn-25c: `435275292806561a03d6172a7678065ded4aa05e200a7ce5697899eb2a8961e6`
* base+knn-25m: `6eb5232d9b3c1ef606604c6fc012abe002c28f7f113b40389288f7152dbe2ae2`
* base+all: `19fc659206d6fdbced7bbea7dba161a2423ba52386f342488875c62421d7cd73`

`knn-25c` is the worst chain standalone. The split-1 cosine scaling (2.5) does not suit CF48 residuals, yet it
adds the most in the blend. All four chains together add **−0.000459 (5/5 folds)** over CF48 alone.
Release asset `rcb-lane1-cf48-residual.tar.gz` (825,358,842 B, sha256 `a604c510…7c06`) holds:
* chain probex and qual vectors and native `.out` files;
* the CF48 trainx and fulltrain exports, with the export source, binary hash and logs;
* the blend known and qual vectors and coefficients for every arm;
* receipts and the sim-stat hashes (the 7.6 GB stats are reproducible and not uploaded).

Members are listed in `package-members.sha256`.
