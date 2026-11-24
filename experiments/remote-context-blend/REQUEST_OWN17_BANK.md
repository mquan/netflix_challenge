# Request to coordinator: own17 bank for the context-blend rerun (owner-approved 2026-11-24)

The owner approved rerunning global vs context v2 on the coordinator's own17 bank.
This machine only has own17 metadata, not the arrays. Please upload to draft
release `remote-context-blend-20261124` (asset prefix `own17-bank-`):

1. `own17-bank-known.npy`: float32 [1,128,164, K], canonical probex row order
   (fresh `data/probex`, ratings sha prefix `5a554ca04854b25a`), every own17 input
   column that the own17 simplex/ridge uses, **as produced without probex outcomes**
   (native holdout vectors, or fold-specific OOF where the column is a learned
   second stage — note which, and the fold rule, per column).
2. `own17-bank-qual.npy`: float32 [2,817,131, K], native qual order, same columns.
3. `own17-bank-columns.json`: column names/families in order, per-column SHA-256
   of both vectors, provenance kind (`native-holdout` | `user-fold-oof` + fold seed),
   plus the own17 simplex known RMSE for reference.
4. Parts ≤ 2 GiB each with byte count + SHA-256 in the JSON.

Known labels, users and contexts are recomputed here from the fresh data tree;
no recovered outcomes are needed or wanted. On arrival this machine runs the
frozen `scripts/rcb_blend.py` (global λ=1000; context v2 grid {1e2…1e5}) unchanged,
plus `own17 + knn3d` as the incremental column test, and returns frozen
qualifying vectors for both candidates.
