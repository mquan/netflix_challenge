# Remote context-blend v1 results (known rows only)

Bank: 43 admitted columns (`bank.json`), 1,128,164 probex rows, original five user
folds. All fits inside training folds; no qualifying labels read.

| Arm (same rows) | Pooled RMSE | Fold 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|---|
| global without knn3/knn3d (41 cols) | 0.863431 | 0.86284 | 0.86084 | 0.86418 | 0.86501 | 0.86429 |
| global + knn3 control | 0.863387 | 0.86279 | 0.86081 | 0.86413 | 0.86497 | 0.86423 |
| global + knn3d | 0.863335 | 0.86275 | 0.86075 | 0.86408 | 0.86493 | 0.86418 |
| **global, all 43 (primary)** | **0.863340** | 0.86275 | 0.86076 | 0.86408 | 0.86493 | 0.86418 |
| context (all 43) | 0.862646 | 0.86194 | 0.86011 | 0.86341 | 0.86425 | 0.86352 |

* **Temporal residual:** knn3d adds −0.000096 to the global blend; the non-temporal
  knn3 control adds −0.000044, so the date kernel itself is worth ≈ −0.00005.
  Standalone probe_3: TX300r 0.87834, knn3 0.87607, knn3d 0.87596.
* **Context:** −0.000694 pooled, 5/5 folds improve (0.00065–0.00080). Below the
  frozen 0.001 bar → **global is primary**; context is frozen as the comparison.
* Inner λ_ctx selected 1e4, the **lowest grid value, in every fold** and in the final
  fit: the frozen grid is boundary-limited, so the context gain is likely
  understated. Any wider grid is a separately declared v2, not a re-run of v1.

Frozen qualifying vectors (2,817,131 rows), coefficients and held known vectors:
draft release `remote-context-blend-20261124`, asset `rcb-blend-v1.tar.gz`
(hashes in `blend/files.sha256`, `blend/archive.sha256`). Recovered audit pending
on the coordinator.

## v2 (declared after v1: λ_ctx grid {1e2…1e5} only)

| Arm | Pooled | Fold 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|---|
| global (unchanged) | 0.863340 | 0.86275 | 0.86076 | 0.86408 | 0.86493 | 0.86418 |
| context v2 | 0.862355 | 0.86157 | 0.85979 | 0.86323 | 0.86401 | 0.86318 |

Gain −0.000985 pooled, 5/5 folds (0.00085–0.00118): **just short of the 0.001 bar,
so global stays primary under the frozen rule.** λ_ctx again picked the grid floor
(1e2) everywhere: the interactions are effectively unshrunk. I stopped extending
the grid because each extension is another adaptive choice. The owner/coordinator
decides whether to admit context v2 as primary. Asset `rcb-blend-v2.tar.gz`.
