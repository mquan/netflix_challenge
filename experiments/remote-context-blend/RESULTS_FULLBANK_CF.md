# Own17x full-bank refit with CF-NADE residual columns (2026-11-24)

Global ridge λ=1000, 5 user folds (seed 20261116), clip [0,6] → [1.05,4.95]. **Own17x CV is non-nested.** 16 of its
columns are probex-fitted OOF stacks, so the native bank v2 companion (N) is the held-out evidence. No duplicate
columns were found. The selection rule was declared before fitting (plan amendment).

Standalone probex: CF96 0.89327; chains knn3 0.88158, knn3d 0.88062, knn-25c 0.93023, knn-25m 0.89077 (CF48 base 0.89496).

| arm | added | cols | Own17x pooled | Δ vs X0 | folds | native pooled | Δ vs N0 | folds | qual sha256 |
|---|---|---|---|---|---|---|---|---|---|
| X0/N0 | — (control) | 82 | 0.858340 | +0.000000 | 0/5 | 0.863155 | +0.000000 | 0/5 | `717f9afab2901055…` |
| X1/N1 | 4 CF48 chains | 86 | 0.858160 | -0.000180 | 5/5 | 0.862696 | -0.000459 | 5/5 | `1737739bced57740…` |
| X2/N2 | CF96 | 83 | 0.858331 | -0.000009 | 4/5 | 0.863067 | -0.000089 | 5/5 | `b73c5ee716f860ad…` |
| X3/N3 | CF96 + 4 CF96 chains | 87 | 0.858130 | -0.000209 | 5/5 | 0.862591 | -0.000564 | 5/5 | `4b11dff09d234faf…` |
| X4/N4 | CF96 + 4 CF96 + 4 CF48 chains | 91 | 0.858131 | -0.000209 | 5/5 | 0.862583 | -0.000572 | 5/5 | `b57a34d0e6418bd9…` |

**Selected (known-only rule): X3**, qual sha256 `4b11dff09d234faf3b8ffb94eea72fe6ebffbbba87f015442a7c009ce1f453db`. The matched control is X0,
`717f9afab2901055331f1865ea3cef8bcd9998ed5a6e6d7d67a59ac296243927`, which reproduces the returned Own17x global exactly. X4 ties X3 on Own17x to about 1e-6: the CF48
chains add nothing once the CF96 chains are in, though they still add 0.000008 on the native bank.

Release asset `rcb-own17x-fullbank-cf.tar.gz` (186,503,034 B, sha256 `bf0c5983…df65`) holds, for every arm X0–X4 and
N0–N4, the known and qual vectors and fixed coefficients, plus the CF96 chain probex/qual vectors, `.out` files, logs
and script. Members: `fullbank-cf/package-members.sha256`; native-vector qual sha256s are in `fullbank-cf/summary.json`.
