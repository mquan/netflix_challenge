# own17 rerun — global vs context v2 (2026-11-24)

Same `rcb_blend.py`, v2 grid, original 5 user folds, contexts from bank v2. **Known-row CV is non-nested.**
16 of the 17 own17 columns are user-fold OOF outputs of probex-fitted models (see the plan amendment).
Read these numbers as relative only. For reference, the coordinator's own non-nested simplex on own17
is 0.858643.

| bank | columns | global | context v2 | context − global | folds improved | λ_ctx (folds / final) |
|---|---|---|---|---|---|---|
| own17 | 17 | **0.858675** | 0.858692 | +0.000016 | 0/5 | 300, 1e4, 1e3, 1e3, 300 / 1e3 |
| own17x = own17 + bank v2 | 82 | **0.858340** | 0.858340 | +0.0000003 | 3/5 | 1e5, 3e4, 1e4, 1e4, 3e4 / 1e3 |

* **The context interactions add nothing on top of own17.** Many own17 columns are already context
  stacks (`*-mps-context`, selected-context neural), so the support/recency adjustment that paid
  0.001 on our raw bank is already absorbed there. The rule makes **global** primary on both banks.
* **Our bank v2 adds 0.000335 to own17** under the global blend (0.858675 → 0.858340). knn3/knn3d on
  tx300r alone add 0.000003, so the gain comes from the breadth of the bank, not the date kernel.
* Qual sha256: own17 global `2e61003e…5d33`, context `1ec6540e…4dc6`; own17x global `717f9afa…3927`,
  context `a632f5b9…8430`.

Release asset `rcb-own17-blends.tar.gz` (64,718,850 B, sha256 `032bf314…f877`) holds the known and
qual vectors, coefficients, summaries and bank manifests. Members: `own17-package-members.sha256`.

**Recommended for the coordinator's recovered audit:** own17x global (`blend-own17x/qual-global.f32`)
vs own17 global (`blend-own17/qual-global.f32`) as its matched control. The audit is the only nested
evidence. Context v2 on bank v1 stays the frozen owner-admitted primary for the bank-only study.
